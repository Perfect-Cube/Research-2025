import os
import tempfile
import threading
import warnings
import torch
import numpy as np
import PyPDF2
import pypdfium2 as pdfium
import nltk
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langdetect import detect, DetectorFactory
import gradio as gr

# Setup reproducible language detection and download nltk data
DetectorFactory.seed = 0
nltk.download('punkt')

# ------------------ Global Setup ------------------ #
qdrant_client = QdrantClient(path="qdrant_storage")
collection_name = "pdf_embeddings"
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
embedding_dim = embedding_model.get_sentence_embedding_dimension()

def ensure_qdrant_collection():
    collections = [col.name for col in qdrant_client.get_collections().collections]
    if collection_name not in collections:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        print(f"Created collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' already exists.")

ensure_qdrant_collection()

# Global list to hold PDF file paths (for mapping document name to file path)
pdf_files = []

# ------------------ Translation Function ------------------ #
def translate_german_to_english(text):
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# ------------------ PDF Processing and Search Functions ------------------ #
def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    page_entries = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            sentences = nltk.sent_tokenize(page_text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    try:
                        lang = detect(sentence)
                    except Exception:
                        lang = "unknown"
                    page_entries.append({
                        "text": sentence,
                        "page": page_number,  # 1-indexed for clarity in UI
                        "sentence_id": i,
                        "language": lang
                    })
    return page_entries

def store_pdf_embeddings(pdf_path, context_window=1):
    doc_name = os.path.basename(pdf_path)
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    filter_condition = Filter(
        must=[FieldCondition(key="document", match=MatchValue(value=doc_name))]
    )
    count_result = qdrant_client.count(collection_name=collection_name, count_filter=filter_condition)
    if count_result.count > 0:
        print(f"Embeddings for {doc_name} already exist. Skipping.")
        return
    pdf_entries = extract_text_from_pdf(pdf_path)
    texts = [entry["text"] for entry in pdf_entries]
    embeddings = embedding_model.encode(texts, batch_size=32).tolist()
    entries_by_page = {}
    for entry in pdf_entries:
        page = entry["page"]
        entries_by_page.setdefault(page, []).append(entry)
    points = []
    for i, entry in enumerate(pdf_entries):
        page = entry["page"]
        sid = entry["sentence_id"]
        page_entries = sorted(entries_by_page[page], key=lambda x: x["sentence_id"])
        context_entries = [e for e in page_entries if abs(e["sentence_id"] - sid) <= context_window]
        full_text = " ".join([e["text"] for e in context_entries])
        uid = f"{doc_name}_{page}_{sid}"
        points.append(PointStruct(
            id=hash(uid),
            vector=embeddings[i],
            payload={
                "text": entry["text"],
                "full_text": full_text,
                "page": page,
                "document": doc_name,
                "language": entry.get("language", "unknown"),
                "sentence_id": sid
            }
        ))
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Stored embeddings for {doc_name} in Qdrant.")

def hybrid_search(query, top_k=3):
    # Compute dense embedding for the query.
    try:
        detect(query)
    except Exception:
        pass
    query_embedding = embedding_model.encode([query]).tolist()[0]
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k * 2,
        with_payload=True
    )
    candidate_texts = [hit.payload.get("full_text", hit.payload.get("text", "")) for hit in search_results if hit.payload]
    tokenized_docs = [word_tokenize(text.lower()) for text in candidate_texts]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    dense_scores = np.array([hit.score for hit in search_results])
    max_dense = max(dense_scores) if max(dense_scores) > 0 else 1
    dense_scores = dense_scores / max_dense
    max_sparse = max(sparse_scores) if max(sparse_scores) > 0 else 1
    sparse_scores = sparse_scores / max_sparse
    hybrid_scores = 0.7 * dense_scores + 0.3 * sparse_scores
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    best_results = []
    for i in sorted_indices[:top_k]:
        hit = search_results[i]
        best_results.append({
            "text": hit.payload.get("full_text", hit.payload.get("text", "")),
            "page": hit.payload["page"],
            "document": hit.payload["document"],
            "score": hybrid_scores[i]
        })
    return best_results

def hybrid_search_dual(query, top_k=5):
    """
    Dual retrieval:
      - If the query is in German, translate it to English.
      - Run hybrid search on both the original and translated query.
      - Merge and re-rank the results.
    """
    try:
        query_lang = detect(query)
    except Exception:
        query_lang = "en"
    if query_lang == "de":
        query_en = translate_german_to_english(query)
        query_de = query
    else:
        query_en = query
        query_de = query

    results_de = hybrid_search(query_de, top_k=top_k)
    results_en = hybrid_search(query_en, top_k=top_k)

    combined = {}
    for res in results_de + results_en:
        key = (res["document"], res["page"])
        if key in combined:
            combined[key]["score"] = (combined[key]["score"] + res["score"]) / 2
        else:
            combined[key] = res

    combined_results = list(combined.values())
    combined_results.sort(key=lambda x: x["score"], reverse=True)
    return combined_results[:top_k]

def get_total_pages(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    return len(reader.pages)

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_number)
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    if output_path:
        pil_image.save(output_path)
        print(f"Saved rendered page {page_number + 1} to: {output_path}")
    page.close()
    pdf.close()
    return pil_image

# Updated: Save the rendered image and return its file path.
def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page {page_number + 1} out of range (Total: {total_pages}).")
        return None
    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{doc_name}_page_{page_number + 1}.png")
    render_page(pdf_path, page_number, scale=scale, output_path=output_path)
    return output_path

# Helper: Given a document name, return the corresponding file path from pdf_files.
def get_pdf_path(doc_name):
    for path in pdf_files:
        if os.path.basename(path) == doc_name:
            return path
    return None

# ------------------ Generation Functions ------------------ #
warnings.filterwarnings("ignore", category=DeprecationWarning)
model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to("cpu")

def generate_text(prompt, search_results):
    """Generate text using the additional prompt and search results (combined text).
       Sets the system prompt based on the language of the prompt.
    """
    try:
        prompt_lang = detect(prompt)
    except Exception:
        prompt_lang = "en"
    if prompt_lang == "de":
        system_prompt = "You are an AI agent of Volkswagen. Give detailed answers using retrieved text only. Respond in German."
    else:
        system_prompt = "You are an AI agent of Volkswagen. Give detailed answers using retrieved text only. Respond in English."
    
    combined_text = ""
    if search_results:
        combined_text = "\n".join([res["text"] for res in search_results])
    combined_prompt = f"{prompt}\n{combined_text}"
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": combined_prompt}]
        }
    ]
    inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to("cpu")
    output = model.generate(**inputs, max_new_tokens=900)
    gen_text = processor.decode(output[0], skip_special_tokens=True)
    return gen_text

def generate_text_from_image(prompt, image):
    """Generate text from a given image and user prompt.
       Sets the system prompt based on the language of the prompt.
    """
    try:
        prompt_lang = detect(prompt)
    except Exception:
        prompt_lang = "en"
    if prompt_lang == "de":
        system_prompt = "Respond in German. Give detailed answers using the retrieved image and text only."
    else:
        system_prompt = "Respond in English.Give detailed answers using the retrieved image and text only."
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to("cpu")
    output = model.generate(**inputs, max_new_tokens=900)
    gen_text = processor.decode(output[0], skip_special_tokens=True)
    return gen_text

# This combined function checks the image option.
def generate_text_final(prompt, image_option, gallery, search_results):
    if image_option.lower() == "yes" and gallery:
        selected = gallery[0] if isinstance(gallery, list) and len(gallery) > 0 else None
        if selected is not None:
            os_path = os.path.abspath(selected).replace("\\", "/")
            print("Selected OS image path:", os_path)
            return generate_text_from_image(prompt, os_path)
    return generate_text(prompt, search_results)

# ------------------ New Functions for Image Preview ------------------ #
def preview_images(search_results):
    image_list = []
    for res in search_results:
        doc_name = res['document']
        page = res['page']
        pdf_path = get_pdf_path(doc_name)
        if pdf_path:
            image_path = process_specific_page(pdf_path, page - 1, scale=7.5)
            if image_path:
                image_list.append(image_path)
    return image_list

def show_images_option(option, search_results):
    if option.lower() == "yes" and search_results:
        return preview_images(search_results)
    return []

# ------------------ Gradio Interface ------------------ #
def process_upload(file_paths):
    status_messages = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        status_messages.append(f"Processing {filename} ...")
        pdf_files.append(file_path)
        store_pdf_embeddings(file_path, context_window=1)
        status_messages.append(f"Finished processing {filename}")
    return "\n".join(status_messages)

def perform_search(query):
    results = hybrid_search_dual(query, top_k=5)
    if not results:
        return "No results found.", results
    output_lines = []
    for res in results:
        output_lines.append(f"Document: {res['document']} | Page: {res['page']} | Score: {res['score']:.4f}\nText: {res['text']}\n" + "-"*40)
    return "\n".join(output_lines), results

with gr.Blocks(theme="default") as demo:
    gr.Markdown("# PDF Embeddings & Text Generation App")
    with gr.Tabs():
        with gr.Tab("Upload & Process PDFs"):
            gr.Markdown("### Upload one or more PDF files to process.")
            pdf_input = gr.File(label="Select PDF Files", file_count="multiple", type="filepath")
            upload_btn = gr.Button("Upload and Process")
            upload_status = gr.Textbox(label="Status", lines=10)
            upload_btn.click(fn=process_upload, inputs=pdf_input, outputs=upload_status)
            
        with gr.Tab("Search & Generate"):
            gr.Markdown("### Search PDFs and Generate Text")
            query_input = gr.Textbox(label="Enter Search Query")
            search_btn = gr.Button("Search")
            search_output = gr.Textbox(label="Search Results", lines=10)
            search_state = gr.State([])
            search_btn.click(fn=perform_search, inputs=query_input, outputs=[search_output, search_state])
            
            gr.Markdown("### Optional: View Page Images for Further Generation")
            image_option = gr.Radio(choices=["Yes", "No"], label="Do you want to view page images?", value="No")
            show_images_btn = gr.Button("Show Images")
            gallery = gr.Gallery(label="Select an Image", show_label=True, columns=3, height="auto")
            show_images_btn.click(fn=show_images_option, inputs=[image_option, search_state], outputs=gallery)
            
            gr.Markdown("### Provide Prompt and Generate Text")
            prompt_input = gr.Textbox(label="Enter Prompt", lines=4)
            generate_btn = gr.Button("Generate Text")
            gen_output = gr.Textbox(label="Generated Text", lines=10)
            generate_btn.click(fn=generate_text_final, inputs=[prompt_input, image_option, gallery, search_state], outputs=gen_output)
            
demo.launch()
