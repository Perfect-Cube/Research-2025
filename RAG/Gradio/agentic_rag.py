import os
import warnings
import json
import torch
import numpy as np
import PyPDF2
import pypdfium2 as pdfium
import nltk
from PIL import Image
from transformers import pipeline, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langdetect import detect, DetectorFactory
import gradio as gr
from duckduckgo_search import DDGS

# Ensure reproducible language detection
DetectorFactory.seed = 0

# Global list to record rendered image paths
img_list = []

# Download NLTK data if not already available.
nltk.download('punkt_tab')
nltk.download('punkt')

# Use Qdrant persistent storage (all Qdrant and embedding work on CPU)
qdrant_client = QdrantClient(path="qdrant_storage1")
collection_name = "pdf_embeddings1"

# Load the embedding model on CPU
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
embedding_dim = embedding_model.get_sentence_embedding_dimension()

def ensure_qdrant_collection():
    """Ensure the Qdrant collection exists with the correct configuration."""
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

def translate_german_to_english(text):
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)  # Loaded on CPU
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of the PDF and returns a list of entries with text, page number,
    sentence ID, and detected language.
    """
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
                        "page": page_number,
                        "sentence_id": i,
                        "language": lang
                    })
    return page_entries

def store_pdf_embeddings(pdf_path, context_window=5):
    """
    Extracts text, generates embeddings, and stores them in Qdrant.
    Combines neighboring sentences (based on context_window) for extra context.
    Skips processing if embeddings for this PDF are already stored.
    """
    doc_name = os.path.basename(pdf_path)

    # Check if embeddings for this document already exist.
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

    # Build a lookup for entries by page.
    entries_by_page = {}
    for entry in pdf_entries:
        page = entry["page"]
        entries_by_page.setdefault(page, []).append(entry)

    points = []
    for i, entry in enumerate(pdf_entries):
        page = entry["page"]
        sid = entry["sentence_id"]
        # Retrieve neighboring sentences based on context_window.
        page_entries_sorted = sorted(entries_by_page[page], key=lambda x: x["sentence_id"])
        context_entries = [e for e in page_entries_sorted if abs(e["sentence_id"] - sid) <= context_window]
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
    print(f"Stored embeddings for {doc_name}.")

def hybrid_search(query, top_k=3):
    """
    Performs hybrid search combining dense retrieval (from Qdrant) and sparse retrieval (using BM25).
    """
    try:
        _ = detect(query)
    except Exception:
        pass

    query_embedding = embedding_model.encode([query]).tolist()[0]

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k * 10,
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
    seen = set()
    for i in sorted_indices:
        hit = search_results[i]
        doc_page_pair = (hit.payload["document"], hit.payload["page"])
        if doc_page_pair in seen:
            continue
        seen.add(doc_page_pair)
        best_results.append({
            "text": hit.payload.get("full_text", hit.payload.get("text", "")),
            "page": hit.payload["page"],
            "document": hit.payload["document"],
            "score": hybrid_scores[i]
        })
        if len(best_results) >= top_k:
            break
    return best_results

def hybrid_search_dual(query, top_k=5):
    """
    If the query is in German, translates it to English.
    Runs hybrid search on both the original and translated query.
    Merges and re-ranks the results.
    """
    try:
        query_lang = detect(query)
    except Exception:
        query_lang = None

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

# PDF Rendering Functions
def get_total_pages(pdf_path):
    """Return total pages in the PDF."""
    reader = PyPDF2.PdfReader(pdf_path)
    return len(reader.pages)

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    """
    Render a specified page using pypdfium2.
    Saves the image if output_path is provided.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_number)
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    if output_path:
        pil_image.save(output_path)
        print(f"Saved rendered page {page_number + 1} to {output_path}")
    page.close()
    pdf.close()
    return pil_image

def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
    """
    Render a specific page and record the image path.
    """
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} out of range. Total pages: {total_pages}")
        return None
    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{doc_name}_page_{page_number + 1}.png")
    # Save the absolute path in global img_list.
    abs_path = os.path.abspath(output_path).replace('\\', '/')
    if abs_path not in img_list:
        img_list.append(abs_path)
    return render_page(pdf_path, page_number, scale=scale, output_path=output_path)

def get_rendered_image_paths(rendered_dir="rendered_page"):
    """
    Retrieve all rendered image file paths from the rendered directory.
    """
    image_paths = []
    for root, _, files in os.walk(rendered_dir):
        for file in files:
            if file.lower().endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_pdf_paths(root_folder):
    """
    Recursively find all PDF files in a root folder.
    """
    pdf_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(subdir, file))
    return pdf_paths

# --- Setup the Gemma pipeline using the Gemma 3B model ---
pipe = pipeline("image-text-to-text", model="google/gemma-3-4b-it", device="cuda", torch_dtype=torch.bfloat16)

# Set the folder containing your PDFs (change as needed)
root_directory = "/content/paper"
pdf_files = get_pdf_paths(root_directory)
for pdf_file in pdf_files:
    store_pdf_embeddings(pdf_file, context_window=5)

# --- Simple Agent for External Web Search ---
class SimpleAgent:
    def __init__(self, tools: dict):
        self.tools = tools

    def call_tool(self, tool_name: str, input_str: str) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return tool(input_str)

def duckduckgo_search_tool(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
    formatted = ""
    for res in results:
        title = res.get("title", "No Title")
        url = res.get("href", "No URL")
        snippet = res.get("body", "No snippet available")
        formatted += f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n\n"
    return formatted.strip()

# Create an agent with only the web search tool.
agent = SimpleAgent(tools={
    "web_search": duckduckgo_search_tool,
})

# --- Gradio UI Functions ---
def search_query(query):
    """
    Takes a search query, performs dual-hybrid search on PDFs, runs web search for additional context,
    renders images for retrieved pages, and returns a summary string, a list of PIL images, and a JSON of best results.
    """
    global img_list
    img_list.clear()
    # Retrieve PDF results using hybrid search.
    best_results = hybrid_search_dual(query, top_k=5)
    # Perform external web search using DuckDuckGo.
    web_search_results = agent.call_tool("web_search", query)
    
    if not best_results:
        return "No PDF results found.", [], json.dumps({"pdf_results": best_results, "web_results": web_search_results})
    
    results_text = "PDF Results:\n"
    for res in best_results:
        results_text += f"Document: {res['document']}, Page: {res['page']}\n"
        results_text += f"Score: {res['score']:.4f}\n"
        results_text += f"Text: {res['text']}\n"
        results_text += "-"*50 + "\n"
    
    results_text += "\nWeb Search Results:\n" + web_search_results

    images = []
    for res in best_results:
        pdf_path = next((path for path in pdf_files if os.path.basename(path) == res["document"]), None)
        if pdf_path:
            pil_img = process_specific_page(pdf_path, res["page"] - 1, scale=10.0)
            if pil_img:
                images.append(pil_img)
    # Return the combined results in a hidden JSON for later use.
    combined_results = {"pdf_results": best_results, "web_results": web_search_results}
    return results_text, images, json.dumps(combined_results)

def generate_text_response(additional_prompt, best_results_json, selected_index):
    """
    Builds a combined prompt from additional text (if any), PDF and web search context,
    and directly passes it to the Gemma 3B model for text generation.
    """
    best_results = json.loads(best_results_json)
    pdf_texts = "\n".join([res["text"] for res in best_results.get("pdf_results", [])])
    web_text = best_results.get("web_results", "")
    
    # Build combined prompt
    combined_context = f"PDF Context:\n{pdf_texts}\n\nWeb Search Context:\n{web_text}"
    if additional_prompt:
        combined_prompt = additional_prompt + "\n" + combined_context
    else:
        combined_prompt = combined_context

    system_prompt_text = ("Using the retrieved PDF context and web search results only. "
                          "Respond in English as an advanced technical agent.")
    
    # Optionally include an image if the user selects one.
    try:
        idx = int(selected_index)
    except Exception:
        idx = 0
    if 0 <= idx < len(img_list):
        selected_image = img_list[idx]
    else:
        selected_image = ""

    if selected_image:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
            {"role": "user", "content": [
                {"type": "image", "url": selected_image},
                {"type": "text", "text": combined_prompt},
            ]},
        ]
    else:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
            {"role": "user", "content": [{"type": "text", "text": combined_prompt}]},
        ]
    
    # Directly pass the combined data to the Gemma model.
    output = pipe(text=conversation, max_new_tokens=1800)
    assistant_response = ""
    for message in output[0]['generated_text']:
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant_response += message.get("content", "")
    return assistant_response

# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Integrated PDF and Web Search with LLM Query Interface")
    gr.Markdown("### Step 1: Enter your search query")
    with gr.Row():
        query_input = gr.Textbox(label="Search Query", placeholder="Enter your query here")
        search_btn = gr.Button("Search")
    search_results = gr.Textbox(label="Combined Search Results", lines=10)
    gr.Markdown("### Retrieved Page Images")
    gallery = gr.Gallery(label="Rendered Images", height="auto")

    gr.Markdown("### Step 2: Generate Text Response")
    gr.Markdown("Enter an additional prompt and select a rendered image by its index (starting at 0).")
    additional_prompt_input = gr.Textbox(label="Additional Prompt", placeholder="Enter additional text prompt here")
    selected_image_index = gr.Number(label="Image Index", value=0)
    hidden_best_results = gr.Textbox(label="Hidden Best Results", visible=False)
    generate_btn = gr.Button("Generate Response")
    generated_text = gr.Textbox(label="Generated Text", lines=10)

    # When search is clicked, update search results, gallery, and hidden best results.
    search_btn.click(
        fn=search_query,
        inputs=query_input,
        outputs=[search_results, gallery, hidden_best_results]
    )

    # When generate is clicked, generate a text response using the additional prompt,
    # the hidden best results JSON, and the image index.
    generate_btn.click(
        fn=generate_text_response,
        inputs=[additional_prompt_input, hidden_best_results, selected_image_index],
        outputs=generated_text
    )

demo.launch()
