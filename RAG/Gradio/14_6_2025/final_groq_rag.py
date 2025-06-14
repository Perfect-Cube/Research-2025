import os
import warnings
import json
import torch
import numpy as np
import PyPDF2
import pypdfium2 as pdfium
import nltk
from PIL import Image, UnidentifiedImageError
import base64
from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langdetect import detect, DetectorFactory
import gradio as gr
import io  # <-- Added for in-memory image handling
import math # <-- Added for resizing calculation

# --- 1. SETUP GROQ CLIENT ---
# IMPORTANT: You must set your Groq API key as an environment variable.
try:
    groq_api_key = "key"
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=groq_api_key)
    print("Groq client initialized successfully.")
except (ValueError, ImportError) as e:
    print(f"Error initializing Groq client: {e}")
    client = None

# Ensure reproducible language detection
DetectorFactory.seed = 0

# Global list to record rendered image paths
img_list = []

# Download NLTK data if not already available.
nltk.download('punkt_tab')

# Use Qdrant persistent storage (all Qdrant and embedding work on CPU)
qdrant_client = QdrantClient(path="qdrant_storage1")
collection_name = "pdf_embeddings1"

# Load the embedding model on CPU
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
embedding_dim = embedding_model.get_sentence_embedding_dimension()

# --- 2. NEW AND MODIFIED HELPER FUNCTIONS ---

def resize_image_if_needed(img: Image.Image, max_pixels: int = 33177600) -> Image.Image:
    """
    Resizes a PIL image only if its total number of pixels exceeds the max_pixels limit.
    Maintains the aspect ratio.
    """
    current_pixels = img.width * img.height
    if current_pixels > max_pixels:
        print(f"Image is too large ({current_pixels} pixels). Resizing to fit under {max_pixels} pixels.")
        # Calculate the resize ratio
        ratio = math.sqrt(max_pixels / current_pixels)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        # Use LANCZOS for high-quality downsampling
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Resized image to {resized_img.width}x{resized_img.height} ({resized_img.width * resized_img.height} pixels).")
        return resized_img
    return img

def encode_pil_image_to_base64(img: Image.Image) -> str:
    """
    Encodes a PIL image object to a Base64-encoded data URI.
    """
    with io.BytesIO() as buffer:
        img.save(buffer, format="PNG") # Save image to a memory buffer
        encoded_bytes = base64.b64encode(buffer.getvalue())
    return f"data:image/png;base64,{encoded_bytes.decode('utf-8')}"


def ensure_qdrant_collection():
    try:
        collections = [col.name for col in qdrant_client.get_collections().collections]
        if collection_name not in collections:
            qdrant_client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE))
            print(f"Created collection '{collection_name}'.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Could not configure Qdrant: {e}")

ensure_qdrant_collection()

def translate_german_to_english(text):
    from transformers import MarianMTModel, MarianTokenizer
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def extract_text_from_pdf(pdf_path):
    page_entries = []
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                sentences = nltk.sent_tokenize(page_text)
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        try: lang = detect(sentence)
                        except Exception: lang = "unknown"
                        page_entries.append({"text": sentence, "page": page_number, "sentence_id": i, "language": lang})
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    return page_entries

def store_pdf_embeddings(pdf_path, context_window=5):
    doc_name = os.path.basename(pdf_path)
    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        filter_condition = Filter(must=[FieldCondition(key="document", match=MatchValue(value=doc_name))])
        if qdrant_client.count(collection_name=collection_name, count_filter=filter_condition).count > 0:
            print(f"Embeddings for {doc_name} already exist. Skipping.")
            return
        pdf_entries = extract_text_from_pdf(pdf_path)
        if not pdf_entries: return
        texts = [entry["text"] for entry in pdf_entries]
        embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True).tolist()
        entries_by_page = {}
        for entry in pdf_entries: entries_by_page.setdefault(entry["page"], []).append(entry)
        points = []
        for i, entry in enumerate(pdf_entries):
            page, sid = entry["page"], entry["sentence_id"]
            page_entries_sorted = sorted(entries_by_page[page], key=lambda x: x["sentence_id"])
            context_entries = [e for e in page_entries_sorted if abs(e["sentence_id"] - sid) <= context_window]
            full_text = " ".join([e["text"] for e in context_entries])
            points.append(PointStruct(id=hash(f"{doc_name}_{page}_{sid}"), vector=embeddings[i], payload={"text": entry["text"], "full_text": full_text, "page": page, "document": doc_name, "language": entry.get("language", "unknown"), "sentence_id": sid}))
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Stored embeddings for {doc_name}.")
    except Exception as e:
        print(f"Failed to store embeddings for {doc_name}: {e}")

def hybrid_search(query, top_k=3):
    try:
        query_embedding = embedding_model.encode([query]).tolist()[0]
        search_results = qdrant_client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k * 10, with_payload=True)
        if not search_results: return []
        candidate_texts = [hit.payload.get("full_text", "") for hit in search_results if hit.payload]
        tokenized_docs = [word_tokenize(text.lower()) for text in candidate_texts]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = word_tokenize(query.lower())
        sparse_scores = np.array(bm25.get_scores(tokenized_query))
        dense_scores = np.array([hit.score for hit in search_results])
        dense_norm, sparse_norm = dense_scores / (np.max(dense_scores) or 1), sparse_scores / (np.max(sparse_scores) or 1)
        hybrid_scores = 0.7 * dense_norm + 0.3 * sparse_norm
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        best_results, seen = [], set()
        for i in sorted_indices:
            hit = search_results[i]
            doc_page_pair = (hit.payload["document"], hit.payload["page"])
            if doc_page_pair in seen: continue
            seen.add(doc_page_pair)
            best_results.append({"text": hit.payload.get("full_text", ""), "page": hit.payload["page"], "document": hit.payload["document"], "score": hybrid_scores[i]})
            if len(best_results) >= top_k: break
        return best_results
    except Exception as e:
        print(f"An error during hybrid search: {e}")
        return []

def hybrid_search_dual(query, top_k=5):
    try:
        query_lang = detect(query)
    except Exception:
        query_lang = "en"
    if query_lang == "de":
        results = hybrid_search(query, top_k=top_k) + hybrid_search(translate_german_to_english(query), top_k=top_k)
    else:
        results = hybrid_search(query, top_k=top_k)
    combined = { (res["document"], res["page"]): res for res in sorted(results, key=lambda x: x["score"], reverse=True) }
    return sorted(list(combined.values()), key=lambda x: x["score"], reverse=True)[:top_k]

def get_total_pages(pdf_path):
    try:
        with open(pdf_path, 'rb') as f: return len(PyPDF2.PdfReader(f).pages)
    except Exception: return 0

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    try:
        pil_image = pdfium.PdfDocument(pdf_path)[page_number].render(scale=scale).to_pil()
        if output_path: pil_image.save(output_path)
        return pil_image
    except Exception as e:
        print(f"Failed to render page {page_number+1} from {pdf_path}: {e}")
        return None

def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
    if not (0 <= page_number < get_total_pages(pdf_path)):
        print(f"Page number {page_number + 1} out of range.")
        return None
    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{doc_name}_page_{page_number + 1}.png")
    abs_path = os.path.abspath(output_path)
    if abs_path not in img_list: img_list.append(abs_path)
    return render_page(pdf_path, page_number, scale=scale, output_path=output_path)

def get_pdf_paths(root_folder):
    return [os.path.join(s, f) for s, _, files in os.walk(root_folder) for f in files if f.lower().endswith(".pdf")]

# Setup PDFs
root_directory = "paper"
if not os.path.exists(root_directory): os.makedirs(root_directory)
pdf_files = get_pdf_paths(root_directory)
if pdf_files:
    print(f"Found {len(pdf_files)} PDF(s). Processing for embeddings...")
    for pdf_file in pdf_files: store_pdf_embeddings(pdf_file, context_window=5)
else:
    print(f"No PDF files found in '{root_directory}'.")


# --- Gradio UI Functions ---
def search_query(query):
    if not query: return "Please enter a search query.", [], json.dumps([])
    global img_list
    img_list.clear()
    best_results = hybrid_search_dual(query, top_k=5)
    if not best_results: return "No results found.", [], json.dumps(best_results)
    results_text = "".join([f"Document: {res['document']}, Page: {res['page']}\nScore: {res['score']:.4f}\nText: {res['text']}\n{'-'*50}\n" for res in best_results])
    images = []
    for res in best_results:
        pdf_path = next((path for path in pdf_files if os.path.basename(path) == res["document"]), None)
        if pdf_path:
            # Use high scale for gallery display
            pil_img = process_specific_page(pdf_path, res["page"] - 1, scale=10.0)
            if pil_img: images.append(pil_img)
    return results_text, images, json.dumps(best_results)

# --- 3. REWRITTEN FUNCTION USING GROQ API WITH IMAGE RESIZING ---
def generate_text_response(additional_prompt, best_results_json, selected_index):
    if not client: return "Error: Groq client not initialized. Please set the GROQ_API_KEY."

    best_results = json.loads(best_results_json)
    retrieved_context = "\n\n--- RETRIEVED CONTEXT ---\n" + "\n".join([res["text"] for res in best_results])
    full_user_prompt = additional_prompt + retrieved_context
    system_prompt = "You are an advanced technical agent. Use the retrieved text and, if provided, the image to answer the user's question. Respond in English."
    
    selected_image_path = None
    try:
        idx = int(selected_index)
        if 0 <= idx < len(img_list): selected_image_path = img_list[idx]
    except (ValueError, TypeError): pass

    messages = [{"role": "system", "content": system_prompt}]
    
    if selected_image_path:
        print(f"Processing image for API: {selected_image_path}")
        try:
            # Open the high-res image, resize it if necessary, and encode it
            with Image.open(selected_image_path) as img:
                resized_img = resize_image_if_needed(img)
                base64_image = encode_pil_image_to_base64(resized_img)
            
            messages.append({"role": "user", "content": [{"type": "text", "text": full_user_prompt}, {"type": "image_url", "image_url": {"url": base64_image}}]})
        except Exception as e:
            print(f"Error processing image: {e}. Falling back to text-only.")
            messages.append({"role": "user", "content": full_user_prompt})
    else:
        print("No valid image selected. Generating response with text only.")
        messages.append({"role": "user", "content": full_user_prompt})

    try:
        response = client.chat.completions.create(
            # IMPORTANT: The model 'meta-llama/llama-4-scout-17b-16e-instruct' is NOT available on Groq's public API.
            # You MUST change this to 'llava-llama-3-8b-vapor' for the code to work.
            # I am leaving your requested model name here as instructed.
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=2048,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"An error occurred while communicating with the Groq API: {e}"


# --- 4. Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PDF Search and Analysis with RAG")
    gr.Markdown("This interface uses a dual-language hybrid search to find relevant pages in your PDFs and then uses a vision-language model via Groq to answer questions about the retrieved text and page images.")
    
    with gr.Tabs():
        with gr.TabItem("Search & Generate"):
            gr.Markdown("### Step 1: Find Relevant Information")
            with gr.Row():
                query_input = gr.Textbox(label="Search Query", placeholder="Enter your question or keywords in English or German...", scale=4)
                search_btn = gr.Button("Search PDFs", variant="primary", scale=1)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Retrieved Page Images")
                    gallery = gr.Gallery(label="Rendered Images", height=600, object_fit="contain", columns=1)
                with gr.Column(scale=2):
                    gr.Markdown("#### Retrieved Text Snippets")
                    search_results = gr.Textbox(label="Search Results", lines=25, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Step 2: Ask a Question About the Results")
            gr.Markdown("Enter a question and select an image by its index (0 for the first image, 1 for the second, etc.). The model will use the retrieved text snippets and the selected image to generate an answer.")
            
            with gr.Row():
                selected_image_index = gr.Number(label="Image Index to Analyze", value=0, minimum=0, step=1, interactive=True)
                additional_prompt_input = gr.Textbox(label="Your Question for the LLM", placeholder="e.g., 'Summarize the main point from this page' or 'Explain the diagram in image 0'", scale=3, interactive=True)
                generate_btn = gr.Button("Generate Response", variant="primary")

            generated_text = gr.Textbox(label="LLM Answer", lines=15, interactive=False)
            hidden_best_results = gr.Textbox(visible=False)

    search_btn.click(fn=search_query, inputs=query_input, outputs=[search_results, gallery, hidden_best_results])
    generate_btn.click(fn=generate_text_response, inputs=[additional_prompt_input, hidden_best_results, selected_image_index], outputs=generated_text)

if __name__ == "__main__":
    demo.launch(debug=True)
