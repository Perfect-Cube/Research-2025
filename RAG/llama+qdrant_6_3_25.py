import os
import warnings
import torch
import numpy as np
import PyPDF2
import pypdfium2 as pdfium
import nltk
from PIL import Image
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import pickle
from langdetect import detect, DetectorFactory
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Ensure reproducible language detection
DetectorFactory.seed = 0

img_list=[]

# Download the NLTK tokenizer data if not already available.
nltk.download('punkt')

# Use Qdrant persistent storage (no Docker required)
qdrant_client = QdrantClient(path="qdrant_storage")
collection_name = "pdf_embeddings"

# Load Embedding Model first so we can obtain its dimension
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

# def store_pdf_embeddings(pdf_path, context_window=1):
#     """
#     Extracts text, generates embeddings, and stores them in Qdrant.
#     Additionally, it combines neighboring sentences (based on context_window)
#     to store extra context for each hit.
#     """
#     doc_name = os.path.basename(pdf_path)
#     pdf_entries = extract_text_from_pdf(pdf_path)
#     texts = [entry["text"] for entry in pdf_entries]
#     embeddings = embedding_model.encode(texts, batch_size=32).tolist()
    
#     # Build a lookup for entries by (page, sentence_id)
#     entries_by_page = {}
#     for entry in pdf_entries:
#         page = entry["page"]
#         entries_by_page.setdefault(page, []).append(entry)
    
#     # For each entry, compute full context by combining neighboring sentences on the same page.
#     points = []
#     for i, entry in enumerate(pdf_entries):
#         page = entry["page"]
#         sid = entry["sentence_id"]
#         # Retrieve all sentences from the same page
#         page_entries = sorted(entries_by_page[page], key=lambda x: x["sentence_id"])
#         # Filter those within the context window
#         context_entries = [e for e in page_entries if abs(e["sentence_id"] - sid) <= context_window]
#         full_text = " ".join([e["text"] for e in context_entries])
        
#         uid = f"{doc_name}_{page}_{sid}"
#         points.append(PointStruct(
#             id=hash(uid),
#             vector=embeddings[i],
#             payload={
#                 "text": entry["text"],
#                 "full_text": full_text,
#                 "page": page,
#                 "document": doc_name,
#                 "language": entry.get("language", "unknown"),
#                 "sentence_id": sid
#             }
#         ))
    
#     qdrant_client.upsert(collection_name=collection_name, points=points)
#     print(f"Stored embeddings for {doc_name} in Qdrant.")
def store_pdf_embeddings(pdf_path, context_window=1):
    """
    Extracts text, generates embeddings, and stores them in Qdrant.
    Additionally, it combines neighboring sentences (based on context_window)
    to store extra context for each hit.
    Skips processing if embeddings for this PDF are already stored.
    """
    doc_name = os.path.basename(pdf_path)
    
    # Check if embeddings for this document already exist in Qdrant.
    # (Requires QdrantClient vX.X+; adjust imports if necessary.)
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    filter_condition = Filter(
        must=[FieldCondition(key="document", match=MatchValue(value=doc_name))]
    )
    count_result = qdrant_client.count(collection_name=collection_name, count_filter=filter_condition)
    if count_result.count > 0:
        print(f"Embeddings for {doc_name} already exist in Qdrant. Skipping embedding computation.")
        return

    # If not found, proceed to extract text and compute embeddings.
    pdf_entries = extract_text_from_pdf(pdf_path)
    texts = [entry["text"] for entry in pdf_entries]
    embeddings = embedding_model.encode(texts, batch_size=32).tolist()

    # Build a lookup for entries by (page, sentence_id)
    entries_by_page = {}
    for entry in pdf_entries:
        page = entry["page"]
        entries_by_page.setdefault(page, []).append(entry)

    # For each entry, compute full context by combining neighboring sentences on the same page.
    points = []
    for i, entry in enumerate(pdf_entries):
        page = entry["page"]
        sid = entry["sentence_id"]
        # Retrieve all sentences from the same page
        page_entries = sorted(entries_by_page[page], key=lambda x: x["sentence_id"])
        # Filter those within the context window
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

def hybrid_search(query, top_k=3, filter_doc=None):
    """Performs hybrid search combining dense retrieval from Qdrant and sparse retrieval using BM25."""
    try:
        query_lang = detect(query)
    except Exception:
        query_lang = None
    
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
            "text": hit.payload.get("full_text", hit.payload.get("text", "")),  # fallback if full_text is missing
            "page": hit.payload["page"],
            "document": hit.payload["document"],
            "score": hybrid_scores[i]
        })
    
    return best_results


# PDF Rendering Functions
def get_total_pages(pdf_path):
    """Return the total number of pages in the PDF using PyPDF2."""
    reader = PyPDF2.PdfReader(pdf_path)
    return len(reader.pages)

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    """
    Render the specified page using pypdfium2 at the given scale.
    Saves the image if an output path is provided.
    """
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

def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
    """
    Render and scale only the specified page of the PDF.
    Saves the image with the document name and page number.
    """
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract document name without extension
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{doc_name}_page_{page_number + 1}.png")
    img_list.append(os.path.abspath(output_path).replace('\\','/'))
    return render_page(pdf_path, page_number, scale=scale, output_path=output_path)

# Setup the Text-Generation Pipeline
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = "cpu"
# AI Model Integration
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
def get_pdf_paths(root_folder):
    """
    Recursively finds all PDF files in the given root folder and its subdirectories.
    
    :param root_folder: The main folder containing subfolders with PDFs.
    :return: A list of full paths to the PDF files.
    """
    pdf_paths = []
    
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(subdir, file))
    
    return pdf_paths

# Example usage
root_directory = "C:/Users/alukkib/Documents/Hybrid RAG/Standards und Normen"  # Change this to your folder path
pdf_files = get_pdf_paths(root_directory)
for pdf_file in pdf_files:
    store_pdf_embeddings(pdf_file, context_window=1)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    best_results = hybrid_search(query, top_k=5)
    if not best_results:
        print("No results found. Please try another query.")
        continue
    
    print("\n--- Top 5 Search Results ---")
    for res in best_results:
        print(f"Document: {res['document']}, Page {res['page']}")
        print(f"Text: {res['text']}")
        print(f"Score: {res['score']:.4f}")
        print("-" * 50)
    show_image = input("Would you like to see the retrieved page as an image? (yes/no): ").strip().lower()
    if show_image == "yes":
        for res in best_results:
            pdf_path = next((path for path in pdf_files if os.path.basename(path) == res["document"]), None)
            if pdf_path:
                process_specific_page(pdf_path, res["page"] - 1, scale=10.0)
        print(img_list)
    user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
    print("\n--- Generated Text ---")
    if user_prompt:
        combined_prompt=""
        if img_list:
            combined_prompt = "<|image|>" + "<|begin_of_text|> " + "Please respond in english" + user_prompt 
            images = [Image.open(img_list[0]).convert("RGB")]
            inputs = processor(images=images, text=combined_prompt, return_tensors="pt", padding=True).to(model.device)
            output = model.generate(**inputs, max_new_tokens=300)
            response = processor.decode(output[0], skip_special_tokens=True)
            print(response)
        else:
            combined_prompt = "<|begin_of_text|> " + "Please respond in english" + user_prompt + "\n".join([res["text"] for res in best_results])
            inputs = processor(text=combined_prompt, return_tensors="pt", padding=True).to(model.device)
            output = model.generate(**inputs, max_new_tokens=300)
            response = processor.decode(output[0], skip_special_tokens=True)
            print(response)
       
        
