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
import multiprocessing
import pickle

# Download the NLTK tokenizer data if not already available.
nltk.download('punkt')

# Load Embedding Model
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# In-memory storage for embeddings and metadata
sentence_embeddings = {}  # {id: embedding}
sentence_metadata = {}    # {id: {"text": str, "page": int, "document": str, "type": str}}
document_embeddings = {}  # {doc_name: embedding}

# --------------------------
# PDF Text Extraction with PyPDF2
# --------------------------
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of the PDF and returns a list of entries with text, page number, and sentence ID.
    """
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        page_entries = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                sentences = nltk.sent_tokenize(page_text)
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        page_entries.append({
                            "text": sentence,
                            "page": page_number,
                            "sentence_id": i
                        })
        return page_entries
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []

# --------------------------
# Store Embeddings in Memory and Cache to Disk
# --------------------------
def store_pdf_embeddings(pdf_path):
    """
    Extracts text, generates embeddings (using cache if available), and stores them in memory.
    Also computes and stores an overall document embedding.
    """
    doc_name = os.path.basename(pdf_path)
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{doc_name}.pkl")
    
    # Check if embeddings are already cached
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            pdf_entries = data["pdf_entries"]
            embeddings = data["embeddings"]
            print(f"Loaded cached embeddings for {doc_name}")
        except Exception as e:
            print(f"Error loading cache for {doc_name}: {e}")
            return
    else:
        pdf_entries = extract_text_from_pdf(pdf_path)
        if not pdf_entries:
            print(f"No text extracted from {doc_name}, skipping.")
            return
        texts = [entry["text"] for entry in pdf_entries]
        try:
            embeddings = embedding_model.encode(texts, batch_size=32).tolist()
            with open(cache_file, "wb") as f:
                pickle.dump({"pdf_entries": pdf_entries, "embeddings": embeddings}, f)
            print(f"Computed and cached embeddings for {doc_name}")
        except Exception as e:
            print(f"Error computing embeddings for {doc_name}: {e}")
            return
    
    # Store sentence-level embeddings in memory
    for i, entry in enumerate(pdf_entries):
        id = f"{doc_name}_{entry['page']}_{entry['sentence_id']}"
        sentence_embeddings[id] = embeddings[i]
        sentence_metadata[id] = {
            "text": entry["text"],
            "page": entry["page"],
            "document": doc_name,
            "type": "sentence"
        }
    print(f"Stored sentence-level embeddings for {doc_name}")
    
    # Compute and store overall document embedding
    if embeddings:
        doc_embedding = np.mean(embeddings, axis=0).tolist()
        document_embeddings[doc_name] = doc_embedding
        sentence_metadata[f"{doc_name}_document"] = {
            "text": f"[Document: {doc_name}]",
            "document": doc_name,
            "type": "document"
        }
        sentence_embeddings[f"{doc_name}_document"] = doc_embedding
        print(f"Stored overall document embedding for {doc_name}")

# --------------------------
# Parallel Processing for Multiple PDFs
# --------------------------
def process_multiple_pdfs(pdf_files):
    with multiprocessing.Pool(processes=min(4, len(pdf_files))) as pool:
        pool.map(store_pdf_embeddings, pdf_files)

# --------------------------
# PDF Image Rendering Functions using pdfium2
# --------------------------
def get_total_pages(pdf_path):
    """Return the total number of pages in the PDF using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error getting total pages for {pdf_path}: {e}")
        return 0

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    """
    Render the specified page using pypdfium2 at the given scale.
    Saves the image if an output path is provided.
    """
    try:
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
    except Exception as e:
        print(f"Error rendering page {page_number + 1} from {pdf_path}: {e}")
        return None

def process_specific_page(pdf_path, page_number, scale=1.0, output_dir="rendered_page"):
    """
    Render and scale only the specified page of the PDF.
    """
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scaled_page_{page_number + 1}.png")
    render_page(pdf_path, page_number, scale=scale, output_path=output_path)

# --------------------------
# Hybrid Search (BM25 + Dense) with Top-3 Retrieval
# --------------------------
def hybrid_search(query, top_k=3, filter_doc=None):
    """
    Performs hybrid search combining dense and sparse retrieval using in-memory storage.
    Returns the top matching results (both sentence-level and document-level) with their metadata.
    """
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # Dense retrieval: compute distances to all embeddings
    distances = []
    for id, emb in sentence_embeddings.items():
        if filter_doc and sentence_metadata[id]["document"] != filter_doc:
            continue
        dist = np.linalg.norm(np.array(emb) - np.array(query_embedding))
        distances.append((id, dist))
    
    if not distances:
        print("No results found from dense retrieval.")
        return []
    
    # Sort by distance (lower is better)
    distances.sort(key=lambda x: x[1])
    top_ids = [id for id, _ in distances[:top_k * 2]]  # Get more for BM25 filtering
    
    # Prepare texts for BM25
    retrieved_texts = [sentence_metadata[id]["text"] for id in top_ids]
    if not retrieved_texts:
        print("No texts retrieved for BM25.")
        return []
    
    tokenized_docs = [word_tokenize(doc.lower()) for doc in retrieved_texts]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Normalize dense scores (convert distances to similarities)
    dense_scores = np.array([dist[1] for dist in distances[:len(top_ids)]])
    if np.max(dense_scores) > 0:
        dense_scores = 1 - (dense_scores / np.max(dense_scores))
    else:
        dense_scores = np.zeros_like(dense_scores)
    
    # Normalize sparse scores
    max_sparse = np.max(sparse_scores) if len(sparse_scores) > 0 else 0
    if max_sparse > 0:
        sparse_scores = sparse_scores / max_sparse
    else:
        sparse_scores = np.zeros_like(dense_scores)
    
    # Combine scores
    hybrid_scores = 0.7 * dense_scores + 0.3 * sparse_scores
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    
    # Get top-k results
    best_results = []
    for i in sorted_indices[:top_k]:
        id = top_ids[i]
        meta = sentence_metadata[id]
        best_results.append({
            "text": meta["text"],
            "page": meta.get("page", None),
            "document": meta["document"],
            "type": meta["type"],
            "score": hybrid_scores[i]
        })
    
    return best_results

# --------------------------
# Setup the Text-Generation Pipeline
# --------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = "cpu"
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

# --------------------------
# Process and Store PDFs
# --------------------------
pdf_files = [
    "C:/Users/alukkib/Documents/Hybrid RAG/AutomotiveSPICE_PAM_31.pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"
]
process_multiple_pdfs(pdf_files)

# Check stored data for debugging
print("Stored Sentence IDs:", list(sentence_embeddings.keys())[:5])  # Sample of first 5
print("Stored Document Embeddings:", list(document_embeddings.keys()))

# --------------------------
# Interactive Query Loop with Top-3 Retrieval and Image Rendering
# --------------------------
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    doc_filter = input("Filter by document name? (Press Enter to skip): ").strip() or None
    best_results = hybrid_search(query, top_k=3, filter_doc=doc_filter)
    if not best_results:
        print("No results found. Please try another query.")
        continue

    print("\n--- Top 3 Search Results ---")
    for res in best_results:
        result_type = "Overall Document" if res["type"] == "document" else f"Page {res['page']}"
        print(f"Document: {res['document']}, {result_type}")
        print(f"Text: {res['text']}")
        print(f"Score: {res['score']:.4f}")
        print("-" * 50)
    
    # Build combined text
    lines = []
    for res in best_results:
        page_info = "Overall" if res["type"] == "document" else f"Page {res['page']}"
        lines.append(f"[{res['document']} - {page_info}] {res['text']}")
    combined_text = "\n".join(lines)
    
    if best_results:
        img_choice = input("Enter the page number from the above results to render an image (or type 'no'): ").strip().lower()
        if img_choice != "no":
            try:
                chosen_page = int(img_choice)
                pages = [res["page"] for res in best_results if res["page"] is not None]
                if chosen_page in pages:
                    for res in best_results:
                        if res.get("page") == chosen_page:
                            pdf_path = next(p for p in pdf_files if res["document"] == os.path.basename(p))
                            break
                    process_specific_page(pdf_path, chosen_page - 1, scale=7.5, output_dir="relevante")
                else:
                    print(f"Page {chosen_page} is not among the retrieved results.")
            except ValueError:
                print("Invalid input. Skipping image rendering.")
    
    user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
    if user_prompt:
        combined_prompt = f"{user_prompt}\nContext:\n{combined_text}"
    else:
        combined_prompt = combined_text
    
    try:
        generated = pipe(combined_prompt, max_new_tokens=900)
        generated_text = generated[0]['generated_text']
    except Exception as e:
        generated_text = f"Error generating text: {e}"
    
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n----------------------\n")
