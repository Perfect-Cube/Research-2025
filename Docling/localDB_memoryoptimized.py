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

# In-memory storage optimized for memory usage
sentence_embeddings_array = np.array([])  # Will be resized as needed
sentence_ids_list = []
sentence_metadata_dict = {}
document_embeddings_dict = {}

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

def store_pdf_embeddings(pdf_path):
    """
    Extracts text, generates embeddings, and stores them in memory with caching.
    """
    doc_name = os.path.basename(pdf_path)
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{doc_name}.pkl")
    
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
            embeddings = embedding_model.encode(texts, batch_size=16).tolist()  # Reduced batch size for memory
            with open(cache_file, "wb") as f:
                pickle.dump({"pdf_entries": pdf_entries, "embeddings": embeddings}, f)
            print(f"Computed and cached embeddings for {doc_name}")
        except Exception as e:
            print(f"Error computing embeddings for {doc_name}: {e}")
            return
    
    # Store in memory, using numpy for efficiency
    global sentence_embeddings_array, sentence_ids_list
    start_len = len(sentence_ids_list)
    new_embeddings = np.array(embeddings, dtype=np.float32)  # Use float32 for memory efficiency
    if sentence_embeddings_array.size == 0:
        sentence_embeddings_array = new_embeddings
    else:
        sentence_embeddings_array = np.vstack((sentence_embeddings_array, new_embeddings))
    
    for i, entry in enumerate(pdf_entries):
        id = f"{doc_name}_{entry['page']}_{entry['sentence_id']}"
        sentence_ids_list.append(id)
        sentence_metadata_dict[id] = {
            "text": entry["text"],
            "page": entry["page"],
            "document": doc_name,
            "type": "sentence"
        }
    
    print(f"Stored sentence-level embeddings for {doc_name}")
    
    if embeddings:
        doc_embedding = np.mean(new_embeddings, axis=0).tolist()
        document_embeddings_dict[doc_name] = doc_embedding
        id = f"{doc_name}_document"
        sentence_ids_list.append(id)
        sentence_embeddings_array = np.vstack((sentence_embeddings_array, np.array(doc_embedding, dtype=np.float32)))
        sentence_metadata_dict[id] = {
            "text": f"[Document: {doc_name}]",
            "document": doc_name,
            "type": "document"
        }
        print(f"Stored overall document embedding for {doc_name}")

def process_pdf_sequentially(pdf_files):
    """
    Process PDFs sequentially to optimize memory usage.
    """
    for pdf_file in pdf_files:
        store_pdf_embeddings(pdf_file)

# PDF Image Rendering Functions using pdfium2 (unchanged for brevity)
def get_total_pages(pdf_path):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error getting total pages for {pdf_path}: {e}")
        return 0

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
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
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scaled_page_{page_number + 1}.png")
    render_page(pdf_path, page_number, scale=scale, output_path=output_path)

def hybrid_search(query, top_k=3, filter_doc=None):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # Filter indices based on document filter
    filtered_indices = []
    for idx, id in enumerate(sentence_ids_list):
        if filter_doc is None or sentence_metadata_dict[id]["document"] == filter_doc:
            filtered_indices.append(idx)
    
    if not filtered_indices:
        print("No results found from dense retrieval.")
        return []
    
    # Compute distances using numpy for efficiency
    all_filtered_embeddings = sentence_embeddings_array[filtered_indices]
    distances = np.linalg.norm(all_filtered_embeddings - np.array([query_embedding]), axis=1)
    
    # Get top 20 for BM25 to manage memory
    top_n = min(20, len(filtered_indices))  # Limit for memory
    sorted_indices = np.argsort(distances)[:top_n]
    top_filtered_indices = [filtered_indices[i] for i in sorted_indices]
    
    # Prepare texts for BM25
    retrieved_texts = [sentence_metadata_dict[sentence_ids_list[idx]]["text"] for idx in top_filtered_indices]
    if not retrieved_texts:
        print("No texts retrieved for BM25.")
        return []
    
    tokenized_docs = [word_tokenize(doc.lower()) for doc in retrieved_texts]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Normalize dense scores
    dense_scores = distances[sorted_indices]
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
    sorted_hybrid_indices = np.argsort(hybrid_scores)[::-1]
    
    # Get top-k results
    best_results = []
    for i in sorted_hybrid_indices[:top_k]:
        idx = top_filtered_indices[i]
        id = sentence_ids_list[idx]
        meta = sentence_metadata_dict[id]
        best_results.append({
            "text": meta["text"],
            "page": meta.get("page", None),
            "document": meta["document"],
            "type": meta["type"],
            "score": hybrid_scores[i]
        })
    
    return best_results

# Setup the Text-Generation Pipeline
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = "cpu"  # Suggest changing to "cuda" if GPU available
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

# Process and Store PDFs
pdf_files = [
    "C:/Users/alukkib/Documents/Hybrid RAG/AutomotiveSPICE_PAM_31.pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"
]
process_pdf_sequentially(pdf_files)  # Changed to sequential for memory optimization

# Interactive Query Loop (unchanged for brevity)
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
