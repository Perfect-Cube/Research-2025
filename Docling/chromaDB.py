# import os
# import warnings
# import torch
# import chromadb
# import numpy as np
# import PyPDF2
# import pypdfium2 as pdfium
# import nltk
# from PIL import Image
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from nltk.tokenize import word_tokenize
# import multiprocessing
# import pickle

# # Download the NLTK tokenizer data if not already available.
# nltk.download('punkt')

# # --------------------------
# # Initialize ChromaDB
# # --------------------------
# chroma_client = chromadb.PersistentClient(path="chroma_db")
# collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

# # --------------------------
# # Load Embedding Model
# # --------------------------
# embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# # --------------------------
# # PDF Text Extraction with PyPDF2
# # --------------------------
# def extract_text_from_pdf(pdf_path):
#     """Extracts text from each page of the PDF and returns a list of entries with text, page number, and sentence ID."""
#     reader = PyPDF2.PdfReader(pdf_path)
#     page_entries = []
#     for page_number, page in enumerate(reader.pages, start=1):
#         page_text = page.extract_text()
#         if page_text:
#             sentences = nltk.sent_tokenize(page_text)
#             for i, sentence in enumerate(sentences):
#                 if sentence.strip():
#                     page_entries.append({
#                         "text": sentence,
#                         "page": page_number,
#                         "sentence_id": i
#                     })
#     return page_entries

# # --------------------------
# # Store Embeddings in ChromaDB and Cache to Disk
# # --------------------------
# def store_pdf_embeddings(pdf_path):
#     """Extracts text, generates embeddings (using cache if available), and stores them in ChromaDB."""
#     doc_name = os.path.basename(pdf_path)
#     cache_dir = "embeddings_cache"
#     os.makedirs(cache_dir, exist_ok=True)
#     cache_file = os.path.join(cache_dir, f"{doc_name}.pkl")
    
#     # Check if embeddings are already cached.
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             data = pickle.load(f)
#         pdf_entries = data["pdf_entries"]
#         embeddings = data["embeddings"]
#         print(f"Loaded cached embeddings for {doc_name}")
#     else:
#         pdf_entries = extract_text_from_pdf(pdf_path)
#         texts = [entry["text"] for entry in pdf_entries]
#         embeddings = embedding_model.encode(texts, batch_size=32).tolist()
#         with open(cache_file, "wb") as f:
#             pickle.dump({"pdf_entries": pdf_entries, "embeddings": embeddings}, f)
#         print(f"Computed and cached embeddings for {doc_name}")
    
#     # Prepare metadata and IDs.
#     ids = [f"{doc_name}_{entry['page']}_{entry['sentence_id']}" for entry in pdf_entries]
#     metadatas = [{"text": entry["text"], "page": entry["page"], "document": doc_name} for entry in pdf_entries]
    
#     # Add the embeddings to ChromaDB.
#     collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
#     print(f"Stored embeddings for {doc_name}")

# # --------------------------
# # Parallel Processing for Multiple PDFs
# # --------------------------
# def process_multiple_pdfs(pdf_files):
#     with multiprocessing.Pool(processes=min(8, len(pdf_files))) as pool:
#         pool.map(store_pdf_embeddings, pdf_files)

# # --------------------------
# # PDF Image Rendering Functions using pdfium2
# # --------------------------
# def get_total_pages(pdf_path):
#     """Return the total number of pages in the PDF using PyPDF2."""
#     reader = PyPDF2.PdfReader(pdf_path)
#     return len(reader.pages)

# def render_page(pdf_path, page_number, scale=1.0, output_path=None):
#     """
#     Render the specified page using pypdfium2 at the given scale.
#     Saves the image if an output path is provided.
#     """
#     pdf = pdfium.PdfDocument(pdf_path)
#     page = pdf.get_page(page_number)
#     bitmap = page.render(scale=scale)
#     pil_image = bitmap.to_pil()
#     if output_path:
#         pil_image.save(output_path)
#         print(f"Saved rendered page {page_number + 1} with image to: {output_path}")
#     page.close()
#     pdf.close()
#     return pil_image

# def process_specific_page(pdf_path, page_number, scale=1.0, output_dir="rendered_page"):
#     """
#     Render and scale only the specified page of the PDF.
#     """
#     total_pages = get_total_pages(pdf_path)
#     if page_number < 0 or page_number >= total_pages:
#         print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
#         return
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"scaled_page_{page_number + 1}.png")
#     render_page(pdf_path, page_number, scale=scale, output_path=output_path)

# # --------------------------
# # Hybrid Search (BM25 + Dense) with Top-3 Retrieval
# # --------------------------
# def hybrid_search(query, top_k=3, filter_doc=None):
#     """Performs hybrid search combining dense and sparse retrieval across PDFs.
#        Returns the top three matching results."""
#     query_embedding = embedding_model.encode([query]).tolist()
#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k,
#         where={"document": filter_doc} if filter_doc else None
#     )
    
#     if not results["metadatas"][0]:
#         print("No results found from dense retrieval.")
#         return []
    
#     retrieved_texts = [meta["text"] for meta in results["metadatas"][0]]
#     if not retrieved_texts:
#         print("No texts retrieved for BM25.")
#         return []
    
#     tokenized_docs = [word_tokenize(doc.lower()) for doc in retrieved_texts]
#     bm25 = BM25Okapi(tokenized_docs)
#     tokenized_query = word_tokenize(query.lower())
#     sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
#     dense_scores = np.array(results["distances"][0])
#     if np.max(dense_scores) > 0:
#         dense_scores = 1 - (dense_scores / np.max(dense_scores))
#     else:
#         dense_scores = np.zeros_like(dense_scores)
    
#     max_sparse = np.max(sparse_scores) if len(sparse_scores) > 0 else 0
#     if max_sparse > 0:
#         sparse_scores = sparse_scores / max_sparse
#     else:
#         sparse_scores = np.zeros_like(sparse_scores)
    
#     hybrid_scores = 0.7 * dense_scores + 0.3 * sparse_scores
#     sorted_indices = np.argsort(hybrid_scores)[::-1]
    
#     best_results = []
#     for i in sorted_indices[:top_k]:
#         best_results.append({
#             "text": retrieved_texts[i],
#             "page": results["metadatas"][0][i]["page"],
#             "document": results["metadatas"][0][i]["document"],
#             "score": hybrid_scores[i]
#         })
    
#     return best_results

# # --------------------------
# # Setup the Text-Generation Pipeline
# # --------------------------
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# device = "cpu"
# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-2-9b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device=device,
# )

# # --------------------------
# # Process and Store PDFs (Run This First)
# # --------------------------
# pdf_files = [
#     "C:/Users/alukkib/Documents/Hybrid RAG/AutomotiveSPICE_PAM_31.pdf",
#     "C:/Users/alukkib/Documents/Hybrid RAG/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"
# ]
# process_multiple_pdfs(pdf_files)

# # Optional: Check stored data in the collection for debugging.
# stored = collection.get()
# print("Stored IDs:", stored["ids"])

# # --------------------------
# # Interactive Query Loop with Top-3 Retrieval and Image Rendering
# # --------------------------
# while True:
#     query = input("Enter your query (or type 'exit' to quit): ")
#     if query.lower() == "exit":
#         break
    
#     doc_filter = input("Filter by document name? (Press Enter to skip): ").strip() or None
#     best_results = hybrid_search(query, top_k=3, filter_doc=doc_filter)
#     if not best_results:
#         print("No results found. Please try another query.")
#         continue

#     print("\n--- Top 3 Search Results ---")
#     for res in best_results:
#         print(f"Document: {res['document']}, Page: {res['page']}")
#         print(f"Text: {res['text']}")
#         print(f"Score: {res['score']:.4f}")
#         print("-" * 50)
    
#     combined_text = "\n".join(
#         [f"[{res['document']} - Page {res['page']}] {res['text']}" for res in best_results]
#     )
    
#     if best_results:
#         img_choice = input("Enter the page number from the above results to render an image (or type 'no'): ").strip().lower()
#         if img_choice != "no":
#             try:
#                 chosen_page = int(img_choice)
#                 pages = [res['page'] for res in best_results]
#                 if chosen_page in pages:
#                     for res in best_results:
#                         if res['page'] == chosen_page:
#                             pdf_file = res['document']
#                             break
#                     process_specific_page(pdf_file, chosen_page - 1, scale=7.5, output_dir="relevante")
#                 else:
#                     print(f"Page {chosen_page} is not among the retrieved results.")
#             except ValueError:
#                 print("Invalid input. Skipping image rendering.")
    
#     user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
#     if user_prompt:
#         combined_prompt = f"{user_prompt}\nContext:\n{combined_text}"
#     else:
#         combined_prompt = combined_text
    
#     try:
#         generated = pipe(combined_prompt, max_new_tokens=900)
#         generated_text = generated[0]['generated_text']
#     except Exception as e:
#         generated_text = f"Error generating text: {e}"
    
#     print("\n--- Generated Text ---")
#     print(generated_text)
#     print("\n----------------------\n")





import os
import warnings
import torch
import chromadb
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

# --------------------------
# Initialize ChromaDB
# --------------------------
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

# --------------------------
# Load Embedding Model
# --------------------------
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# --------------------------
# PDF Text Extraction with PyPDF2
# --------------------------
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of the PDF and returns a list of entries with text, page number, and sentence ID.
    """
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

# --------------------------
# Store Embeddings in ChromaDB and Cache to Disk
# --------------------------
def store_pdf_embeddings(pdf_path):
    """
    Extracts text, generates embeddings (using cache if available), and stores them in ChromaDB.
    In addition to storing sentence-level embeddings with page and sentence metadata,
    this function computes and stores an overall document embedding.
    """
    doc_name = os.path.basename(pdf_path)
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{doc_name}.pkl")
    
    # Check if embeddings are already cached.
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        pdf_entries = data["pdf_entries"]
        embeddings = data["embeddings"]
        print(f"Loaded cached embeddings for {doc_name}")
    else:
        pdf_entries = extract_text_from_pdf(pdf_path)
        texts = [entry["text"] for entry in pdf_entries]
        embeddings = embedding_model.encode(texts, batch_size=32).tolist()
        with open(cache_file, "wb") as f:
            pickle.dump({"pdf_entries": pdf_entries, "embeddings": embeddings}, f)
        print(f"Computed and cached embeddings for {doc_name}")
    
    # Prepare metadata and IDs for sentence-level entries.
    sentence_ids = [f"{doc_name}_{entry['page']}_{entry['sentence_id']}" for entry in pdf_entries]
    sentence_metadatas = [{"text": entry["text"], "page": entry["page"], "document": doc_name, "type": "sentence"} 
                            for entry in pdf_entries]
    
    # Add the sentence-level embeddings to ChromaDB.
    collection.add(ids=sentence_ids, embeddings=embeddings, metadatas=sentence_metadatas)
    print(f"Stored sentence-level embeddings for {doc_name}")
    
    # Compute and store an overall document embedding.
    # Here, we compute a simple average of all sentence embeddings.
    if embeddings:
        doc_embedding = np.mean(embeddings, axis=0).tolist()
        doc_id = f"{doc_name}_document"
        doc_metadata = {"document": doc_name, "type": "document", "index": "overall"}
        collection.add(ids=[doc_id], embeddings=[doc_embedding], metadatas=[doc_metadata])
        print(f"Stored overall document embedding for {doc_name}")

# --------------------------
# Parallel Processing for Multiple PDFs
# --------------------------
def process_multiple_pdfs(pdf_files):
    with multiprocessing.Pool(processes=min(8, len(pdf_files))) as pool:
        pool.map(store_pdf_embeddings, pdf_files)

# --------------------------
# PDF Image Rendering Functions using pdfium2
# --------------------------
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
        print(f"Saved rendered page {page_number + 1} with image to: {output_path}")
    page.close()
    pdf.close()
    return pil_image

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
    Performs hybrid search combining dense and sparse retrieval across PDFs.
    Returns the top matching results (both sentence-level and document-level) with their metadata.
    """
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"document": filter_doc} if filter_doc else None
    )
    
    if not results["metadatas"][0]:
        print("No results found from dense retrieval.")
        return []
    
    # For dense retrieval, get the texts and metadata.
    retrieved_entries = results["metadatas"][0]
    retrieved_texts = [entry["text"] if entry.get("type") == "sentence" else f"[Document: {entry['document']}]" 
                         for entry in retrieved_entries]
    if not retrieved_texts:
        print("No texts retrieved for BM25.")
        return []
    
    tokenized_docs = [word_tokenize(doc.lower()) for doc in retrieved_texts]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
    dense_scores = np.array(results["distances"][0])
    if np.max(dense_scores) > 0:
        dense_scores = 1 - (dense_scores / np.max(dense_scores))
    else:
        dense_scores = np.zeros_like(dense_scores)
    
    max_sparse = np.max(sparse_scores) if len(sparse_scores) > 0 else 0
    if max_sparse > 0:
        sparse_scores = sparse_scores / max_sparse
    else:
        sparse_scores = np.zeros_like(sparse_scores)
    
    hybrid_scores = 0.7 * dense_scores + 0.3 * sparse_scores
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    
    best_results = []
    for i in sorted_indices[:top_k]:
        meta = retrieved_entries[i]
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
# Process and Store PDFs (Run This First)
# --------------------------
pdf_files = [
    "C:/Users/alukkib/Documents/Hybrid RAG/AutomotiveSPICE_PAM_31.pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"
    # Add paths for additional PDFs as needed.
]
process_multiple_pdfs(pdf_files)

# Optional: Check stored data in the collection for debugging.
stored = collection.get()
print("Stored IDs:", stored["ids"])

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
        result_type = 'Overall Document' if res['type'] == 'document' else f"Page {res['page']}"
        print(f"Document: {res['document']}, {result_type}")
        print(f"Text: {res['text']}")
        print(f"Score: {res['score']:.4f}")
        print("-" * 50)
    
    combined_text = "\n".join(
        [f"[{res['document']} - {'Overall' if res['type']=='document' else f'Page {res['page']}'}] {res['text']}" 
         for res in best_results]
    )
    
    if best_results:
        img_choice = input("Enter the page number from the above results to render an image (or type 'no'): ").strip().lower()
        if img_choice != "no":
            try:
                chosen_page = int(img_choice)
                pages = [res['page'] for res in best_results if res['page'] is not None]
                if chosen_page in pages:
                    for res in best_results:
                        if res.get('page') == chosen_page:
                            pdf_file = res['document']
                            break
                    process_specific_page(pdf_file, chosen_page - 1, scale=7.5, output_dir="relevante")
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
