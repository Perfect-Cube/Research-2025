import os
import warnings
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
import pickle

# Ensure reproducible language detection
DetectorFactory.seed = 0

img_list = []

# Download the NLTK tokenizer data if not already available.
nltk.download('punkt_tab')
nltk.download('punkt')

# Use Qdrant persistent storage (all Qdrant and embedding work on CPU)
qdrant_client = QdrantClient(path="qdrant_storage1")
collection_name = "pdf_embeddings1"

# Load the embedding model on CPU (default behavior)
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
    model = MarianMTModel.from_pretrained(model_name)  # Loaded on CPU by default
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
    Additionally, it combines neighboring sentences (based on context_window)
    to store extra context for each hit.
    Skips processing if embeddings for this PDF are already stored.
    """
    doc_name = os.path.basename(pdf_path)

    # Check if embeddings for this document already exist in Qdrant.
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    filter_condition = Filter(
        must=[FieldCondition(key="document", match=MatchValue(value=doc_name))]
    )
    count_result = qdrant_client.count(collection_name=collection_name, count_filter=filter_condition)
    if count_result.count > 0:
        print(f"Embeddings for {doc_name} already exist in Qdrant. Skipping embedding computation.")
        return

    pdf_entries = extract_text_from_pdf(pdf_path)
    texts = [entry["text"] for entry in pdf_entries]
    embeddings = embedding_model.encode(texts, batch_size=32).tolist()

    # Build a lookup for entries by page
    entries_by_page = {}
    for entry in pdf_entries:
        page = entry["page"]
        entries_by_page.setdefault(page, []).append(entry)

    points = []
    for i, entry in enumerate(pdf_entries):
        page = entry["page"]
        sid = entry["sentence_id"]
        # Retrieve all sentences from the same page and filter by context_window
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
    """
    Performs hybrid search combining dense retrieval from Qdrant and sparse retrieval using BM25.
    """
    try:
        query_lang = detect(query)
    except Exception:
        query_lang = None

    query_embedding = embedding_model.encode([query]).tolist()[0]

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k * 10,  # Adjusted limit for more candidate results
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
            continue  # Skip duplicate pages from the same document
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
    Performs dual retrieval:
    - If query is in German, translate it to English.
    - Run hybrid search on both the original query and its translation.
    - Merge and re-rank the results, removing duplicate (document, page) entries.
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
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{doc_name}_page_{page_number + 1}.png")
    img_list.append(os.path.abspath(output_path).replace('\\', '/'))
    return render_page(pdf_path, page_number, scale=scale, output_path=output_path)

def get_pdf_paths(root_folder):
    """
    Recursively finds all PDF files in the given root folder and its subdirectories.
    """
    pdf_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(subdir, file))
    return pdf_paths

# Setup the Gemma pipeline (LLM only on GPU; other tasks remain on CPU)
pipe = pipeline("image-text-to-text", model="google/gemma-3-4b-it", device="cuda", torch_dtype=torch.bfloat16)

# Example usage: set the root directory to your PDF folder
root_directory = "/content/paper"  # Change this to your folder path
pdf_files = get_pdf_paths(root_directory)
for pdf_file in pdf_files:
    store_pdf_embeddings(pdf_file, context_window=5)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    try:
        query_lang = detect(query)
    except Exception:
        query_lang = "en"

    system_prompt_text = "using the retrieved text and image only. Respond in English."

    best_results = hybrid_search_dual(query, top_k=5)
    if not best_results:
        print("No results found. Please try another query.")
        continue

    print("\n--- Top 5 Search Results ---")
    for res in best_results:
        print(f"Document: {res['document']}, Page {res['page']}")
        print(f"Text: {res['text']}")
        print(f"Score: {res['score']:.4f}")
        print("-" * 50)

    show_image = input("Would you like to see the retrieved pages as images? (yes/no): ").strip().lower()
    if show_image == "yes":
        img_list.clear()
        for res in best_results:
            pdf_path = next((path for path in pdf_files if os.path.basename(path) == res["document"]), None)
            if pdf_path:
                process_specific_page(pdf_path, res["page"] - 1, scale=10.0)
        if img_list:
            print("\nRendered images:")
            for idx, img_path in enumerate(img_list, start=1):
                print(f"{idx}: {img_path}")
        else:
            print("No images were rendered.")
    else:
        print("Skipping image rendering.")

    selected_image = None
    if img_list:
        image_selection = input("Enter the number of the image you want to send to the LLM (or press Enter to skip): ").strip()
        if image_selection:
            try:
                idx = int(image_selection) - 1
                if 0 <= idx < len(img_list):
                    selected_image = img_list[idx]
                    print(f"Selected image: {selected_image}")
                else:
                    print("Invalid index. No image will be sent.")
            except ValueError:
                print("Invalid input. No image will be sent.")

    user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
    if user_prompt:
        try:
            prompt_lang = detect(user_prompt)
        except Exception:
            prompt_lang = "en"
        if prompt_lang == "de":
            user_prompt = translate_german_to_english(user_prompt)

    combined_prompt = f"{user_prompt}\n" + "\n".join([res["text"] for res in best_results])
    print("\n--- Generated Text ---")

    # Build conversation based on whether an image is selected.
    if selected_image:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": selected_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    else:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt_text}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": combined_prompt}],
            },
        ]

    # Directly call the Gemma pipeline (running on GPU) with the conversation.
    output = pipe(text=conversation, max_new_tokens=900)
    print(output[0]['generated_text'])
