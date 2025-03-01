![image-1](https://github.com/user-attachments/assets/1483f82e-4a1b-441f-a3ec-e0f56e944d97)



```
 ┌────────────┐
 │    Start    │
 └────────────┘
         ↓
 ┌────────────────────────────────────────────────────┐
 │1. PDF Extraction (pyPDF2 + pdfPlumber, BSD License)│
 └────────────────────────────────────────────────────┘
         ↓
 ┌───────────────────────────────────────────────────────────────────┐
 │2. Query System:                                                  │
 │   - Transform text → encoder (dense + sparse embeddings)         │
 │   - Generate recommendations (≈85% accuracy)                     │
 └───────────────────────────────────────────────────────────────────┘
         ↓
 ┌───────────────────────────────────────────────────────────────────┐
 │3. User Interface:                                                │
 │   - Display details of the page                                  │
 │   - Ask: "Is an image needed?"                                   │
 └───────────────────────────────────────────────────────────────────┘
         ↓                         ┌────────────┐
        Yes ──────────────────────▶│ Extract    │
         │                         │ image from │
         │                         │ PDF page   │
         │                         └────────────┘
         │                                   ↓
         │                          (image extracted)
         │                                   ↓
         └───────────────────────────────────▶┐
                                             │
                                     ┌─────────────────────────────────────────┐
                                     │4. Llama 3.2 Vision:                    │
                                     │   - Combine <image> + text → Generation│
                                     └─────────────────────────────────────────┘
                                             ↓
                                     ┌─────────────────┐
                                     │       End       │
                                     └─────────────────┘

```




# Building a RAG System for Complex PDFs

Creating a Retrieval-Augmented Generation (RAG) system for PDFs with mixed content (text, tables, images) and substantial length requires several components working together. Here's a comprehensive approach:

## 1. PDF Extraction Pipeline

First, you need to extract and process all content types:

```python
# Core extraction libraries
import fitz (PyMuPDF)  # For comprehensive PDF handling
import pytesseract     # For OCR on images
import pandas as pd    # For table processing
```

### Text Extraction
```python
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_blocks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Extract text while preserving some layout information
        text = page.get_text("dict")
        blocks = text["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Type 0 is text
                block_text = " ".join([line["spans"][0]["text"] for line in block["lines"]])
                text_blocks.append({
                    "text": block_text,
                    "page": page_num + 1,
                    "block_id": len(text_blocks),
                    "type": "text"
                })
    
    return text_blocks
```

### Table Extraction
```python
def extract_tables_from_pdf(pdf_path):
    # Using tabula-py or camelot for table extraction
    import camelot
    
    tables = []
    table_dfs = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
    
    for i, table in enumerate(table_dfs):
        tables.append({
            "type": "table",
            "content": table.df.to_dict(),
            "page": table.parsing_report['page'],
            "table_id": i
        })
    
    return tables
```

### Image Handling
```python
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Use OCR to extract text from image
            image_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
            
            images.append({
                "type": "image",
                "text": image_text,
                "page": page_num + 1,
                "image_id": f"{page_num}_{img_index}"
            })
    
    return images
```

## 2. Chunking Strategy for Long Documents

For 100-page documents, strategic chunking is critical:

```python
def chunk_document(text_blocks, tables, images, chunk_size=1000, overlap=200):
    all_content = text_blocks + tables + images
    all_content.sort(key=lambda x: (x["page"], x.get("block_id", 0)))
    
    chunks = []
    current_chunk = ""
    current_chunk_metadata = {
        "pages": set(),
        "contains_tables": False,
        "contains_images": False
    }
    
    for item in all_content:
        if item["type"] == "text":
            content = item["text"]
        elif item["type"] == "table":
            content = f"TABLE: {str(item['content'])}"
            current_chunk_metadata["contains_tables"] = True
        else:  # image
            content = f"IMAGE: {item['text']}"
            current_chunk_metadata["contains_images"] = True
        
        current_chunk_metadata["pages"].add(item["page"])
        
        if len(current_chunk) + len(content) > chunk_size:
            chunks.append({
                "text": current_chunk,
                "metadata": current_chunk_metadata
            })
            
            # Create overlap for context continuity
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk_metadata = {
                "pages": set(),
                "contains_tables": False,
                "contains_images": False
            }
        
        current_chunk += content + " "
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "metadata": current_chunk_metadata
        })
    
    return chunks
```

## 3. Vector Database Setup

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def create_vector_store(chunks):
    # Create texts and metadatas for vector store
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Convert set to list for JSON serialization
    for metadata in metadatas:
        metadata["pages"] = list(metadata["pages"])
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vector_store
```

## 4. RAG Implementation

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def setup_rag(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    llm = OpenAI(temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def query_pdf(qa_chain, query):
    result = qa_chain({"query": query})
    
    return {
        "answer": result["result"],
        "source_pages": list(set([doc.metadata["pages"] for doc in result["source_documents"]]))
    }
```

## 5. Putting It All Together

```python
def build_pdf_rag(pdf_path):
    # Extract content
    text_blocks = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    
    # Create chunks
    chunks = chunk_document(text_blocks, tables, images)
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    # Set up RAG
    qa_chain = setup_rag(vector_store)
    
    return qa_chain
```

## 6. Improvements for Complex PDFs

For better handling of complex PDFs:

1. **Hybrid search**: Combine vector search with keyword search
2. **Hierarchical chunking**: Maintain document structure (sections, subsections)
3. **Content-aware chunking**: Adjust chunk sizes based on content type
4. **Multi-modal embeddings**: Use models that understand both text and images
5. **Table semantics**: Convert tables to meaningful textual descriptions

Would you like me to elaborate on any of these components or suggest specific libraries for implementation?
