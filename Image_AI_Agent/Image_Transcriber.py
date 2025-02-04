# %% [code]
# Install required libraries (run this cell)
!pip install transformers faiss-cpu torch torchvision langchain-groq pillow


import os
import torch
import numpy as np
from getpass import getpass
from langchain_groq import ChatGroq
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# --- Set up the Groq API key ---
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter API key for Groq: ")

# Initialize the Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")  # Updated model name

# --- Load the BLIP model for image captioning ---
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --- Function to generate image captions using BLIP ---
def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# --- Function to retrieve unique images from the vector DB ---
def retrieve_images(query_embedding, index, metadata, top_k=5):
    query_embedding = np.expand_dims(query_embedding, axis=0).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    
    unique_results = []
    seen = set()
    
    for idx in indices[0]:
        if idx < 0:
            continue
        file_path = metadata[idx]["file_path"]
        if file_path not in seen:
            unique_results.append(metadata[idx])
            seen.add(file_path)
    
    return unique_results

# --- Updated RAG Pipeline with BLIP Captioning ---
def rag_pipeline(user_query, clip_processor, clip_model, faiss_index, image_metadata, llm, top_k=5):
    # Compute text embedding using CLIP
    text_inputs = clip_processor(text=[user_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    
    text_embedding = text_features[0] / text_features[0].norm(p=2)
    query_embedding = text_embedding.cpu().numpy()
    
    # Retrieve images from FAISS index
    retrieved = retrieve_images(query_embedding, faiss_index, image_metadata, top_k)

    # Generate captions for retrieved images
    captioned_images = []
    for item in retrieved:
        file_path = item["file_path"]
        caption = generate_caption(file_path, blip_processor, blip_model)
        captioned_images.append({"file_path": file_path, "caption": caption})

    # Construct LLM prompt with captions
    context_str = "\n".join([f"Image file: {img['file_path']}, Caption: {img['caption']}" for img in captioned_images])
    
    prompt = (
        f"User Query: '{user_query}'\n\n"
        f"Retrieved images and captions:\n{context_str}\n\n"
        "Based on the above images and captions, please provide a summary of the common visual themes."
    )

    # Invoke the Groq LLM
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content, captioned_images

# --- Example Usage ---
user_query = "Peacock is walking in the street and in the background some stairs are there"
answer, retrieved_images = rag_pipeline(user_query, clip_processor, clip_model, faiss_index, image_metadata, llm, top_k=5)

# Output results
print("LLM Response:\n", answer)
print("\nRetrieved Images and Captions:")
for item in retrieved_images:
    print(f"File: {item['file_path']}, Caption: {item['caption']}")
