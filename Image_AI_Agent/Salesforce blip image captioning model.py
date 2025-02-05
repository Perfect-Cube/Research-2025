import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return processor, model, embedder

def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

def get_text_embedding(text, embedder):
    embedding = embedder.encode(text, convert_to_tensor=True)
    return embedding

def process_images(image_folder):
    processor, model, embedder = load_model()
    results = {}

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            caption = generate_caption(image_path, processor, model)
            embedding = get_text_embedding(caption, embedder)
            results[filename] = {
                "caption": caption,
                "embedding": embedding.cpu().numpy()
            }

    return results

# Example usage:
image_folder = "/content/sample_data"  # Change this to your image folder path
results = process_images(image_folder)

# Print results
for image, data in results.items():
    print(f"Image: {image}\nCaption: {data['caption']}\nEmbedding: {data['embedding'][:5]}...\n")
