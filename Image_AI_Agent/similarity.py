import torch
import os
import json
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import IPython.display as display

# Set image folder
IMAGE_FOLDER = "/content/sample_data"
CAPTION_FILE = "/content/captions.json"

# Load models
def load_models():
    print("Loading models...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return processor, model, embedder

processor, model, embedder = load_models()

# Generate caption for an image
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = model.generate(**inputs)
    return processor.decode(caption_ids[0], skip_special_tokens=True)

# Get text embedding
def get_text_embedding(text):
    return embedder.encode(text, convert_to_tensor=True).cpu().numpy()

# Process all images in the folder
def process_images():
    captions = {}

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            caption = generate_caption(image_path)
            embedding = get_text_embedding(caption).tolist()
            captions[filename] = {"caption": caption, "embedding": embedding}

    # Save captions
    with open(CAPTION_FILE, "w") as f:
        json.dump(captions, f)

    return captions

# Run processing
print("Processing images...")
captions_data = process_images()
print("Processing complete!")

# Function to search for the best-matching image
def search_image(query):
    with open(CAPTION_FILE, "r") as f:
        captions_data = json.load(f)

    best_match = None
    best_score = float("-inf")
    query_embedding = get_text_embedding(query)

    for filename, data in captions_data.items():
        caption_embedding = np.array(data["embedding"])
        similarity = np.dot(query_embedding, caption_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(caption_embedding)
        )

        if similarity > best_score:
            best_score = similarity
            best_match = filename

    if best_match:
        print(f"Best match: {best_match}")
        print(f"Caption: {captions_data[best_match]['caption']}")
        display.display(Image.open(os.path.join(IMAGE_FOLDER, best_match)))
    else:
        print("No match found.")

# Example --------------------    :)
query = "a horse playing" # a dog playing
search_image(query)
