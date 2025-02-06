import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import faiss
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from groq import Groq

# Set the Groq API key as an environment variable
os.environ["groq_api_key"] = "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"  # Replace with your actual API key

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
    return embedding.cpu().numpy()


def store_in_vector_db(embeddings, image_names):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "vector_db.index")
    with open("image_names.txt", "w") as f:
        for name in image_names:
            f.write(name + "\n")


def load_vector_db():
    index = faiss.read_index("vector_db.index")
    with open("image_names.txt", "r") as f:
        image_names = [line.strip() for line in f.readlines()]
    return index, image_names


def query_groq_api(user_prompt, image_captions):
    """
    Queries the Groq API to rank image captions based on the user prompt.

    Args:
        user_prompt: The user's query.
        image_captions: A dictionary mapping image names to their captions.

    Returns:
        The ranked list of image captions as a string.
    """
    client = Groq(api_key=os.environ.get("groq_api_key"))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Rank the following image captions based on how well they match the user prompt.",
            },
            {
                "role": "user",
                "content": f"User Prompt: {user_prompt}\nCaptions: {json.dumps(image_captions)}",
            },
        ],
        model="llama-3.3-70b-versatile",  # or another suitable Groq model
    )

    return chat_completion.choices[0].message.content


def process_images(image_folder):
    processor, model, embedder = load_model()
    results = {}
    image_names = []
    embeddings_list = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, filename)
            caption = generate_caption(image_path, processor, model)
            embedding = get_text_embedding(caption, embedder)
            results[filename] = {"caption": caption, "embedding": embedding}
            image_names.append(filename)
            embeddings_list.append(embedding)

    embeddings_array = np.vstack(embeddings_list)
    store_in_vector_db(embeddings_array, image_names)
    return results


def rank_images(user_prompt, image_folder):
    _, _, embedder = load_model()
    index, image_names = load_vector_db()
    query_embedding = get_text_embedding(user_prompt, embedder)
    distances, indices = index.search(np.array([query_embedding]), k=len(image_names))
    ranked_results = [(image_names[i], distances[0][j]) for j, i in enumerate(indices[0])]
    ranked_results.sort(key=lambda x: x[1])  # Sort by distance (lower is better)

    image_captions = {name: generate_caption(os.path.join(image_folder, name), *load_model()[:2]) for name, _ in ranked_results}
    refined_ranking = query_groq_api(user_prompt, image_captions)
    print("Refined Ranking by Groq API:")
    print(refined_ranking)

    # Display the highest-ranked image
    best_match_image = os.path.join(image_folder, ranked_results[0][0])
    img = Image.open(best_match_image)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Best Match: {ranked_results[0][0]}\nScore: {ranked_results[0][1]:.4f}")
    plt.show()


# Example usage:
image_folder = "/content/sample_data"  # Change this to your image folder path
results = process_images(image_folder)

# Example query:
user_prompt = "A yellow snake"
rank_images(user_prompt, image_folder)
