#pip install gliclass


from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# Load model and tokenizer
model = GLiClassModel.from_pretrained("knowledgator/gliclass-large-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-large-v1.0")

# Initialize pipeline
pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device='cuda:0'
)

# Input queries and labels
queries = [
    "What is 2+2?",
    "How do I build a RAG system with multimodal embeddings?",
    "Hi there!",
    "Write a function to simulate drone trajectory with wind drag",
    "Who is the president of USA?",
]
labels = ["trivial", "non-trivial", "greetings", "technology"]

# Run classification
for i in queries:
    results = pipeline(i, labels, threshold=0.5)[0]  # Because we have one text
    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    for result in sorted_results:
        print(result["label"], "=>", result["score"])
    print("-----------End----------------")
