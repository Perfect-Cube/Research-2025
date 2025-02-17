# import torch
import warnings
import torch
from transformers import pipeline

# Suppress the batch_size deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure the correct device (CPU for Ryzen)
device = "cpu"

# Load the text-generation model
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

# User input message
messages = [
    {"role": "user", "content": "Tell about gemini flash 2"},
]

# Generate response
outputs = pipe(messages, max_new_tokens=256)

# Extract assistant response
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

# Print output
print(assistant_response)
