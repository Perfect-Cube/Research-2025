import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a local transformer model and its tokenizer
model_name = "gpt2"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare a prompt for the model
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Measure time for generation
start_time = time.time()

# Generate tokens; adjust max_new_tokens as needed
output_ids = model.generate(input_ids, max_new_tokens=50)
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Decode the generated tokens and count them (excluding the prompt tokens if desired)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
generated_tokens = tokenizer.tokenize(output_text)  # Counts tokens in generated text
tokens_count = len(generated_tokens)

# Compute tokens per second
tokens_per_second = tokens_count / elapsed_time

print(f"Generated {tokens_count} tokens in {elapsed_time:.2f} seconds ({tokens_per_second:.2f} tokens/second)")
