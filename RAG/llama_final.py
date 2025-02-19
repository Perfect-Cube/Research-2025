import torch 
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor
) 

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

local_path = "C:/Users/S1QNEOV/Downloads/Gemma_RAG/saved images/saved images_5.png"
image = Image.open(local_path).convert("RGB")

prompt = "<|image|><|begin_of_text|>Describe document number,version,date."
inputs = processor(image, prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=250)
print(processor.decode(output[0]))
