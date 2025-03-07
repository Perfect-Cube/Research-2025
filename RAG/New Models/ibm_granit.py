from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch
 
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "C:/Users/alukkib/Documents/Hybrid RAG/Final/rendered_page/AutomotiveSPICE_PRM_v45_page_47.png"

model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
 
# prepare image and text prompt, using the appropriate prompt template
 
# img_path = hf_hub_download(repo_id=model_path, filename=image_path)
 
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "What is the relation between acceptance testing and system according to the flowchart in the image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)
 
 
# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=900)
print(processor.decode(output[0], skip_special_tokens=True))
