# # pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision-release'
# from transformers import AutoProcessor, AutoModelForImageTextToText
# import torch

# model_id = "CohereForAI/aya-vision-8b"

# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id, device_map="auto", torch_dtype=torch.float16
# )

# # Format message with the aya-vision chat template
# messages = [
#     {"role": "user",
#      "content": [
#        {"type": "image", "url": "https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&name=medium"},
#         {"type": "text", "text": "चित्र में लिखा पाठ क्या कहता है?"},
#     ]},
#     ]

# inputs = processor.apply_chat_template(
#     messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
# ).to(model.device)

# gen_tokens = model.generate(
#     **inputs, 
#     max_new_tokens=300, 
#     do_sample=True, 
#     temperature=0.3,
# )

# print(processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))



from transformers import pipeline

pipe = pipeline(model="CohereForAI/aya-vision-8b", task="image-text-to-text", device_map="auto")

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "C:/Users/S1QNEOV/Downloads/Complex LLM/PEP-HB-AGG_090508_jw_page_7.png"},
        {"type": "text", "text": "Describe the gantt chart present in lower side of page"},
    ]},
    ]
outputs = pipe(text=messages, max_new_tokens=400, return_full_text=False)

print(outputs)
