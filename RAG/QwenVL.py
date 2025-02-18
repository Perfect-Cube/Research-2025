# # Use a pipeline as a high-level helper
# from transformers import pipeline

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-7B-Instruct")
# pipe(messages)

# from transformers import pipeline

# # Define the messages with an image URL and a text instruction
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Create the pipeline for image-text-to-text generation using the Qwen2.5-VL model
# pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-7B-Instruct")

# # Run the pipeline with the provided messages and print the output
# result = pipe(messages)
# print(result)


# from transformers import pipeline

# # Create the pipeline for image-text-to-text generation using the Qwen2.5-VL model
# pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-7B-Instruct")

# # Provide a dictionary with "image" and "text" keys
# input_data = {
#     "images": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#     "text": "Describe this image."
# }

# # Run the pipeline with the provided input and print the output
# result = pipe(input_data)
# print(result)
# import torch
# from transformers import pipeline
# from PIL import Image

# # Define the model checkpoint or local path
# model_checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"  # or "./your_local_checkpoint" if saved locally

# # Create the pipeline for image-text-to-text generation
# pipe = pipeline(
#     "image-text-to-text",
#     model=model_checkpoint,
#     torch_dtype=torch.float16,  # adjust dtype as needed
#     device=-1  # use GPU (set to -1 for CPU)
# )

# # Load your local image using PIL
# local_image_path = "image02.jpg"  # replace with your image path
# image = Image.open(local_image_path).convert("RGB")

# # Provide the input with the proper key and include the <image> token in the text prompt
# input_data = {
#     "images": image,
#     "text": "<image> Describe this image."
# }

# # Run the pipeline and print the output
# result = pipe(input_data)
# print(result)



import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

# Define your model checkpoint or local path.
model_checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"  # or your local checkpoint path

# Load the processor and model.
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load your local image (ensure it's RGB).
local_image_path = "image04.jpg"  # update this with your local image path
image = Image.open(local_image_path).convert("RGB")

# Create a messages list for the chat template.
# The processor will automatically insert the appropriate number of <image> tokens.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Generate the chat prompt text with proper <image> tokens.
# This ensures that the number of tokens in the text matches the number of image features.
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process the vision inputs (this will extract image features).
image_inputs, video_inputs = process_vision_info(messages)

# Create the model inputs using the processed text and image features.
inputs = processor(
    text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cpu")  # or .to("cpu") if you don't have a GPU

# Generate the output.
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# Trim off the input prompt tokens and decode the output.
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text)
