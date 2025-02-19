import gradio as gr
import requests
import torch
from PIL import Image
import spaces
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
from huggingface_hub import login

huggingface_token = os.getenv("SECRET_ENV_VARIABLE")
login(huggingface_token)

# Load the Llama 3.2 Vision Model
def load_llama_model():
    model_id = "meta-llama/Llama-3.2-11B-Vision"

    # Load model and processor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload", 
    )
    model.tie_weights() 
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

# Function to generate predictions for text and image
@spaces.GPU
def process_input(text, image=None):
    model, processor = load_llama_model()

    if image:
        # If an image is uploaded, process it as a PIL Image object
        vision_input = image.convert("RGB").resize((224, 224))

        prompt = f"<|image|><|begin_of_text|>{text}"

        # Process image and text together
        inputs = processor(vision_input, prompt, return_tensors="pt").to(model.device)
    else:
        # If no image is uploaded, just process the text
        prompt = f"<|begin_of_text|>{text}"
        inputs = processor(prompt, return_tensors="pt").to(model.device)

    # Generate output from the model
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode the output to return a readable text
    decoded_output = processor.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

def demo():
    # Define Gradio input and output components
    text_input = gr.Textbox(label="Text Input", placeholder="Enter text here", lines=5)
    image_input = gr.Image(label="Upload an Image", type="pil")
    output = gr.Textbox(label="Model Output", lines=3)

    # Add two examples for multimodal analysis
    examples = [
        ["The llama is ", "./examples/llama.png"],
        ["The cute hampster is wearing ", "./examples/hampster.png"]
    ]

    # Define the interface layout
    interface = gr.Interface(
        fn=process_input,
        inputs=[text_input, image_input],
        outputs=output,
        examples=examples,
        title="Llama 3.2 Multimodal Text-Image Analyzer",
        description="Upload an image and/or provide text for analysis using the Llama 3.2 Vision Model. You can also try out the provided examples.",
    )

    # Launch the demo
    interface.launch()

# Run the demo
if __name__ == "__main__":
    demo()
