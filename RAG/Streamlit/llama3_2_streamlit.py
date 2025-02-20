import streamlit as st
import fitz  # PyMuPDF
import os
import io
from PIL import Image
import re
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch

def join_images_vertically(image_paths, output_path):
    """Joins multiple images vertically into a single long image."""
    images = [Image.open(img_path) for img_path in image_paths]
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)
    
    combined_image = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    combined_image.save(output_path)
    return output_path

st.title("Visual Intelligence")
st.header("Intelligent PDF & Visual Analyzer")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
scale = st.slider("Scale Factor", 1.0, 10.0, 6.0)
selected_images = []
selected_image_paths = []

if uploaded_files:
    output_base = "extracted_images"
    os.makedirs(output_base, exist_ok=True)
    mat = fitz.Matrix(scale, scale)
    extracted_images = {}
    
    for uploaded_file in uploaded_files:
        pdf_name = os.path.splitext(uploaded_file.name)[0]
        pdf_output_dir = os.path.join(output_base, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
        
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        extracted_images[pdf_name] = []
        
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_bytes))
            
            img_path = os.path.join(pdf_output_dir, f"high_quality_page_{page_index}.png")
            img.save(img_path)
            extracted_images[pdf_name].append(img_path)
    
    # Display extracted images and allow selection
    st.subheader("Select Pages to Join")
    for pdf_name, img_paths in extracted_images.items():
        st.write(f"**{pdf_name}**")
        for img_path in img_paths:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
            with col2:
                if st.checkbox(f"Select {os.path.basename(img_path)}", key=img_path):
                    selected_images.append(img_path)
                    selected_image_paths.append(os.path.abspath(img_path).replace('\\','/'))
    
    if selected_image_paths:
        st.subheader("Selected Image Paths:")
        st.write([path for path in selected_image_paths])
    
    if st.button("Join Selected Pages") and selected_images:
        output_path = os.path.join(output_base, "joined_image.png")
        joined_image_path = join_images_vertically(selected_images, output_path)
        st.image(joined_image_path, caption="Joined Image", use_column_width=True)
        st.download_button("Download Joined Image", data=open(joined_image_path, "rb"), file_name="joined_image.png", mime="image/png")
    
    # AI Model Integration
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    user_prompt = st.text_area("Enter your prompt:")
    
    if st.button("Generate AI Response") and selected_image_paths:
        formatted_prompt = "<|image|>" * len(selected_image_paths) + "<|begin_of_text|> " + user_prompt
        images = [Image.open(path).convert("RGB") for path in selected_image_paths]
        inputs = processor(images=images, text=formatted_prompt, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=300)
        response = processor.decode(output[0], skip_special_tokens=True)
        st.subheader("AI Response:")
        st.write(response)
    
    
st.success("Processing complete.")
