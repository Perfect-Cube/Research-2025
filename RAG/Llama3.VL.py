import base64
from transformers import pipeline
from PyPDF2 import PdfReader
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text if text else "No readable text found."

def analyze_file(file_path, text_prompt):
    # Create the pipeline using the meta-llama/Llama-3.2-11B-Vision-Instruct model.
    pipe = pipeline("image-text-to-text", model="meta-llama/Llama-3.2-11B-Vision-Instruct")
    
    # If the file is a PDF, extract its text and build a text-only message.
    if file_path.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
        messages = [
            {
                "role": "user",
                "content": f"{text_prompt}\nExtracted Text:\n{extracted_text}"
            }
        ]
    else:
        # Assume it's an image. Load it locally.
        image = Image.open(file_path).convert("RGB")
        # Build a message that includes the image.
        # IMPORTANT: The image must be wrapped in a list with a dict indicating its type.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
    
    # Call the pipeline with the messages.
    result = pipe(messages)
    return result

# Example usage
if __name__ == "__main__":
    # Replace with your local file path (either a .pdf or an image file like .jpg or .png)
    file_path = "PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"  # or "your_document.pdf"
    text_prompt = "Analyze this file and provide insights."
    analysis = analyze_file(file_path, text_prompt)
    print("AI Analysis:", analysis)
