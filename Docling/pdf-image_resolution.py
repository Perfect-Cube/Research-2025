# install both pypdfium and pypdfium2 , nhi to gye
import pypdfium2 as pdfium
import os

# Path to your PDF and output directory for saving images.
pdf_path = "/content/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"
output_dir = "saved_images"
os.makedirs(output_dir, exist_ok=True)

# Open the PDF document.
pdf = pdfium.PdfDocument(pdf_path)
num_pages = len(pdf)
print(f"Total pages: {num_pages}")

scale = 7.0  # Scale factor to achieve a high-resolution rendering.

# Iterate over the pages and render each page.
for i in range(num_pages):
    page = pdf.get_page(i)
    # Render the page at the specified scale and convert to a PIL Image.
    bitmap = page.render(scale=scale) # Render the page to a bitmap
    pil_image = bitmap.to_pil() # Convert the bitmap to a PIL image
    
    output_path = os.path.join(output_dir, f"saved_image_page_{i+1}.png")
    pil_image.save(output_path)
    print(f"Saved page {i+1} as: {output_path}")
    
    # It's good practice to close the page once done.
    page.close()

pdf.close()
