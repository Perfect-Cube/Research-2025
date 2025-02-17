import fitz  # PyMuPDF
import os

def split_pdf(input_pdf, output_folder):
    """
    Splits a PDF file into individual pages.
    
    :param input_pdf: Path to the input PDF file.
    :param output_folder: Folder where the split pages will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    doc = fitz.open(input_pdf)
    num_pages = len(doc)  # Get the number of pages before closing the document
    for page_num in range(num_pages):
        output_pdf_path = os.path.join(output_folder, f"page_{page_num + 1}.pdf")
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        new_doc.save(output_pdf_path)
        new_doc.close()
    
    doc.close()
    print(f"PDF split into {num_pages} pages and saved in {output_folder}")  # Use num_pages here

def zoom_image_in_pdf(input_pdf, output_pdf, zoom_factor):
    """
    Zooms images in a PDF file.
    
    :param input_pdf: Path to the input PDF file.
    :param output_pdf: Path to the output PDF file with zoomed images.
    :param zoom_factor: Scaling factor for zooming.
    """
    doc = fitz.open(input_pdf)
    mat = fitz.Matrix(zoom_factor, zoom_factor)
    
    new_doc = fitz.open()
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_page = new_doc.new_page(width=pix.width, height=pix.height)
        img_page.insert_image(img_page.rect, pixmap=pix)
    
    new_doc.save(output_pdf)
    new_doc.close()
    doc.close()
    print(f"Zoomed PDF saved as {output_pdf}")

# Example usage
split_pdf("/content/Solar Pond Power Plant.pdf", "split_pages")
zoom_image_in_pdf("/content/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf", "zoomed_output.pdf", 6.0)
