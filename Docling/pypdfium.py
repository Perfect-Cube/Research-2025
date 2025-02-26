!pip install PyPDF2 pypdfium2 pillow

import os
from PyPDF2 import PdfReader
import pypdfium2 as pdfium
from PIL import Image
 
def get_total_pages(pdf_path):
    """Return the total number of pages in the PDF using PyPDF2."""
    reader = PdfReader(pdf_path)
    return len(reader.pages)
 
# def page_has_image(reader, page_number):
#     """
#     Check if the specified page (using a PyPDF2 PdfReader)
#     contains any embedded images by inspecting its '/XObject' resources.
#     """
#     page = reader.pages[page_number]
#     resources = page.get("/Resources")
#     if not resources:
#         return False
#     xobjects = resources.get("/XObject")
#     if not xobjects:
#         return False
#     for xobj_name in xobjects:
#         xobj = xobjects[xobj_name]
#         # Check if the XObject's subtype is Image
#         if xobj.get("/Subtype") == "/Image":
#             return True
#     return False

def page_has_image(reader, page_number):
    """
    Check if the specified page (using a PyPDF2 PdfReader)
    contains any embedded images by inspecting its '/XObject' resources.
    """
    page = reader.pages[page_number]
    resources = page.get("/Resources")
    if not resources:
        return False
    # Resolve the indirect object if necessary.
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    xobjects = resources.get("/XObject")
    if not xobjects:
        return False
    # Resolve xobjects if it's an indirect object.
    if hasattr(xobjects, "get_object"):
        xobjects = xobjects.get_object()
    for xobj_name in xobjects:
        xobj = xobjects[xobj_name]
        # Resolve each xobj if needed.
        if hasattr(xobj, "get_object"):
            xobj = xobj.get_object()
        if xobj.get("/Subtype") == "/Image":
            return True
    return False

 
def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    """
    Render the specified page using pypdfium2 at the given scale.
    Saves the image if an output path is provided.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_number)
    # Render the page at high resolution (scale acts like a transformation matrix)
    bitmap = page.render(scale=scale) # Render the page to a bitmap
    pil_image = bitmap.to_pil()
    if output_path:
        pil_image.save(output_path)
        print(f"Saved rendered page {page_number + 1} with image to: {output_path}")
    page.close()
    pdf.close()
    return pil_image
 
def process_pdf_pages(pdf_path, scale=1.0, output_dir="rendered_pages"):
    """
    Loop over all pages in the PDF. For each page, if an embedded image is detected
    (using PyPDF2), render the entire page with pypdfium2 and save the result.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_pages = get_total_pages(pdf_path)
    print(f"Total pages: {total_pages}")
    # Open PDF once with PyPDF2 for inspection.
    reader = PdfReader(pdf_path)
    for page_num in range(total_pages):
        if page_has_image(reader, page_num):
            output_path = os.path.join(output_dir, f"page_{page_num+1}.png")
            render_page(pdf_path, page_num, scale=scale, output_path=output_path)
        else:
            print(f"Page {page_num+1} does not contain an embedded image; skipping rendering.")
 
# Example usage:
pdf_path = "/content/PS_2.1_011_1756_01 (2).pdf"
process_pdf_pages(pdf_path, scale=10.0, output_dir="rendered_pages")
