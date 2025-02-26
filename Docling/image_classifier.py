import os
import pikepdf
import pypdfium2 as pdfium
from PIL import Image
 
def get_total_pages(pdf_path):
    """Return the total number of pages in the PDF using pikepdf."""
    with pikepdf.Pdf.open(pdf_path) as pdf:
        return len(pdf.pages)
 
def page_has_image(pdf, page_number):
    """
    Check if the specified page (from an open pikepdf.Pdf object)
    contains any embedded images.
    """
    page = pdf.pages[page_number]
    resources = page.get("/Resources")
    if resources is None:
        return False
    xobjects = resources.get("/XObject")
    if xobjects is None:
        return False
    # Iterate through all XObjects in the resources.
    for key, xobj in xobjects.items():
        try:
            # Check if the XObject subtype is 'Image'
            if xobj.get("/Subtype") == "/Image":
                return True
        except Exception:
            continue
    return False
 
def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    """
    Render the specified page using pypdfium2 at the given scale,
    and optionally save the output as an image.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_number)
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
    For each page in the PDF, if an embedded image is detected,
    render the full page using pypdfium2 and save it.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_pages = get_total_pages(pdf_path)
    print(f"Total pages: {total_pages}")
    # Open the PDF once with pikepdf to check for images.
    with pikepdf.Pdf.open(pdf_path) as pdf:
        for page_num in range(total_pages):
            if page_has_image(pdf, page_num):
                output_path = os.path.join(output_dir, f"page_{page_num+1}.png")
                render_page(pdf_path, page_num, scale=scale, output_path=output_path)
            else:
                print(f"Page {page_num+1} does not contain an embedded image; skipping rendering.")
 
# Example usage:
pdf_path = "/content/PS_2.1_011_1756_01 (2).pdf"
process_pdf_pages(pdf_path, scale=10.0, output_dir="rendered_page")
