import math
from pdf2image import convert_from_path
from PIL import Image
 
def render_pdf_page(pdf_path: str, page_number: int, dpi: int = 300, scale: float = 1.0) -> Image.Image:
    """
    Render a single PDF page to a PIL Image using pdf2image and then (optionally) upscale it.
 
    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Zero-indexed page number to render.
        dpi (int): The DPI at which to render the page (e.g. 300 for 300 DPI).
        scale (float): Upscaling factor applied after rendering.
                       (scale=1.0 means no additional scaling)
 
    Returns:
        PIL.Image.Image: The rendered (and optionally upscaled) page image.
    """
    # pdf2image uses 1-indexed page numbers.
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_number + 1, last_page=page_number + 1)
    if not pages:
        raise ValueError("No pages were rendered. Check your PDF path and parameters.")
    # Get the first (and only) page.
    image = pages[0]
    # If scale factor is not 1, resize (upscale) the image using Pillow.
    if scale != 1.0:
        new_width = math.ceil(image.width * scale)
        new_height = math.ceil(image.height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image
 
# Example usage:
if __name__ == "__main__":
    pdf_file = "/content/AutomotiveSPICE_PAM_31.pdf"  # Replace with your PDF file path.
    page_num = 11              # Render the first page (0-indexed).
    dpi_setting = 300         # Render at 300 DPI.
    upscale_factor = 7.5      # Optionally upscale the image 2×.
    rendered_image = render_pdf_page(pdf_file, page_num, dpi=dpi_setting, scale=upscale_factor)
    # rendered_image.show()  # Display the image.
    # Optionally, save the image:
    rendered_image.save("rendered_page_automotive.png")
