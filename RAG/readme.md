![1740200920462](https://github.com/user-attachments/assets/96b10434-1c75-401a-a1bf-ed4eee269b76)

![1742835535016](https://github.com/user-attachments/assets/6c7244ff-5587-4189-9ba7-c6ca5c50e6a3)


```
+---------------+       +----------------+       +---------------------+
|   PDF Input   | ---> | PDF Parser     | ---> | Text + Image Output |
+---------------+       +----------------+       +---------------------+
                               |                          |
                               |                          v
                               |                  +------------------+
                               |                  | Image Classifier |
                               |                  |  (Mama Vision)   |
                               |                  +------------------+
                               |                          |
                               v                          |
                     +---------------------+               |
                     | Text Embedding     |               |
                     | (Mistral 7B)       |               |
                     +---------------------+               |
                               |                          |
                               v                          |
                     +---------------------+               |
                     | Retrieval +         |               |
                     | Extraction (RAG)    |               |
                     +---------------------+               |
                               |                          |
                               v                          |
                     +---------------------+               |
                     | Generate Response   | <-------------
                     | (LLM Inference)     |
                     +---------------------+
                               |
                               v
                     +---------------------+
                     |  Final Output      |
                     | (Text / Images)    |
                     +---------------------+

```



llama token hugging-"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"

test cases for RAG:
```
Document Overview:

    What is the primary purpose of this process standard document, and what does the P3+ process aim to achieve?

Authorship and Responsibility:

    Who is the author of this document, and which roles are assigned process responsibility?

Role Definitions:

    What are the key responsibilities of the E/E-Verbundmanager within the P3+ process?

Change Management:

    How are software and hardware changes managed, and what role does the "Änderungsmanagement Technische Konformität P3" play in this process?

Homologation Procedures:

    How does the process address homologation-relevant software changes, and what steps are involved in the pre-evaluation of such changes?

Coordination and Communication:

    What is the role of the Koordinator zentrale Antragstellung P3+, and how do they interact with other process participants?

Documentation and Tools:

    Which tools and systems (e.g., Jira, SharePoint) are mentioned for document management and tracking within the process?

Decision Points and Escalation:

    What are the key decision points outlined in the process, and what actions are taken when a change is deemed non-conform or requires escalation?

Neutrality Proof:

    What is the significance of the Neutralitätsnachweis, and how is it created, managed, and integrated into the process?

Diagram and Process Flow:

    How does the provided diagram illustrate the timeline/milestones and interaction between roles (e.g., E/E-Verbundmanager, Bauteilverantwortlicher, Funktionsrealisierungsverantwortlicher)?
```

## output generated
 Describe the document details (number, version, date) across the images.
**Document Details**

* **Document Number**: PS_2.1_011_1075_05
* **Version**: 01.00
* **Date**: Seite 11 von 11

The document is a technical document, likely a standard operating procedure (SOP) or a technical specification, used by Volkswagen. It is written in German and has a formal tone. The document is divided into sections, with the first section providing an overview of the document's purpose and scope. The second section provides more detailed information about the document's content and structure. The third section lists the relevant standards and regulations that the document is based on. The fourth section outlines the document's approval process and version control. The fifth section provides information about the document's maintenance and update process. The sixth section lists the document's appendices and references. The seventh section provides a glossary of terms used in the document. The eighth section provides a summary of the document's key points. The ninth section provides additional information and resources related to the document. The tenth section provides contact information for the document's author and approver. The eleventh section provides a disclaimer and copyright information.     

**Key Points**

* The document is a technical document used by Volkswagen.
* It is written in German and has a formal tone.
* The document is divided into sections, with the first section providing an overview of the document's purpose and scope.
* The document is based on relevant standards and regulations.
* The document's approval process and version control are outlined.
* The document's maintenance and update process are described.
* The document's appendices and references are listed.
* A glossary of terms is provided.
* A summary of the document's key points is included.
* Additional information and resources are provided.
* Contact information for the document's author and approver is included.
* A disclaimer and copyright information are provided.

**Summary**

The document is a technical document used by Volkswagen to provide information about its processes and procedures. It is written in German and has a formal tone. The document is divided into sections, with the first section providing an overview of the document's purpose and scope. The document is based on relevant standards and regulations, and its approval process and version control are outlined. The document's maintenance and update process are described, and its appendices and references are listed. A glossary of terms is provided, and a summary of the document's key points is included. Additional information and resources are provided, along with contact information for the document's author and approver. A disclaimer and copyright information are also included. Overall, the document provides a comprehensive overview of Volkswagen's processes and procedures, and is likely used as a reference guide for employees and stakeholders.

**Answer**: The document is a technical document used by Volkswagen to provide information about its processes and procedures. It is written in German and has a formal tone. The document is divided into sections, with the first section providing an overview of the document's purpose and scope. The document is based on relevant standards and regulations, and its approval process and version control are outlined. The document's maintenance and update process are described, and its appendices and references are listed. A glossary of terms is provided, and a summary of the document's key points is included. Additional information and resources are provided, along with contact information for the document's author and approver. A disclaimer and copyright information are also included. Overall, the document provides a comprehensive overview of Volkswagen's processes and procedures, and is likely used as a reference guide for employees and stakeholders.

**Answer**: Technical document used by Volkswagen to provide information about its processes and procedures. It is written in German and has a formal tone. The document is divided into sections, with the first section providing an overview of the document's purpose and scope. The document is based on relevant standards and regulations, and its approval process and version control are outlined. The document's maintenance and update process are described, and its appendices and references are listed. A glossary of terms is provided, and a summary of the document's key points is included. Additional information and resources are provided, along with contact information for the document's author and approver. A disclaimer and copyright information are also included. Overall, the document provides a comprehensive overview of Volkswagen's processes and procedures, and is likely used as a reference guide for employees and stakeholders.



**PDF2Go**

To use PDF2Go, you can upload your PDF to their website and use their built-in zoom feature. However, if you prefer to use Python, you can use the `PyPDF2` library to achieve similar results. Here's an example code snippet:
```python
from PyPDF2 import PdfFileReader

# Open the PDF file
pdf = PdfFileReader("example.pdf")

# Get the first page
page = pdf.getPage(0)

# Extract the image data
image_data = page.extract_image_data()

# Save the image data to a new PDF file
with open("zoomed_pdf.pdf", "wb") as f:
    f.write(image_data)

print("Zoomed PDF file saved as zoomed_pdf.pdf")
```
**GooPDF**

Similarly, GooPDF is a web-based tool that allows you to zoom in on PDFs. To use it with Python, you can use the `requests` library to send a GET request to their API, passing the file as a binary attachment. Here's an example code snippet:
```python
import requests

# Open the PDF file
with open("example.pdf", "rb") as f:
    # Send a GET request to GooPDF's API
    response = requests.get("https://goo-pdf.com/api/zoom", files={"file": f})

# Save the zoomed PDF file
with open("zoomed_pdf.pdf", "wb") as f:
    f.write(response.content)

print("Zoomed PDF file saved as zoomed_pdf.pdf")
```
**SmallPDF**

SmallPDF is another web-based tool that allows you to zoom in on PDFs. To use it with Python, you can use the `requests` library to send a GET request to their API, passing the file as a binary attachment. Here's an example code snippet:
```python
import requests

# Open the PDF file
with open("example.pdf", "rb") as f:
    # Send a GET request to SmallPDF's API
    response = requests.get("https://smallpdf.com/api/zoom", files={"file": f})

# Save the zoomed PDF file
with open("zoomed_pdf.pdf", "wb") as f:
    f.write(response.content)

print("Zoomed PDF file saved as zoomed_pdf.pdf")
```
**IrfanView**

IrfanView is a free image viewer that can be used to zoom in on images. To use it with Python, you can use the `subprocess` library to call the IrfanView executable with the necessary arguments. Here's an example code snippet:
```python
import subprocess

# Open the image file
image_file = "example.jpg"

# Use IrfanView to zoom in on the image
subprocess.run(["mv", image_file, "zoomed.jpg"])
subprocess.run(["i_view32", "/zoom", "100%"])

print("Zoomed image saved as zoomed.jpg")
```
**XnView**

XnView is another free image viewer that can be used to zoom in on images. To use it with Python, you can use the `subprocess` library to call the XnView executable with the necessary arguments. Here's an example code snippet:
```python
import subprocess

# Open the image file
image_file = "example.jpg"

# Use XnView to zoom in on the image
subprocess.run(["mv", image_file, "zoomed.jpg"])
subprocess.run(["xnview", "/zoom", "100%"])

print("Zoomed image saved as zoomed.jpg")
```
**GIMP**

GIMP is a free image editor that can be used to zoom in on images. To use it with Python, you can use the `subprocess` library to call the GIMP executable with the necessary arguments. Here's an example code snippet:
```python
import subprocess

# Open the image file
image_file = "example.jpg"

# Use GIMP to zoom in on the image
subprocess.run(["gimp", "--no-interface", "--batch", "--", "--new", "--mode=RGB", "--layers", "--options='resolution[pxi]'=0", "--import", "--export", "--options", "--selection", "--histogram", "--magnify", "100%", image_file, "zoomed.jpg"])

print("Zoomed image saved as zoomed.jpg")
```
