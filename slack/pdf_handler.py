# pdf_handler.py

from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from a PDF document.
    
    Args:
        pdf_file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF document.
    """
    reader = PdfReader(pdf_file_path)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text