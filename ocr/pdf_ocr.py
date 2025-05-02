# ocr/pdf_ocr.py

import pytesseract
from PIL import Image
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF and Tesseract OCR.
    Returns the full combined text from all pages.
    """
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text


def extract_text_from_image(image_path):
    """
    Extracts text from a single image (PNG/JPG).
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text
