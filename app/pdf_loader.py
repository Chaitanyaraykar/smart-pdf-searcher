# app/pdf_loader.py

import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    """
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # remove extra spaces
    return text


if __name__ == "__main__":
    pdf_path = "sample.pdf"
    if not os.path.exists(pdf_path):
        print("Please place a PDF named 'sample.pdf' in the root directory.")
    else:
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        print("Extracted Text Preview:\n")
        print(cleaned_text[:100])
