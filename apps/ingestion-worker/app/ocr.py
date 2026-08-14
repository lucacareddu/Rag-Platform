from pathlib import Path

import pdfplumber
import pytesseract
from docx import Document
from PIL import Image


def extract_text(path: str) -> str:
    """Extracts text based on file extension. PDFs use embedded text first,
    falling back to OCR per page if a page has no extractable text.
    Images go straight through tesseract OCR."""
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return pytesseract.image_to_string(Image.open(path))
    if ext in {".txt", ".md"}:
        return Path(path).read_text(errors="ignore")

    raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                # Scanned page with no embedded text — OCR it.
                image = page.to_image(resolution=200).original
                page_text = pytesseract.image_to_string(image)
            text_parts.append(page_text or "")
    return "\n".join(text_parts)


def _extract_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]
