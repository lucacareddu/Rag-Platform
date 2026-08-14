from unstructured.partition.auto import partition


def extract_text(path: str) -> str:
    """OCR/parses PDFs, images, docx, etc. via unstructured."""
    elements = partition(filename=path)
    return "\n".join(str(e) for e in elements)


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]
