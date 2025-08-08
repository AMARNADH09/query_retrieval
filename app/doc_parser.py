
import fitz  # pymupdf
from docx import Document
from typing import Tuple, Dict

def parse_pdf(file_path: str) -> Tuple[str, Dict]:
    doc = fitz.open(file_path)
    pages = []
    for p in doc:
        pages.append(p.get_text())
    return "\n".join(pages), {"type": "pdf", "pages": len(pages)}

def parse_docx(file_path: str) -> Tuple[str, Dict]:
    d = Document(file_path)
    paras = [p.text for p in d.paragraphs]
    return "\n".join(paras), {"type": "docx", "pages": None}

def parse_document(file_path: str) -> Tuple[str, Dict]:
    fp = file_path.lower()
    if fp.endswith(".pdf"): return parse_pdf(file_path)
    if fp.endswith(".docx"): return parse_docx(file_path)
    raise ValueError("Unsupported file type")
