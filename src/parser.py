import os
import pymupdf
from llama_index.core import Document
from typing import List

def load_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    """Load a PDF and convert each page to a LlamaIndex Document with metadata."""
    doc = pymupdf.open(pdf_path)
    documents = []

    for i, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": os.path.basename(pdf_path),
                    "page_number": i + 1,
                    "total_pages": len(doc),
                },
            )
        )
    doc.close()
    return documents
