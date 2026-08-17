"""PDF embedded text-layer extraction (not OCR) via PyMuPDF.

Distinct from `preprocessing.pdf_to_images`, which rasterizes pages for the
image-based OCR engines. This reads the PDF's actual text objects when
present — exact characters, no OCR-induced misreads — but many production
PCB drawings have no usable text layer at all (scanned pages, or CAD exports
with text converted to vector outlines rather than real text objects), so
callers must check `has_substantial_text_layer` before trusting the result.
"""

from __future__ import annotations

import pymupdf


def extract_pdf_text_layer(pdf_path: str) -> str:
    """Extract the PDF's embedded text layer, concatenated across pages.

    Returns:
        Extracted text, or an empty string if the PDF has no text layer.
    """
    doc = pymupdf.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def has_substantial_text_layer(text: str, min_chars: int = 200) -> bool:
    """Heuristic: is there enough real extractable text to bother parsing?

    Counts only alphanumeric characters so a page that's mostly whitespace,
    a lone watermark, or punctuation artifacts doesn't false-positive.
    """
    alnum_count = sum(1 for c in text if c.isalnum())
    return alnum_count >= min_chars
