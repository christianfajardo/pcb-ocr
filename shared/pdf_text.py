"""PDF embedded text-layer extraction (not OCR) via PyMuPDF.

Distinct from `preprocessing.pdf_to_images`, which rasterizes pages for the
image-based OCR engines. This reads the PDF's actual text objects when
present — exact characters, no OCR-induced misreads — but many production
PCB drawings have no usable text layer at all (scanned pages, or CAD exports
with text converted to vector outlines rather than real text objects), so
callers must check `has_substantial_text_layer` before trusting the result.
"""

from __future__ import annotations

import os

import pymupdf

# Mirrors shared.preprocessing.MAX_PAGES. PyMuPDF must honor the same cap as
# the image engines: if it read text from pages they never saw, reconciliation
# would be voting on inconsistent inputs.
MAX_PAGES = int(os.environ.get("MAX_PAGES", "5"))


def extract_pdf_text_layer(pdf_path: str, max_pages: int | None = None) -> str:
    """Extract the PDF's embedded text layer, concatenated across pages.

    Args:
        pdf_path: Path to the PDF.
        max_pages: Stop after this many pages (defaults to MAX_PAGES; pass 0
            or negative for no limit).

    Returns:
        Extracted text, or an empty string if the PDF has no text layer.
    """
    limit = MAX_PAGES if max_pages is None else max_pages
    doc = pymupdf.open(pdf_path)
    try:
        pages = list(doc)
        if limit and limit > 0:
            pages = pages[:limit]
        return "\n".join(page.get_text() for page in pages)
    finally:
        doc.close()


def get_pdf_page_count(pdf_path: str) -> int:
    """Real page count from the PDF itself.

    Exists because the caller previously inferred this as
    `text.count("\f") + 1`, but pages are joined with "\n" and PyMuPDF's
    default text mode emits no form feed — so that always evaluated to 1.
    """
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def has_substantial_text_layer(text: str, min_chars: int = 200) -> bool:
    """Heuristic: is there enough real extractable text to bother parsing?

    Counts only alphanumeric characters so a page that's mostly whitespace,
    a lone watermark, or punctuation artifacts doesn't false-positive.
    """
    alnum_count = sum(1 for c in text if c.isalnum())
    return alnum_count >= min_chars
