"""Accept either a PDF upload or pages the supervisor already rasterized.

The supervisor rasterizes each PDF once in `ingest_pdf_node` and hands the
same PNGs to every engine, so the file isn't rasterized four separate times.
Engines still accept a plain PDF, which keeps each service usable standalone
(they each have their own documented /extract, auth, and health endpoints).

Shared because all three image engines need identical handling.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import structlog
from fastapi import HTTPException, UploadFile
from PIL import Image

from .preprocessing import bytes_to_images, pdf_to_images

logger = structlog.get_logger(__name__)

# The supervisor tags each page part as "<original>::page-N.png" so the engine
# can recover the client's filename for per-PDF log grouping.
_PAGE_SUFFIX = "::page-"


@dataclass
class PageInput:
    """Pages to process, plus where they came from."""

    images: list[Image.Image]
    filename: str | None
    #: Temp PDF the caller must unlink when done; None in pre-rasterized mode.
    pdf_path: str | None

    def cleanup(self) -> None:
        if self.pdf_path and os.path.exists(self.pdf_path):
            os.unlink(self.pdf_path)


def _original_filename(part_name: str | None) -> str | None:
    """Recover the client's filename from a "<original>::page-N.png" part."""
    if not part_name:
        return None
    return part_name.split(_PAGE_SUFFIX)[0] if _PAGE_SUFFIX in part_name else part_name


async def load_page_input(
    file: UploadFile | None,
    pages: list[UploadFile] | None,
    dpi: int,
    max_pages: int | None = None,
) -> PageInput:
    """Resolve an /extract request to a list of page images.

    Exactly one of `file` (a PDF) or `pages` (pre-rasterized PNGs) must be
    supplied.

    Raises:
        HTTPException: 400 if neither or both were supplied.
    """
    if pages and file:
        raise HTTPException(
            status_code=400, detail="Provide either 'file' or 'pages', not both"
        )
    if not pages and not file:
        raise HTTPException(status_code=400, detail="Provide a PDF 'file' or 'pages'")

    if pages:
        blobs = [await p.read() for p in pages]
        images = bytes_to_images(blobs)
        filename = _original_filename(pages[0].filename)
        logger.info("Using pre-rasterized pages", pages=len(images), filename=filename)
        return PageInput(images=images, filename=filename, pdf_path=None)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    images = pdf_to_images(tmp_path, dpi, max_pages=max_pages)
    return PageInput(images=images, filename=file.filename, pdf_path=tmp_path)
