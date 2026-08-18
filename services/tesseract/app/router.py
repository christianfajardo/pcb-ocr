"""Tesseract OCR API router."""

from __future__ import annotations

import structlog
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.confidence import build_confidence_map
from shared.logging_config import bind_pdf_filename
from shared.pcb_parser import parse_pcb_text_with_provenance
from shared.schemas import PCBDataWithConfidence, normalize_units

from .ocr_engine import TesseractOCR

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["tesseract"])

DPI = int(__import__("os").environ.get("OCR_DPI", "300"))
ocr_engine = TesseractOCR(base_dpi=DPI)


@router.post("")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract PCB data from a PDF using Tesseract.

    Args:
        file: PDF file.

    Returns:
        PCBDataWithConfidence as dict.
    """
    import os
    import tempfile

    start = time.monotonic()

    with bind_pdf_filename(file.filename):
        logger.info("Tesseract extract started", filename=file.filename)

        try:
            # Save upload to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            try:
                # Run OCR
                ocr_result = ocr_engine.extract(tmp_path)

                # Parse structured data
                data, provenance = parse_pcb_text_with_provenance(ocr_result["raw_text"])

                # Normalize units
                data = normalize_units(data)

                elapsed_ms = (time.monotonic() - start) * 1000

                # Build confidence map
                confidence = build_confidence_map(
                    data=data,
                    ocr_engine="tesseract",
                    tesseract_avg_confidence=ocr_result.get("word_confidences"),
                    provenance=provenance,
                )

                result = PCBDataWithConfidence(
                    data=data,
                    confidence=confidence,
                    ocr_engine="tesseract",
                    processing_time_ms=elapsed_ms,
                    page_count=ocr_result["page_count"],
                )

                logger.info(
                    "Tesseract extract complete",
                    elapsed_ms=round(elapsed_ms, 1),
                    pages=result.page_count,
                )
                return result.model_dump()

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error("Tesseract extraction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
