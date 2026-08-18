"""GLM-OCR API router."""

from __future__ import annotations

import base64
import io
import structlog
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.confidence import build_confidence_map
from shared.logging_config import bind_pdf_filename
from shared.pcb_parser import parse_pcb_text_with_provenance
from shared.preprocessing import pdf_to_images
from shared.schemas import PCBDataWithConfidence, normalize_units

from .vlm_client import GLMOCRClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["glm-ocr"])

DPI = int(__import__("os").environ.get("OCR_DPI", "300"))
glm_client = GLMOCRClient()


@router.post("")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract PCB data using GLM-OCR.

    GLM-OCR is a document-vision model, not a general reasoning LLM: its raw
    text recognition is strong (dense text, small fonts), but asking it to
    also map that text onto a complex nested JSON schema (this project's
    PCBData) produces empty or misattributed fields — verified directly
    against the model, both with our full schema and with a minimal prompt
    matching the model card's own "Information Extraction" example. So GLM-OCR
    is used here only for OCR; the proven shared regex parser (same one
    Tesseract and PyMuPDF use) turns its raw text into structured fields, with
    per-field source-text attribution as a side effect.

    Args:
        file: PDF file.

    Returns:
        PCBDataWithConfidence as dict.
    """
    import os
    import tempfile

    start = time.monotonic()

    with bind_pdf_filename(file.filename):
        logger.info("GLM-OCR extract started", filename=file.filename)

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            try:
                # Convert PDF to images
                images = pdf_to_images(tmp_path, DPI)
                page_count = len(images)

                # OCR each page
                all_text_parts: list[str] = []
                for img in images:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    text = await glm_client.extract_text(b64)
                    all_text_parts.append(text)

                raw_text = "\n".join(all_text_parts)

                # Parse structured data from the raw OCR text
                data, provenance = parse_pcb_text_with_provenance(raw_text)
                data.ocr_raw_text = raw_text[:5000]

                # Normalize units
                data = normalize_units(data)

                elapsed_ms = (time.monotonic() - start) * 1000

                # Confidence
                confidence = build_confidence_map(
                    data=data,
                    ocr_engine="glm-ocr",
                    raw_text_length=len(raw_text),
                    provenance=provenance,
                )

                result = PCBDataWithConfidence(
                    data=data,
                    confidence=confidence,
                    ocr_engine="glm-ocr",
                    processing_time_ms=elapsed_ms,
                    page_count=page_count,
                )

                logger.info(
                    "GLM-OCR extract complete",
                    elapsed_ms=round(elapsed_ms, 1),
                    pages=page_count,
                )
                return result.model_dump()

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error("GLM-OCR extraction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
