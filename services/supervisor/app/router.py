"""Supervisor API router."""

from __future__ import annotations

import time

import structlog

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.logging_config import bind_pdf_filename

from .graph import compiled
from .state import create_initial_state

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["supervisor"])


def _sec(ms: float | None) -> float | None:
    """Convert a millisecond duration to seconds, rounded for display."""
    return round(ms / 1000, 3) if ms is not None else None


# (engine display name, state key prefix)
_ENGINES = [
    ("tesseract", "tesseract"),
    ("glm-ocr", "glm_ocr"),
    ("qwen-vl", "qwen_vl"),
    ("pymupdf", "pymupdf"),
]


@router.post("")
async def extract(file: UploadFile = File(...)) -> dict:
    """Run the full OCR pipeline on a PDF.

    Args:
        file: PDF file.

    Returns:
        Reconciled PCBData as dict.
    """
    import os
    import tempfile

    request_start = time.monotonic()

    with bind_pdf_filename(file.filename):
        logger.info("Pipeline started", filename=file.filename)

        try:
            # Save upload to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(await file.read())
                pdf_path = tmp.name

            try:
                # Create initial state
                state = create_initial_state(pdf_path, file.filename)

                # Run the graph
                result = await compiled.ainvoke(state)

                merged = result.get("reconciled")
                errors = result.get("errors", [])
                log = result.get("reconciliation_log", [])
                attribution = result.get("attribution", {})

                if not merged:
                    raise HTTPException(
                        status_code=500,
                        detail=f"No output: {errors}",
                    )

                total_duration_sec = round(time.monotonic() - request_start, 3)

                logger.info(
                    "Pipeline complete",
                    layer_count=merged.layer_count,
                    material=merged.material,
                    total_duration_sec=total_duration_sec,
                )

                response = merged.model_dump()
                response["reconciliation_log"] = log
                response["attribution"] = attribution
                response["errors"] = errors
                response["engine_durations_sec"] = [
                    {
                        "engine": name,
                        "duration_sec": _sec(result.get(f"{prefix}_duration_ms")),
                        "start_time": result.get(f"{prefix}_start_time"),
                        "end_time": result.get(f"{prefix}_end_time"),
                    }
                    for name, prefix in _ENGINES
                ]
                response["total_duration_sec"] = total_duration_sec

                return response

            finally:
                os.unlink(pdf_path)

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
