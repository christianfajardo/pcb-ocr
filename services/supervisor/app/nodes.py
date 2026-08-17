"""LangGraph node implementations for the OCR pipeline."""

from __future__ import annotations

import asyncio
import structlog
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from shared.confidence import build_confidence_map, safe_merge_results
from shared.pcb_parser import parse_pcb_text
from shared.pdf_text import extract_pdf_text_layer, has_substantial_text_layer
from shared.preprocessing import image_to_bytes, pdf_to_images
from shared.schemas import PCBDataWithConfidence, normalize_units

logger = structlog.get_logger(__name__)

# Service URLs
TESSERACT_URL = os.environ.get("TESSERACT_URL", "http://tesseract-ocr:8001/extract")
GLM_OCR_URL = os.environ.get("GLM_OCR_URL", "http://glm-ocr-api:8002/extract")
QWEN_VL_URL = os.environ.get("QWEN_VL_URL", "http://qwen-vl-api:8003/extract")

DPI = int(os.environ.get("OCR_DPI", "300"))
PDF_TEXT_LAYER_MIN_CHARS = int(os.environ.get("PDF_TEXT_LAYER_MIN_CHARS", "200"))

# Node timestamps are reported in Pacific time (auto-handles PST/PDT), not UTC.
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


async def ingest_pdf_node(state: dict) -> dict:
    """Convert PDF to images for downstream OCR nodes.

    Args:
        state: Pipeline state with pdf_path.

    Returns:
        Updated state with page_images.
    """
    pdf_path = state["pdf_path"]
    logger.info("Ingesting PDF", pdf_path=pdf_path)

    start = time.monotonic()
    images = pdf_to_images(pdf_path, DPI)

    page_images: list[bytes] = []
    for img in images:
        page_images.append(image_to_bytes(img))

    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "PDF ingestion complete",
        pages=len(images),
        elapsed_ms=round(elapsed, 1),
    )

    return {"page_images": page_images}


async def tesseract_extract_node(state: dict) -> dict:
    """Call Tesseract OCR service.

    Args:
        state: Pipeline state with pdf_path.

    Returns:
        Updated state with tesseract_result.
    """
    logger.info("Tesseract extraction started")
    return await _call_ocr_node(state, TESSERACT_URL, "tesseract", "tesseract")


async def glm_ocr_extract_node(state: dict) -> dict:
    """Call GLM-OCR service.

    Args:
        state: Pipeline state with pdf_path.

    Returns:
        Updated state with glm_ocr_result.
    """
    logger.info("GLM-OCR extraction started")
    return await _call_ocr_node(state, GLM_OCR_URL, "glm_ocr", "glm-ocr")


async def qwen_vl_extract_node(state: dict) -> dict:
    """Call Qwen3-VL service.

    Args:
        state: Pipeline state with pdf_path.

    Returns:
        Updated state with qwen_vl_result.
    """
    logger.info("Qwen-VL extraction started")
    return await _call_ocr_node(state, QWEN_VL_URL, "qwen_vl", "qwen-vl")


async def pymupdf_extract_node(state: dict) -> dict:
    """Extract PCB data from the PDF's own embedded text layer, when present.

    Runs locally in-process (no HTTP call, no GPU) — it's just reading text
    objects out of the PDF. Skips itself (contributes no result) when the
    PDF has no substantial text layer, which is common in production: scans,
    or CAD exports with text converted to vector outlines. When it does have
    real text, it's exact-character extraction with no OCR-induced errors.

    Args:
        state: Pipeline state with pdf_path.

    Returns:
        Updated state with pymupdf_result (or None if skipped).
    """
    logger.info("PyMuPDF extraction started")
    pdf_path = state["pdf_path"]
    node_start = time.monotonic()
    node_start_dt = datetime.now(PACIFIC_TZ)

    try:
        text = extract_pdf_text_layer(pdf_path)
        node_elapsed = (time.monotonic() - node_start) * 1000
        end_time = datetime.now(PACIFIC_TZ).isoformat()

        if not has_substantial_text_layer(text, PDF_TEXT_LAYER_MIN_CHARS):
            logger.info(
                "PyMuPDF skipped — no substantial text layer",
                text_len=len(text),
            )
            return {
                "pymupdf_duration_ms": node_elapsed,
                "pymupdf_start_time": node_start_dt.isoformat(),
                "pymupdf_end_time": end_time,
            }

        data = parse_pcb_text(text)
        data = normalize_units(data)
        confidence = build_confidence_map(data=data, ocr_engine="pymupdf")

        result = PCBDataWithConfidence(
            data=data,
            confidence=confidence,
            ocr_engine="pymupdf",
            processing_time_ms=node_elapsed,
            page_count=text.count("\f") + 1,  # PyMuPDF separates pages with \f
        )

        logger.info(
            "PyMuPDF extraction complete",
            elapsed_ms=round(node_elapsed, 1),
            text_len=len(text),
        )

        return {
            "pymupdf_result": result,
            "pymupdf_duration_ms": node_elapsed,
            "pymupdf_start_time": node_start_dt.isoformat(),
            "pymupdf_end_time": end_time,
        }

    except Exception as e:
        node_elapsed = (time.monotonic() - node_start) * 1000
        logger.error("PyMuPDF extraction failed", error=str(e))
        return {
            "pymupdf_duration_ms": node_elapsed,
            "pymupdf_start_time": node_start_dt.isoformat(),
            "pymupdf_end_time": datetime.now(PACIFIC_TZ).isoformat(),
            "errors": [f"pymupdf failed: {e!s}"],
        }


async def _call_ocr_node(
    state: dict,
    url: str,
    key_prefix: str,
    engine_name: str,
) -> dict:
    """Call an OCR service with retry logic.

    Args:
        state: Pipeline state.
        url: Service URL.
        key_prefix: State-key prefix for this engine (e.g. "tesseract"),
            used to derive the result/duration/timestamp state keys.
        engine_name: Engine name for logging.

    Returns:
        Updated state.
    """
    result_key = f"{key_prefix}_result"
    duration_key = f"{key_prefix}_duration_ms"
    start_time_key = f"{key_prefix}_start_time"
    end_time_key = f"{key_prefix}_end_time"

    pdf_path = state["pdf_path"]
    # Forward the client's original filename, not the temp file's random
    # name — downstream engines use it to group their own logs under the
    # same /logs/{filename}.log as the rest of the pipeline.
    upload_filename = state.get("original_filename") or os.path.basename(pdf_path)
    new_errors: list[str] = []
    node_start = time.monotonic()
    node_start_dt = datetime.now(PACIFIC_TZ)

    for attempt in range(3):
        try:
            start = time.monotonic()
            async with httpx.AsyncClient(timeout=300) as client:
                with open(pdf_path, "rb") as f:
                    response = await client.post(
                        url,
                        files={"file": (upload_filename, f, "application/pdf")},
                    )
                response.raise_for_status()

                data = response.json()
                result = PCBDataWithConfidence(**data)
                elapsed = (time.monotonic() - start) * 1000

                logger.info(
                    f"{engine_name} extraction complete",
                    engine=engine_name,
                    elapsed_ms=round(elapsed, 1),
                    pages=result.page_count,
                )

                node_elapsed = (time.monotonic() - node_start) * 1000
                return {
                    result_key: result,
                    duration_key: node_elapsed,
                    start_time_key: node_start_dt.isoformat(),
                    end_time_key: datetime.now(PACIFIC_TZ).isoformat(),
                }

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            new_errors.append(
                f"{engine_name} attempt {attempt + 1} failed after {round(elapsed, 1)}ms: {e!s}"
            )
            logger.error(
                f"{engine_name} extraction failed",
                engine=engine_name,
                attempt=attempt + 1,
                error=str(e),
            )
            if attempt < 2:
                await asyncio.sleep(2**attempt)  # exponential backoff

    # All attempts failed
    logger.error(f"{engine_name} all attempts exhausted", errors=new_errors)
    node_elapsed = (time.monotonic() - node_start) * 1000
    return {
        result_key: None,
        duration_key: node_elapsed,
        start_time_key: node_start_dt.isoformat(),
        end_time_key: datetime.now(PACIFIC_TZ).isoformat(),
        "errors": new_errors,
    }


async def reconcile_node(state: dict) -> dict:
    """Reconcile 3 OCR results into one merged PCBData.

    Args:
        state: Pipeline state with all 3 OCR results.

    Returns:
        Updated state with reconciled data and log.
    """
    logger.info("Reconciliation started")

    start = time.monotonic()

    try:
        results = [
            r
            for r in (
                state.get("tesseract_result"),
                state.get("glm_ocr_result"),
                state.get("qwen_vl_result"),
                state.get("pymupdf_result"),
            )
            if r is not None
        ]
        if not results:
            raise ValueError("No OCR engine produced a result")

        merged, log = safe_merge_results(results)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Reconciliation complete",
            elapsed_ms=round(elapsed, 1),
            log_entries=len(log),
        )

        return {
            "reconciled": merged,
            "reconciliation_log": log,
            "processing_time_ms": elapsed,
        }

    except Exception as e:
        logger.error("Reconciliation failed", error=str(e))
        return {"errors": [f"Reconciliation failed: {e!s}"]}


async def validate_node(state: dict) -> dict:
    """Validate the reconciled output.

    Args:
        state: Pipeline state with reconciled data.

    Returns:
        Updated state with validation errors (if any).
    """
    new_errors: list[str] = []
    merged = state.get("reconciled")

    if not merged:
        new_errors.append("Validation failed: no reconciled data")
        return {"errors": new_errors}

    # Validate dimensional ranges
    if merged.board_thickness and merged.board_thickness.nominal:
        if merged.board_thickness.nominal <= 0 or merged.board_thickness.nominal > 1.0:
            new_errors.append(
                f"Board thickness {merged.board_thickness.nominal} out of "
                f"reasonable range (0-1 inches)"
            )

    for row in merged.drill_table:
        if row.size_mils and (row.size_mils <= 0 or row.size_mils > 500):
            new_errors.append(
                f"Drill size {row.size_mils} mils out of range (likely wrong units)"
            )

    logger.info(
        "Validation complete",
        errors_found=len(new_errors),
    )

    return {"errors": new_errors}
