"""LangGraph node implementations for the OCR pipeline."""

from __future__ import annotations

import asyncio
import structlog
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from shared.auth import API_KEY
from shared.confidence import build_confidence_map, merge_with_attribution
from shared.cross_checks import run_cross_checks
from shared.pcb_parser import parse_pcb_text_with_provenance
from shared.pdf_text import extract_pdf_text_layer, has_substantial_text_layer
from shared.pdf_text import get_pdf_page_count as pdf_text_page_count
from shared.preprocessing import get_pdf_page_count, image_to_bytes, pdf_to_images
from shared.schemas import PCBDataWithConfidence, normalize_units

logger = structlog.get_logger(__name__)

# Service URLs
TESSERACT_URL = os.environ.get("TESSERACT_URL", "http://tesseract-ocr:8001/extract")
GLM_OCR_URL = os.environ.get("GLM_OCR_URL", "http://glm-ocr-api:8002/extract")
QWEN_VL_URL = os.environ.get("QWEN_VL_URL", "http://qwen-vl-api:8003/extract")

# Downstream engines require the same API_KEY the supervisor itself does —
# forwarded as a Bearer token on every outbound call. A no-op header when
# API_KEY isn't configured (see shared/auth.py).
_AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

DPI = int(os.environ.get("OCR_DPI", "300"))
PDF_TEXT_LAYER_MIN_CHARS = int(os.environ.get("PDF_TEXT_LAYER_MIN_CHARS", "200"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "5"))
# Per-page budget for a downstream engine call. Engines process pages
# sequentially, so a fixed timeout that suits 1 page guarantees failure on
# N — Qwen-VL alone runs ~90-215s/page on this hardware.
ENGINE_TIMEOUT_PER_PAGE_SEC = int(os.environ.get("ENGINE_TIMEOUT_PER_PAGE_SEC", "300"))

# Node timestamps are reported in Pacific time (auto-handles PST/PDT), not UTC.
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def _capped_page_count(total: int) -> int:
    """Pages actually processed, given the MAX_PAGES cap (0/neg = uncapped)."""
    return min(total, MAX_PAGES) if MAX_PAGES > 0 else total


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
    total_pages = get_pdf_page_count(pdf_path)
    images = pdf_to_images(pdf_path, DPI, max_pages=MAX_PAGES)

    page_images: list[bytes] = []
    for img in images:
        page_images.append(image_to_bytes(img))

    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "PDF ingestion complete",
        pages=len(images),
        total_pages=total_pages,
        elapsed_ms=round(elapsed, 1),
    )

    # Truncation must be visible — otherwise a 20-page PDF silently reports
    # page_count 5 and the caller has no idea the rest was never looked at.
    new_errors: list[str] = []
    if total_pages > len(images):
        msg = (
            f"Only first {len(images)} of {total_pages} pages scanned "
            f"(MAX_PAGES={MAX_PAGES})"
        )
        logger.warning("PDF truncated to page cap", pages=len(images), total_pages=total_pages)
        new_errors.append(msg)

    return {"page_images": page_images, "errors": new_errors}


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

        data, provenance = parse_pcb_text_with_provenance(text)
        data = normalize_units(data)
        confidence = build_confidence_map(
            data=data, ocr_engine="pymupdf", provenance=provenance
        )

        result = PCBDataWithConfidence(
            data=data,
            confidence=confidence,
            ocr_engine="pymupdf",
            processing_time_ms=node_elapsed,
            # 'pages processed', not the PDF's true length — matches what
            # the image engines report under the same MAX_PAGES cap.
            page_count=_capped_page_count(pdf_text_page_count(pdf_path)),
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
    page_images: list[bytes] = state.get("page_images") or []
    new_errors: list[str] = []
    node_start = time.monotonic()
    node_start_dt = datetime.now(PACIFIC_TZ)

    for attempt in range(3):
        try:
            start = time.monotonic()
            # Scale with page count: 1 page keeps the original 300s.
            timeout_sec = ENGINE_TIMEOUT_PER_PAGE_SEC * max(1, len(page_images))
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                # Send the pages the supervisor already rasterized in
                # ingest_pdf_node, rather than the PDF — otherwise every
                # engine re-rasterizes the same file (4x the work overall).
                # Falls back to the PDF if ingestion produced nothing.
                if page_images:
                    files = [
                        (
                            "pages",
                            (f"{upload_filename}::page-{i + 1}.png", blob, "image/png"),
                        )
                        for i, blob in enumerate(page_images)
                    ]
                    response = await client.post(url, files=files, headers=_AUTH_HEADERS)
                else:
                    with open(pdf_path, "rb") as f:
                        response = await client.post(
                            url,
                            files={"file": (upload_filename, f, "application/pdf")},
                            headers=_AUTH_HEADERS,
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

        merged, log, attribution = merge_with_attribution(results)

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Reconciliation complete",
            elapsed_ms=round(elapsed, 1),
            log_entries=len(log),
        )

        return {
            "reconciled": merged,
            "reconciliation_log": log,
            "attribution": attribution,
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

    # Self-consistency checks across fields (drill size/qty ranges included).
    # These can only run post-reconciliation, which is why they live here and
    # not in schemas.normalize_units — that runs per-engine, pre-merge.
    new_errors.extend(run_cross_checks(merged))

    logger.info(
        "Validation complete",
        errors_found=len(new_errors),
    )

    return {"errors": new_errors}
