"""LangGraph state definition for the OCR pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from shared.schemas import PCBData, PCBDataWithConfidence


class PipelineState(TypedDict, total=False):
    """State for the OCR pipeline graph.

    Fields:
        pdf_path: Input PDF file path (a temp file, not the uploaded name).
        original_filename: The filename as uploaded by the client — forwarded
            to downstream engines and used to group logs under /logs.
        page_images: PNG images of each page (bytes).
        tesseract_result: Tesseract OCR result.
        glm_ocr_result: GLM-OCR result.
        qwen_vl_result: Qwen3-VL result.
        pymupdf_result: PyMuPDF text-layer result (None when the PDF has no
            substantial embedded text layer — common for scans/CAD exports).
        tesseract_duration_ms: Wall-clock time spent in the Tesseract node (all attempts).
        glm_ocr_duration_ms: Wall-clock time spent in the GLM-OCR node (all attempts).
        qwen_vl_duration_ms: Wall-clock time spent in the Qwen-VL node (all attempts).
        pymupdf_duration_ms: Wall-clock time spent in the PyMuPDF node.
        tesseract_start_time / tesseract_end_time: ISO-8601 Pacific-time timestamps
            bracketing the Tesseract node's execution.
        glm_ocr_start_time / glm_ocr_end_time: ISO-8601 Pacific-time timestamps
            bracketing the GLM-OCR node's execution.
        qwen_vl_start_time / qwen_vl_end_time: ISO-8601 Pacific-time timestamps
            bracketing the Qwen-VL node's execution.
        pymupdf_start_time / pymupdf_end_time: ISO-8601 Pacific-time timestamps
            bracketing the PyMuPDF node's execution.
        reconciled: Merged/reconciled PCBData.
        reconciliation_log: Audit trail of merge decisions.
        attribution: Per-field provenance — winning engine, the drawing text
            it attributed the value to, and all contributing engines.
        errors: List of error messages, appended to by concurrent nodes.
        processing_time_ms: Total processing time.
    """

    pdf_path: str
    original_filename: str | None
    page_images: list[bytes]
    tesseract_result: PCBDataWithConfidence | None
    glm_ocr_result: PCBDataWithConfidence | None
    qwen_vl_result: PCBDataWithConfidence | None
    pymupdf_result: PCBDataWithConfidence | None
    tesseract_duration_ms: float | None
    glm_ocr_duration_ms: float | None
    qwen_vl_duration_ms: float | None
    pymupdf_duration_ms: float | None
    tesseract_start_time: str | None
    tesseract_end_time: str | None
    glm_ocr_start_time: str | None
    glm_ocr_end_time: str | None
    qwen_vl_start_time: str | None
    qwen_vl_end_time: str | None
    pymupdf_start_time: str | None
    pymupdf_end_time: str | None
    reconciled: PCBData | None
    reconciliation_log: list[Any]
    attribution: dict[str, Any]
    errors: Annotated[list[str], operator.add]
    processing_time_ms: float


def create_initial_state(pdf_path: str, original_filename: str | None = None) -> dict:
    """Create initial pipeline state.

    Args:
        pdf_path: Path to input PDF (a temp file — not the uploaded name).
        original_filename: The filename as uploaded by the client, used for
            forwarding to downstream engines and for per-PDF log grouping.

    Returns:
        Initial state dict.
    """
    return {
        "pdf_path": pdf_path,
        "original_filename": original_filename,
        "page_images": [],
        "tesseract_result": None,
        "glm_ocr_result": None,
        "qwen_vl_result": None,
        "pymupdf_result": None,
        "tesseract_duration_ms": None,
        "glm_ocr_duration_ms": None,
        "qwen_vl_duration_ms": None,
        "pymupdf_duration_ms": None,
        "tesseract_start_time": None,
        "tesseract_end_time": None,
        "glm_ocr_start_time": None,
        "glm_ocr_end_time": None,
        "qwen_vl_start_time": None,
        "qwen_vl_end_time": None,
        "pymupdf_start_time": None,
        "pymupdf_end_time": None,
        "reconciled": None,
        "reconciliation_log": [],
        "attribution": {},
        "errors": [],
        "processing_time_ms": 0.0,
    }
