"""GLM-OCR API router."""

from __future__ import annotations

import base64
import io
import structlog
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.confidence import build_confidence_map
from shared.logging_config import bind_pdf_filename
from shared.preprocessing import pdf_to_images
from shared.schemas import PCBData, PCBDataWithConfidence, normalize_units

from .vlm_client import GLMOCRClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["glm-ocr"])

DPI = int(__import__("os").environ.get("OCR_DPI", "300"))
glm_client = GLMOCRClient()


@router.post("")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract PCB data using GLM-OCR.

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
                    # Convert to base64
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    text = await glm_client.extract_text(b64)
                    all_text_parts.append(text)

                raw_text = "\n".join(all_text_parts)

                # Structured extraction
                structured = await glm_client.structured_extract(raw_text)

                # Build PCBData from structured output
                data = _build_pcb_data(structured, raw_text)
                data = normalize_units(data)

                elapsed_ms = (time.monotonic() - start) * 1000

                # Confidence
                confidence = build_confidence_map(
                    data=data,
                    ocr_engine="glm-ocr",
                    raw_text_length=len(raw_text),
                    json_parse_success=structured is not None,
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


def _safe_int(value: object) -> int | None:
    """Best-effort int conversion; None for anything unparseable (e.g. "")."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_pcb_data(structured: dict | None, raw_text: str) -> PCBData:
    """Build PCBData from structured GLM-OCR output."""
    from shared.schemas import (
        BoardThickness,
        CopperWeights,
        DrillRow,
        ImpedanceControl,
        IPCClass,
        LayerSpec,
        PCBData,
        Silkscreen,
        SolderMask,
    )

    if not structured:
        return PCBData(ocr_raw_text=raw_text[:5000])

    data = PCBData()
    data.part_number = structured.get("part_number")
    data.manufacturer = structured.get("manufacturer")
    data.drawing_title = structured.get("drawing_title")
    data.is_itar = structured.get("is_itar", False)
    data.ipc_specs = structured.get("ipc_specs", [])
    data.layer_count = _safe_int(structured.get("layer_count"))
    data.material = structured.get("material")
    data.surface_finish = structured.get("surface_finish")
    data.fabrication_notes = structured.get("fabrication_notes")

    # IPC class
    ipc_class = structured.get("ipc_class")
    if ipc_class:
        try:
            data.ipc_class = IPCClass(ipc_class)
        except ValueError:
            pass

    # Board thickness
    bt = structured.get("board_thickness")
    if bt and isinstance(bt, dict):
        try:
            data.board_thickness = BoardThickness(
                nominal=bt.get("nominal"),
                plus_tol=bt.get("plus_tol"),
                minus_tol=bt.get("minus_tol"),
                unit="in",
            )
        except Exception as e:
            logger.warning("Skipping malformed board_thickness", error=str(e))

    # Copper weights
    cw = structured.get("copper_weights")
    if cw and isinstance(cw, dict):
        try:
            data.copper_weights = CopperWeights(**cw)
        except Exception as e:
            logger.warning("Skipping malformed copper_weights", error=str(e))

    # Solder mask
    sm = structured.get("solder_mask")
    if sm and isinstance(sm, dict):
        try:
            data.solder_mask = SolderMask(**sm)
        except Exception as e:
            logger.warning("Skipping malformed solder_mask", error=str(e))

    # Silkscreen
    sk = structured.get("silkscreen")
    if sk and isinstance(sk, dict):
        try:
            data.silkscreen = Silkscreen(**sk)
        except Exception as e:
            logger.warning("Skipping malformed silkscreen", error=str(e))

    # Impedance
    imp = structured.get("impedance_control")
    if imp and isinstance(imp, dict):
        try:
            if not isinstance(imp.get("single_ended"), dict):
                imp = {**imp, "single_ended": None}
            data.impedance_control = ImpedanceControl(**imp)
        except Exception as e:
            logger.warning("Skipping malformed impedance_control", error=str(e))

    # Layer stackup
    stackup = structured.get("layer_stackup")
    if stackup and isinstance(stackup, list):
        try:
            data.layer_stackup = [LayerSpec(**s) for s in stackup]
        except Exception as e:
            logger.warning("Skipping malformed layer_stackup", error=str(e))

    # Drill table
    dt = structured.get("drill_table")
    if dt and isinstance(dt, list):
        try:
            data.drill_table = [DrillRow(**d) for d in dt]
        except Exception as e:
            logger.warning("Skipping malformed drill_table", error=str(e))

    return data
