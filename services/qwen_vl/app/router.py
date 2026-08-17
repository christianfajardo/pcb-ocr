"""Qwen3-VL API router."""

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

from .vlm_client import QwenVLClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["qwen-vl"])

DPI = int(__import__("os").environ.get("OCR_DPI", "300"))
qwen_client = QwenVLClient()


@router.post("")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract PCB data using Qwen3-VL.

    Args:
        file: PDF file.

    Returns:
        PCBDataWithConfidence as dict.
    """
    import os
    import tempfile

    start = time.monotonic()

    with bind_pdf_filename(file.filename):
        logger.info("Qwen-VL extract started", filename=file.filename)

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            try:
                # Convert PDF to images
                images = pdf_to_images(tmp_path, DPI)
                page_count = len(images)

                # Extract from each page
                all_results: list[dict] = []
                for img in images:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    result = await qwen_client.structured_extract(b64)
                    if result:
                        all_results.append(result)

                # Merge results from multiple pages
                merged = _merge_page_results(all_results)

                # Build PCBData
                data = _build_pcb_data(merged)
                data = normalize_units(data)

                elapsed_ms = (time.monotonic() - start) * 1000

                # Confidence
                confidence = build_confidence_map(
                    data=data,
                    ocr_engine="qwen-vl",
                    json_parse_success=len(all_results) > 0,
                )

                result = PCBDataWithConfidence(
                    data=data,
                    confidence=confidence,
                    ocr_engine="qwen-vl",
                    processing_time_ms=elapsed_ms,
                    page_count=page_count,
                )

                logger.info(
                    "Qwen-VL extract complete",
                    elapsed_ms=round(elapsed_ms, 1),
                    pages=page_count,
                )
                return result.model_dump()

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error("Qwen-VL extraction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


def _merge_page_results(results: list[dict]) -> dict:
    """Merge structured results from multiple pages."""
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    merged = dict(results[0])
    for r in results[1:]:
        for k, v in r.items():
            if (
                k == "ipc_specs"
                and isinstance(v, list)
                or k == "drill_table"
                and isinstance(v, list)
                or k == "layer_stackup"
                and isinstance(v, list)
            ):
                merged.setdefault(k, []).extend(v)
            elif k == "fabrication_notes" and v:
                if merged.get(k):
                    merged[k] += "\n" + v
                else:
                    merged[k] = v
            elif merged.get(k) is None:
                merged[k] = v

    # Dedupe ipc_specs
    if isinstance(merged.get("ipc_specs"), list):
        merged["ipc_specs"] = list(dict.fromkeys(merged["ipc_specs"]))

    return merged


def _safe_int(value: object) -> int | None:
    """Best-effort int conversion; None for anything unparseable (e.g. "")."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_pcb_data(structured: dict) -> PCBData:
    """Build PCBData from structured Qwen-VL output."""
    from shared.schemas import (
        BoardThickness,
        CopperWeights,
        DrillRow,
        ImpedanceControl,
        ImpedanceRange,
        IPCClass,
        LayerSpec,
        PCBData,
        Silkscreen,
        SolderMask,
    )

    if not structured:
        return PCBData()

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
                raw=bt.get("raw"),
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
            se = imp.get("single_ended")
            if not isinstance(se, dict):
                imp = {**imp, "single_ended": None}
            else:
                imp = {**imp, "single_ended": ImpedanceRange(**se)}
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
