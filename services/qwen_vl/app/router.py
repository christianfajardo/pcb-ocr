"""Qwen3-VL API router."""

from __future__ import annotations

import base64
import io
import json
import structlog
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from shared.auth import require_api_key
from shared.confidence import build_confidence_map
from shared.logging_config import bind_pdf_filename
from shared.page_input import load_page_input
from shared.preprocessing import MAX_PAGES
from shared.schemas import PCBData, PCBDataWithConfidence, normalize_units

from .vlm_client import QwenVLClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["qwen-vl"])

DPI = int(__import__("os").environ.get("OCR_DPI", "300"))
qwen_client = QwenVLClient()


@router.post("", dependencies=[Depends(require_api_key)])
async def extract(
    file: UploadFile | None = File(default=None),
    pages: list[UploadFile] | None = File(default=None),
) -> dict:
    """Extract PCB data using Qwen3-VL.

    Args:
        file: PDF file.

    Returns:
        PCBDataWithConfidence as dict.
    """
    start = time.monotonic()

    page_input = await load_page_input(file, pages, DPI, max_pages=MAX_PAGES)

    with bind_pdf_filename(page_input.filename):
        logger.info("Qwen-VL extract started", filename=page_input.filename)

        try:
            try:
                images = page_input.images
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
                page_input.cleanup()

        except Exception as e:
            logger.error("Qwen-VL extraction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


#: Fields accumulated across pages rather than taken from the first page.
_LIST_FIELDS = ("ipc_specs", "drill_table", "layer_stackup")


def _is_blank(value: object) -> bool:
    """True for values carrying no information (None or empty/whitespace str)."""
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _row_key(row: object) -> str:
    """Stable identity for a table row, for cross-page dedup."""
    if isinstance(row, dict):
        return json.dumps(row, sort_keys=True, default=str)
    return str(row)


def _dedupe_rows(rows: list) -> list:
    """Drop repeated rows while preserving order.

    Multi-sheet fab packages routinely repeat the drill chart or stackup on
    more than one sheet. Without this, those rows are counted twice — which
    doubles drill quantities and produces duplicate layer numbers.
    """
    seen: set[str] = set()
    unique = []
    for row in rows:
        key = _row_key(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _merge_page_results(results: list[dict]) -> dict:
    """Merge structured results from multiple pages."""
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    merged = dict(results[0])
    # Copy list values so accumulating across pages never mutates the caller's
    # page-1 dict in place (dict() above is only a shallow copy).
    for field in _LIST_FIELDS:
        if isinstance(merged.get(field), list):
            merged[field] = list(merged[field])

    for r in results[1:]:
        for k, v in r.items():
            if k in _LIST_FIELDS and isinstance(v, list):
                # Not setdefault(): if the key exists holding None, setdefault
                # returns that None and .extend() raises.
                existing = merged.get(k)
                merged[k] = existing + v if isinstance(existing, list) else list(v)
            elif k == "fabrication_notes" and v:
                if merged.get(k):
                    merged[k] += "\n" + v
                else:
                    merged[k] = v
            elif _is_blank(merged.get(k)):
                # Blank, not just None — an empty string from an earlier page
                # must not permanently block a real value from a later one.
                merged[k] = v

    for field in _LIST_FIELDS:
        if isinstance(merged.get(field), list):
            merged[field] = _dedupe_rows(merged[field])

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
