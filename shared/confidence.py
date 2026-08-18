"""Confidence scoring utilities for OCR engine outputs."""

from __future__ import annotations

import logging
from typing import Any

from .schemas import FieldConfidence, PCBData, PCBDataWithConfidence

logger = logging.getLogger(__name__)

# ── Base confidence scores per engine ─────────────────────────────────────────

TESSERACT_BASE: dict[str, float] = {
    "ocr_raw_text": 0.85,
    "fabrication_notes": 0.85,
    "part_number": 0.70,
    "drawing_title": 0.70,
    "manufacturer": 0.65,
    "ipc_class": 0.70,
    "layer_count": 0.70,
    "material": 0.70,
    "surface_finish": 0.70,
    "board_thickness": 0.70,
    "copper_weights": 0.65,
    "solder_mask": 0.65,
    "silkscreen": 0.65,
    "impedance_control": 0.60,
    "drill_table": 0.50,
    "layer_stackup": 0.55,
    "ipc_specs": 0.60,
    "is_itar": 0.75,
}

GLM_OCR_BASE: dict[str, float] = {
    # GLM-OCR's own structured JSON output was unreliable for this schema
    # (verified against the model directly: empty fields with our full
    # schema, misattributed fields even with a minimal prompt matching its
    # own model card example) — a capability gap, not a prompting bug. So
    # structured fields are now extracted by the same regex parser Tesseract
    # uses, over GLM-OCR's raw OCR text. Same mechanism as Tesseract, hence
    # the same base scores; GLM-OCR has no per-word confidence signal to
    # scale by, unlike Tesseract (see tesseract_word_confidence below).
    "ocr_raw_text": 0.85,
    "fabrication_notes": 0.85,
    "part_number": 0.70,
    "drawing_title": 0.70,
    "manufacturer": 0.65,
    "ipc_class": 0.70,
    "layer_count": 0.70,
    "material": 0.70,
    "surface_finish": 0.70,
    "board_thickness": 0.70,
    "copper_weights": 0.65,
    "solder_mask": 0.65,
    "silkscreen": 0.65,
    "impedance_control": 0.60,
    "drill_table": 0.50,
    "layer_stackup": 0.55,
    "ipc_specs": 0.60,
    "is_itar": 0.75,
}

QWEN_VL_BASE: dict[str, float] = {
    "ocr_raw_text": 0.80,
    "fabrication_notes": 0.70,
    "part_number": 0.80,
    "drawing_title": 0.80,
    "manufacturer": 0.80,
    "ipc_class": 0.80,
    "layer_count": 0.80,
    "material": 0.80,
    "surface_finish": 0.80,
    "board_thickness": 0.82,
    "copper_weights": 0.82,
    "solder_mask": 0.80,
    "silkscreen": 0.80,
    "impedance_control": 0.85,
    "drill_table": 0.85,
    "layer_stackup": 0.85,
    "ipc_specs": 0.80,
    "is_itar": 0.85,
}

PYMUPDF_BASE: dict[str, float] = {
    # Exact-character extraction from the PDF's own text layer — no OCR
    # guessing — so confidence runs higher than any OCR engine wherever it
    # has data. Same blind spot as Tesseract for symbol/table-heavy fields
    # (shape symbols, spatial table layout aren't recoverable from raw text).
    "ocr_raw_text": 0.95,
    "fabrication_notes": 0.90,
    "part_number": 0.88,
    "drawing_title": 0.88,
    "manufacturer": 0.85,
    "ipc_class": 0.88,
    "layer_count": 0.85,
    "material": 0.88,
    "surface_finish": 0.88,
    "board_thickness": 0.85,
    "copper_weights": 0.78,
    "solder_mask": 0.78,
    "silkscreen": 0.78,
    "impedance_control": 0.60,
    "drill_table": 0.50,
    "layer_stackup": 0.55,
    "ipc_specs": 0.85,
    "is_itar": 0.90,
}

ENGINE_BASE_MAP: dict[str, dict[str, float]] = {
    "tesseract": TESSERACT_BASE,
    "glm-ocr": GLM_OCR_BASE,
    "qwen-vl": QWEN_VL_BASE,
    "pymupdf": PYMUPDF_BASE,
}

STRUCTURED_FIELDS: set[str] = {
    "part_number",
    "drawing_title",
    "manufacturer",
    "ipc_class",
    "layer_count",
    "material",
    "surface_finish",
    "board_thickness",
    "copper_weights",
    "solder_mask",
    "silkscreen",
    "impedance_control",
    "is_itar",
}

TABLE_FIELDS: set[str] = {"drill_table", "layer_stackup"}

TEXT_FIELDS: set[str] = {"fabrication_notes", "ocr_raw_text"}


def build_confidence_entry(
    field_name: str,
    ocr_engine: str,
    base_override: float | None = None,
    reasoning: str | None = None,
    tesseract_word_confidence: float | None = None,
    json_parse_success: bool = True,
    is_null: bool = False,
    raw_text_length: int | None = None,
) -> FieldConfidence:
    """Build a FieldConfidence entry for a single extracted field.

    Args:
        field_name: Name of the PCBData field.
        ocr_engine: Engine identifier ("tesseract", "glm-ocr", "qwen-vl").
        base_override: Override the default base confidence for this field.
        reasoning: Human-readable explanation of the confidence level.
        tesseract_word_confidence: Average word confidence from Tesseract (0-100).
        json_parse_success: Whether the model returned valid JSON.
        is_null: Whether the field value is null/missing.
        raw_text_length: Length of raw OCR text.

    Returns:
        FieldConfidence instance.
    """
    bases = ENGINE_BASE_MAP.get(ocr_engine, {})
    base = base_override if base_override is not None else bases.get(field_name, 0.60)

    if is_null:
        return FieldConfidence(
            value=0.0,
            source=ocr_engine,
            reasoning="Field value is null / not detected",
        )

    if ocr_engine == "tesseract" and tesseract_word_confidence is not None:
        factor = tesseract_word_confidence / 100.0
        base = base * factor

    if ocr_engine == "glm-ocr" and raw_text_length is not None and raw_text_length < 200:
        base = max(0.0, base - 0.1)

    if ocr_engine == "qwen-vl" and json_parse_success:
        base = min(1.0, base + 0.05)

    if reasoning is None:
        reasoning = f"Base confidence {base:.2f} from {ocr_engine} for {field_name}"

    return FieldConfidence(
        value=max(0.0, min(1.0, base)),
        source=ocr_engine,
        reasoning=reasoning,
    )


def build_confidence_map(
    data: PCBData,
    ocr_engine: str,
    tesseract_avg_confidence: float | None = None,
    raw_text_length: int | None = None,
    json_parse_success: bool = True,
    field_overrides: dict[str, float] | None = None,
    provenance: dict[str, str] | None = None,
) -> dict[str, FieldConfidence]:
    """Build a full confidence map for all PCBData fields.

    Args:
        data: The extracted PCBData.
        ocr_engine: Engine identifier.
        tesseract_avg_confidence: Average Tesseract word confidence (0-100).
        raw_text_length: Length of raw OCR text.
        json_parse_success: Whether JSON parse succeeded.
        field_overrides: Per-field confidence overrides.
        provenance: Optional per-field source text (the drawing text that
            produced the value), attached as FieldConfidence.source_text.

    Returns:
        Dictionary mapping field names to FieldConfidence.
    """
    field_overrides = field_overrides or {}
    provenance = provenance or {}
    confidence: dict[str, FieldConfidence] = {}

    for field_name in PCBData.model_fields:
        value = getattr(data, field_name, None)
        is_null = _is_field_null(value, field_name)
        base_override = field_overrides.get(field_name)

        entry = build_confidence_entry(
            field_name=field_name,
            ocr_engine=ocr_engine,
            base_override=base_override,
            tesseract_word_confidence=tesseract_avg_confidence,
            json_parse_success=json_parse_success,
            is_null=is_null,
            raw_text_length=raw_text_length,
        )
        if not is_null:
            entry.source_text = provenance.get(field_name)
        confidence[field_name] = entry

    return confidence


def _is_field_null(value: Any, field_name: str) -> bool:
    """Check if a field value is effectively null/empty."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and len(value) == 0:
        if field_name in ("ipc_specs", "drill_table", "layer_stackup"):
            return True
    if isinstance(value, bool):
        return False
    return False


def select_best_value(
    field_name: str,
    results: list[PCBDataWithConfidence],
) -> tuple[Any, float, str]:
    """Select the best value for a field from multiple OCR results.

    Strategy (generalized over however many engines produced a non-null
    value — not hardcoded to 3, since some engines like PyMuPDF skip
    themselves conditionally and the count varies per request):
    1. Collect all non-null values + their confidences
    2. If every contributing engine agrees -> use the value,
       confidence = max(confidences) * 1.1
    3. If a strict majority agree (more than half of the contributors) ->
       majority value, confidence = avg of agreeing confidences
    4. Otherwise (no agreement, or a tie with no group over half) ->
       highest-confidence value
    5. If only 1 non-null -> use it with 0.8x penalty

    Args:
        field_name: Name of the PCBData field.
        results: List of PCBDataWithConfidence from each engine.

    Returns:
        Tuple of (best_value, final_confidence, selection_reason).
    """
    value, conf, reason, _attr = select_best_value_detailed(field_name, results)
    return (value, conf, reason)


def select_best_value_detailed(
    field_name: str,
    results: list[PCBDataWithConfidence],
) -> tuple[Any, float, str, dict[str, Any]]:
    """`select_best_value`, plus attribution for the winning value.

    The 4th element describes *where the answer came from*: the engine whose
    value won, the drawing text that engine attributed it to (when it can
    report one), and every engine that contributed a value — so a disagreement
    can be audited without re-running the pipeline.
    """
    # (value, confidence, engine, source_text)
    candidates: list[tuple[Any, float, str, str | None]] = []

    for result in results:
        raw_value = getattr(result.data, field_name, None)
        conf_map = result.confidence.get(
            field_name,
            FieldConfidence(value=0.0, source=result.ocr_engine),
        )
        if not _is_field_null(raw_value, field_name):
            candidates.append(
                (raw_value, conf_map.value, result.ocr_engine, conf_map.source_text)
            )

    def _attribution(winner: tuple[Any, float, str, str | None] | None) -> dict[str, Any]:
        source_text = winner[3] if winner else None
        source_text_from = winner[2] if (winner and source_text) else None
        if winner is not None and source_text is None:
            # The winning engine can't cite source text (the vision models
            # generally can't). Borrow it from another engine that arrived at
            # the *same* value — the evidence supports that value regardless
            # of which engine won the confidence tie-break. Never borrow from
            # an engine that disagreed.
            winner_key = _normalize_for_comparison(winner[0], field_name)
            for v, _c, e, st in candidates:
                if st and _normalize_for_comparison(v, field_name) == winner_key:
                    source_text = st
                    source_text_from = e
                    break
        return {
            "engine": winner[2] if winner else None,
            "source_text": source_text,
            "source_text_from": source_text_from,
            "contributors": [
                {"engine": e, "value": v, "confidence": round(c, 3), "source_text": st}
                for v, c, e, st in candidates
            ],
        }

    if not candidates:
        return (None, 0.0, "No engine detected this field", _attribution(None))

    if len(candidates) == 1:
        val, conf, engine, _st = candidates[0]
        return (
            val,
            max(0.0, conf * 0.8),
            f"Only source ({engine}), confidence penalized to 0.8x",
            _attribution(candidates[0]),
        )

    agreeing_groups: dict[str, list[tuple[Any, float, str, str | None]]] = {}
    for cand in candidates:
        key = _normalize_for_comparison(cand[0], field_name)
        agreeing_groups.setdefault(key, []).append(cand)

    if len(agreeing_groups) == 1:
        all_confs = [c for _, c, _, _ in candidates]
        best_conf = min(1.0, max(all_confs) * 1.1)
        best = max(candidates, key=lambda x: x[1])
        return (
            best[0],
            best_conf,
            f"All {len(candidates)} engines agree, confidence boosted to {best_conf:.2f}",
            _attribution(best),
        )

    majority = max(agreeing_groups.values(), key=len)
    if len(majority) > len(candidates) / 2:
        avg_conf = sum(c for _, c, _, _ in majority) / len(majority)
        best = max(majority, key=lambda x: x[1])
        return (
            best[0],
            avg_conf,
            f"{len(majority)}/{len(candidates)} engines agree",
            _attribution(best),
        )

    best = max(candidates, key=lambda x: x[1])
    return (
        best[0],
        best[1],
        f"Engines disagree ({len(candidates)} candidates), "
        f"picked highest confidence ({best[2]})",
        _attribution(best),
    )


def _normalize_for_comparison(value: Any, field_name: str) -> str:
    """Normalize a value for equality comparison."""
    if value is None:
        return "<null>"

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, str):
        normalized = value.strip().lower()
        if field_name == "ipc_class":
            normalized = f"class {normalized.replace('class', '').strip()}"
        return normalized

    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    if isinstance(value, list):
        return str(sorted([str(v) for v in value]))

    return str(value)


def safe_merge_results(
    results: list[PCBDataWithConfidence],
) -> tuple[PCBData, list[str]]:
    """Merge multiple OCR results into a single PCBData with reconciliation log.

    Thin wrapper over `merge_with_attribution` for callers that don't need
    per-field attribution.

    Args:
        results: List of PCBDataWithConfidence from each engine.

    Returns:
        Tuple of (merged PCBData, reconciliation log entries).
    """
    merged, log, _attribution = merge_with_attribution(results)
    return merged, log


def merge_with_attribution(
    results: list[PCBDataWithConfidence],
) -> tuple[PCBData, list[str], dict[str, dict[str, Any]]]:
    """Merge OCR results, also returning per-field attribution.

    Returns:
        (merged PCBData, reconciliation log, attribution) where attribution
        maps each field to the winning engine, the drawing text that engine
        attributed the value to, and every engine that contributed a value.
    """
    log: list[str] = []
    merged_fields: dict[str, Any] = {}
    attribution: dict[str, dict[str, Any]] = {}

    for field_name in PCBData.model_fields:
        value, conf, reason, attr = select_best_value_detailed(field_name, results)

        # List-typed fields default to [], never None, per PCBData's schema
        if field_name in TABLE_FIELDS and value is None:
            value = []

        # Special handling for is_itar
        if field_name == "is_itar":
            if any(getattr(r.data, "is_itar", False) for r in results):
                value = True
                conf = max(
                    r.confidence.get(
                        "is_itar",
                        FieldConfidence(value=0.0, source=r.ocr_engine),
                    ).value
                    for r in results
                )
                reason = "SPECIAL RULE: ITAR detected by at least one engine"

        log.append(f"[{field_name}] value={value}, conf={conf:.2f}, reason={reason}")
        merged_fields[field_name] = value
        attribution[field_name] = {
            "value": value,
            "confidence": round(conf, 3),
            "reason": reason,
            **attr,
        }

    # ipc_specs = union of all found specs, deduplicated
    all_specs: list[str] = []
    for r in results:
        all_specs.extend(r.data.ipc_specs or [])
    merged_fields["ipc_specs"] = list(dict.fromkeys(all_specs))
    log.append(f"[ipc_specs] Union of all engines: {merged_fields['ipc_specs']}")

    # fabrication_notes = longest non-null value
    notes_candidates = [r.data.fabrication_notes for r in results if r.data.fabrication_notes]
    if notes_candidates:
        merged_fields["fabrication_notes"] = max(notes_candidates, key=len)
        log.append(
            f"[fabrication_notes] Used longest value "
            f"({len(merged_fields['fabrication_notes'])} chars)"
        )

    try:
        merged = PCBData(**merged_fields)
    except Exception as e:
        log.append(f"[ERROR] Failed to construct PCBData: {e}")
        merged = PCBData()
        for k, v in merged_fields.items():
            try:
                setattr(merged, k, v)
            except Exception:
                pass

    return merged, log, attribution
