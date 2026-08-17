"""Unit tests for the reconciliation/merge logic."""

from __future__ import annotations

from shared.confidence import (
    build_confidence_map,
    safe_merge_results,
    select_best_value,
)
from shared.schemas import (
    IPCClass,
    PCBData,
    PCBDataWithConfidence,
)


def _make_result(
    engine: str,
    layer_count: int | None = None,
    material: str | None = None,
    ipc_class: str | None = None,
    surface_finish: str | None = None,
    is_itar: bool | None = None,
    ipc_specs: list[str] | None = None,
    fabrication_notes: str | None = None,
) -> PCBDataWithConfidence:
    """Create a mock PCBDataWithConfidence."""
    data = PCBData(
        layer_count=layer_count,
        material=material,
        ipc_class=IPCClass(ipc_class) if ipc_class else None,
        surface_finish=surface_finish,
        is_itar=is_itar,
        ipc_specs=ipc_specs or [],
        fabrication_notes=fabrication_notes,
    )
    confidence = build_confidence_map(data, engine)
    return PCBDataWithConfidence(
        data=data,
        confidence=confidence,
        ocr_engine=engine,
        processing_time_ms=100.0,
        page_count=1,
    )


class TestSelectBestValue:
    """Tests for select_best_value reconciliation."""

    def test_all_agree(self):
        """When all 3 engines agree, use the value with boosted confidence."""
        r1 = _make_result("tesseract", layer_count=5)
        r2 = _make_result("glm-ocr", layer_count=5)
        r3 = _make_result("qwen-vl", layer_count=5)

        value, conf, reason = select_best_value("layer_count", [r1, r2, r3])
        assert value == 5
        assert conf > 0.7  # boosted
        assert "agree" in reason.lower()

    def test_only_one_nonnull(self):
        """When only 1 engine found a value, use it with penalty."""
        r1 = _make_result("tesseract", layer_count=5)
        r2 = _make_result("glm-ocr", layer_count=None)
        r3 = _make_result("qwen-vl", layer_count=None)

        value, conf, reason = select_best_value("layer_count", [r1, r2, r3])
        assert value == 5
        assert conf < 0.7  # penalized

    def test_all_null(self):
        """When no engine found a value, return None."""
        r1 = _make_result("tesseract")
        r2 = _make_result("glm-ocr")
        r3 = _make_result("qwen-vl")

        value, conf, reason = select_best_value("layer_count", [r1, r2, r3])
        assert value is None
        assert conf == 0.0

    def test_two_agree(self):
        """When 2/3 agree, use majority."""
        r1 = _make_result("tesseract", layer_count=5)
        r2 = _make_result("glm-ocr", layer_count=5)
        r3 = _make_result("qwen-vl", layer_count=8)

        value, conf, reason = select_best_value("layer_count", [r1, r2, r3])
        assert value == 5

    def test_three_of_four_agree(self):
        """A 3-of-4 majority (e.g. with PyMuPDF as a 4th engine) is recognized
        as a majority and averaged, not misclassified as 'all disagree' —
        the majority branch used to be hardcoded to exactly 3 candidates."""
        r1 = _make_result("tesseract", layer_count=5)
        r2 = _make_result("glm-ocr", layer_count=5)
        r3 = _make_result("qwen-vl", layer_count=5)
        r4 = _make_result("pymupdf", layer_count=9)

        value, conf, reason = select_best_value("layer_count", [r1, r2, r3, r4])
        assert value == 5
        assert "3/4" in reason

    def test_two_of_four_is_not_a_majority(self):
        """2 of 4 agreeing is only a tie, not a strict majority — falls back
        to highest confidence rather than being averaged as if it won."""
        r1 = _make_result("tesseract", layer_count=5)
        r2 = _make_result("glm-ocr", layer_count=5)
        r3 = _make_result("qwen-vl", layer_count=8)
        r4 = _make_result("pymupdf", layer_count=9)

        value, conf, reason = select_best_value("layer_count", [r1, r2, r3, r4])
        assert "disagree" in reason.lower()


class TestSafeMerge:
    """Tests for safe_merge_results."""

    def test_itar_override(self):
        """If ANY engine detects ITAR, result MUST be True."""
        r1 = _make_result("tesseract", is_itar=True)
        r2 = _make_result("glm-ocr", is_itar=False)
        r3 = _make_result("qwen-vl", is_itar=False)

        merged, _log = safe_merge_results([r1, r2, r3])
        assert merged.is_itar is True

    def test_ipc_specs_union(self):
        """IPC specs should be union of all engines."""
        r1 = _make_result("tesseract", ipc_specs=["IPC-6012", "IPC-A-600"])
        r2 = _make_result("glm-ocr", ipc_specs=["IPC-A-600", "IPC-SM-840"])
        r3 = _make_result("qwen-vl", ipc_specs=["IPC-6012"])

        merged, _log = safe_merge_results([r1, r2, r3])
        assert "IPC-6012" in merged.ipc_specs
        assert "IPC-A-600" in merged.ipc_specs
        assert "IPC-SM-840" in merged.ipc_specs
        # No duplicates
        assert len(merged.ipc_specs) == len(set(merged.ipc_specs))

    def test_fabrication_notes_longest(self):
        """Use longest fabrication notes."""
        r1 = _make_result("tesseract", fabrication_notes="Short notes")
        r2 = _make_result("glm-ocr", fabrication_notes="Much longer notes with more detail")
        r3 = _make_result("qwen-vl", fabrication_notes="Medium")

        merged, _log = safe_merge_results([r1, r2, r3])
        assert merged.fabrication_notes == "Much longer notes with more detail"


class TestConfidenceBuilding:
    """Tests for confidence map building."""

    def test_null_field_zero_confidence(self):
        """Null fields get 0.0 confidence."""
        data = PCBData(layer_count=None, material=None)
        conf_map = build_confidence_map(data, "tesseract")
        assert conf_map["layer_count"].value == 0.0
        assert conf_map["material"].value == 0.0

    def test_nonnull_field_positive_confidence(self):
        """Non-null fields get positive confidence."""
        data = PCBData(layer_count=5, material="FR4")
        conf_map = build_confidence_map(data, "tesseract")
        assert conf_map["layer_count"].value > 0.0
        assert conf_map["material"].value > 0.0

    def test_source_in_confidence(self):
        """Confidence entries track the source engine."""
        data = PCBData(layer_count=5)
        conf_map = build_confidence_map(data, "qwen-vl")
        assert conf_map["layer_count"].source == "qwen-vl"
