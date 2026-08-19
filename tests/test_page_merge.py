"""Unit tests for Qwen-VL's multi-page result merging.

All 4 sample PDFs are single-page, so this merge path had never executed with
more than one page before these tests existed.
"""

from __future__ import annotations

from services.qwen_vl.app.router import (
    _dedupe_rows,
    _is_blank,
    _merge_page_results,
)


class TestIsBlank:
    def test_none_and_empty_are_blank(self):
        assert _is_blank(None)
        assert _is_blank("")
        assert _is_blank("   ")

    def test_real_values_are_not(self):
        assert not _is_blank("FR4")
        assert not _is_blank(0)  # a real zero must not read as "missing"
        assert not _is_blank(False)
        assert not _is_blank([])


class TestDedupeRows:
    def test_removes_repeats_preserving_order(self):
        rows = [{"size_mils": 18.0}, {"size_mils": 36.0}, {"size_mils": 18.0}]
        assert _dedupe_rows(rows) == [{"size_mils": 18.0}, {"size_mils": 36.0}]

    def test_key_order_does_not_defeat_dedup(self):
        rows = [{"a": 1, "b": 2}, {"b": 2, "a": 1}]
        assert len(_dedupe_rows(rows)) == 1

    def test_near_identical_rows_kept(self):
        # Adjacent drill rows can be genuinely close (.124 vs .125) — those
        # are separate rows and must survive.
        rows = [{"size_mils": 124.0}, {"size_mils": 125.0}]
        assert len(_dedupe_rows(rows)) == 2

    def test_unhashable_rows_are_safe(self):
        rows = [{"layers": [1, 2]}, {"layers": [1, 2]}, {"layers": [3]}]
        assert len(_dedupe_rows(rows)) == 2


class TestMergePageResults:
    def test_empty(self):
        assert _merge_page_results([]) == {}

    def test_single_page_passthrough(self):
        page = {"material": "FR4", "drill_table": [{"size_mils": 18.0}]}
        assert _merge_page_results([page]) == page

    def test_identical_pages_do_not_double_tables(self):
        # The core multi-sheet bug: the same drill chart repeated on two
        # sheets previously doubled every row and every quantity.
        page = {
            "layer_count": 4,
            "drill_table": [{"size_mils": 18.0, "qty": 90}],
            "layer_stackup": [{"number": 1, "function": "signal"}],
            "ipc_specs": ["IPC-6012"],
        }
        merged = _merge_page_results([page, dict(page)])
        assert len(merged["drill_table"]) == 1
        assert merged["drill_table"][0]["qty"] == 90
        assert len(merged["layer_stackup"]) == 1
        assert merged["ipc_specs"] == ["IPC-6012"]

    def test_distinct_rows_accumulate(self):
        p1 = {"drill_table": [{"size_mils": 18.0}]}
        p2 = {"drill_table": [{"size_mils": 36.0}]}
        assert len(_merge_page_results([p1, p2])["drill_table"]) == 2

    def test_empty_string_does_not_block_later_value(self):
        # Regression: the old `is None` guard let "" from page 1 permanently
        # suppress a real value found on page 2.
        merged = _merge_page_results([{"material": ""}, {"material": "FR4"}])
        assert merged["material"] == "FR4"

    def test_none_filled_from_later_page(self):
        merged = _merge_page_results([{"material": None}, {"material": "370HR"}])
        assert merged["material"] == "370HR"

    def test_first_real_value_wins(self):
        merged = _merge_page_results([{"material": "FR4"}, {"material": "370HR"}])
        assert merged["material"] == "FR4"

    def test_zero_is_not_overwritten(self):
        merged = _merge_page_results([{"layer_count": 0}, {"layer_count": 8}])
        assert merged["layer_count"] == 0

    def test_notes_concatenate(self):
        merged = _merge_page_results(
            [{"fabrication_notes": "sheet one"}, {"fabrication_notes": "sheet two"}]
        )
        assert merged["fabrication_notes"] == "sheet one\nsheet two"

    def test_ipc_specs_union_deduped(self):
        merged = _merge_page_results(
            [{"ipc_specs": ["IPC-6012", "IPC-A-600"]}, {"ipc_specs": ["IPC-6012"]}]
        )
        assert merged["ipc_specs"] == ["IPC-6012", "IPC-A-600"]

    def test_non_list_field_does_not_crash(self):
        merged = _merge_page_results([{"drill_table": None}, {"drill_table": [{"a": 1}]}])
        assert merged["drill_table"] == [{"a": 1}]
