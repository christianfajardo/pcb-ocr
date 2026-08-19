"""Unit tests for shared/cross_checks.py — post-reconciliation self-consistency."""

from __future__ import annotations

from shared.cross_checks import (
    check_drill_table,
    check_impedance,
    check_layer_consistency,
    find_stated_hole_total,
    run_cross_checks,
)
from shared.schemas import (
    DrillRow,
    ImpedanceControl,
    ImpedanceRange,
    LayerSpec,
    PCBData,
)


class TestFindStatedHoleTotal:
    def test_plain(self):
        assert find_stated_hole_total("TOTAL HOLES: 180") == 180

    def test_ocr_garbled_separator(self):
        # The real sample1 drawing reads exactly this — an "= " sneaks in.
        # A stricter [:=]? pattern misses it, which is why the regex is loose.
        assert find_stated_hole_total("votes TOTAL HOLES: = 180\nPS. VENDOR") == 180

    def test_variants(self):
        assert find_stated_hole_total("total hole 42") == 42
        assert find_stated_hole_total("TOTAL  HOLES   =   7") == 7

    def test_absent_is_none(self):
        # 3 of 4 real samples state no total at all — absence must be normal.
        assert find_stated_hole_total("no total stated here") is None
        assert find_stated_hole_total(None) is None
        assert find_stated_hole_total(None, "") is None

    def test_falls_through_to_later_text(self):
        assert find_stated_hole_total(None, "TOTAL HOLES: 12") == 12


class TestLayerConsistency:
    def test_matching_is_clean(self):
        data = PCBData(
            layer_count=3,
            layer_stackup=[LayerSpec(number=n, function="signal") for n in (1, 2, 3)],
        )
        assert check_layer_consistency(data) == []

    def test_count_mismatch(self):
        data = PCBData(
            layer_count=4,
            layer_stackup=[LayerSpec(number=n) for n in (1, 2)],
        )
        issues = check_layer_consistency(data)
        assert len(issues) == 1
        assert "disagrees" in issues[0]

    def test_duplicate_numbers(self):
        # The signature of the same sheet being merged twice.
        data = PCBData(
            layer_count=4,
            layer_stackup=[LayerSpec(number=n) for n in (1, 2, 1, 2)],
        )
        issues = check_layer_consistency(data)
        assert any("duplicate layer numbers" in i for i in issues)

    def test_missing_numbers(self):
        data = PCBData(
            layer_count=3,
            layer_stackup=[LayerSpec(number=n) for n in (1, 2, 5)],
        )
        issues = check_layer_consistency(data)
        assert any("missing layer numbers" in i for i in issues)

    def test_empty_stackup_is_skipped(self):
        assert check_layer_consistency(PCBData(layer_count=4)) == []


class TestDrillTable:
    def test_plausible_rows_clean(self):
        data = PCBData(drill_table=[DrillRow(size_mils=18.0, qty=270)])
        assert check_drill_table(data) == []

    def test_size_above_range(self):
        data = PCBData(drill_table=[DrillRow(size_mils=5000.0, qty=1)])
        issues = check_drill_table(data)
        assert any("outside plausible range" in i for i in issues)

    def test_size_below_range(self):
        data = PCBData(drill_table=[DrillRow(size_mils=0.02, qty=1)])
        assert any("outside plausible range" in i for i in check_drill_table(data))

    def test_boundaries_inclusive(self):
        data = PCBData(drill_table=[DrillRow(size_mils=1.0), DrillRow(size_mils=1000.0)])
        assert check_drill_table(data) == []

    def test_nonpositive_qty(self):
        data = PCBData(drill_table=[DrillRow(size_mils=18.0, qty=0)])
        assert any("non-positive qty" in i for i in check_drill_table(data))

    def test_qty_sum_matches_stated_total(self):
        data = PCBData(
            drill_table=[DrillRow(size_mils=18.0, qty=100), DrillRow(size_mils=36.0, qty=80)]
        )
        assert check_drill_table(data, stated_total=180) == []

    def test_qty_sum_mismatch(self):
        data = PCBData(drill_table=[DrillRow(size_mils=18.0, qty=100)])
        issues = check_drill_table(data, stated_total=180)
        assert any("sum to 100" in i and "180" in i for i in issues)

    def test_doubled_table_caught_by_total(self):
        # Exactly the multi-page duplication failure mode.
        rows = [DrillRow(size_mils=18.0, qty=90)] * 2
        assert check_drill_table(PCBData(drill_table=rows), stated_total=90)

    def test_no_stated_total_skips_sum_check(self):
        data = PCBData(drill_table=[DrillRow(size_mils=18.0, qty=1)])
        assert check_drill_table(data, stated_total=None) == []


class TestImpedance:
    def test_valid_range(self):
        data = PCBData(
            impedance_control=ImpedanceControl(
                controlled=True, single_ended=ImpedanceRange(min=45.0, max=55.0)
            )
        )
        assert check_impedance(data) == []

    def test_equal_is_valid(self):
        data = PCBData(
            impedance_control=ImpedanceControl(
                controlled=True, single_ended=ImpedanceRange(min=90.0, max=90.0)
            )
        )
        assert check_impedance(data) == []

    def test_inverted_range(self):
        data = PCBData(
            impedance_control=ImpedanceControl(
                controlled=True, single_ended=ImpedanceRange(min=99.0, max=50.0)
            )
        )
        assert any("inverted" in i for i in check_impedance(data))

    def test_missing_is_skipped(self):
        assert check_impedance(PCBData()) == []


class TestRunCrossChecks:
    def test_empty_data_is_clean(self):
        assert run_cross_checks(PCBData()) == []

    def test_pulls_stated_total_from_notes(self):
        data = PCBData(
            fabrication_notes="TOTAL HOLES: = 180",
            drill_table=[DrillRow(size_mils=18.0, qty=170)],
        )
        assert any("180" in i for i in run_cross_checks(data))

    def test_aggregates_across_checks(self):
        data = PCBData(
            layer_count=9,
            layer_stackup=[LayerSpec(number=1)],
            drill_table=[DrillRow(size_mils=9999.0, qty=1)],
            impedance_control=ImpedanceControl(
                controlled=True, single_ended=ImpedanceRange(min=99.0, max=1.0)
            ),
        )
        assert len(run_cross_checks(data)) >= 3
