"""Unit tests for the heuristic text->PCBData regex parser."""

from __future__ import annotations

from shared.pcb_parser import extract_board_thickness


class TestExtractBoardThickness:
    def test_tolerance_pair_still_works(self):
        """Existing 'X +/- Y' pattern is unaffected by the new stackup pattern."""
        bt = extract_board_thickness("TOTAL THICKNESS OF PCB SHALL BE .062 +/- .005")
        assert bt is not None
        assert abs(bt.nominal - 0.062) < 0.0001
        assert abs(bt.plus_tol - 0.005) < 0.0001

    def test_bare_metric_near_rigid_label(self):
        """A bare metric value next to a RIGID/FLEX stackup label (no
        'thickness' keyword nearby at all) is recognized — this is a
        standard PCB stackup-diagram convention, not specific to one
        drawing's layout."""
        text = "NOTES:\n7\n7\n6\n6\n1.80238 MM\n__________\nRIGID\n1-4\n* SOLDERMASK"
        bt = extract_board_thickness(text)
        assert bt is not None
        assert abs(bt.nominal - 0.0710) < 0.001
        assert bt.raw == "1.80238 MM"

    def test_bare_metric_near_flex_label(self):
        """Same pattern, label before the value, FLEX instead of RIGID."""
        text = "STACKUP\nFLEX\n1.5 MM\nCOVERLAY"
        bt = extract_board_thickness(text)
        assert bt is not None
        assert abs(bt.nominal - (1.5 / 25.4)) < 0.001

    def test_implausible_value_near_rigid_is_rejected(self):
        """A number near RIGID/FLEX that converts outside any plausible PCB
        thickness range is not treated as board thickness (avoids false
        matches on unrelated nearby dimensions)."""
        text = "RIGID 500.0 MM CONNECTOR SPACING"
        bt = extract_board_thickness(text)
        assert bt is None

    def test_no_match_returns_none(self):
        assert extract_board_thickness("NO RELEVANT TEXT HERE") is None
