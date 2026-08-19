"""Self-consistency checks over a fully reconciled PCBData.

These compare extracted fields against *each other* and against totals the
drawing states about itself (e.g. "TOTAL HOLES: 180" vs the sum of the drill
table's quantities). That makes them fundamentally different from
`schemas.normalize_units`, which fixes one field at a time and — critically —
runs per-engine *before* reconciliation. A cross-check can only run after the
merge, so it lives here and is called from the supervisor's validate_node.

Findings are advisory: they describe an inconsistency, they don't decide
whether the job succeeded.
"""

from __future__ import annotations

import re

from .constants import DRILL_SIZE_MAX_MILS, DRILL_SIZE_MIN_MILS
from .schemas import PCBData

# Matches a stated hole total. The separator is deliberately loose — this text
# comes out of OCR, and the real sample1 drawing reads "TOTAL HOLES: = 180",
# which a stricter `[:=]?` pattern silently misses.
_TOTAL_HOLES_RE = re.compile(r"TOTAL\s*HOLES?[\s:=]*(\d+)", re.IGNORECASE)


def find_stated_hole_total(*texts: str | None) -> int | None:
    """Return the hole count the drawing states about itself, if any.

    Only 1 of the 4 current samples states one at all, so callers must treat
    absence as normal rather than as a problem.
    """
    for text in texts:
        if not text:
            continue
        m = _TOTAL_HOLES_RE.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def check_layer_consistency(data: PCBData) -> list[str]:
    """layer_count vs the stackup: count match, no dupes, no gaps."""
    issues: list[str] = []
    stackup = data.layer_stackup or []
    if not stackup:
        return issues

    if data.layer_count is not None and data.layer_count != len(stackup):
        issues.append(
            f"layer_count ({data.layer_count}) disagrees with layer_stackup "
            f"({len(stackup)} layers)"
        )

    numbers = [layer.number for layer in stackup if layer.number is not None]
    if not numbers:
        return issues

    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        # A prime symptom of the same sheet being merged twice.
        issues.append(f"layer_stackup has duplicate layer numbers: {duplicates}")

    expected = set(range(1, len(numbers) + 1))
    missing = sorted(expected - set(numbers))
    if missing and not duplicates:
        issues.append(f"layer_stackup is missing layer numbers: {missing}")

    return issues


def check_drill_table(data: PCBData, stated_total: int | None = None) -> list[str]:
    """Drill row sanity, plus the quantity sum against a stated total."""
    issues: list[str] = []
    rows = data.drill_table or []
    if not rows:
        return issues

    for row in rows:
        if row.size_mils is not None and not (
            DRILL_SIZE_MIN_MILS <= row.size_mils <= DRILL_SIZE_MAX_MILS
        ):
            issues.append(
                f"Drill size {row.size_mils} mils outside plausible range "
                f"({DRILL_SIZE_MIN_MILS}-{DRILL_SIZE_MAX_MILS}) — likely wrong units"
            )
        if row.qty is not None and row.qty <= 0:
            issues.append(f"Drill row has non-positive qty ({row.qty})")

    if stated_total is not None:
        quantities = [r.qty for r in rows if r.qty is not None]
        if quantities:
            actual = sum(quantities)
            if actual != stated_total:
                issues.append(
                    f"Drill quantities sum to {actual} but the drawing states "
                    f"{stated_total} total holes"
                )

    return issues


def check_impedance(data: PCBData) -> list[str]:
    """Impedance range sanity."""
    issues: list[str] = []
    imp = data.impedance_control
    if not imp or not imp.single_ended:
        return issues

    lo, hi = imp.single_ended.min, imp.single_ended.max
    if lo is not None and hi is not None and lo > hi:
        issues.append(f"Impedance range inverted: min ({lo}) > max ({hi})")
    return issues


def run_cross_checks(data: PCBData) -> list[str]:
    """Run every cross-check; returns a flat list of human-readable findings."""
    stated_total = find_stated_hole_total(data.fabrication_notes, data.ocr_raw_text)
    return [
        *check_layer_consistency(data),
        *check_drill_table(data, stated_total),
        *check_impedance(data),
    ]
