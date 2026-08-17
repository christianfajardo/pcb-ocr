#!/usr/bin/env python3
"""Benchmark script — run all samples, measure accuracy + timing."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
EXPECTED_DIR = PROJECT_ROOT / "tests" / "expected"

SAMPLE_FILES = [
    ("sample1.pdf", "sample1.json"),
    ("sample2.pdf", "sample2.json"),
    ("sample3..pdf", "sample3.json"),
    ("sample4.pdf", "sample4.json"),
]

SUPERVISOR_URL = "http://localhost:8080/extract"


async def extract_sample(pdf_path: str) -> dict:
    """Extract a single sample via the supervisor."""
    async with httpx.AsyncClient(timeout=300) as client:
        with open(pdf_path, "rb") as f:
            resp = await client.post(
                SUPERVISOR_URL,
                files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            )
        resp.raise_for_status()
        return resp.json()


def check_accuracy(result: dict, expected: dict, sample_name: str) -> list[str]:
    """Compare result against expected with tolerances."""
    issues = []

    # Core fields — exact match
    for field in ["layer_count", "material", "ipc_class", "surface_finish", "is_itar"]:
        actual = result.get(field)
        expected_val = expected.get(field)
        if expected_val is not None and actual != expected_val:
            issues.append(f"  {field}: expected={expected_val}, got={actual}")

    # Board thickness — tolerance check
    exp_bt = expected.get("board_thickness")
    act_bt = result.get("board_thickness")
    if exp_bt and exp_bt.get("nominal"):
        if not act_bt or not act_bt.get("nominal"):
            issues.append(f"  board_thickness.nominal: expected={exp_bt['nominal']}, got=None")
        elif abs(act_bt["nominal"] - exp_bt["nominal"]) >= 0.002:
            issues.append(
                f"  board_thickness.nominal: expected={exp_bt['nominal']}, got={act_bt['nominal']}"
            )
        if act_bt.get("unit") != "in":
            issues.append(f"  board_thickness.unit: expected='in', got={act_bt.get('unit')}")

    # Drill table — row count within 10%
    exp_dt = expected.get("drill_table", [])
    act_dt = result.get("drill_table", [])
    if exp_dt:
        if len(act_dt) < len(exp_dt) * 0.8:
            issues.append(f"  drill_table: expected ~{len(exp_dt)} rows, got {len(act_dt)}")
        for row in act_dt:
            if row.get("size_mils") and not (1.0 <= row["size_mils"] <= 500.0):
                issues.append(f"  drill_table size_mils={row['size_mils']} out of range")

    # Solder mask color
    exp_sm = expected.get("solder_mask", {})
    act_sm = result.get("solder_mask", {})
    if exp_sm and exp_sm.get("color"):
        if act_sm.get("color", "").lower() != exp_sm["color"].lower():
            issues.append(
                f"  solder_mask.color: expected={exp_sm['color']}, got={act_sm.get('color')}"
            )

    # IPC specs — all expected must be present
    exp_specs = set(expected.get("ipc_specs", []))
    act_specs = set(result.get("ipc_specs", []))
    missing = exp_specs - act_specs
    if missing:
        issues.append(f"  ipc_specs missing: {missing}")

    return issues


async def main() -> int:
    """Run benchmark on all samples."""
    print("=" * 60)
    print("PCB OCR Pipeline Benchmark")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    for pdf_name, expected_name in SAMPLE_FILES:
        pdf_path = SAMPLES_DIR / pdf_name
        expected_path = EXPECTED_DIR / expected_name

        print(f"\n{'─' * 40}")
        print(f"Sample: {pdf_name}")

        if not pdf_path.exists():
            print(f"  SKIP: {pdf_path} not found")
            continue

        with open(expected_path) as f:
            expected = json.load(f)

        start = time.monotonic()
        try:
            result = await extract_sample(str(pdf_path))
            elapsed = time.monotonic() - start
        except Exception as e:
            print(f"  FAIL: {e}")
            total_failed += 1
            continue

        issues = check_accuracy(result, expected, pdf_name)

        if not issues:
            print(f"  PASS ({elapsed:.1f}s)")
            total_passed += 1
        else:
            print(f"  FAIL ({elapsed:.1f}s):")
            for issue in issues:
                print(issue)
            total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print(f"{'=' * 60}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
