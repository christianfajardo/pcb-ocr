#!/usr/bin/env python3
"""Local extraction pipeline using Tesseract OCR + host LLM for structured extraction.

This bypasses Docker and uses the host's vLLM at 172.17.0.1:8000 for structured extraction.
Validates output against tests/expected/*.json files.

Usage:
    python scripts/local_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.tesseract.app.ocr_engine import TesseractOCR
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

VLLM_URL = "http://172.17.0.1:8000/v1/chat/completions"
VLLM_MODEL = "nvidia/Qwen3.6-27B-NVFP4"

SAMPLES = [
    ("samples/sample1.pdf", "tests/expected/sample1.json"),
    ("samples/sample2.pdf", "tests/expected/sample2.json"),
    ("samples/sample3..pdf", "tests/expected/sample3.json"),
    ("samples/sample4.pdf", "tests/expected/sample4.json"),
]

STRUCTURED_PROMPT = """You are a PCB fabrication drawing expert. Given raw OCR text extracted from a PCB fabrication drawing, extract the following fields as JSON.

UNIT RULES:
- Board thickness: INCHES (e.g. 0.062). Convert mm if needed.
- Drill sizes: MILS (thousandths of an inch). Convert mm if needed (mm * 39.37 = mils).
- Trace widths: MILS. Convert mm if needed.
- Copper weight: oz/ft².

The OCR text may contain artifacts (wrong characters, merged lines). Use your domain knowledge to correct obvious OCR errors.

Return ONLY valid JSON with this exact structure:
{{
  "part_number": null,
  "manufacturer": null,
  "drawing_title": null,
  "is_itar": false,
  "ipc_class": "Class 1 or Class 2 or Class 3 or null",
  "ipc_specs": ["IPC-6012", ...],
  "layer_count": null,
  "layer_stackup": [{{"number": 1, "function": "signal"}}, ...],
  "material": "FR4 or 370HR or null",
  "board_thickness": {{"nominal": 0.062, "plus_tol": 0.005, "minus_tol": 0.005, "unit": "in", "raw": ".062 +/- .005"}} or null,
  "copper_weights": {{"signal_layers_oz": 1.0, "plane_layers_oz": null, "external_finished_oz": null}} or null,
  "surface_finish": "ENIG or HASL or null",
  "solder_mask": {{"present": true, "type": "LPI", "color": "Green", "spec": null, "sides": "both"}} or null,
  "silkscreen": {{"present": true, "color": "White", "ink": "non-conductive epoxy", "sides": "both"}} or null,
  "impedance_control": {{"controlled": true, "single_ended": {{"min": 50, "max": 50, "unit": "ohm", "raw": "50 OHMS"}}, "trace_width_mils": 5.0, "layers": [2, 5]}} or null,
  "drill_table": [{{"size_mils": 8.0, "qty": 3, "symbol": "circle", "plated": true}}, ...] or [],
  "fabrication_notes": "Full text of notes" or null
}}

OCR text:
---
{raw_text}
---"""


async def llm_extract(raw_text: str) -> dict | None:
    """Send raw OCR text to the host LLM for structured extraction."""
    prompt = STRUCTURED_PROMPT.format(raw_text=raw_text[:4000])  # Keep prompt shorter

    # Use synchronous request (worked before with httpx)
    loop = asyncio.get_event_loop()

    def _sync_call():
        return httpx.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
            timeout=300,  # 5 minutes for 27B model
        )

    resp = await loop.run_in_executor(None, _sync_call)
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or data["choices"][0]["message"].get(
        "reasoning", ""
    )

    if not content:
        print("WARNING: LLM returned empty content")
        return None

    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find JSON block
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Try stripping markdown fences
    if "```" in content:
        lines = content.split("\n")
        in_json = False
        json_lines = []
        for line in lines:
            if "```" in line:
                in_json = not in_json
                continue
            if in_json:
                json_lines.append(line)
        if json_lines:
            try:
                return json.loads("\n".join(json_lines))
            except json.JSONDecodeError:
                pass

    print(f"WARNING: Could not parse JSON from LLM (first 100 chars: {content[:100]})")
    return None


def build_pcb_data(structured: dict) -> PCBData:
    """Build PCBData from structured LLM output."""
    data = PCBData()

    data.part_number = structured.get("part_number")
    data.manufacturer = structured.get("manufacturer")
    data.drawing_title = structured.get("drawing_title")
    data.is_itar = structured.get("is_itar", False)
    data.ipc_specs = structured.get("ipc_specs", [])
    data.layer_count = structured.get("layer_count")
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
        data.board_thickness = BoardThickness(
            nominal=bt.get("nominal"),
            plus_tol=bt.get("plus_tol"),
            minus_tol=bt.get("minus_tol"),
            unit="in",
            raw=bt.get("raw"),
        )

    # Copper weights
    cw = structured.get("copper_weights")
    if cw and isinstance(cw, dict):
        data.copper_weights = CopperWeights(**cw)

    # Solder mask
    sm = structured.get("solder_mask")
    if sm and isinstance(sm, dict):
        data.solder_mask = SolderMask(**sm)

    # Silkscreen
    sk = structured.get("silkscreen")
    if sk and isinstance(sk, dict):
        data.silkscreen = Silkscreen(**sk)

    # Impedance
    imp = structured.get("impedance_control")
    if imp and isinstance(imp, dict):
        se = imp.get("single_ended")
        if se and isinstance(se, dict):
            imp["single_ended"] = ImpedanceRange(**se)
        data.impedance_control = ImpedanceControl(**imp)

    # Layer stackup
    stackup = structured.get("layer_stackup")
    if stackup and isinstance(stackup, list):
        data.layer_stackup = [LayerSpec(**s) for s in stackup]

    # Drill table
    dt = structured.get("drill_table")
    if dt and isinstance(dt, list):
        data.drill_table = [DrillRow(**d) for d in dt]

    return data


def load_expected(expected_path: str) -> dict[str, Any]:
    with open(expected_path) as f:
        return json.load(f)


def check_sample(result: PCBData, expected: dict[str, Any], sample_name: str) -> list[str]:
    """Compare result against expected with tolerances."""
    issues: list[str] = []

    if expected.get("layer_count") is not None:
        if result.layer_count != expected["layer_count"]:
            issues.append(
                f"layer_count: expected={expected['layer_count']}, got={result.layer_count}"
            )

    if expected.get("material"):
        if not result.material or result.material.upper() != expected["material"].upper():
            issues.append(f"material: expected={expected['material']}, got={result.material}")

    if expected.get("ipc_class"):
        act_class = result.ipc_class.value if result.ipc_class else None
        if act_class != expected["ipc_class"]:
            issues.append(f"ipc_class: expected={expected['ipc_class']}, got={act_class}")

    if expected.get("surface_finish"):
        act_sf = (result.surface_finish or "").upper()
        if act_sf != expected["surface_finish"].upper():
            issues.append(
                f"surface_finish: expected={expected['surface_finish']}, got={result.surface_finish}"
            )

    if expected.get("is_itar") is not None:
        if result.is_itar != expected["is_itar"]:
            issues.append(f"is_itar: expected={expected['is_itar']}, got={result.is_itar}")

    exp_bt = expected.get("board_thickness")
    if exp_bt and exp_bt.get("nominal"):
        if not result.board_thickness or not result.board_thickness.nominal:
            issues.append(f"board_thickness.nominal: expected={exp_bt['nominal']}, got=None")
        elif abs(result.board_thickness.nominal - exp_bt["nominal"]) >= 0.002:
            issues.append(
                f"board_thickness.nominal: expected={exp_bt['nominal']}, got={result.board_thickness.nominal}"
            )
        if result.board_thickness and result.board_thickness.unit != "in":
            issues.append(f"board_thickness.unit: expected='in', got={result.board_thickness.unit}")

    exp_dt = expected.get("drill_table", [])
    if exp_dt:
        if not result.drill_table:
            issues.append(f"drill_table: expected ~{len(exp_dt)} rows, got 0")
        elif len(result.drill_table) < len(exp_dt) * 0.8:
            issues.append(
                f"drill_table: expected ~{len(exp_dt)} rows, got {len(result.drill_table)}"
            )
        for row in result.drill_table:
            if row.size_mils and not (1.0 <= row.size_mils <= 500.0):
                issues.append(f"drill_table size_mils={row.size_mils} out of range")

    exp_sm = expected.get("solder_mask", {})
    if exp_sm and exp_sm.get("color"):
        act_color = (result.solder_mask.color or "") if result.solder_mask else ""
        if act_color.lower() != exp_sm["color"].lower():
            issues.append(f"solder_mask.color: expected={exp_sm['color']}, got={act_color}")

    exp_specs = set(expected.get("ipc_specs", []))
    act_specs = set(result.ipc_specs) if result.ipc_specs else set()
    missing = exp_specs - act_specs
    if missing:
        issues.append(f"ipc_specs missing: {missing}")

    exp_ls = expected.get("layer_stackup", [])
    if exp_ls:
        if not result.layer_stackup:
            issues.append(f"layer_stackup: expected ~{len(exp_ls)} layers, got 0")
        elif len(result.layer_stackup) < len(exp_ls) * 0.8:
            issues.append(
                f"layer_stackup: expected ~{len(exp_ls)} layers, got {len(result.layer_stackup)}"
            )

    exp_cw = expected.get("copper_weights", {})
    if exp_cw and exp_cw.get("signal_layers_oz"):
        act_cw = result.copper_weights
        if not act_cw or not act_cw.signal_layers_oz:
            issues.append(
                f"copper_weights.signal_layers_oz: expected={exp_cw['signal_layers_oz']}, got={act_cw.signal_layers_oz if act_cw else None}"
            )

    if expected.get("manufacturer"):
        exp_man = expected["manufacturer"].lower()
        act_man = (result.manufacturer or "").lower()
        if exp_man and not any(word in act_man for word in exp_man.split()):
            issues.append(
                f"manufacturer: expected '{expected['manufacturer']}', got '{result.manufacturer}'"
            )

    if expected.get("impedance_control", {}).get("controlled") is not None:
        exp_ic = expected["impedance_control"]["controlled"]
        act_ic = result.impedance_control.controlled if result.impedance_control else False
        if act_ic != exp_ic:
            issues.append(f"impedance_control.controlled: expected={exp_ic}, got={act_ic}")

    if expected.get("fabrication_notes"):
        if not result.fabrication_notes:
            issues.append("fabrication_notes: expected notes, got None")

    return issues


async def main() -> int:
    os.chdir(PROJECT_ROOT)

    ocr = TesseractOCR(base_dpi=300)
    total_passed = 0
    total_failed = 0

    for pdf_rel, expected_rel in SAMPLES:
        pdf_path = PROJECT_ROOT / pdf_rel
        expected_path = PROJECT_ROOT / expected_rel

        print(f"\n{'=' * 60}")
        print(f"Sample: {pdf_rel}")
        print(f"{'=' * 60}")

        if not pdf_path.exists():
            print(f"  SKIP: {pdf_path} not found")
            continue

        expected = load_expected(str(expected_path))

        # Step 1: Tesseract OCR
        start = time.monotonic()
        try:
            ocr_result = ocr.extract(str(pdf_path))
            raw_text = ocr_result["raw_text"]
            ocr_time = time.monotonic() - start
            print(f"  Tesseract OCR: {len(raw_text)} chars in {ocr_time:.1f}s")

            # Step 2: LLM structured extraction
            llm_start = time.monotonic()
            structured = await llm_extract(raw_text)
            llm_time = time.monotonic() - llm_start
            print(f"  LLM extraction: {llm_time:.1f}s")

            if structured:
                data = build_pcb_data(structured)
            else:
                # Fallback to Tesseract parser
                from shared.pcb_parser import parse_pcb_text

                data = parse_pcb_text(raw_text)
                print("  WARNING: LLM returned nothing, using Tesseract parser")

        except Exception as e:
            import traceback

            print(f"  FAIL: {e}")
            traceback.print_exc()
            total_failed += 1
            continue

        issues = check_sample(data, expected, pdf_rel)

        if not issues:
            print("  PASS")
            total_passed += 1
        else:
            print("  FAIL:")
            for issue in issues:
                print(f"    {issue}")
            total_failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {total_passed} passed, {total_failed} failed out of {len(SAMPLES)}")
    print(f"{'=' * 60}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(main()))
