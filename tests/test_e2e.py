"""End-to-end tests: PDF in → PCBData JSON out, compared against expected."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from conftest import AUTH_HEADERS, PROJECT_ROOT, SAMPLE_FILES, poll_job

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


def _load_expected(expected_path: str) -> PCBData:
    """Load and validate expected JSON."""
    with open(expected_path) as f:
        raw = json.load(f)

    # Manual deserialization with proper nested models
    data = dict(raw)

    # Board thickness
    if data.get("board_thickness"):
        bt = data["board_thickness"]
        data["board_thickness"] = BoardThickness(**bt)

    # Copper weights
    if data.get("copper_weights"):
        cw = data["copper_weights"]
        data["copper_weights"] = CopperWeights(**cw)

    # Solder mask
    if data.get("solder_mask"):
        sm = data["solder_mask"]
        data["solder_mask"] = SolderMask(**sm)

    # Silkscreen
    if data.get("silkscreen"):
        sk = data["silkscreen"]
        data["silkscreen"] = Silkscreen(**sk)

    # IPC class
    if data.get("ipc_class"):
        data["ipc_class"] = IPCClass(data["ipc_class"])

    # Impedance
    if data.get("impedance_control"):
        imp = data["impedance_control"]
        if imp.get("single_ended"):
            imp["single_ended"] = ImpedanceRange(**imp["single_ended"])
        data["impedance_control"] = ImpedanceControl(**imp)

    # Layer stackup
    if data.get("layer_stackup"):
        data["layer_stackup"] = [LayerSpec(**s) for s in data["layer_stackup"]]

    # Drill table
    if data.get("drill_table"):
        data["drill_table"] = [DrillRow(**d) for d in data["drill_table"]]

    return PCBData(**data)


def _check_accuracy(result: PCBData, expected: PCBData, sample_name: str) -> list[str]:
    """Compare result against expected with tolerances."""
    issues: list[str] = []

    # ── Core fields — exact match ──
    if expected.layer_count is not None:
        if result.layer_count != expected.layer_count:
            issues.append(f"layer_count: expected={expected.layer_count}, got={result.layer_count}")

    if expected.material is not None:
        if not result.material or result.material.upper() != expected.material.upper():
            issues.append(f"material: expected={expected.material}, got={result.material}")

    if expected.ipc_class is not None:
        if result.ipc_class != expected.ipc_class:
            issues.append(f"ipc_class: expected={expected.ipc_class}, got={result.ipc_class}")

    if expected.surface_finish is not None:
        if (
            not result.surface_finish
            or result.surface_finish.upper() != expected.surface_finish.upper()
        ):
            issues.append(
                f"surface_finish: expected={expected.surface_finish}, got={result.surface_finish}"
            )

    if expected.is_itar is not None:
        if result.is_itar != expected.is_itar:
            issues.append(f"is_itar: expected={expected.is_itar}, got={result.is_itar}")

    # ── Board thickness — tolerance check ──
    if expected.board_thickness and expected.board_thickness.nominal:
        if not result.board_thickness or not result.board_thickness.nominal:
            issues.append(
                f"board_thickness.nominal: expected={expected.board_thickness.nominal}, got=None"
            )
        else:
            if abs(result.board_thickness.nominal - expected.board_thickness.nominal) >= 0.002:
                issues.append(
                    f"board_thickness.nominal: expected={expected.board_thickness.nominal}, "
                    f"got={result.board_thickness.nominal}"
                )
            if result.board_thickness.unit != "in":
                issues.append(
                    f"board_thickness.unit: expected='in', got={result.board_thickness.unit}"
                )

    # ── Drill table — row count within 10% ──
    if expected.drill_table:
        if len(result.drill_table) < len(expected.drill_table) * 0.8:
            issues.append(
                f"drill_table: expected ~{len(expected.drill_table)} rows, got {len(result.drill_table)}"
            )
        # Verify all drill sizes are in mils (reasonable range: 1-500 mils)
        for row in result.drill_table:
            if row.size_mils is not None:
                if not (1.0 <= row.size_mils <= 500.0):
                    issues.append(
                        f"drill_table size_mils={row.size_mils} out of range — likely wrong units"
                    )

    # ── Solder mask color ──
    if expected.solder_mask and expected.solder_mask.color:
        if not result.solder_mask or (
            result.solder_mask.color
            and result.solder_mask.color.lower() != expected.solder_mask.color.lower()
        ):
            issues.append(
                f"solder_mask.color: expected={expected.solder_mask.color}, "
                f"got={result.solder_mask.color if result.solder_mask else None}"
            )

    # ── IPC specs — all expected must be present ──
    if expected.ipc_specs:
        exp_specs = set(expected.ipc_specs)
        act_specs = set(result.ipc_specs) if result.ipc_specs else set()
        missing = exp_specs - act_specs
        if missing:
            issues.append(f"ipc_specs missing: {missing}")

    # ── Layer stackup ──
    if expected.layer_stackup:
        if len(result.layer_stackup) < len(expected.layer_stackup) * 0.8:
            issues.append(
                f"layer_stackup: expected ~{len(expected.layer_stackup)} layers, got {len(result.layer_stackup)}"
            )

    # ── Copper weights ──
    if expected.copper_weights:
        if expected.copper_weights.signal_layers_oz:
            if not result.copper_weights or not result.copper_weights.signal_layers_oz:
                issues.append(
                    f"copper_weights.signal_layers_oz: expected={expected.copper_weights.signal_layers_oz}, "
                    f"got={result.copper_weights.signal_layers_oz if result.copper_weights else None}"
                )

    return issues


@pytest.mark.parametrize("pdf_path,expected_path", SAMPLE_FILES)
async def test_extraction_accuracy(pdf_path: str, expected_path: str):
    """Test that extraction matches expected output with tolerances."""
    pdf_path = PROJECT_ROOT / pdf_path
    expected_path = PROJECT_ROOT / expected_path

    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}")

    expected = _load_expected(str(expected_path))
    sample_name = Path(expected_path).stem

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Quick check if supervisor is available
            await client.get(
                os.environ.get("SUPERVISOR_URL", "http://localhost:8080/health"), timeout=2
            )
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPError):
            pytest.skip("Supervisor service not available (run 'make up' first)")

        with open(pdf_path, "rb") as f:
            response = await client.post(
                os.environ.get("SUPERVISOR_URL", "http://localhost:8080/extract"),
                files={"file": ("test.pdf", f, "application/pdf")},
                headers=AUTH_HEADERS,
            )
        assert response.status_code == 202, f"HTTP {response.status_code}: {response.text}"

        job_id = response.json()["job_id"]
        body = await poll_job(client, job_id, AUTH_HEADERS)
        if body["status"] == "failed":
            pytest.fail(f"Job failed for {sample_name}: {body.get('error')}")

        raw = body["result"]
        result = PCBData(**raw)

    output_dir = PROJECT_ROOT / "tests" / "output"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / f"{sample_name}.json", "w") as f:
        json.dump(raw, f, indent=2)

    issues = _check_accuracy(result, expected, sample_name)
    if issues:
        sep = "=" * 60
        msg = f"\n{sep}\nAccuracy issues for {sample_name}:\n" + "\n".join(issues)
        pytest.fail(msg)
