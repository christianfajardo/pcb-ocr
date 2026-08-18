"""Canonical Pydantic schemas for PCB fabrication drawing extraction."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .constants import normalize_material

# ── Enum types ──────────────────────────────────────────────────────────────


class IPCClass(str, Enum):
    """IPC performance class. str-mixin => JSON-serializes to its value."""

    CLASS_1 = "Class 1"
    CLASS_2 = "Class 2"
    CLASS_3 = "Class 3"


# ── Nested models ───────────────────────────────────────────────────────────


class LayerSpec(BaseModel):
    """A single layer in the board stackup."""

    number: int
    function: str | None = None


class BoardThickness(BaseModel):
    """Board thickness with tolerances — ALL values in INCHES."""

    nominal: float | None = None
    plus_tol: float | None = None
    minus_tol: float | None = None
    unit: str = "in"
    raw: str | None = None


class CopperWeights(BaseModel):
    """Copper weight specifications — oz/ft²."""

    signal_layers_oz: float | None = None
    plane_layers_oz: float | None = None
    external_finished_oz: float | None = None


class SolderMask(BaseModel):
    """Solder mask specification."""

    present: bool = False
    type: str | None = None
    color: str | None = None
    spec: str | None = None
    sides: str | None = None


class Silkscreen(BaseModel):
    """Silkscreen/legend specification."""

    present: bool = False
    color: str | None = None
    ink: str | None = None
    sides: str | None = None


class ImpedanceRange(BaseModel):
    """Impedance target range."""

    min: float | None = None
    max: float | None = None
    unit: str = "ohm"
    raw: str | None = None


class ImpedanceControl(BaseModel):
    """Impedance control specification."""

    controlled: bool = False
    single_ended: ImpedanceRange | None = None
    trace_width_mils: float | None = None  # ALWAYS in mils
    layers: list[int] = Field(default_factory=list)


class DrillRow(BaseModel):
    """A single row from the drill table."""

    size_mils: float | None = None  # ALWAYS in mils
    qty: int | None = None
    symbol: str | None = None
    plated: bool | None = None


# ── Top-level extraction result ─────────────────────────────────────────────


class PCBData(BaseModel):
    """Top-level extracted record for one fabrication drawing."""

    part_number: str | None = None
    manufacturer: str | None = None
    drawing_title: str | None = None
    is_itar: bool | None = None
    ipc_class: IPCClass | None = None
    ipc_specs: list[str] = Field(default_factory=list)
    layer_count: int | None = None
    layer_stackup: list[LayerSpec] = Field(default_factory=list)
    material: str | None = None
    board_thickness: BoardThickness | None = None
    copper_weights: CopperWeights | None = None
    surface_finish: str | None = None
    solder_mask: SolderMask | None = None
    silkscreen: Silkscreen | None = None
    impedance_control: ImpedanceControl | None = None
    drill_table: list[DrillRow] = Field(default_factory=list)
    fabrication_notes: str | None = None
    ocr_raw_text: str | None = None


# ── Confidence envelope ─────────────────────────────────────────────────────


class FieldConfidence(BaseModel):
    """Confidence score for a single extracted field."""

    value: float = Field(ge=0.0, le=1.0, description="0.0 = no confidence, 1.0 = certain")
    source: str = Field(
        description="Which OCR engine produced this: tesseract | glm-ocr | qwen-vl | pymupdf"
    )
    reasoning: str | None = Field(default=None, description="Why this confidence level")
    source_text: str | None = Field(
        default=None,
        description=(
            "The text on the drawing that produced this value, when the engine "
            "can attribute it. Null when the value was inferred rather than read "
            "from a specific span (the vision models don't always report one)."
        ),
    )


class PCBDataWithConfidence(BaseModel):
    """OCR node output: extraction + per-field confidence."""

    data: PCBData
    confidence: dict[str, FieldConfidence]  # keys match PCBData field names
    ocr_engine: str
    processing_time_ms: float
    page_count: int


# ── Unit normalization ──────────────────────────────────────────────────────


def normalize_units(data: PCBData) -> PCBData:
    """
    Ensure all dimensional values are in inches/mils. Convert from mm if detected.

    Heuristics for detecting metric input:
    - Board thickness > 1.0 is likely mm (typical range 0.8-3.2 mm vs 0.031-0.125 in)
    - Drill size > 1000 is likely microns (real mils values commonly exceed 100
      for mounting/tooling holes, so that range is left alone)
    - Drill size < 1 is a raw decimal-inch value never converted to mils
    - A drill table (3+ rows) where every row has the identical size is a
      degenerate/hallucinated extraction, not a real hole schedule — discarded
    - Trace width > 50 is likely mm*1000 or microns

    Call this function before returning ANY PCBData instance from any OCR node
    or from the reconciliation step.
    """
    if data.material:
        data.material = normalize_material(data.material)

    if data.board_thickness and data.board_thickness.nominal:
        if data.board_thickness.nominal > 1.0:  # likely mm
            data.board_thickness.nominal /= 25.4
            if data.board_thickness.plus_tol:
                data.board_thickness.plus_tol /= 25.4
            if data.board_thickness.minus_tol:
                data.board_thickness.minus_tol /= 25.4
        data.board_thickness.unit = "in"

    for row in data.drill_table:
        if row.size_mils and row.size_mils > 1000:
            # Likely in microns — convert. (Values in 100-1000 are left alone:
            # that's a normal range for real drill sizes in mils, e.g. mounting
            # holes, and a genuine unconverted mm value would be a sub-10 decimal.)
            row.size_mils /= 25.4  # microns -> mils
        elif row.size_mils and 0 < row.size_mils < 1:
            # No real PCB drill is sub-1-mil — this is almost always a raw
            # decimal-inch value straight off the drawing (e.g. ".024") that
            # was never converted to mils (multiply by 1000: inches -> mils).
            row.size_mils *= 1000

    if len(data.drill_table) >= 3:
        sizes = {row.size_mils for row in data.drill_table if row.size_mils is not None}
        if len(sizes) == 1:
            # A real hole schedule always has multiple distinct sizes; every
            # row reporting the identical size means the model gave up and
            # echoed a single constant (often a unit-conversion factor from
            # its own prompt) instead of reading the table. Discard it so
            # reconciliation falls back to another engine.
            data.drill_table = []

    if data.impedance_control and data.impedance_control.trace_width_mils:
        if data.impedance_control.trace_width_mils > 50:
            # Likely microns or mm
            if data.impedance_control.trace_width_mils > 1000:
                data.impedance_control.trace_width_mils /= 25.4
            else:
                data.impedance_control.trace_width_mils /= 0.0254

    return data
