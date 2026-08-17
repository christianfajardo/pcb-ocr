"""Domain constants: IPC standards, materials, regex patterns, and lookup tables."""

from __future__ import annotations

import re

# ── IPC Standards ──────────────────────────────────────────────────────────────

IPC_STANDARDS: list[str] = [
    "IPC-2611",
    "IPC-4101",
    "IPC-4204",
    "IPC-4203",
    "IPC-6012",
    "IPC-6013",
    "IPC-A-600",
    "IPC-SM-840",
    "IPC-4761",
    "IPC-J-STD-003",
    "IPC-9252",
    "IPC-4781",
    "IPC-1066",
    "J-STD-609",
]

IPC_REGEX = re.compile(
    r"(?:IPC[-–]\s*)?"
    r"("
    r"2611|4101|4204|4203|6012|6013|A-?600"
    r"|SM-?840|4761|J-?STD-?003|9252|4781|1066"
    r"|J-?STD-?609"
    r")",
    re.IGNORECASE,
)

# ── Materials ─────────────────────────────────────────────────────────────────

FR4_MATERIALS: list[str] = [
    "FR4",
    "FR-4",
    "FR 4",
    "FR-406",
    "FR406",
    "370HR",
    "370-HR",
    "NEMA GR-18",
    "NEMA-G18",
    "TG130",
    "TG150",
    "TG170",
    "TG180",
    "PTFE",
    "Teflon",
    "Rogers",
    "Polyimide",
    "Kapton",
]

# ── Surface Finishes ──────────────────────────────────────────────────────────

SURFACE_FINISHES: dict[str, list[str]] = {
    "ENIG": ["ENIG", "electroless nickel immersion gold", "ENi/ImAu"],
    "HASL": ["HASL", "hot air solder leveling", "hast", "lead free hasl", "hasl lead free"],
    "HASL-Leaded": ["hasl leaded", "hasl lead", "leaded hasl", "tin lead hasl"],
    "OSP": ["OSP", "organic solderability preservative", "organic solder preservative"],
    "Immersion Tin": ["immersion tin", "immersion Sn", "immersion tin/silver"],
    "Immersion Silver": ["immersion silver", "immersion Ag"],
    "EHASL": ["E-HASL", "electrolytic HASL"],
    "Tin/Lead": ["tin/lead", "tin lead", "tin-lead", "SnPb"],
}

# ── Solder Mask Colors ────────────────────────────────────────────────────────

SOLDER_MASK_COLORS: list[str] = [
    "green",
    "red",
    "blue",
    "black",
    "white",
    "yellow",
    "natural",
    "brown",
]

SILKSCREEN_COLORS: list[str] = [
    "white",
    "yellow",
    "black",
    "blue",
    "red",
]

# ── IPC Class patterns ────────────────────────────────────────────────────────

IPC_CLASS_REGEX = re.compile(
    r"(?:class\s*\d|class\s*\w+)",
    re.IGNORECASE,
)

IPC_CLASS_MAP: dict[str, str] = {
    "class 1": "Class 1",
    "class 2": "Class 2",
    "class 3": "Class 3",
    "class i": "Class 1",
    "class ii": "Class 2",
    "class iii": "Class 3",
}

# ── ITAR / Export Control detection ───────────────────────────────────────────

ITAR_KEYWORDS: list[str] = [
    "itar",
    "export control",
    "export-controlled",
    "subject to export",
    "restricted data",
    "u.s. munitions",
    "us munitions",
    "itars",
    "information subject to export control laws",
    "ddt",
    "dod controlled",
    "evs",
]

# ── Board thickness regex ─────────────────────────────────────────────────────

THICKNESS_REGEX = re.compile(
    r"(?:thickness|thk|board\s*thk)\s*"
    r"[=:\s]*"
    r"(.+?)",
    re.IGNORECASE,
)

THICKNESS_VALUE_REGEX = re.compile(
    r"(\d+\.?\d*)\s*[+\-]\s*(\d+\.?\d*)",
)

THICKNESS_PERCENT_REGEX = re.compile(
    r"(\d+\.?\d*)\s*[+\-]\s*(\d+)%",
)

THICKNESS_MIL_REGEX = re.compile(
    r"(\d+\.?\d*)\s*mil",
    re.IGNORECASE,
)

# ── Copper weight patterns ────────────────────────────────────────────────────

COPPER_WEIGHT_REGEX = re.compile(
    r"(\d+\.?\d*)\s*(?:oz|ounce|ounces)\s*/\s*ft\s*\^?\s*2",
    re.IGNORECASE,
)

# ── Drill table patterns ──────────────────────────────────────────────────────

PLATED_REGEX = re.compile(
    r"(?:non\s*-{0,2}\s*plated|plated|non-plated|non plated|n\.p\.|unplated)",
    re.IGNORECASE,
)

# ── Layer function normalization ──────────────────────────────────────────────

LAYER_FUNCTION_ALIASES: dict[str, str] = {
    "signal": ["signal", "trace", "routing", "component", "comp"],
    "plane": ["plane", "power", "gnd", "ground", "ground plane", "power plane"],
    "core": ["core", "prepreg", "pre-preg"],
}


def normalize_layer_function(func: str) -> str:
    """Normalize layer function name to canonical form."""
    func_lower = func.lower().strip()
    for canonical, aliases in LAYER_FUNCTION_ALIASES.items():
        if func_lower in aliases:
            return canonical
    return func_lower


def normalize_material(material: str) -> str:
    """Normalize a material name to a canonical form.

    Collapses formatting variants that refer to the same material (e.g.
    'FR-4' / 'FR 4' -> 'FR4') by comparing with hyphens/spaces stripped.
    Distinct materials (e.g. TG130 vs TG180) are never merged, since they
    only match here if they're identical once punctuation is stripped.
    """
    if not material:
        return material

    def _key(s: str) -> str:
        return re.sub(r"[\s\-]", "", s).upper()

    target_key = _key(material)
    for canonical in FR4_MATERIALS:
        if _key(canonical) == target_key:
            return canonical
    return material


# ── Fabrication note section detection ────────────────────────────────────────

NOTE_HEADER_PATTERNS: list[str] = [
    r"^\s*notes?\s*:?\s*$",
    r"^\s*fab\s*notes?\s*:?\s*$",
    r"^\s*fabrication\s*notes?\s*:?\s*$",
    r"^\s*manufacturing\s*instructions?\s*:?\s*$",
]

# ── Quantity / count patterns ─────────────────────────────────────────────────

QUANTITY_REGEX = re.compile(r"qty[=:\s]+(\d+)", re.IGNORECASE)
QTY_IN_TEXT_REGEX = re.compile(r"(\d+)\s*(?:holes?|pcs?|qty)", re.IGNORECASE)

# ── Impedance patterns ────────────────────────────────────────────────────────

IMPEDANCE_REGEX = re.compile(
    r"(?:impedance|z0|z\s*0|controlled\s*z)\s*"
    r"[=:\s]*\s*"
    r"(\d+)\s*(?:\+-|±|plus-minus|plus/\-)\s*(\d+)%",
    re.IGNORECASE,
)

IMPEDANCE_VALUE_REGEX = re.compile(
    r"(?:single\s*ended|single-ended|z0)\s*[=:\s]*(\d+)\s*(?:ohm|Ω)?",
    re.IGNORECASE,
)

TRACE_WIDTH_REGEX = re.compile(
    r"(?:trace\s*width|w\s*=?\s*)\s*[=:\s]*(\d+\.?\d*)\s*(?:mil|mils?)?",
    re.IGNORECASE,
)

# ── Hole size unit detection ──────────────────────────────────────────────────


def is_mm_value(value_str: str) -> bool:
    """Check if a numeric value is likely in millimeters (drill sizes)."""
    try:
        val = float(value_str)
        if val > 50:
            return False
        if val < 1.0:
            return True
        return False
    except (ValueError, TypeError):
        return False


def mils_to_mm(mils: float) -> float:
    """Convert mils to millimeters."""
    return mils * 0.0254


def mm_to_mils(mm: float) -> float:
    """Convert millimeters to mils."""
    return mm / 0.0254
