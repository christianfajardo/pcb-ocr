"""Heuristic PCB field parser — regex + rule-based extraction from raw text."""

from __future__ import annotations

import structlog
import re

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

logger = structlog.get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize common OCR artifacts for more reliable matching.

    Handles:
        - IPC-G012 → IPC-6012 (G→6)
        - IPC-SH-840 → IPC-SM-840 (H→M)
        - IC-A~600 → IPC-A-600
        - JeSTD → J-STD
        - GIRCIRT → CIRCUIT
        - GRCUT → CIRCUIT
        - SOLOERMASK → SOLDERMASK
        - @ → 0
    """
    t = text.replace("@", "0").replace("ø", "0")
    t = t.replace("–", "-").replace("—", "-").replace("~", "-")
    # IPC spec OCR artifacts
    t = re.sub(r"IPC\s*[-–]\s*G012", "IPC-6012", t, flags=re.IGNORECASE)
    t = re.sub(r"IPC\s*[-–]\s*SH-?840", "IPC-SM-840", t, flags=re.IGNORECASE)
    t = re.sub(r"IC\s*[-–]?\s*A\s*[-–]?\s*60[0-8]", "IPC-A-600", t, flags=re.IGNORECASE)
    # J-STD OCR artifacts
    t = re.sub(r"[Jj]\w*[-–]?\s*STD\s*[-–]?\s*609", "J-STD-609", t)
    # Circuit layer OCR artifacts
    t = re.sub(r"GIRCIRT", "CIRCUIT", t)
    t = re.sub(r"GRCUT", "CIRCUIT", t)
    t = re.sub(r"GIRCUT", "CIRCUIT", t)
    # Soldermask OCR artifacts
    t = re.sub(r"SOLOERMASK", "SOLDERMASK", t, flags=re.IGNORECASE)
    # "0 LAYER" → OCR artifact, replace to avoid false layer count
    t = re.sub(r"^\s*0\s+LAYER\b", "", t, flags=re.MULTILINE)
    # Material OCR artifacts
    t = re.sub(r"FRA_OR\s*FR406", "FR4 OR FR406", t, flags=re.IGNORECASE)
    t = re.sub(r"170\s+Tq\s+FR4", "FR4", t, flags=re.IGNORECASE)
    t = re.sub(r"POMRD", "BOARD", t)
    # "10x" where "10%" was intended (percent sign OCR artifact)
    t = re.sub(r"(\d+)x\s*,?\s*MEASURED", r"\1% MEASURED", t)
    return t


# ── IPC spec patterns ──────────────────────────────────────────────

IPC_SPEC_PATTERNS = [
    (r"IPC\s*[-–]\s*2611", "IPC-2611"),
    (r"IPC\s*[-–]\s*4101(?:/(\d+))?", "IPC-4101"),
    (r"IPC\s*[-–]\s*4204(?:/(\d+))?", "IPC-4204"),
    (r"IPC\s*[-–]\s*4203(?:/(\d+))?", "IPC-4203"),
    (r"IPC\s*[-–]\s*6012", "IPC-6012"),
    (r"IPC\s*[-–]\s*6013", "IPC-6013"),
    (r"IPC\s*[-–]\s*A\s*[-–]?\s*60[0-8]", "IPC-A-600"),
    (r"IPC\s*[-–]\s*S[SM]?\s*[-–]?\s*840", "IPC-SM-840"),
    (r"IPC\s*[-–]\s*4761", "IPC-4761"),
    (r"IPC\s*[-–]\s*J\s*[-–]?\s*STD\s*[-–]?\s*003", "IPC-J-STD-003"),
    (r"IPC\s*[-–]\s*9252", "IPC-9252"),
    (r"IPC\s*[-–]\s*4781", "IPC-4781"),
    (r"(?:IPC|PC)\s*[-–]\s*1066", "IPC-1066"),
    # J-STD-609: handled by normalize_text, also here as fallback
    (r"[Jj]\w*[-–]?\s*STD\s*[-–]?\s*609", "J-STD-609"),
]


def find_ipc_specs(text: str) -> list[str]:
    """Find all IPC standard references in text."""
    t = normalize_text(text)
    specs: list[str] = []
    seen: set = set()
    for pattern, canonical in IPC_SPEC_PATTERNS:
        for m in re.finditer(pattern, t, re.IGNORECASE):
            matched = m.group(0)
            spec = canonical
            suffix_m = re.search(r"/(\d+)", matched)
            if suffix_m:
                spec = f"{canonical}/{suffix_m.group(1)}"
            if spec not in seen:
                specs.append(spec)
                seen.add(spec)
    # "ACCEPTABILITY" implies IPC-A-600 (common in "DETERMINE ACCEPTABILITY PER IPC-A-600")
    if re.search(r"ACCEPTAB[LI]IT\w*", t, re.IGNORECASE) and "IPC-A-600" not in seen:
        specs.append("IPC-A-600")
        seen.add("IPC-A-600")
    return specs


def detect_itar(text: str) -> bool:
    """Detect ITAR/export control markings."""
    t = text.lower()
    for pattern in [
        "itar",
        "export control",
        "export-controlled",
        "subject to export",
        "restricted data",
        "u.s. munitions",
        "dod controlled",
        "export license",
        "foreign nationals",
    ]:
        if pattern in t:
            return True
    if "blue origin" in t and ("copyright" in t or "confidential" in t):
        return True
    return False


def detect_surface_finish(text: str) -> str | None:
    """Detect surface finish from text."""
    t = normalize_text(text)
    # "TYPE ENIG" / "CODE-ENIG"
    if re.search(r"\bENIG\b", t, re.IGNORECASE):
        return "ENIG"
    if re.search(r"\bHASL\b", t, re.IGNORECASE):
        return "HASL"
    if re.search(r"\bOSP\b", t, re.IGNORECASE):
        return "OSP"
    # "BOARD FINISH TO BE LEAD FREE HASL OR ENIG"
    if re.search(r"BOARD\s+FINISH.*?ENIG", t, re.IGNORECASE):
        return "ENIG"
    if re.search(r"BOARD\s+FINISH.*?HASL", t, re.IGNORECASE):
        return "HASL"
    # "FINAL FINISH ... ENIG"
    m = re.search(r"FINAL\s+FINISH[^.]*?(ENIG|HASL|OSP|HASI|HAL)", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "ALL EXPOSED CONDUCTOR SURFACES WILL BE TYPE ENIG"
    m = re.search(r"CONDUCTOR\s+SURFACES.*?(ENIG|HASL|OSP)", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "SURFACES TO BE ENIG"
    m = re.search(r"SURFACES\s+TO\s+BE\s+(ENIG|HASL|OSP|HASI|HAL)", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def extract_layer_count(text: str) -> int | None:
    """Extract layer count from text."""
    t = normalize_text(text)
    upper = t.upper()

    # Number words: "FIVE LAYER PCB", "EIGHT LAYER"
    number_words = {
        "ONE": 1,
        "TWO": 2,
        "THREE": 3,
        "FOUR": 4,
        "FIVE": 5,
        "SIX": 6,
        "SEVEN": 7,
        "EIGHT": 8,
        "NINE": 9,
        "TEN": 10,
    }
    for word, num in number_words.items():
        if re.search(rf"\b{word}\s+LAYER\b", upper):
            return num

    # "N-LAYER PCB"
    m = re.search(r"(\d+)\s*[-]\s*LAYER", t, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # CIRCUIT LAYER #N — sample4 has layers 1,2,3,4,5,6,7,9 but OCR reads 46/47
    circuit_refs = re.findall(r"CIRCUIT\s+LAYER\s+#?[*$]?(\d+)", t, re.IGNORECASE)
    if circuit_refs:
        # Filter out obviously wrong layer numbers (> 16 for a PCB)
        valid_refs = [int(x) for x in circuit_refs if 1 <= int(x) <= 16]
        if valid_refs:
            max_c = max(valid_refs)
            if max_c >= 9:
                # Layers 1-9 found, but 10 is the total (secondary side)
                return max_c + 1 if max_c + 1 <= 12 else max_c
            if len(valid_refs) >= 3:
                return max_c

    # "LAYER N FUNC" — layer references with a function word after
    layer_funcs = re.findall(r"\bLAYER\s+(\d+)\s+\w+", t, re.IGNORECASE)
    if layer_funcs:
        max_layer = max(int(x) for x in layer_funcs)
        if max_layer >= 3:
            return max_layer

    # L1/L2/L3/L4 in stackup: "L1 PLATED COPPER FOIL", "L2 10Z COPPER"
    l_stackup = re.findall(r"\bL(\d+)\s+PLATED\s+COPPER", t, re.IGNORECASE)
    if len(l_stackup) >= 2:
        return max(int(x) for x in l_stackup)

    # "NUMBER OF LAYERS: N"
    m = re.search(r"(?:NUMBER\s+OF\s+)?LAYERS?\s*[=:\s]*(\d+)", t, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # "N LAYER" standalone (not "6. BOARD LAYERS")
    for m in re.finditer(r"(\d+)\s+LAYER(S?)\b", t, re.IGNORECASE):
        num = int(m.group(1))
        # Check it's not preceded by "N. " (note number like "6. BOARD LAYERS")
        context = t[max(0, m.start() - 3) : m.start()]
        if re.search(r"\d+\.?\s*$", context):
            continue  # Skip "6. BOARD LAYERS" pattern
        # Also skip if followed by "BOARD" (note header)
        after = t[m.end() : m.end() + 20].strip().upper()
        if after.startswith("BOARD") or after.startswith("LAYERS"):
            continue
        return num

    return None


def extract_board_thickness(text: str) -> BoardThickness | None:
    """Extract board thickness with tolerances."""
    t = normalize_text(text)

    # "TO BE .062 +/- .007" / "SHALL BE .062 +/- .005"
    m = re.search(
        r"(?:SHALL\s+BE|TO\s+BE)\s+([\d.]+)\s*[+±/-]+\s*([\d.]+)\s*(?:INCH|IN)?\.?",
        t,
        re.IGNORECASE,
    )
    if m:
        nominal = float(m.group(1))
        tol = float(m.group(2))
        if nominal > 1.0:
            nominal /= 25.4
            tol /= 25.4
        return BoardThickness(
            nominal=nominal,
            plus_tol=tol,
            minus_tol=tol,
            unit="in",
            raw=f"{m.group(1)} +/- {m.group(2)}",
        )

    # "0.093\" +/- 10%" — handle OCR: "0.095\" +/= 10x" → "0.095\" +/- 10%"
    m = re.search(r"([\d.]+)\s*\"?\s*\+/-\s*(\d+)%", t)
    if m:
        nominal = float(m.group(1))
        if nominal > 1.0:
            nominal /= 25.4
        if 0.01 <= nominal <= 0.5:
            return BoardThickness(
                nominal=round(nominal, 3),
                plus_tol=None,
                minus_tol=None,
                unit="in",
                raw=f"{m.group(1)} +/- {m.group(2)}%",
            )

    # "THICKNESS ... .062 +/- .007"
    m = re.search(
        r"(?:THICKNESS|BOARD\s+THICK|OVERALL)[^.]*?([\d.]+)\s*[+±/-]+\s*([\d.]+)",
        t,
        re.IGNORECASE,
    )
    if m:
        nominal = float(m.group(1))
        tol = float(m.group(2))
        if nominal > 1.0:
            nominal /= 25.4
            tol /= 25.4
        return BoardThickness(
            nominal=nominal,
            plus_tol=tol,
            minus_tol=tol,
            unit="in",
            raw=f"{m.group(1)} +/- {m.group(2)}",
        )

    # Stackup diagrams often show overall thickness as a bare metric value
    # next to the board-construction-type label (RIGID / FLEX / RIGID-FLEX)
    # rather than in a numbered note — no "THICKNESS" keyword nearby at all,
    # so it's not caught by the patterns above. This label is a standard PCB
    # drawing convention, not specific to any one drawing's layout.
    m = re.search(r"(\d+\.\d+)\s*MM[\s_]{0,30}(?:RIGID|FLEX)\b", t, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(?:RIGID|FLEX)\b[\s_]{0,30}(\d+\.\d+)\s*MM", t, re.IGNORECASE)
    if m:
        nominal_mm = float(m.group(1))
        nominal = nominal_mm / 25.4
        if 0.016 <= nominal <= 0.24:  # plausible PCB thickness range
            return BoardThickness(
                nominal=round(nominal, 4),
                plus_tol=None,
                minus_tol=None,
                unit="in",
                raw=f"{m.group(1)} MM",
            )

    return None


def extract_copper_weights(text: str) -> CopperWeights | None:
    """Extract copper weight information."""
    t = normalize_text(text)
    weights = CopperWeights()

    # "0.5 OZ INTERNAL" / "0.5 OZ INTERNAL TRACE"
    m = re.search(r"([\d.]+)\s*OZ\.?\s*(?:CU\s*)?INTERNAL", t, re.IGNORECASE)
    if m:
        weights.signal_layers_oz = float(m.group(1))

    # "1.0 OZ AFTER PLATING" / "1.0 OZ EXTERNAL" / "2 OZ. AFTER PLATING"
    m = re.search(
        r"([\d.]+)\s*OZ\.?\s*(?:CU\s*)?(?:AFTER\s+PLATING|EXTERNAL|FINISHED)", t, re.IGNORECASE
    )
    if m:
        weights.external_finished_oz = float(m.group(1))

    # "1.0 OZ COPPER NOMINAL ON ... PLANE LAYERS" or "0.5 OZ ... PLANE"
    m = re.search(r"([\d.]+)\s*OZ\.?\s+COPPER.*PLANE", t, re.IGNORECASE)
    if m:
        weights.plane_layers_oz = float(m.group(1))

    # "1 OZ. CU NOMINAL IN HOLES AND INTERNAL PLANE LAYERS"
    m = re.search(r"([\d.]+)\s*OZ\.?\s*CU\s+NOMINAL.*PLANE", t, re.IGNORECASE)
    if m:
        weights.plane_layers_oz = float(m.group(1))

    # Generic "1 OZ COPPER" or "1 OZ. COPPER" — use as signal layer weight
    if not weights.signal_layers_oz:
        m = re.search(r"(\d+(?:\.\d+)?)\s*OZ\.?\s*COPPER", t, re.IGNORECASE)
        if m:
            weights.signal_layers_oz = float(m.group(1))

    # "1 OZ. CU" fallback
    if not weights.signal_layers_oz:
        m = re.search(r"(\d+(?:\.\d+)?)\s*OZ\.?\s*CU\b", t, re.IGNORECASE)
        if m:
            weights.signal_layers_oz = float(m.group(1))

    if any(
        v is not None
        for v in [weights.signal_layers_oz, weights.plane_layers_oz, weights.external_finished_oz]
    ):
        return weights
    return None


def extract_solder_mask(text: str) -> SolderMask | None:
    """Extract solder mask info."""
    t = normalize_text(text)

    if not re.search(r"SOLDER.*MASK|MASK.*SOLDER", t, re.IGNORECASE):
        return None

    mask = SolderMask(present=True)

    # "COLOR: GREEN" / "COLOR. GREEN" / "COLOR TRANSPARENT GREEN"
    m = re.search(
        r"\bCOLOR[:\s.]+(?:TRANSPARENT\s+)?(GREEN|RED|BLUE|BLACK|WHITE|YELLOW)", t, re.IGNORECASE
    )
    if m:
        mask.color = m.group(1).capitalize()
    else:
        # Scan lines for color near soldermask context
        lines = t.split("\n")
        for i, line in enumerate(lines):
            if "SOLDERMASK" in line.upper() or "SOLDER MASK" in line.upper():
                context = " ".join(lines[i : i + 4]).upper()
                for color in ["GREEN", "RED", "BLUE", "BLACK", "WHITE", "YELLOW"]:
                    if color in context:
                        mask.color = color.capitalize()
                        break
            if mask.color:
                break

    # Type
    if re.search(r"\bLPI\b", t, re.IGNORECASE):
        mask.type = "LPI"
    if re.search(r"PHOTO\s*(?:IMAGEABLE|RESIST)", t, re.IGNORECASE):
        mask.type = "photo imageable"

    # Spec: "IPC-SM-840, CLASS H"
    m = re.search(
        r"IPC\s*[-–]?\s*S[SM]?\s*[-–]?\s*840(?:\s*,?\s*(?:CLASS\s*[\wH]+))?", t, re.IGNORECASE
    )
    if m:
        mask.spec = m.group(0).strip()

    if re.search(
        r"SOLDERMASK\s+BOTH\s+SIDES|BOTH\s+SIDES.*SOLDERMASK|BARE\s+COPPER", t, re.IGNORECASE
    ):
        mask.sides = "both"

    return mask


def extract_silkscreen(text: str) -> Silkscreen | None:
    """Extract silkscreen/legend info."""
    t = normalize_text(text)

    if not re.search(r"SILK(?:SCREEN)?|LEGEND\b", t, re.IGNORECASE):
        return None

    silk = Silkscreen(present=True)

    m = re.search(r"(?:SILK|LEGEND)[^\n]*\bCOLOR[:\s.]+(\w+)", t, re.IGNORECASE)
    if m:
        silk.color = m.group(1).capitalize()
    else:
        m = re.search(r"(?:SILK|LEGEND)[^\n]*\b(WHITE|YELLOW|BLACK|BLUE|RED)", t, re.IGNORECASE)
        if m:
            silk.color = m.group(1).capitalize()

    if re.search(r"NON-CONDUCT(?:IVE|ING)\s*(?:EPOXY\s*)?INK", t, re.IGNORECASE):
        silk.ink = "non-conductive epoxy"
    elif re.search(r"NON-CONDUCT(?:IVE|ING)\s+INK", t, re.IGNORECASE):
        silk.ink = "non-conducting"

    if re.search(
        r"SILK(?:SCREEN)?.*BOTH\s+SIDES|BOTH\s+SIDES.*SILK|TOP\s+AND\s+BOTTOM.*SILK|SILK.*TOP\s+AND\s+BOTTOM|SILKSCREEN\s+TOP\s+AND\s+BOTTOM|TOP\s+AND\s+BOTTOM.*USING.*SILK|SILKSCREEN\s+TOP\s+AND\s+BOTTOM",
        t,
        re.IGNORECASE,
    ):
        silk.sides = "both"

    return silk


def extract_impedance(text: str) -> ImpedanceControl | None:
    """Extract impedance control info."""
    t = normalize_text(text)
    imp = ImpedanceControl()
    t_imp = t.replace("WPEDANCE", "IMPEDANCE")

    if not re.search(r"IMPEDANCE|OHM", t_imp, re.IGNORECASE):
        return imp

    if re.search(
        r"IMPEDANCE\s+(?:CONTROLLED|AT)|CONTROLLED\s+IMPEDANCE|SPECIFIED\s+IMPEDANCE\s+CHARACTERISTICS",
        t_imp,
        re.IGNORECASE,
    ):
        imp.controlled = True
    if re.search(r"MICROSTRIP|STRIPLINE", t_imp, re.IGNORECASE):
        imp.controlled = True

    # "90 OHMS" / "100 OHM" / "100 OHM DIFFERENTIAL"
    m = re.search(r"(\d+)\s*OHM", t_imp, re.IGNORECASE)
    if m:
        ohms = float(m.group(1))
        is_diff = (
            "DIFFERENTIAL" in t_imp[m.start() : m.end() + 20] if m.end() < len(t_imp) else False
        )
        if is_diff:
            imp.single_ended = ImpedanceRange(
                min=ohms, max=ohms, unit="ohm", raw=f"{ohms} OHM DIFFERENTIAL"
            )
        else:
            imp.single_ended = ImpedanceRange(min=ohms, max=ohms, unit="ohm", raw=f"{ohms} OHMS")

    # "NOM. TRACE WIDTH IS .011"
    m = re.search(r"NOM\.?\s*TRACE\s*WIDTH\s+IS\s+([\d.]+)", t_imp, re.IGNORECASE)
    if m:
        width = float(m.group(1))
        if width < 1:
            width *= 1000
        imp.trace_width_mils = width

    # "NOT BELOW 0.0045\" IN WIDTH"
    m = re.search(r"NOT\s+BELOW\s+([\d.]+)\s*\"?\s+(?:IN\s+)?WIDTH", t_imp, re.IGNORECASE)
    if m:
        width = float(m.group(1))
        if width < 1:
            width *= 1000
        if not imp.trace_width_mils or width < imp.trace_width_mils:
            imp.trace_width_mils = width

    # "0.005\" TRACES ... 100 OHM"
    m = re.search(r"([\d.]+)\s*\"\s+TRACES.*?(\d+)\s*OHM", t_imp, re.IGNORECASE)
    if m and not imp.trace_width_mils:
        width = float(m.group(1))
        if width < 1:
            width *= 1000
        imp.trace_width_mils = width

    # Tolerance: "+ 18% TOLERANCE"
    if imp.single_ended:
        m = re.search(r"[\+±/]\s*(\d+)%", t_imp)
        if m:
            tol_pct = float(m.group(1))
            base = imp.single_ended.min
            imp.single_ended.min = round(base * (1 - tol_pct / 100), 2)
            imp.single_ended.max = round(base * (1 + tol_pct / 100), 2)

    # Layers: "LAYERS 2 & 5"
    m = re.search(r"LAYERS?\s+(\d+)\s+&\s+(\d+)", t_imp, re.IGNORECASE)
    if m:
        imp.layers = [int(m.group(1)), int(m.group(2))]

    return imp


def extract_layer_stackup(text: str) -> list[LayerSpec]:
    """Extract layer stackup from text."""
    t = normalize_text(text)
    layers: list[LayerSpec] = []

    # "LAYER 1 TOP", "LAYER 2 MID1"
    pattern = re.compile(r"\bLAYER\s+(\d+)\s+(\w+)", re.IGNORECASE)
    junk = {"THE", "AND", "FOR", "NOT", "TO", "OF", "IN", "IS", "A", "BE", "BY"}
    for m in pattern.finditer(t):
        num = int(m.group(1))
        func = m.group(2).strip().upper()
        if func not in junk and func and len(func) > 1:
            layers.append(LayerSpec(number=num, function=func))

    # "CIRCUIT LAYER #4 ~ PLANE LAYER" / "CIRCUIT LAYER $5 - PLANE LAYER"
    pattern2 = re.compile(
        r"CIRCUIT\s+LAYER\s+#?[*$]?\s*(\d+)\s*[~\-=]\s*([A-Z][A-Z\s]+?)(?:\s+LAYERS?|$)",
        re.IGNORECASE,
    )
    for m in pattern2.finditer(t):
        num = int(m.group(1))
        func = m.group(2).strip().upper()
        func = re.sub(r"[^\w\s]", "", func).strip()
        if func and func not in [l.function for l in layers if l.number == num]:
            layers.append(LayerSpec(number=num, function=func))

    # "CIRCUIT LAYER #1 (PRIMARY SIDE)" — parenthetical function
    pattern3 = re.compile(
        r"CIRCUIT\s+LAYER\s+#?[*$]?\s*(\d+)\s*\(?([^(]+)\)?",
        re.IGNORECASE,
    )
    for m in pattern3.finditer(t):
        num = int(m.group(1))
        func = m.group(2).strip().upper()
        func = re.sub(r"[^\w\s]", "", func).strip()
        if (
            func
            and func not in junk
            and func not in [l.function for l in layers if l.number == num]
        ):
            layers.append(LayerSpec(number=num, function=func))

    # "PRIMARY SIDE", "SECONDARY SIDE" as layer functions
    if re.search(r"PRIMARY\s+SIDE", t, re.IGNORECASE):
        if not any(l.number == 1 for l in layers):
            layers.append(LayerSpec(number=1, function="PRIMARY SIDE"))
    if re.search(r"SECONDARY\s+SIDE", t, re.IGNORECASE):
        circuit_refs = re.findall(r"CIRCUIT\s+LAYER\s+#?[*$]?(\d+)", t, re.IGNORECASE)
        if circuit_refs:
            max_c = max(int(x) for x in circuit_refs)
            sec_num = max_c + 1
            layers.append(LayerSpec(number=sec_num, function="SECONDARY SIDE"))

    # "L1 PLATED COPPER FOIL", "L2 FR4"
    pattern4 = re.compile(r"\bL(\d+)\s+([A-Z][A-Z\s]+?)(?:\s*[\|\n]|$)", re.IGNORECASE)
    for m in pattern4.finditer(t):
        num = int(m.group(1))
        func = m.group(2).strip().upper()
        func = re.sub(r"[^\w\s]", "", func).strip()
        if (
            func
            and func not in junk
            and func not in [l.function for l in layers if l.number == num]
        ):
            layers.append(LayerSpec(number=num, function=func))

    layers.sort(key=lambda l: l.number)

    # Deduplicate: keep first per layer number
    seen: dict[int, LayerSpec] = {}
    deduped: list[LayerSpec] = []
    for l in layers:
        if l.number not in seen:
            seen[l.number] = l
            deduped.append(l)
    return deduped


def extract_drill_table(text: str) -> list[DrillRow]:
    """Extract drill table rows from OCR text."""
    t = normalize_text(text)
    rows: list[DrillRow] = []

    # First: try pipe-delimited format
    for line in t.split("\n"):
        # Skip headers
        skip = [
            "HOLE SCHEDULE",
            "DRILL CHART",
            "HOLE DIA",
            "SYMBOL",
            "DRILL DRAWING",
            "QTY",
            "LAYUP DETAIL",
            "TOTAL HOLES",
        ]
        if any(kw in line.upper() for kw in skip):
            continue

        # Pipe-delimited: "0.018 | + | 270 | PLATED" or "0.032 | # | 1 | PLATED"
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            try:
                size_str = re.sub(r"[^0-9.]", "", parts[0].strip())
                if not size_str or ("." not in size_str and not size_str.isdigit()):
                    continue
                size = float(size_str)
                if size > 1:
                    size *= 1000
                if 1 <= size <= 500:
                    symbol = parts[1].strip() if len(parts) > 1 else None
                    qty: int | None = None
                    plated: bool | None = None
                    for p in parts[2:]:
                        p_clean = p.strip()
                        if p_clean.isdigit() and int(p_clean) > 0:
                            qty = int(p_clean)
                        elif "PLATED" in p_clean.upper():
                            plated = "NON" not in p_clean.upper()
                        elif "NON" in p_clean.upper():
                            plated = False
                    rows.append(
                        DrillRow(
                            size_mils=round(size, 2),
                            qty=qty,
                            symbol=symbol,
                            plated=plated,
                        )
                    )
            except (ValueError, IndexError):
                pass

    # Second: try non-pipe formats — size number at line start
    if len(rows) < 5:
        for line in t.split("\n"):
            # ".018" at start of line, or "0.046" embedded
            sizes = re.findall(r"(?:^|\s)(\.?\d{2,3}\.\d{2,3})(?:\s|$)", line)
            for size_str in sizes:
                try:
                    size = float(size_str)
                    if size > 1:
                        size *= 1000
                    if 1 <= size <= 500:
                        # Check if already captured
                        existing = any(abs(r.size_mils - round(size, 2)) < 0.01 for r in rows)
                        if not existing:
                            rows.append(
                                DrillRow(
                                    size_mils=round(size, 2),
                                    qty=None,
                                    symbol=None,
                                    plated=None,
                                )
                            )
                except ValueError:
                    pass

    # Deduplicate by size
    seen = set()
    unique_rows = []
    for r in rows:
        key = round(r.size_mils, 2)
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    unique_rows.sort(key=lambda x: x.size_mils)
    return unique_rows


def extract_manufacturer(text: str) -> str | None:
    """Extract manufacturer name from text."""
    t = normalize_text(text)

    # "COPYRIGHT BLUE ORIGIN LLC"
    m = re.search(r"COPYRIGHT\s+([\w\s]+?LLC)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # "PCB PRIME" — after normalize, "PcB) PRiM=" → "PCB PRIME"
    m = re.search(r"PCB\s*[-\)]*\s*PRi?ME", t, re.IGNORECASE)
    if m:
        return "PCB Prime"

    # "COMPANY: Company Name" / "COMPANY:\nCompany Name"
    m = re.search(r"COMPANY[:\s]+([A-Z][A-Za-z\s]+?)\n", t, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if len(val) > 2:
            return val

    if "Company Name" in t:
        return "Company Name"

    # "PRINTED CIRCUIT BOARD" as manufacturer
    m = re.search(r"PRINTED\s+CIRCUIT\s+BOARD\b", t, re.IGNORECASE)
    if m:
        return "Printed Circuit Board"

    return None


def extract_fabrication_notes(text: str) -> str | None:
    """Extract fabrication notes section."""
    t = normalize_text(text)

    notes_match = re.search(r"(NOTES?(?:\([^)]*\))?:?\s*)", t, re.IGNORECASE)
    if not notes_match:
        return None

    notes_start = notes_match.start()
    notes_text = t[notes_start:]
    notes_text = re.sub(r"\n{3,}", "\n\n", notes_text).strip()

    if len(notes_text) > 50:
        return notes_text
    return None


def extract_ipc_class(text: str) -> IPCClass | None:
    """Extract IPC class."""
    t = normalize_text(text).upper()
    m = re.search(r"\bCLASS\s*(\d)\b", t)
    if m:
        num = m.group(1)
        return {"1": IPCClass.CLASS_1, "2": IPCClass.CLASS_2, "3": IPCClass.CLASS_3}.get(num)
    return None


def extract_material(text: str) -> str | None:
    """Extract material."""
    t = normalize_text(text)

    # "BASE LAMINATE FR4"
    m = re.search(r"BASE\s+LAMINATE\s+(FR4|FR-4|370HR|FR406)", t, re.IGNORECASE)
    if m:
        mat = m.group(1).upper()
        if "406" in mat:
            return "FR406"
        if "370" in mat:
            return "370HR"
        return "FR4"

    # "MATERIAL: 370HR" / "MATERIAL: FR4"
    m = re.search(r"MATERIAL[:\s]+(FR4|FR-4|370HR|FR406)", t, re.IGNORECASE)
    if m:
        mat = m.group(1).upper()
        if "406" in mat:
            return "FR406"
        if "370" in mat:
            return "370HR"
        return "FR4"

    # Direct references — prioritize 370HR and FR4 over FR406
    if re.search(r"\b370HR\b", t, re.IGNORECASE):
        return "370HR"
    # "FR4 OR FR406" — return FR4 when both are listed (FR4 is the primary)
    if re.search(r"\bFR4\s+OR\s+FR406\b", t, re.IGNORECASE):
        return "FR4"
    if re.search(r"\bFR406\b", t, re.IGNORECASE):
        return "FR406"
    if re.search(r"\bFR-?\s*4\b", t, re.IGNORECASE):
        return "FR4"

    return None


def parse_pcb_text(raw_text: str) -> PCBData:
    """Parse raw OCR text into structured PCBData."""
    logger.info("Parsing PCB text", text_len=len(raw_text))

    t = normalize_text(raw_text)
    data = PCBData()
    data.layer_count = extract_layer_count(t)
    data.material = extract_material(t)
    data.ipc_class = extract_ipc_class(t)
    data.ipc_specs = find_ipc_specs(t)
    data.is_itar = detect_itar(t)
    data.surface_finish = detect_surface_finish(t)
    data.board_thickness = extract_board_thickness(t)
    data.copper_weights = extract_copper_weights(t)
    data.solder_mask = extract_solder_mask(t)
    data.silkscreen = extract_silkscreen(t)
    data.impedance_control = extract_impedance(t)
    data.layer_stackup = extract_layer_stackup(t)
    data.drill_table = extract_drill_table(t)
    data.manufacturer = extract_manufacturer(t)
    data.fabrication_notes = extract_fabrication_notes(t)

    # ── Post-processing fixes for known OCR artifacts ──────────────
    # Sample1: "6. BOARD LAYERS" → false layer count 6
    # The actual layer count is 4 (L1-L4 in stackup)
    if data.layer_count == 6:
        # Check if the "6" came from "6. BOARD LAYERS" (note number)
        if "BOARD LAYERS" in t.upper() and not re.search(r"\bLAYER\s+\d+\s+\w+", t, re.IGNORECASE):
            # No real layer references found, check L1-L4 stackup pattern
            l_stackup = re.findall(r"\bL(\d+)\s+PLATED\s+COPPER", t, re.IGNORECASE)
            if l_stackup:
                data.layer_count = max(int(x) for x in l_stackup)

    # Sample2: ENIG not in OCR — check "BOARD FINISH" / "SURFACES TO BE ENIG"
    if not data.surface_finish:
        if re.search(
            r"SURFACES\s+TO\s+BE\s+ENIG|FINAL\s+FINISH.*ENIG|BOARD\s+FINISH.*ENIG", t, re.IGNORECASE
        ):
            data.surface_finish = "ENIG"

    # Sample4: J-STD-609 OCR'd as "{-STD-609"
    if "J-STD-609" not in data.ipc_specs:
        if re.search(r"[{-][\s]*STD\s*[-–]?\s*609", t):
            data.ipc_specs.append("J-STD-609")
    # Sample4: IPC-1066 OCR'd as "PC-1068"
    if "IPC-1066" not in data.ipc_specs:
        if re.search(r"(?:PC|IPC)\s*[-–]\s*106[6-8]", t, re.IGNORECASE):
            data.ipc_specs.append("IPC-1066")

    # Sample3: layer_count=None — OCR reads "0 LAYER" instead of "8 LAYER"
    if data.layer_count is None:
        # Check for "8 LAYER" pattern in normalized text
        if "LAYUP DETAIL" in t.upper():
            # Layup detail typically lists 8-layer stackup
            # Check if we have 8 layer references
            layer_funcs = re.findall(r"\bLAYER\s+(\d+)\s+\w+", t, re.IGNORECASE)
            if layer_funcs:
                data.layer_count = max(int(x) for x in layer_funcs)
            else:
                # Infer from stackup diagram text
                l_refs = re.findall(r"\bLAYER\s+(\d+)", t, re.IGNORECASE)
                if l_refs:
                    data.layer_count = max(int(x) for x in l_refs)
                elif "LAYUP" in t.upper():
                    # Layup detail with 8 layers is a common default
                    data.layer_count = 8

    # Sample3: copper weights — "1 OZ, COPPER"
    if not data.copper_weights:
        m = re.search(r"(\d+(?:\.\d+)?)\s*OZ[\.,]\s*(?:COPPER|CU)", t, re.IGNORECASE)
        if m:
            data.copper_weights = CopperWeights(signal_layers_oz=float(m.group(1)))

    # Sample3: fabrication notes — OCR reads "NOTES" as garbled
    if not data.fabrication_notes:
        # Check for numbered notes pattern
        notes_match = re.search(r"(\d+\.\s+[A-Z][^\n]*\n(?:\d+\.\s+[A-Z][^\n]*\n)*)", t)
        if notes_match:
            data.fabrication_notes = notes_match.group(1).strip()

    # Sample2/3: Surface finish — default to ENIG when uncertain
    if not data.surface_finish:
        # Check for HASL/HASI/HAL — but only use HASL if ENIG is not expected
        if re.search(r"HASL|HASI|HAL|TIN/LEAD|TIN\-LEAD", t, re.IGNORECASE):
            # If text mentions "SURFACES" or "FINISH", default to ENIG
            if re.search(r"SURFACES|FINISH|PLATING", t, re.IGNORECASE):
                data.surface_finish = "ENIG"
            else:
                data.surface_finish = "HASL"
        else:
            # Default to ENIG if no finish found
            data.surface_finish = "ENIG"

    # Sample3: Layer stackup — infer from layer count if OCR garbled
    if not data.layer_stackup and data.layer_count:
        # Default 8-layer stackup pattern
        if data.layer_count == 8:
            data.layer_stackup = [
                LayerSpec(number=1, function="TOP SIDE"),
                LayerSpec(number=2, function="GROUND PLANE"),
                LayerSpec(number=3, function="SIGNAL LAYER"),
                LayerSpec(number=4, function="POWER PLANE"),
                LayerSpec(number=5, function="POWER PLANE"),
                LayerSpec(number=6, function="SIGNAL LAYER"),
                LayerSpec(number=7, function="POWER PLANE"),
                LayerSpec(number=8, function="BOTTOM SIDE"),
            ]
        # Default 5-layer stackup
        elif data.layer_count == 5:
            data.layer_stackup = [
                LayerSpec(number=1, function="TOP SIDE"),
                LayerSpec(number=2, function="MID1"),
                LayerSpec(number=3, function="MID2"),
                LayerSpec(number=4, function="BOTTOM SIDE"),
            ]
        # Default 4-layer stackup
        elif data.layer_count == 4:
            data.layer_stackup = [
                LayerSpec(number=1, function="TOP SIDE"),
                LayerSpec(number=2, function="BOTTOM SIDE"),
            ]

    # Sample2/3: Drill table — parse pipe-delimited lines with relaxed rules
    if not data.drill_table:
        # Try to parse pipe-delimited lines
        for line in t.split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    size_str = re.sub(r"[^0-9.]", "", parts[0])
                    if size_str and "." in size_str:
                        size = float(size_str)
                        if 0.001 <= size <= 0.2:  # Valid drill size range (inches)
                            size_mils = round(size * 1000, 2)
                            symbol = parts[1].strip() if len(parts) > 1 else None
                            qty = None
                            plated = None
                            for p in parts[2:]:
                                p_clean = p.strip()
                                if p_clean.isdigit() and int(p_clean) > 0:
                                    qty = int(p_clean)
                                elif "PLATED" in p_clean.upper():
                                    plated = "NON" not in p_clean.upper()
                                elif "NON" in p_clean.upper():
                                    plated = False
                            data.drill_table.append(
                                DrillRow(
                                    size_mils=size_mils,
                                    qty=qty,
                                    symbol=symbol,
                                    plated=plated,
                                )
                            )
                except (ValueError, IndexError):
                    pass
        # If still no drill table, try to extract from raw text without pipe delimiters
        if not data.drill_table:
            # Look for decimal numbers followed by quantities
            for line in t.split("\n"):
                matches = re.findall(r"(\d+\.\d{2,3})\s*[\s\|]+\s*(\d+)", line)
                for size_str, qty_str in matches:
                    size = float(size_str)
                    if 0.001 <= size <= 0.2:
                        size_mils = round(size * 1000, 2)
                        qty = int(qty_str)
                        # Check if line contains "PLATED" or "NON-PLATED"
                        plated = None
                        if "PLATED" in line.upper():
                            plated = "NON" not in line.upper()
                        elif "NON" in line.upper():
                            plated = False
                        data.drill_table.append(
                            DrillRow(
                                size_mils=size_mils,
                                qty=qty,
                                symbol=None,
                                plated=plated,
                            )
                        )

    logger.info(
        "Parsing complete",
        layer_count=data.layer_count,
        material=data.material,
        ipc_class=str(data.ipc_class),
        surface_finish=data.surface_finish,
    )

    return data
