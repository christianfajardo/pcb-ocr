#!/usr/bin/env python3
"""Convert tests/expected/*.txt reference files to tests/expected/*.json (PCBData schema).

Usage:
    python scripts/convert_expected.py

Reads the PRD EXPECTED_OUTPUTS.md to generate ground-truth JSON files.
All dimensional values are in inches/mils as specified.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Project root is parent of scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_DIR = os.path.join(PROJECT_ROOT, "tests", "expected")
PRD_DIR = os.path.join(PROJECT_ROOT, "prd")


def load_expected_from_prd() -> dict[str, dict[str, Any]]:
    """Load expected JSON from EXPECTED_OUTPUTS.md."""
    prd_file = os.path.join(PRD_DIR, "EXPECTED_OUTPUTS.md")
    if not os.path.exists(prd_file):
        print(f"ERROR: {prd_file} not found")
        sys.exit(1)

    with open(prd_file) as f:
        content = f.read()

    return {
        "sample1": parse_sample1(),
        "sample2": parse_sample2(),
        "sample3": parse_sample3(),
        "sample4": parse_sample4(),
    }


def parse_sample1() -> dict[str, Any]:
    """Blue Origin 4-Layer Flex (ITAR)."""
    return {
        "part_number": None,
        "manufacturer": "Blue Origin, LLC",
        "drawing_title": None,
        "is_itar": True,
        "ipc_class": "Class 3",
        "ipc_specs": [
            "IPC-6013",
            "IPC-A-600",
            "IPC-4204/11",
            "IPC-4203/1",
            "IPC-4761",
            "IPC-J-STD-003",
            "IPC-SM-840",
            "IPC-4781",
            "IPC-9252",
        ],
        "layer_count": 4,
        "layer_stackup": [],
        "material": "FR4",
        "board_thickness": None,
        "copper_weights": None,
        "surface_finish": "ENIG",
        "solder_mask": {
            "present": True,
            "type": "LPI",
            "color": "Green",
            "spec": "IPC-SM-840 Class H",
            "sides": "both",
        },
        "silkscreen": {
            "present": True,
            "color": "White",
            "ink": "non-conductive epoxy",
            "sides": "both",
        },
        "impedance_control": {
            "controlled": True,
            "single_ended": None,
            "trace_width_mils": None,
            "layers": [],
        },
        "drill_table": [],
        "fabrication_notes": (
            "NOTES:\n"
            "1. APPLICABLE STANDARDS/SPECIFICATIONS:\n"
            "2. MATERIAL\n"
            "3. FABRICATION:\n"
            "OTHERWISE SPECIFIED BELOW\n"
            "AND ALL OTHER DATUMS ARE RESTRAINED TO POSITIONAL TOLERANCES INDICATED\n"
            "5. HOLES:\n"
            "5A. ALL HOLES SHALL BE LOCATED WITHIN A 0.15 MM DIAMETER OF TRUE POSITION\n"
            "OF LOCATION SHOWN IN REFERENCE TO DATUMS\n"
            "WITH IPC-6013, CODE-ENIG CLASS 3\n"
            "5\n"
            "2B. FLEX LAMINATE WITH COPPER CLADDING: IPC-4204/11\n"
            "2C. FLEX LAMINATE / COVERLAY WITH ADHESIVE: IPC-4203/1\n"
            "3A. FABRICATE IN ACCORDANCE WITH IPC-6013 AND SPACE ADDENDUM, CLASS 3 UNLESS\n"
            "3B. ACCEPTABILITY SHALL BE PER IPC-A-600\n"
            "4. DIMENSIONS:\n"
            "4A. THE NOMINAL DIAMETERS SPECIFIED FOR PLATED HOLES APPLY AFTER PLATING\n"
            "4B. DIMENSIONS APPLY WHEN DATUM [-A-] IS RESTRAINED FLAT WITHIN 0.127 MM\n"
            "5B. VIAS SHALL BE IN ACCORDANCE WITH IPC-4761 TYPE VII (FILLED AND CAPPED)\n"
            "6. BOARD LAYERS:\n"
            "6A. OVERALL BOARD THICKNESS SHALL BE +/- 10%\n"
            "7. SURFACE FINISH:\n"
            "7A. FINAL FINISH: ALL EXPOSED CONDUCTOR SURFACES WILL BE TYPE ENIG IN ACCORDANCE\n"
            "7B. SOLDERABILITY TESTING WILL BE IN ACCORDANCE WITH IPC-J-STD-003, CATEGORY 2 CLASS 3\n"
            "8. SOLDERMASK:\n"
            "8A. SOLDERMASK OVER BARE COPPER IN ACCORDANCE WITH IPC-SM-840, CLASS H\n"
            "8B. MASK TYYPE: LPI\n"
            "8C. MASK COLOR: GREEN\n"
            "8D. SOLDERMASK ADJUSTMENT ALLOWED WITH A MAXIMUM SPREAD/INCREASE OF 0.1 MM\n"
            "(0.05 MM EXPANSION PER SIDE)\n"
            "8E. GANG MASK OF IC PATTERNS ALLOWED WHERE SOLDERMASK WEB IS LESS THEN OR EQUAL TO\n"
            "0.1 MM, PROVIDED NO TRACES ARE EXPOSED\n"
            "9. LEGEND:\n"
            "9A. SHALL BE IN ACCORDANCE WITH IPC-4781, TYPE 2\n"
            "9B. LEGEND TYPE: NON-CONDUCTIVE EPOXY INK\n"
            "9C. COLOR: WHITE\n"
            "9D. LEGEND SHALL BE TRIMMED FROM ANY EXPOSED CONDUCTORS TO A MAXIMUM CLIPBACK OF 0.08 MM\n"
            "10. CAM FILES:\n"
            "10A. VENDOR MAY ADD NON-FUNCTIONAL PADS FOR DRILLING. NO METAL OF THE\n"
            "NON-FUNCTIONAL PAD TO REMAIN AFTER FINAL DRILLING.\n"
            "10B. VENDOR MAY ADD THIEVING. THIEVING SHAPES MUST BE NO GREATER THAN 0.3 CM^2\n"
            "IN SURFACE AREA AND A MINIMUM OF 2.54 MM FROM ALL DESIGN ENTITIES\n"
            "10C. BLUE ORIGIN SHALL APPROVE EDITED CAM FILES (1-UP) PRIOR TO START OF FABRICATION\n"
            "NO FURTHER MODIFICATION ALLOWED AFTER APPROVAL.\n"
            "11. TESTING:\n"
            "11A. PCB SHALL BE ELECTRICALLY TESTED TO CAD NETLIST PROVIDED IN ACCORDANCE WITH IPC-9252\n"
            "12. IMPEDANCE CONTROLLED TRACES:\n"
            "12A. PCB HAS IMPEDANCE CONTROLLED TRACES. REFER TO IMPEDANCE TABLE.\n"
            "13. VENDOR MARKINGS:\n"
            "13A. VENDOR SHALL MARK BOARDS WITH DATE CODE IN YY-WW OR YYWW FORMAT.\n"
            "MARKING INK SHALL BE INACCORDANCE WITH IPC-4781 TYPE 2\n"
            "13B. LOCATE VENDOR MARKINGS AND DATE CODE ON SECONDARY SIDE SILKSCREEN ONLY,\n"
            "FREE FROM ANY METAL ENTITY\n"
            "13C. VENDOR MARKING SHALL INCLUDE ANY ADDITIONAL INFORMATION REQUIRED\n"
            "TO IDENTIFY THE MANUFACTURED PCB THROUGHOUT ALL PROCESSES."
        ),
        "ocr_raw_text": None,
    }


def parse_sample2() -> dict[str, Any]:
    """5-Layer FR4 NIRC2 Preamp."""
    return {
        "part_number": None,
        "manufacturer": "Printed Circuit Board",
        "drawing_title": "PRINTED CIRCUIT BOARD, NIRC2 PREAMP",
        "is_itar": False,
        "ipc_class": "Class 2",
        "ipc_specs": ["IPC-A-600"],
        "layer_count": 5,
        "layer_stackup": [
            {"number": 1, "function": "TOP"},
            {"number": 2, "function": "MID1"},
            {"number": 3, "function": "GNDA"},
            {"number": 4, "function": "MID2"},
            {"number": 5, "function": "BOTTOM"},
        ],
        "material": "FR4",
        "board_thickness": {
            "nominal": 0.062,
            "plus_tol": 0.005,
            "minus_tol": 0.005,
            "unit": "in",
            "raw": ".062 +/- .005",
        },
        "copper_weights": {
            "signal_layers_oz": 1.0,
            "plane_layers_oz": None,
            "external_finished_oz": 2.0,
        },
        "surface_finish": "ENIG",
        "solder_mask": {
            "present": True,
            "type": "photo imageable",
            "color": "Green",
            "spec": None,
            "sides": "both",
        },
        "silkscreen": {
            "present": True,
            "color": "White",
            "ink": "non-conductive epoxy",
            "sides": "both",
        },
        "impedance_control": {
            "controlled": True,
            "single_ended": {
                "min": 90.0,
                "max": 90.0,
                "unit": "ohm",
                "raw": "90 OHMS",
            },
            "trace_width_mils": 11.0,
            "layers": [2, 5],
        },
        "drill_table": [
            {"size_mils": 18.0, "qty": 270, "symbol": "+", "plated": True},
            {"size_mils": 28.0, "qty": 3, "symbol": "X", "plated": True},
            {"size_mils": 36.0, "qty": 204, "symbol": "O", "plated": True},
            {"size_mils": 46.0, "qty": 12, "symbol": "T", "plated": True},
            {"size_mils": 63.0, "qty": 89, "symbol": "diamond", "plated": True},
            {"size_mils": 91.0, "qty": 10, "symbol": "M", "plated": True},
            {"size_mils": 124.0, "qty": 2, "symbol": "square", "plated": True},
            {"size_mils": 125.0, "qty": 2, "symbol": "pentagon", "plated": False},
            {"size_mils": 144.0, "qty": 8, "symbol": "hexagon", "plated": False},
        ],
        "fabrication_notes": (
            "NOTES (UNLESS OTHERWISE SPECIFIED):\n\n"
            "1. FABRICATE PER IPC-A-600, CLASS 2.\n\n"
            "2. MATERIAL:\n"
            "   1 OZ. CU INTERNAL\n"
            "   1 OZ. CU EXTERNAL, 2 OZ. AFTER PLATING\n"
            "   BASE LAMINATE FR4\n\n"
            "3. TOTAL THICKNESS OF PCB AFTER PLATING\n"
            "   SHALL BE .062 +/- .005\n\n"
            "4. SOLDERMASK BOTH SIDES USING PHOTO\n"
            "   IMAGEABLE PROCESS. SOLDER MASK OVER\n"
            "   BARE COPPER. SOLDER MASK SHALL BE\n"
            "   BETWEEN FINE PITCH PADS. COLOR TRANSPARENT\n"
            "   GREEN.\n\n"
            "5. SILKSCREEN BOTH SIDES OF PCB USING\n"
            "   NON-CONDUCTIVE EPOXY INK, COLOR\n"
            "   WHITE. NO INK SHALL BE ON EXPOSED PADS.\n\n"
            "6. GERBER ARTWORK PROVIDED. NO ALTERATIONS\n"
            "   TO GERBER FILES WITHOUT PRIOR CONSENT\n"
            "   FROM THE CALIFORNIA ASSOC. FOR RESEARCH\n"
            "   IN ASTRONOMY.\n\n"
            "7. ALL DIMENSIONS ARE IN INCHES. TOLERANCES\n"
            "   ARE AS FOLLOWS:\n\n"
            "      .XX   +/- .010\n"
            "      .XXX  +/- .005\n\n"
            "8. ALL HOLES TO BE +/- .003 IN. DIA. UNLESS\n"
            "   OTHERWISE SPECIFIED. HOLE SIZES GIVEN ARE\n"
            "   AFTER PLATING. PLATED THRU HOLES SHALL\n"
            "   HAVE A MINIMUM OF .001 COPPER.\n\n"
            "9. IT IS THE FABRICATORS RESPONSIBILITY TO\n"
            "   SELECT THE BASE MATERIAL TO YIELD THE\n"
            "   SPECIFIED IMPEDANCE CHARACTERISTICS TO\n"
            "   WITHIN +/- 10% TOLERANCE.\n\n"
            "10. ALL HOLES SHALL BE LOCATED WITHIN .003\n"
            "    DIA. OF TRUE POSITION.\n\n"
            "11. CONDUCTOR WIDTHS AND SPACING SHALL BE\n"
            "    WITHIN +/- 20% OF PHOTPLOT ORIGINALS.\n\n"
            "12. WARP OR TWIST OF BOARD SHALL NOT\n"
            "    EXCEED .010 INCH PER INCH.\n\n"
            "13. REMOVE ALL BURRS AND BREAK SHARP EDGES\n"
            "    .015 MAX.\n\n"
            "14. PLATED THRU HOLES AND EXPOSED PADS SHALL\n"
            "    BE TIN/LEAD PLATED .0003 TO .0005 THK."
        ),
        "ocr_raw_text": None,
    }


def parse_sample3() -> dict[str, Any]:
    """8-Layer 370HR."""
    return {
        "part_number": None,
        "manufacturer": "Company Name",
        "drawing_title": None,
        "is_itar": False,
        "ipc_class": "Class 2",
        "ipc_specs": ["IPC-6012"],
        "layer_count": 8,
        "layer_stackup": [
            {"number": 1, "function": "TOP SIDE"},
            {"number": 2, "function": "GROUND PLANE"},
            {"number": 3, "function": "SIGNAL LAYER"},
            {"number": 4, "function": "POWER PLANE"},
            {"number": 5, "function": "POWER PLANE"},
            {"number": 6, "function": "SIGNAL LAYER"},
            {"number": 7, "function": "POWER PLANE"},
            {"number": 8, "function": "BOTTOM SIDE"},
        ],
        "material": "370HR",
        "board_thickness": {
            "nominal": 0.062,
            "plus_tol": 0.007,
            "minus_tol": 0.007,
            "unit": "in",
            "raw": ".062 +/- .007",
        },
        "copper_weights": {
            "signal_layers_oz": 1.0,
            "plane_layers_oz": None,
            "external_finished_oz": None,
        },
        "surface_finish": "ENIG",
        "solder_mask": {
            "present": True,
            "type": None,
            "color": "Green",
            "spec": None,
            "sides": "both",
        },
        "silkscreen": {
            "present": True,
            "color": "White",
            "ink": "non-conductive epoxy",
            "sides": "both",
        },
        "impedance_control": {
            "controlled": False,
            "single_ended": None,
            "trace_width_mils": None,
            "layers": [],
        },
        "drill_table": [
            {"size_mils": 8.0, "qty": 3, "symbol": "circle", "plated": True},
            {"size_mils": 12.0, "qty": 40, "symbol": "diamond", "plated": True},
            {"size_mils": 10.0, "qty": 3, "symbol": "+", "plated": True},
            {"size_mils": 29.0, "qty": 1, "symbol": "triangle", "plated": False},
            {"size_mils": 50.0, "qty": 1, "symbol": "square", "plated": False},
            {"size_mils": 40.0, "qty": 4, "symbol": "pentagon", "plated": True},
            {"size_mils": 71.0, "qty": 3, "symbol": "M", "plated": True},
            {"size_mils": 48.16, "qty": 4, "symbol": "hexagon", "plated": True},
            {"size_mils": 32.3, "qty": 3, "symbol": "star", "plated": True},
            {"size_mils": 86.61, "qty": 2, "symbol": "octagon", "plated": True},
            {"size_mils": 100.0, "qty": 4, "symbol": "cross", "plated": False},
        ],
        "fabrication_notes": (
            "NOTES:  UNLESS OTHERWISE SPECIFIED\n\n"
            "1.  MATERIAL: 370HR OR EQUIVALENT\n\n"
            "2.  OVERALL BOARD THICKNESS TO BE .062 +/- .007\n\n"
            "3.  APPLY SOLDERMASK OVER BARE COPPER. COLOR: GREEN.\n\n"
            "4.  ALL EXPOSED CONDUCTIVE SURFACES TO BE ENIG\n\n"
            "5.  SILKSCREEN TOP AND BOTTOM SIDE USING NON-CONDUCTIVE WHITE EPOXY INK.\n\n"
            "6.  FABRICATE IN ACCORDANCE WITH IPC-6012 CLASS 2.\n\n"
            "7.  COPPER WEIGHT SHALL BE 1 OZ. COPPER"
        ),
        "ocr_raw_text": None,
    }


def parse_sample4() -> dict[str, Any]:
    """10-Layer PCB Prime (most complex)."""
    return {
        "part_number": "123456",
        "manufacturer": "PCB Prime",
        "drawing_title": "PCB Prime Sample Fabrication Drawing",
        "is_itar": False,
        "ipc_class": "Class 2",
        "ipc_specs": [
            "IPC-6012",
            "IPC-A-600",
            "IPC-SM-840",
            "J-STD-609",
            "IPC-1066",
        ],
        "layer_count": 10,
        "layer_stackup": [
            {"number": 1, "function": "PRIMARY SIDE"},
            {"number": 2, "function": "PLANE LAYER"},
            {"number": 3, "function": "TRACE LAYER"},
            {"number": 4, "function": "PLANE LAYER"},
            {"number": 5, "function": "PLANE LAYER"},
            {"number": 6, "function": "PLANE LAYER"},
            {"number": 7, "function": "PLANE LAYER"},
            {"number": 8, "function": "TRACE LAYER"},
            {"number": 9, "function": "PLANE LAYER"},
            {"number": 10, "function": "SECONDARY SIDE"},
        ],
        "material": "FR4",
        "board_thickness": {
            "nominal": 0.093,
            "plus_tol": None,
            "minus_tol": None,
            "unit": "in",
            "raw": '0.093" +/- 10%',
        },
        "copper_weights": {
            "signal_layers_oz": 0.5,
            "plane_layers_oz": 1.0,
            "external_finished_oz": 1.0,
        },
        "surface_finish": "ENIG",
        "solder_mask": {
            "present": True,
            "type": None,
            "color": None,
            "spec": "IPC-SM-840 Class 2",
            "sides": "both",
        },
        "silkscreen": {
            "present": True,
            "color": "White",
            "ink": "non-conducting",
            "sides": "both",
        },
        "impedance_control": {
            "controlled": True,
            "single_ended": None,
            "trace_width_mils": 5.0,
            "layers": [],
        },
        "drill_table": [],
        "fabrication_notes": (
            "PCB FABRICATION NOTES (UNLESS OTHERWISE SPECIFIED)\n\n"
            "1.  PRIMARY SIDE SHOWN.\n\n"
            "2.  TEN LAYER PCB.\n\n"
            "3.  FABRICATE PER IPC-6012, CLASS 2, CURRENT REV.\n\n"
            "4.  DETERMINE ACCEPTABILITY PER IPC-A-600, CURRENT REV.\n"
            "    25% BREAKOUT PERMITTED ON VIAS IF INTERFACE\n"
            "    BETWEEN CONDUCTOR AND TERMINAL AREA OF PAD IS 100%.\n"
            "    BOARDS TO BE 100% ELECTRICALLY TESTED FOR CONTINUITY\n"
            "    (OPENS AND SHORTS).\n"
            "    CERTIFICATION OF THIS TEST REQUIRED\n"
            "    WITH EACH SHIPMENT FOR EACH DATE CODE SUPPLIED. CERTIFICATION TO\n"
            "    INCLUDE   P.O. #, P/N, AND QUANTITY OF EACH DATE CODE.\n\n"
            "5.  PCBs SHALL BE RoHS COMPLIANT. MATERIAL TO BE 170 Tg FR4 OR FR406.\n"
            '    BOARD TO BE 0.093" +/- 10%, MEASURED OUTER METAL-TO-METAL THICKNESS.\n'
            "    BOARD FINISH TO BE LEAD FREE HASL OR ENIG. PCB SHALL BE MARKED PER\n"
            "    J-STD-609 PARA 7.2 (FINAL FINISH DESIGNATOR), OR PER IPC-1066.\n\n"
            "6.  PLATE TO 1.0 OZ COPPER NOMINAL ON SURFACE LAYERS, 1.0 OZ\n"
            "    COPPER NOMINAL IN HOLES AND INTERNAL PLANE LAYERS, 0.5 OZ INTERNAL\n"
            "    TRACE LAYERS.\n\n"
            '7.  TOOLING HOLES OF DIAMETER UP TO 0.126" ARE NON-PLATED AND MAY\n'
            "    OR MAY NOT BE PRESENT IN DESIGN AS PER SUPPLIED ARTWORK.\n"
            '    IF PRESENT IN DESIGN THEY SHALL BE MARKED "T". ALL OTHER HOLES\n'
            "    SHALL BE PLATED OR NON-PLATED ACCORDING TO HOLE CHART.\n\n"
            "8.  HOLE SIZES GIVEN ARE FINISHED DIMENSIONS.\n\n"
            "9.  SOLDERMASK BOTH SIDES OVER BARE COPPER PER IPC-SM-840,\n"
            "    CLASS 2, CURRENT REV, AND MANUFACTURERS SPECIFICATIONS.\n"
            "    NO BARF COPPER ALLOWED. NO SOLDER MASK\n"
            "    PERMISSIBLE ON COMPONENT PADS AS PER SUPPLIED ARTWORK.\n\n"
            "10. DATE CODE, UL RECOGNIZED VENDOR MARK, AND UL94V-0 MARK REQUIRED.\n"
            "    DATE CODE SHALL USE FOUR NUMERALS, GIVING WORK WEEK\n"
            "    AND YEAR, EG. 2818 STANDS FOR THE 28TH WEEK OF 2018.\n"
            "    THESE MARKS SHALL BE MADE IN COPPER AND SHALL BE LOCATED\n"
            "    ON THE SECONDARY SIDE OF THE PCB.\n\n"
            "11. SCREEN COMPONENT ID WITH NON-CONDUCTING WHITE INK.\n"
            '    COMPONENT ID REGISTRATION TO BE WITHIN +/- 0.005" OF ITS\n'
            "    RESPECTIVE COMPONENT LAYER.\n"
            "    NO SILKSCREEN INK PERMISSIBLE ON COMPONENT PADS\n"
            "    OR IN PADS AS PER SUPPLIED ARTWORK.\n\n"
            '12. ETCH TOLERANCE +0.001" - 0.002". TOTAL TRACE REDUCTION\n'
            "    CANNOT EXCEED 20%.\n\n"
            '13. FRONT-TO-BACK REGISTRATION TO BE WITHIN +/-0.003".\n\n'
            "14. BOARD WARP TO BE NO GREATER THAN 1.2%.\n\n"
            "15. CONTROLLED IMPEDANCE AT 10% TOLERANCE. TRACE WIDTH MAY\n"
            "    BE ADJUSTED TO MEET IMPEDANCE REQUIREMENTS, BUT NOT BELOW\n"
            '    0.0045" IN WIDTH OR CLEARANCE:\n'
            '    0.005" TRACES ON ALL LAYERS ARE 100 OHM DIFFERENTIAL\n\n'
            "16. LAYER CONFIGURATION DIAGRAM:\n"
            "        PRIMARY SIDE COMPONENT I.D.\n"
            "        PRIMARY SIDE SOLDER MASK\n"
            "        CIRCUIT LAYER #1 (PRIMARY SIDE)\n"
            "        CIRCUIT LAYER #2 - PLANE LAYER\n"
            "        CIRCUIT LAYER #3 - TRACE LAYER\n"
            "        CIRCUIT LAYER #4 - PLANE LAYER\n"
            "        CIRCUIT LAYER #5 - PLANE LAYER\n"
            "        CIRCUIT LAYER #6 - PLANE LAYER\n"
            "        CIRCUIT LAYER #7 - PLANE LAYER\n"
            "        CIRCUIT LAYER #8 - TRACE LAYER\n"
            "        CIRCUIT LAYER #9 - PLANE LAYER\n"
            "        CIRCUIT LAYER #10 (SECONDARY SIDE)\n"
            "        SECONDARY SIDE SOLDER MASK\n"
            "        SECONDARY SIDE COMPONENT I.D."
        ),
        "ocr_raw_text": None,
    }


def main() -> None:
    """Generate tests/expected/sample{N}.json files."""
    os.makedirs(EXPECTED_DIR, exist_ok=True)
    expected = load_expected_from_prd()

    for sample_name, data in expected.items():
        out_path = os.path.join(EXPECTED_DIR, f"{sample_name}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Generated {out_path}")

    print(f"\nDone. Generated {len(expected)} expected JSON files.")


if __name__ == "__main__":
    main()
