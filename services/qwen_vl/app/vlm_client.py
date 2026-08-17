"""vLLM client for Qwen3-VL model."""

from __future__ import annotations

import json
import structlog
import os

from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)

VLLM_URL = os.environ.get("QWEN_VL_VLLM_URL", "http://vllm-qwen-vl:8011/v1")
MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


class QwenVLClient:
    """Client for the Qwen3-VL vLLM endpoint."""

    def __init__(self, base_url: str = VLLM_URL, model: str = MODEL) -> None:
        """
        Args:
            base_url: vLLM OpenAI-compatible API URL.
            model: Model name.
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    async def structured_extract(self, image_base64: str) -> dict | None:
        """Single-shot structured extraction from a PCB drawing image.

        Args:
            image_base64: Base64-encoded PNG image.

        Returns:
            Parsed JSON dict or None.
        """
        prompt = (
            "You are an expert PCB fabrication drawing analyst. "
            "Analyze this engineering drawing image and extract ALL data "
            "into the following JSON structure.\n\n"
            "UNIT RULES — MANDATORY:\n"
            "- Board thickness: output in INCHES. "
            "Convert from mm if needed (divide by 25.4).\n"
            "- Drill sizes: output in MILS (thousandths of an inch). "
            "Convert from mm if needed (multiply by 39.3701).\n"
            "- Trace widths: output in MILS. Convert from mm if needed.\n"
            "- Copper weight: output in oz/ft² (no conversion needed).\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Read every note, table, callout, and title block entry\n"
            "- For drill charts: this is a row-by-row hole schedule table. Scan it "
            "top to bottom and emit ONE JSON entry per visible row, in order, with no "
            "skipping. Adjacent rows can have very close or nearly identical hole "
            "sizes (e.g. two rows of .124 and .125) — these are still SEPARATE rows; "
            "never merge, dedupe, or combine rows just because their values look "
            "similar. Before answering, count the visible rows in the table and make "
            "sure drill_table has exactly that many entries.\n"
            "- For layer stackups: capture layer number and function "
            "(signal, plane, ground)\n"
            "- For impedance tables: capture impedance values, "
            "trace widths in mils, layer references\n"
            "- Look for ITAR/export control markings — set is_itar accordingly\n"
            "- Board thickness is usually one self-contained note, most often "
            "'X +/- Y' (nominal and tolerance together, e.g. '.062 +/- .005') "
            "but sometimes a bare metric measurement with no tolerance in the "
            "same note (e.g. '1.80238 MM'). Take the nominal and tolerance "
            "ONLY from that one note; record the raw as-written text too. "
            "Never pull a tolerance from a different, unrelated note "
            "elsewhere on the drawing (e.g. an impedance or trace-width "
            "tolerance) just because it also mentions a percentage — if the "
            "board-thickness note itself has no tolerance, leave "
            "plus_tol/minus_tol null.\n"
            "- IPC class is usually stated as 'Class 1/2/3' near an IPC spec reference\n"
            "- Surface finish: look for ENIG, HASL, OSP, immersion tin/silver\n"
            "- Solder mask: color (green/red/blue/black), type (LPI/liquid), spec (IPC-SM-840)\n"
            "- Silkscreen/legend: color (white/yellow), ink type, sides\n\n"
            "Below, angle brackets like <float> or <string> mark where a real "
            "extracted value goes — never output the literal placeholder "
            "text, only actual data read from the image (or null / [] if "
            "genuinely absent).\n\n"
            "Return this exact JSON structure:\n"
            "{\n"
            '  "part_number": "<string or null>",\n'
            '  "manufacturer": "<string or null>",\n'
            '  "drawing_title": "<string or null>",\n'
            '  "is_itar": <true or false>,\n'
            '  "ipc_class": "<Class 1, Class 2, or Class 3, or null>",\n'
            '  "ipc_specs": ["<IPC spec references found>"],\n'
            '  "layer_count": <integer or null>,\n'
            '  "layer_stackup": [{"number": <integer>, "function": "<string>"}],\n'
            '  "material": "<e.g. FR4, 370HR, or null>",\n'
            '  "board_thickness": {"nominal": <float in inches>, '
            '"plus_tol": <float in inches>, "minus_tol": <float in inches>, '
            '"unit": "in", "raw": "<the raw text as written on the drawing>"},\n'
            '  "copper_weights": {"signal_layers_oz": <float>, '
            '"plane_layers_oz": <float>, "external_finished_oz": <float>},\n'
            '  "surface_finish": "<e.g. ENIG, HASL, or null>",\n'
            '  "solder_mask": {"present": <true or false>, "type": "<string>", '
            '"color": "<string>", "spec": "<string>", "sides": "<string>"},\n'
            '  "silkscreen": {"present": <true or false>, "color": "<string>", '
            '"ink": "<string>", "sides": "<string>"},\n'
            '  "impedance_control": {"controlled": <true or false>, '
            '"single_ended": {"min": <float>, "max": <float>, "unit": "ohm"}, '
            '"trace_width_mils": <float>, "layers": []},\n'
            '  "drill_table": [{"size_mils": <float>, "qty": <integer>, '
            '"symbol": "<string>", "plated": <true or false>}],\n'
            '  "fabrication_notes": "<full text of all numbered fabrication notes, or null>"\n'
            "}\n\n"
            "Return ONLY valid JSON with every <...> placeholder replaced by "
            "real data or null/[]. No markdown fences. No explanation."
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.0,
        )

        content = response.choices[0].message.content or ""
        return self._parse_json(content)

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from model response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block (strip markdown fences)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from Qwen-VL response")
        return None
