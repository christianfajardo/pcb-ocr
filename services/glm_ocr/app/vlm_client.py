"""vLLM client for GLM-OCR model."""

from __future__ import annotations

import json
import structlog
import os

from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)

VLLM_URL = os.environ.get("GLM_OCR_VLLM_URL", "http://vllm-glm-ocr:8010/v1")
MODEL = os.environ.get("GLM_OCR_MODEL", "zai-org/GLM-OCR")


class GLMOCRClient:
    """Client for the GLM-OCR vLLM endpoint."""

    def __init__(self, base_url: str = VLLM_URL, model: str = MODEL) -> None:
        """
        Args:
            base_url: vLLM OpenAI-compatible API URL.
            model: Model name.
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    async def extract_text(self, image_base64: str) -> str:
        """Full text extraction from a PCB drawing image.

        Args:
            image_base64: Base64-encoded PNG image.

        Returns:
            Raw extracted text.
        """
        prompt = (
            "OCR the following engineering drawing image. "
            "Extract ALL text visible in the image, preserving layout and "
            "structure as much as possible. Include:\n"
            "- All numbered notes and fabrication instructions\n"
            "- All table data (drill charts, impedance tables, hole schedules)\n"
            "- Title block information (part number, company, revision, date)\n"
            "- Layer stackup details\n"
            "- Any callouts, labels, or annotations\n"
            "- Export control or ITAR markings if present\n\n"
            "Output the complete extracted text."
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

        return response.choices[0].message.content or ""

    async def structured_extract(self, ocr_text: str) -> dict | None:
        """Structured JSON extraction from OCR text.

        Args:
            ocr_text: Raw text from full OCR extraction.

        Returns:
            Parsed JSON dict or None.
        """
        prompt = f"""You are a PCB fabrication drawing expert. Given this text extracted from a PCB fab drawing, extract the following fields as JSON.

UNIT RULES: All dimensions must be in inches or mils (thousandths of an inch). Convert from mm if needed (1 mm = 39.3701 mils, 1 mm = 0.03937 inches).

Below, angle brackets like <integer> or <float> mark where a real extracted value goes — never output the literal text inside the brackets, only actual data (or null / [] if genuinely absent from the text).

A "HOLE SCHEDULE" or "DRILL CHART" table (columns like HOLE/DIA/SYM/QTY) maps to drill_table — include EVERY row, even if there are many; do not skip, merge, or truncate rows.

{{
  "part_number": "<string or null>",
  "manufacturer": "<string or null>",
  "drawing_title": "<string or null>",
  "is_itar": false,
  "ipc_class": "<Class 1, Class 2, or Class 3, or null>",
  "ipc_specs": ["<IPC spec references found>"],
  "layer_count": <integer or null>,
  "material": "<e.g. FR4, 370HR, or null>",
  "board_thickness": {{"nominal": <float in inches>, "plus_tol": <float in inches>, "minus_tol": <float in inches>, "unit": "in"}},
  "surface_finish": "<e.g. ENIG, HASL, or null>",
  "solder_mask": {{"present": <true or false>, "color": "<string>", "type": "<string>", "spec": "<string>", "sides": "<string>"}},
  "silkscreen": {{"present": <true or false>, "color": "<string>", "ink": "<string>", "sides": "<string>"}},
  "copper_weights": {{"signal_layers_oz": <float>, "plane_layers_oz": <float>, "external_finished_oz": <float>}},
  "impedance_control": {{"controlled": <true or false>, "single_ended": null, "trace_width_mils": <float>, "layers": []}},
  "layer_stackup": [{{"number": <integer>, "function": "<string>"}}],
  "drill_table": [{{"size_mils": <float>, "qty": <integer>, "symbol": "<string>", "plated": <true or false>}}],
  "fabrication_notes": "<full text of all numbered fabrication notes, or null>"
}}

Extracted text:
---
{ocr_text}
---

Return ONLY valid JSON with every <...> placeholder replaced by real data or null/[]."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
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

        # Try to find JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from GLM-OCR response")
        return None
