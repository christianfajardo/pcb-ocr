"""vLLM client for GLM-OCR model."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

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
