#!/usr/bin/env python3
"""Run OCR on all samples and cache results to avoid re-running Tesseract."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tesseract.app.ocr_engine import TesseractOCR

SAMPLES = [
    "samples/sample1.pdf",
    "samples/sample2.pdf",
    "samples/sample3..pdf",
    "samples/sample4.pdf",
]
CACHE_DIR = Path(__file__).parent.parent / "cache"


def main():
    os.chdir(Path(__file__).parent.parent)
    CACHE_DIR.mkdir(exist_ok=True)
    ocr = TesseractOCR(base_dpi=300)

    for pdf_rel in SAMPLES:
        pdf_path = Path(pdf_rel)
        cache_file = CACHE_DIR / (pdf_path.stem + ".json")

        if cache_file.exists():
            print(f"  SKIP (cached): {pdf_rel}")
            continue

        print(f"  OCR: {pdf_rel} ...")
        start = time.monotonic()
        result = ocr.extract(str(pdf_path))
        elapsed = time.monotonic() - start
        print(f"    {len(result['raw_text'])} chars in {elapsed:.1f}s")

        cache_data = {
            "pdf": pdf_rel,
            "raw_text": result["raw_text"],
            "page_count": result["page_count"],
            "ocr_time": elapsed,
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))
        print(f"    Cached to {cache_file}")


if __name__ == "__main__":
    main()
