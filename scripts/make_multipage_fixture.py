#!/usr/bin/env python3
"""Build multi-page test fixtures by repeating a single-page sample.

All four real samples are single-page, so the pipeline's per-page merge logic
had never run with more than one page. Duplicating one page gives an
unambiguous expectation: the merged result must equal the single-page result,
because the second sheet carries nothing new. Any doubling of the drill table
or layer stackup is a dedup bug.

Usage:
    python scripts/make_multipage_fixture.py            # writes the defaults
    python scripts/make_multipage_fixture.py src.pdf 3 out.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).parent.parent

# (source, page repeats, output) — 2 pages exercises dedup; 7 exceeds the
# default MAX_PAGES=5 cap so truncation can be tested.
DEFAULTS = [
    ("samples/sample3..pdf", 2, "samples/sample3_2page.pdf"),
    ("samples/sample3..pdf", 7, "samples/sample3_7page.pdf"),
]


def build(src: str, repeats: int, dest: str) -> None:
    src_path, dest_path = PROJECT_ROOT / src, PROJECT_ROOT / dest
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    out = pymupdf.open()
    try:
        with pymupdf.open(src_path) as source:
            for _ in range(repeats):
                out.insert_pdf(source, from_page=0, to_page=0)
        out.save(dest_path)
    finally:
        out.close()

    with pymupdf.open(dest_path) as check:
        print(f"{dest}: {check.page_count} pages ({dest_path.stat().st_size / 1e3:.0f} KB)")


def main() -> int:
    if len(sys.argv) == 4:
        build(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    else:
        for src, repeats, dest in DEFAULTS:
            build(src, repeats, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
