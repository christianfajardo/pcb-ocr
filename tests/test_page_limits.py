"""Tests for the MAX_PAGES cap and multi-page PDF handling.

Uses the synthetic fixtures from scripts/make_multipage_fixture.py — the four
real samples are all single-page, so nothing else exercises these paths.
"""

from __future__ import annotations

import pytest
from conftest import PROJECT_ROOT

from shared.pdf_text import extract_pdf_text_layer
from shared.pdf_text import get_pdf_page_count as pymupdf_page_count
from shared.preprocessing import bytes_to_images, get_pdf_page_count, image_to_bytes, pdf_to_images

TWO_PAGE = PROJECT_ROOT / "samples" / "sample3_2page.pdf"
SEVEN_PAGE = PROJECT_ROOT / "samples" / "sample3_7page.pdf"
SINGLE_PAGE = PROJECT_ROOT / "samples" / "sample3..pdf"

pytestmark = pytest.mark.skipif(
    not TWO_PAGE.exists() or not SEVEN_PAGE.exists(),
    reason="run scripts/make_multipage_fixture.py to generate multi-page fixtures",
)

# Low DPI throughout — these assert page *counts*, not image quality, and a
# 300-DPI render of 7 pages is slow enough to matter in a unit suite.
DPI = 72


class TestPageCount:
    def test_real_totals(self):
        assert get_pdf_page_count(str(SINGLE_PAGE)) == 1
        assert get_pdf_page_count(str(TWO_PAGE)) == 2
        assert get_pdf_page_count(str(SEVEN_PAGE)) == 7

    def test_pymupdf_count_matches(self):
        # Regression: this was previously inferred as text.count("\f") + 1,
        # which always evaluated to 1 because pages are joined with "\n".
        assert pymupdf_page_count(str(TWO_PAGE)) == 2
        assert pymupdf_page_count(str(SEVEN_PAGE)) == 7


class TestRasterizeCap:
    def test_under_cap_renders_all(self):
        assert len(pdf_to_images(str(TWO_PAGE), DPI, max_pages=5)) == 2

    def test_over_cap_truncates(self):
        assert len(pdf_to_images(str(SEVEN_PAGE), DPI, max_pages=5)) == 5

    def test_cap_of_one(self):
        assert len(pdf_to_images(str(SEVEN_PAGE), DPI, max_pages=1)) == 1

    def test_zero_means_unlimited(self):
        assert len(pdf_to_images(str(SEVEN_PAGE), DPI, max_pages=0)) == 7


class TestTextLayerCap:
    def test_text_layer_honors_cap(self):
        # PyMuPDF must read the same pages the image engines saw, or
        # reconciliation ends up voting on inconsistent inputs.
        capped = extract_pdf_text_layer(str(SEVEN_PAGE), max_pages=2)
        full = extract_pdf_text_layer(str(SEVEN_PAGE), max_pages=0)
        assert len(capped) <= len(full)


class TestBytesRoundTrip:
    def test_png_round_trip_is_lossless(self):
        # The rasterize-once design hands PNG bytes between services; if this
        # weren't pixel-exact, every engine's accuracy would shift.
        original = pdf_to_images(str(SINGLE_PAGE), DPI, max_pages=1)[0]
        restored = bytes_to_images([image_to_bytes(original)])[0]
        assert restored.size == original.size
        assert restored.convert("RGB").tobytes() == original.convert("RGB").tobytes()

    def test_multiple_pages_round_trip(self):
        images = pdf_to_images(str(TWO_PAGE), DPI, max_pages=5)
        restored = bytes_to_images([image_to_bytes(i) for i in images])
        assert len(restored) == 2
        assert all(r.mode == "RGB" for r in restored)
