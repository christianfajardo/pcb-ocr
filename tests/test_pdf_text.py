"""Unit tests for PyMuPDF text-layer extraction and its substantiality heuristic."""

from __future__ import annotations

from shared.pdf_text import extract_pdf_text_layer, has_substantial_text_layer


class TestHasSubstantialTextLayer:
    def test_empty_text_is_not_substantial(self):
        assert has_substantial_text_layer("") is False

    def test_whitespace_only_is_not_substantial(self):
        assert has_substantial_text_layer("   \n\n\t  \f  ") is False

    def test_few_stray_characters_is_not_substantial(self):
        # e.g. a lone watermark or page-number artifact
        assert has_substantial_text_layer("1", min_chars=200) is False

    def test_long_real_text_is_substantial(self):
        text = "FABRICATION NOTES " * 20  # well over 200 alnum chars
        assert has_substantial_text_layer(text, min_chars=200) is True

    def test_threshold_is_configurable(self):
        text = "A" * 50
        assert has_substantial_text_layer(text, min_chars=200) is False
        assert has_substantial_text_layer(text, min_chars=10) is True

    def test_counts_only_alphanumeric_characters(self):
        # Lots of punctuation/whitespace padding but few real characters
        text = "-" * 500 + "AB" + " " * 500
        assert has_substantial_text_layer(text, min_chars=200) is False


class TestExtractPdfTextLayer:
    def test_extracts_real_text_layer(self, tmp_path):
        pymupdf = __import__("pymupdf")
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "HOLE SCHEDULE\nFABRICATE PER IPC-6012 CLASS 2")
        pdf_path = tmp_path / "synthetic.pdf"
        doc.save(str(pdf_path))
        doc.close()

        text = extract_pdf_text_layer(str(pdf_path))
        assert "HOLE SCHEDULE" in text
        assert "IPC-6012" in text

    def test_blank_page_has_no_text(self, tmp_path):
        pymupdf = __import__("pymupdf")
        doc = pymupdf.open()
        doc.new_page()
        pdf_path = tmp_path / "blank.pdf"
        doc.save(str(pdf_path))
        doc.close()

        text = extract_pdf_text_layer(str(pdf_path))
        assert has_substantial_text_layer(text) is False
