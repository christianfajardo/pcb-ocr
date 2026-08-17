"""Tesseract OCR extraction engine with adaptive page type detection."""

from __future__ import annotations

import structlog
import time

import cv2
import numpy as np
import pytesseract
from PIL import Image

from shared.preprocessing import (
    PageType,
    cv2_to_pil,
    detect_page_type,
    image_to_bytes,
    pdf_to_images,
    pil_to_cv2,
)

logger = structlog.get_logger(__name__)

# Tesseract 5 LSTM engine
TESSERACT_OEM = 1  # LSTM
TESSERACT_PSM = 6  # Uniform block of text


class TesseractOCR:
    """Wraps Tesseract 5 for PCB drawing text extraction with adaptive preprocessing.

    Per-page workflow:
        1. Convert PDF page to image at base DPI
        2. Detect page type (VECTOR / SCAN / LOW_CONTRAST / MIXED)
        3. Apply page-type-specific enhancement
        4. Run Tesseract OCR
    """

    def __init__(
        self,
        base_dpi: int = 300,
        max_dpi: int = 600,
        adaptive_enhancement: bool = True,
    ) -> None:
        self.base_dpi = base_dpi
        self.max_dpi = max_dpi
        self.adaptive_enhancement = adaptive_enhancement

    def extract(self, pdf_path: str) -> dict[str, object]:
        """Run full Tesseract extraction with adaptive preprocessing.

        Args:
            pdf_path: Path to the input PDF.

        Returns:
            dict with keys: raw_text, page_images, page_count,
            word_confidences, enhanced_images, page_types
        """
        start = time.monotonic()
        images = pdf_to_images(pdf_path, self.base_dpi)
        page_count = len(images)
        logger.info("PDF pages converted", page_count=page_count)

        all_text_parts: list[str] = []
        all_confidences: list[float] = []
        enhanced_images: list[Image.Image] = []
        page_types: list[PageType] = []

        for i, img in enumerate(images):
            cv_img = pil_to_cv2(img)
            logger.info(f"Processing page {i + 1}/{page_count}")

            # Detect page type
            page_type = detect_page_type(cv_img)
            page_types.append(page_type)
            logger.info(f"Page {i + 1} type: {page_type.value}")

            # Enhance based on page type
            enhanced = self._enhance(cv_img, page_type)
            enhanced_pil = cv2_to_pil(enhanced)
            enhanced_images.append(enhanced_pil)

            # OCR
            text, word_confs = self._ocr(enhanced_pil)
            all_text_parts.append(text)
            all_confidences.extend(word_confs)

            avg_conf = sum(word_confs) / len(word_confs) if word_confs else 0
            logger.info(
                f"Page {i + 1} done",
                type=str(page_type),
                text_len=len(text),
                avg_conf=round(avg_conf, 2),
            )

        raw_text = "\n".join(all_text_parts)
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        page_images_b64: list[bytes] = []
        for img in images:
            page_images_b64.append(image_to_bytes(img))

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("Tesseract extraction complete", elapsed_ms=round(elapsed_ms, 1))

        return {
            "raw_text": raw_text,
            "page_images": page_images_b64,
            "page_count": page_count,
            "word_confidences": avg_conf,
            "enhanced_images": enhanced_images,
            "page_types": page_types,
        }

    def _enhance(self, cv_img: np.ndarray, page_type: PageType) -> np.ndarray:
        """Enhance image based on page type.

        For scans: Otsu threshold (preserves all text)
        For vector: adaptive threshold (sharper)
        For mixed: adaptive threshold (default)
        """
        gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Deskew
        from shared.preprocessing import detect_skew, rotate_image

        angle = detect_skew(gray)
        if abs(angle) > 0.5:
            gray = rotate_image(gray, angle)

        if page_type == PageType.SCAN:
            # Scans need strong denoising + Otsu threshold
            denoised = cv2.fastNlMeansDenoising(gray, h=20)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif page_type == PageType.LOW_CONTRAST:
            # Contrast stretch + CLAHE + Otsu
            stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            enhanced = clahe.apply(stretched.astype(np.uint8))
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif page_type == PageType.MIXED:
            # Adaptive threshold (good for mixed vector/raster)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
            return binary
        else:  # VECTOR — default adaptive threshold
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
            return binary

    def _ocr(self, image: Image.Image, psm: int = TESSERACT_PSM) -> tuple[str, list[int]]:
        """Run Tesseract on a single image."""
        config = f"--psm {psm} --oem {TESSERACT_OEM}"
        ocr_data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )
        raw_text = pytesseract.image_to_string(image, config=config)
        word_confs = [int(c) for c in ocr_data.get("conf", []) if int(c) > -1]
        return raw_text, word_confs

    def extract_region(
        self,
        image: Image.Image,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> str:
        """OCR a specific region of an image."""
        cropped = image.crop((x, y, x + w, y + h))
        text, _ = self._ocr(cropped)
        return text
