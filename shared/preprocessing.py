"""PDF-to-image conversion, image enhancement, and ROI detection for PCB drawings."""

from __future__ import annotations

import logging
import os
from enum import Enum
from io import BytesIO

import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300

# Cap on pages rasterized per PDF. Fab drawings put the relevant data on the
# first few sheets; scanning further is wasted GPU time. 0/negative = no limit.
MAX_PAGES = int(os.environ.get("MAX_PAGES", "5"))


# ── Page Type Detection ──────────────────────────────────────────────────────


class PageType(Enum):
    """Classification of a PDF page for optimal OCR preprocessing."""

    VECTOR = "vector"  # Clean vector/line-art — sharp text, black-on-white
    SCAN = "scan"  # Scanned page — degraded, noisy, possibly colored
    LOW_CONTRAST = "low_contrast"  # Gray-on-gray or light text
    MIXED = "mixed"  # Mix of vector elements and scan artifacts
    PHOTO = "photo"  # Photo-realistic content (rare for PCB docs)


def detect_page_type(image: np.ndarray) -> PageType:
    """Classify a page image to determine optimal OCR strategy.

    Heuristics:
        - Edge density / text content ratio
        - Color channel analysis
        - Histogram shape
        - Sharpness (Laplacian variance)

    Args:
        image: BGR image from PIL/PDF conversion.

    Returns:
        PageType enum value.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Laplacian variance (sharpness measure)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Histogram analysis
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_flat = hist.flatten()

    # Bimodal histogram check (clear black-on-white)
    white_pixels = np.sum(hist_flat[230:]) / image.size
    black_pixels = np.sum(hist_flat[:30]) / image.size
    dark_ratio = black_pixels / (white_pixels + 1e-6)

    # 3. Color channel analysis (detect grayscale scans vs true color)
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        color_deviation = float(np.std([r.std(), g.std(), b.std()]))
        is_color = color_deviation > 10  # Significant difference between channels
    else:
        color_deviation = 0
        is_color = False

    # 4. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 5. Mean intensity
    mean_intensity = float(np.mean(gray))

    # Classification logic
    if lap_var > 800 and dark_ratio > 0.5:
        # High sharpness, clear black-on-white → vector/line-art
        return PageType.VECTOR
    elif lap_var < 100:
        # Very low sharpness → scan/raster
        return PageType.SCAN
    elif mean_intensity > 180 and lap_var < 400:
        # High mean intensity, low sharpness → low contrast
        return PageType.LOW_CONTRAST
    elif is_color or (0.5 < color_deviation < 50):
        return PageType.MIXED
    elif edge_density > 0.15:
        return PageType.VECTOR
    elif lap_var < 200:
        return PageType.SCAN
    else:
        return PageType.MIXED


# ── PDF to Images ─────────────────────────────────────────────────────────────


def _optimal_dpi_for_page(page_type: PageType, base_dpi: int = 300) -> int:
    """Return optimal DPI for a given page type.

    Scans benefit from higher DPI (400-600) for text clarity.
    Vector pages are sharp at lower DPI (200-300).
    """
    dpi_map = {
        PageType.VECTOR: min(base_dpi, 300),  # 300 is enough for clean vector
        PageType.SCAN: max(base_dpi, 400),  # Higher DPI for scan clarity
        PageType.LOW_CONTRAST: max(base_dpi, 400),  # Higher DPI for contrast enhancement
        PageType.MIXED: max(base_dpi, 350),  # Moderate DPI
        PageType.PHOTO: max(base_dpi, 300),
    }
    return dpi_map[page_type]


def get_pdf_page_count(pdf_path: str) -> int:
    """Total pages in the PDF, without rendering any of them."""
    return int(pdfinfo_from_path(pdf_path)["Pages"])


def pdf_to_images(
    pdf_path: str,
    dpi: int = DEFAULT_DPI,
    max_pages: int | None = None,
) -> list[Image.Image]:
    """Convert PDF pages to PIL Images at specified DPI.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution in dots per inch.
        max_pages: Stop after this many pages. Defaults to the MAX_PAGES env
            var (5) — most fab drawings carry the relevant data on the first
            few sheets, and rendering the rest is wasted GPU time downstream.
            Pass 0 or a negative number for no limit.

    Returns:
        List of PIL Image objects, one per page (at most `max_pages`).

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If PDF cannot be parsed.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    limit = MAX_PAGES if max_pages is None else max_pages

    logger.info("Converting PDF to images", pdf_path=pdf_path, dpi=dpi, max_pages=limit)
    # first_page/last_page are passed through to poppler, so pages beyond the
    # cap are never rendered at all — work avoided, not work discarded.
    page_range = {"first_page": 1, "last_page": limit} if limit and limit > 0 else {}
    images = convert_from_path(pdf_path, dpi=dpi, fmt="png", **page_range)
    logger.info("PDF conversion complete", page_count=len(images))
    return images


def bytes_to_images(blobs: list[bytes]) -> list[Image.Image]:
    """Decode already-rasterized page images (PNG bytes) back to PIL Images.

    Lets the supervisor rasterize a PDF once and hand the same pages to every
    engine, instead of each engine re-rasterizing the same file. PNG is
    lossless, so these are pixel-identical to what `pdf_to_images` produced.
    """
    images: list[Image.Image] = []
    for blob in blobs:
        # .convert() also forces the lazy PIL load, so the caller isn't
        # holding a file-backed handle to a closed BytesIO.
        images.append(Image.open(BytesIO(blob)).convert("RGB"))
    logger.info("Decoded pre-rasterized pages", page_count=len(images))
    return images


def pdf_to_images_adaptive(
    pdf_path: str,
    base_dpi: int = DEFAULT_DPI,
    max_dpi: int = 600,
) -> list[tuple[Image.Image, PageType, int]]:
    """Convert PDF pages to images with adaptive DPI per page type.

    First does a quick low-DPI pass to detect page types, then converts
    at the optimal DPI for each type.

    Args:
        pdf_path: Path to the PDF file.
        base_dpi: Base DPI for vector pages.
        max_dpi: Maximum DPI for scan pages.

    Returns:
        List of (image, page_type, dpi) tuples.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1: Quick low-res pass to detect page types
    logger.info("Quick scan pass for page type detection", dpi=150)
    quick_images = convert_from_path(pdf_path, dpi=150, fmt="png")

    page_types: list[PageType] = []
    for img in quick_images:
        cv_img = pil_to_cv2(img)
        ptype = detect_page_type(cv_img)
        page_types.append(ptype)
        logger.info(
            "Page type detected",
            page=quick_images.index(img) + 1,
            type=str(ptype),
        )

    # Step 2: Convert at optimal DPI for each page type
    results = []
    for i, ptype in enumerate(page_types):
        effective_dpi = min(_optimal_dpi_for_page(ptype, base_dpi), max_dpi)
        logger.info(
            "Converting page at adaptive DPI",
            page=i + 1,
            type=str(ptype),
            dpi=effective_dpi,
        )

        # Convert single page at the right DPI
        imgs = convert_from_path(
            pdf_path,
            dpi=effective_dpi,
            fmt="png",
            first_page=i + 1,
            last_page=i + 1,
        )
        if imgs:
            results.append((imgs[0], ptype, effective_dpi))
        else:
            # Fallback
            imgs_fallback = convert_from_path(pdf_path, dpi=base_dpi, fmt="png")
            if i < len(imgs_fallback):
                results.append((imgs_fallback[i], ptype, base_dpi))

    return results


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def images_to_base64(images: list[Image.Image]) -> list[str]:
    """Convert PIL Images to base64-encoded strings for API calls."""
    import base64

    result = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append(b64)
    return result


# ── Image Enhancement ─────────────────────────────────────────────────────────


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV numpy array (BGR)."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV numpy array (BGR) to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Pre-process image for optimal OCR accuracy (default strategy).

    Steps:
        1. Convert to grayscale
        2. Deskew if needed (threshold > 0.5°)
        3. Adaptive threshold for line drawings
        4. Non-local means denoising

    Args:
        image: Input image (BGR numpy array).

    Returns:
        Enhanced grayscale image.
    """
    return enhance_page(image, PageType.VECTOR)


def enhance_page(
    image: np.ndarray,
    page_type: PageType,
) -> np.ndarray:
    """Enhance an image for OCR based on detected page type.

    Different page types need different preprocessing:
        - VECTOR: Preserve sharpness, light cleanup
        - SCAN: Strong denoising, contrast boost, possible unsharp mask
        - LOW_CONTRAST: CLAHE histogram equalization, contrast stretch
        - MIXED: Balanced approach

    Args:
        image: Input image (BGR numpy array).
        page_type: Detected page type.

    Returns:
        Enhanced grayscale image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Deskew (all page types benefit)
    angle = detect_skew(gray)
    if abs(angle) > 0.5:
        logger.info("Deskewing image", angle=round(angle, 2))
        gray = rotate_image(gray, angle)

    # 2. Page-type-specific enhancement
    if page_type == PageType.SCAN:
        gray = _enhance_scan(gray)
    elif page_type == PageType.LOW_CONTRAST:
        gray = _enhance_low_contrast(gray)
    elif page_type == PageType.MIXED:
        gray = _enhance_mixed(gray)
    else:  # VECTOR — default
        gray = _enhance_vector(gray)

    return gray


def _enhance_vector(gray: np.ndarray) -> np.ndarray:
    """Enhancement for clean vector/line-art pages."""
    # Light denoising to remove artifacts without blurring
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold — good for sharp text
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return binary


def _enhance_scan(gray: np.ndarray) -> np.ndarray:
    """Enhancement for scanned/degraded pages.

    Uses unsharp mask, CLAHE, and stronger denoising.
    """
    # 1. Denoise first (scans are noisy)
    denoised = cv2.fastNlMeansDenoising(gray, h=20)

    # 2. Contrast enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. Unsharp mask to sharpen text
    enhanced = unsharp_mask(enhanced, amount=1.5)

    # 4. Threshold — use Otsu for scans (global threshold often better)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def _enhance_low_contrast(gray: np.ndarray) -> np.ndarray:
    """Enhancement for low-contrast pages.

    Focuses on contrast stretching and CLAHE.
    """
    # 1. Contrast stretching
    min_val, max_val = np.min(gray), np.max(gray)
    if max_val > min_val:
        stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    else:
        stretched = gray.copy()

    # 2. CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(stretched.astype(np.uint8))

    # 3. Light denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, h=15)

    # 4. Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def _enhance_mixed(gray: np.ndarray) -> np.ndarray:
    """Enhancement for mixed page types.

    Balanced approach: moderate denoising + adaptive threshold.
    """
    # 1. Moderate denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=15)

    # 2. Slight CLAHE boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. Adaptive threshold
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12
    )

    return binary


def unsharp_mask(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Apply unsharp mask for sharpening.

    Args:
        image: Grayscale image.
        amount: Sharpening amount (higher = sharper).

    Returns:
        Sharpened image.
    """
    blur = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharp = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    return sharp


def detect_skew(image: np.ndarray) -> float:
    """Detect skew angle of an image in degrees.

    Uses Hough line transform to find the dominant text line angle.

    Args:
        image: Grayscale image.

    Returns:
        Skew angle in degrees (positive = clockwise).
    """
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        pts = line.flatten() if hasattr(line, "flatten") else line
        if len(pts) < 4:
            continue
        x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
        if x2 - x1 == 0:
            continue
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by the given angle.

    Args:
        image: Grayscale image.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ── Multi-Strategy Enhancement ───────────────────────────────────────────────


def best_enhancement(
    image: np.ndarray,
    strategies: list[str] | None = None,
) -> np.ndarray:
    """Try multiple enhancement strategies and return the one with best
    visual quality for OCR.

    Quality metric: Shannon entropy of thresholded output (higher = more
    structure/contrast, which Tesseract can work with).

    Args:
        image: Input BGR image.
        strategies: List of strategy names to try.
            Options: "vector", "scan", "low_contrast", "mixed", "otsu",
                     "simple_thresh", "none"
            If None, tries all strategies.

    Returns:
        Best enhanced grayscale image.
    """
    if strategies is None:
        strategies = ["vector", "scan", "low_contrast", "mixed", "otsu", "simple_thresh"]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Deskew first (all strategies share this)
    angle = detect_skew(gray)
    if abs(angle) > 0.5:
        gray = rotate_image(gray, angle)

    enhanced_images = {}

    for strat in strategies:
        if strat == "vector":
            enhanced_images[strat] = _enhance_vector(gray)
        elif strat == "scan":
            enhanced_images[strat] = _enhance_scan(gray)
        elif strat == "low_contrast":
            enhanced_images[strat] = _enhance_low_contrast(gray)
        elif strat == "mixed":
            enhanced_images[strat] = _enhance_mixed(gray)
        elif strat == "otsu":
            denoised = cv2.fastNlMeansDenoising(gray, h=15)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_images[strat] = binary
        elif strat == "simple_thresh":
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            enhanced_images[strat] = binary
        elif strat == "none":
            enhanced_images[strat] = gray
        else:
            enhanced_images[strat] = _enhance_vector(gray)

    # Score each: entropy of the binary output (more structure = better for OCR)
    best_strat = None
    best_score = -1
    for strat, img in enhanced_images.items():
        score = _enhancement_score(img)
        logger.debug(f"Enhancement score [{strat}]: {score:.4f}")
        if score > best_score:
            best_score = score
            best_strat = strat

    logger.info("Best enhancement strategy", strategy=best_strat, score=round(best_score, 4))
    return enhanced_images[best_strat]


def _enhancement_score(image: np.ndarray) -> float:
    """Score an enhanced image for OCR suitability.

    Metrics:
        - Shannon entropy of binary threshold (structure)
        - Edge density (text presence)
        - Dark pixel ratio (sufficient text content)

    Returns:
        Combined score (higher = better for OCR).
    """
    # Count dark/white pixels (binary image)
    dark = np.sum(image == 0)
    light = np.sum(image == 255)
    total = dark + light
    dark_ratio = dark / (total + 1e-6)

    # Ideal dark ratio for OCR: 5-30% of pixels are text
    if 0.03 <= dark_ratio <= 0.40:
        ratio_score = 1.0
    elif dark_ratio < 0.03:
        ratio_score = dark_ratio / 0.03  # Penalize too little text
    else:
        ratio_score = max(0, 1.0 - (dark_ratio - 0.40) / 0.20)  # Penalize too much

    # Edge density
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Combine scores
    score = edge_density * 2.0 + ratio_score * 1.5

    return float(score)


# ── ROI Detection ─────────────────────────────────────────────────────────────


def detect_regions(image: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    """Detect functional regions in a PCB fab drawing.

    Returns dict of region_name -> (x, y, w, h) bounding boxes.

    Args:
        image: Grayscale image.

    Returns:
        Dictionary mapping region names to (x, y, w, h) tuples.
    """
    h, w = image.shape[:2]
    regions: dict[str, tuple[int, int, int, int]] = {}

    # Title block: typically bottom-right ~20% x 15% of page
    tb_x = int(w * 0.6)
    tb_y = int(h * 0.75)
    tb_w = int(w * 0.4)
    tb_h = int(h * 0.2)
    regions["title_block"] = (tb_x, tb_y, tb_w, tb_h)

    # Notes area: typically left side, top ~75% of page
    notes_x = 0
    notes_y = 0
    notes_w = int(w * 0.4)
    notes_h = int(h * 0.75)
    regions["notes_area"] = (notes_x, notes_y, notes_w, notes_h)

    # Drill table: right side, below notes area or bottom
    dt_x = int(w * 0.55)
    dt_y = int(h * 0.4)
    dt_w = int(w * 0.45)
    dt_h = int(h * 0.35)
    regions["drill_table"] = (dt_x, dt_y, dt_w, dt_h)

    # Board outline: center of page
    bo_x = int(w * 0.1)
    bo_y = int(h * 0.1)
    bo_w = int(w * 0.8)
    bo_h = int(h * 0.65)
    regions["board_outline"] = (bo_x, bo_y, bo_w, bo_h)

    # Stackup diagram: varies, typically center-left or center
    stackup_x = int(w * 0.4)
    stackup_y = int(h * 0.15)
    stackup_w = int(w * 0.25)
    stackup_h = int(h * 0.2)
    regions["stackup_diagram"] = (stackup_x, stackup_y, stackup_w, stackup_h)

    # Refine using contour detection for table-like structures
    regions = _refine_with_contours(image, regions)

    return regions


def _refine_with_contours(
    image: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]],
) -> dict[str, tuple[int, int, int, int]]:
    """Refine region detection using contour analysis."""
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect = cw / ch if ch > 0 else 0

        if cw > 200 and ch > 100 and 0.5 < aspect < 3.0:
            h, w = image.shape[:2]
            if x > w * 0.4 and y > h * 0.3:
                regions["drill_table"] = (x, y, cw, ch)
            elif x > w * 0.5 and y > h * 0.7:
                regions["title_block"] = (x, y, cw, ch)

    return regions


def extract_roi(
    image: np.ndarray,
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract a region of interest from an image.

    Args:
        image: Source image (grayscale).
        region: (x, y, w, h) bounding box.

    Returns:
        Cropped image.
    """
    x, y, w, h = region
    h_max, w_max = image.shape[:2]
    x2 = min(x + w, w_max)
    y2 = min(y + h, h_max)
    x = max(0, x)
    y = max(0, y)
    return image[y:y2, x:x2]
