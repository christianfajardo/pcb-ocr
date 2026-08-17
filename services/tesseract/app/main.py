"""Tesseract OCR FastAPI application."""

from __future__ import annotations

import structlog

from fastapi import FastAPI

from shared.logging_config import configure_logging

from .router import router as extract_router

configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Tesseract OCR Service",
    description="Tesseract-based OCR for PCB fabrication drawings",
    version="0.1.0",
)

# ── Health endpoints ──────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "engine": "tesseract", "version": "0.1.0"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness check — Tesseract is always ready (no downstream deps)."""
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
        return {"ready": True}
    except Exception:
        return {"ready": False}


# ── Mount routes ──────────────────────────────────────────────────────────────

app.include_router(extract_router)
