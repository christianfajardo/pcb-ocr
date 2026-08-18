"""Supervisor FastAPI application — LangGraph orchestrator."""

from __future__ import annotations

import structlog

from fastapi import FastAPI

from shared.logging_config import configure_logging

from .router import router as extract_router

configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="PCB OCR Agentic AI Pipeline",
    description="Multi-model OCR pipeline for extracting structured data from PCB fabrication drawings (PDF).",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "engine": "supervisor", "version": "0.1.0"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness check — all downstream services available."""
    import os

    import httpx

    urls = [
        os.environ.get("TESSERACT_URL", "http://tesseract-ocr:8001/extract"),
        os.environ.get("GLM_OCR_URL", "http://glm-ocr-api:8002/extract"),
        os.environ.get("QWEN_VL_URL", "http://qwen-vl-api:8003/extract"),
    ]

    all_ready = True
    status = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for url in urls:
            try:
                resp = await client.get(url.replace("/extract", "/health"))
                status[url] = resp.status_code == 200
                if resp.status_code != 200:
                    all_ready = False
            except Exception:
                status[url] = False
                all_ready = False

    return {"ready": all_ready, "services": status}


app.include_router(extract_router)
