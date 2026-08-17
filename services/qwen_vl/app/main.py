"""Qwen3-VL FastAPI application."""

from __future__ import annotations

import structlog

from fastapi import FastAPI

from shared.logging_config import configure_logging

from .router import router as extract_router

configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Qwen3-VL Service",
    description="Qwen3-VL vision model wrapper for PCB fabrication drawings",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "engine": "qwen-vl", "version": "0.1.0"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness check — vLLM availability."""
    import httpx

    vllm_url = __import__("os").environ.get("QWEN_VL_VLLM_URL", "http://vllm-qwen-vl:8011/v1")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(vllm_url.replace("/v1", "/health"))
            return {"ready": resp.status_code == 200}
    except Exception:
        return {"ready": False}


app.include_router(extract_router)
