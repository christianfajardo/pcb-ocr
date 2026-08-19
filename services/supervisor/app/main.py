"""Supervisor FastAPI application — LangGraph orchestrator."""

from __future__ import annotations

import structlog

from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.logging_config import configure_logging

from .jobs import fail_orphaned_jobs
from .router import jobs_router
from .router import router as extract_router
from .schemas import HealthResponse, ReadyResponse

configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reap jobs orphaned by a previous process before serving traffic."""
    try:
        orphaned = await fail_orphaned_jobs()
        if orphaned:
            logger.info(
                "Marked orphaned jobs as failed",
                count=len(orphaned),
                job_ids=orphaned,
            )
    except Exception as e:
        # Deliberately non-fatal. docker-compose's `depends_on` for redis has
        # no `condition: service_healthy`, so on a cold `make up` this can run
        # before Redis accepts connections — crashing here would put the
        # container into a restart loop over a transient condition. Requests
        # that actually need Redis still fail loudly on their own.
        logger.error("Could not reap orphaned jobs at startup", error=str(e))
    yield


app = FastAPI(
    title="PCB OCR Agentic AI Pipeline",
    description="Multi-model OCR pipeline for extracting structured data from PCB fabrication drawings (PDF).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "engine": "supervisor", "version": "0.1.0"}


@app.get("/ready", response_model=ReadyResponse)
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
app.include_router(jobs_router)
