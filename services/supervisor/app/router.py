"""Supervisor API router."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time

import structlog

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder

from shared.auth import require_api_key
from shared.logging_config import bind_pdf_filename

from .graph import compiled
from .jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    create_job,
    get_job,
    list_jobs,
    mark_completed,
    mark_failed,
)
from .schemas import (
    ErrorDetail,
    JobAccepted,
    JobList,
    JobStatus,
)
from .state import create_initial_state

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["supervisor"])
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])


def _sec(ms: float | None) -> float | None:
    """Convert a millisecond duration to seconds, rounded for display."""
    return round(ms / 1000, 3) if ms is not None else None


# (engine display name, state key prefix)
_ENGINES = [
    ("tesseract", "tesseract"),
    ("glm-ocr", "glm_ocr"),
    ("qwen-vl", "qwen_vl"),
    ("pymupdf", "pymupdf"),
]


@router.post(
    "",
    status_code=202,
    response_model=JobAccepted,
    responses={401: {"model": ErrorDetail, "description": "Missing or invalid API key"}},
    dependencies=[Depends(require_api_key)],
)
async def extract(file: UploadFile = File(...)) -> dict:
    """Accept a PDF and start the OCR pipeline as a background job.

    Args:
        file: PDF file.

    Returns:
        {"job_id": ..., "status": "processing"} — poll GET /jobs/{job_id}
        for the result.
    """
    with bind_pdf_filename(file.filename):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(await file.read())
                pdf_path = tmp.name
        except Exception as e:
            logger.error("Upload save failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

        job_id = await create_job(file.filename)
        asyncio.create_task(_run_pipeline_job(job_id, pdf_path, file.filename))

        logger.info("Job accepted", filename=file.filename, job_id=job_id)

    return {"job_id": job_id, "status": STATUS_PROCESSING}


async def _run_pipeline_job(job_id: str, pdf_path: str, original_filename: str | None) -> None:
    """Run the full OCR pipeline for one job.

    Never raises — always resolves the job to completed or failed via the
    Redis-backed registry, so an exception here can't leave a job stuck in
    "processing" forever. Owns its own bind_pdf_filename() scope rather than
    relying on inheriting the request handler's, so correctness doesn't
    depend on exactly where asyncio.create_task() was called.
    """
    request_start = time.monotonic()

    with bind_pdf_filename(original_filename):
        logger.info("Pipeline started", filename=original_filename, job_id=job_id)

        try:
            try:
                # Create initial state
                state = create_initial_state(pdf_path, original_filename)

                # Run the graph
                result = await compiled.ainvoke(state)

                merged = result.get("reconciled")
                errors = result.get("errors", [])
                log = result.get("reconciliation_log", [])
                attribution = result.get("attribution", {})

                if not merged:
                    raise RuntimeError(f"No output: {errors}")

                total_duration_sec = round(time.monotonic() - request_start, 3)

                logger.info(
                    "Pipeline complete",
                    layer_count=merged.layer_count,
                    material=merged.material,
                    total_duration_sec=total_duration_sec,
                    job_id=job_id,
                )

                response = merged.model_dump()
                response["reconciliation_log"] = log
                response["attribution"] = attribution
                response["errors"] = errors
                response["engine_durations_sec"] = [
                    {
                        "engine": name,
                        "duration_sec": _sec(result.get(f"{prefix}_duration_ms")),
                        "start_time": result.get(f"{prefix}_start_time"),
                        "end_time": result.get(f"{prefix}_end_time"),
                    }
                    for name, prefix in _ENGINES
                ]
                response["total_duration_sec"] = total_duration_sec

                # attribution/contributors can carry raw Pydantic model
                # instances (e.g. LayerSpec, DrillRow) inside list/nested
                # fields — FastAPI's own response serialization handled this
                # transparently when this dict was returned directly; now
                # that it's json.dumps()'d into Redis instead, it must be
                # made JSON-safe explicitly first.
                await mark_completed(job_id, jsonable_encoder(response))

            finally:
                os.unlink(pdf_path)

        except Exception as e:
            logger.error("Pipeline failed", error=str(e), job_id=job_id)
            await mark_failed(job_id, str(e))


@jobs_router.get(
    "",
    response_model=JobList,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid `status` filter"},
        401: {"model": ErrorDetail, "description": "Missing or invalid API key"},
    },
    dependencies=[Depends(require_api_key)],
)
async def list_all_jobs(
    status: str | None = Query(
        default=None,
        description="Optional filter: processing | completed | failed",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    """List all unexpired jobs, newest first.

    Returns per-job metadata only — not the full extraction `result`, which
    runs tens of KB per job and would make this response enormous. Fetch
    GET /jobs/{job_id} for a specific job's result.
    """
    jobs = await list_jobs()

    if status is not None:
        valid = {STATUS_PROCESSING, STATUS_COMPLETED, STATUS_FAILED}
        if status not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status filter: {status}. Expected one of {sorted(valid)}",
            )
        jobs = [j for j in jobs if j["status"] == status]

    total = len(jobs)
    summaries = []
    for job in jobs[:limit]:
        summary = {
            "job_id": job["job_id"],
            "status": job["status"],
            "filename": job.get("filename"),
            "submitted_at": job.get("created_at"),
            "completed_at": job.get("completed_at"),
        }
        if job["status"] == STATUS_FAILED:
            summary["error"] = job.get("error")
        summaries.append(summary)

    return {"total": total, "returned": len(summaries), "jobs": summaries}


@jobs_router.get(
    "/{job_id}",
    response_model=JobStatus,
    responses={
        401: {"model": ErrorDetail, "description": "Missing or invalid API key"},
        404: {"model": ErrorDetail, "description": "Unknown or expired job_id"},
    },
    dependencies=[Depends(require_api_key)],
)
async def get_job_status(job_id: str) -> dict:
    """Poll the status/result of a job created via POST /extract."""
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown or expired job_id: {job_id}")

    submitted_at = job["created_at"]
    if job["status"] == STATUS_COMPLETED:
        return {
            "job_id": job_id,
            "status": STATUS_COMPLETED,
            "submitted_at": submitted_at,
            "result": job["result"],
        }
    if job["status"] == STATUS_FAILED:
        return {
            "job_id": job_id,
            "status": STATUS_FAILED,
            "submitted_at": submitted_at,
            "error": job["error"],
        }
    return {"job_id": job_id, "status": STATUS_PROCESSING, "submitted_at": submitted_at}
