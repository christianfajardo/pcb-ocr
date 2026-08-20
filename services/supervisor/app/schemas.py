"""API response models for the supervisor's public endpoints.

These exist so the OpenAPI schema (and therefore Swagger UI) describes what
the endpoints actually return, instead of a bare `{}` from a `-> dict`
annotation. Supervisor-only, so they live here rather than in `shared/`
(which is reserved for code used by 2+ services).

The job-status response is a *discriminated* union on `status` rather than
one model with optional fields: a processing job's body genuinely has no
`result`/`error` key, and a single model with optionals would make FastAPI
serialize them as explicit nulls — changing the wire format. The union keeps
each state's body byte-identical to what the handlers already return.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from shared.schemas import PCBData


class ErrorDetail(BaseModel):
    """FastAPI's standard error envelope."""

    detail: str


class JobAccepted(BaseModel):
    """202 response from POST /extract."""

    job_id: str = Field(description="Poll GET /jobs/{job_id} with this id")
    status: Literal["processing"]


class EngineDuration(BaseModel):
    """Per-engine timing for one pipeline run."""

    engine: Literal["tesseract", "glm-ocr", "qwen-vl", "pymupdf"]
    duration_sec: float | None = Field(
        default=None, description="Null when the engine was skipped (e.g. PyMuPDF on a scan)"
    )
    start_time: str | None = Field(
        default=None, description="ISO-8601, US Pacific (America/Los_Angeles) — not UTC"
    )
    end_time: str | None = Field(default=None, description="ISO-8601, US Pacific")


class AttributionContributor(BaseModel):
    """One engine's value for a field, whether or not it won."""

    engine: str
    value: Any = None
    confidence: float
    source_text: str | None = Field(
        default=None,
        description="Drawing text this engine matched. Null for the vision models, which "
        "emit JSON without reporting which pixels produced it.",
    )


class AttributionEntry(BaseModel):
    """Where one merged field's value came from."""

    value: Any = None
    confidence: float
    reason: str = Field(description="Which voting rule selected this value")
    engine: str | None = Field(default=None, description="Whose value won")
    source_text: str | None = None
    source_text_from: str | None = Field(
        default=None,
        description="Which engine supplied source_text — may differ from `engine` when the "
        "winner was a VLM and an agreeing engine provided the evidence",
    )
    contributors: list[AttributionContributor] = Field(default_factory=list)


class ExtractionResult(PCBData):
    """The full extraction payload — every PCBData field, plus run metadata."""

    reconciliation_log: list[str] = Field(default_factory=list)
    attribution: dict[str, AttributionEntry] = Field(
        default_factory=dict, description="Per-field provenance, keyed by PCBData field name"
    )
    errors: list[str] = Field(default_factory=list)
    engine_durations_sec: list[EngineDuration] = Field(default_factory=list)
    total_duration_sec: float | None = None


class JobProcessing(BaseModel):
    """A job still running."""

    job_id: str
    status: Literal["processing"]
    submitted_at: str = Field(description="ISO-8601, US Pacific")


class JobCompleted(BaseModel):
    """A finished job, with its extraction result."""

    job_id: str
    status: Literal["completed"]
    submitted_at: str
    result: ExtractionResult


class JobFailed(BaseModel):
    """A job that failed. The GET itself succeeded — the outcome is data."""

    job_id: str
    status: Literal["failed"]
    submitted_at: str
    error: str


JobStatus = Annotated[
    JobProcessing | JobCompleted | JobFailed,
    Field(discriminator="status"),
]


class _JobSummaryBase(BaseModel):
    """Shared fields for a GET /jobs row — metadata only, never the result."""

    job_id: str
    filename: str | None = None
    submitted_at: str | None = None
    completed_at: str | None = Field(
        default=None, description="Null while still processing"
    )


class JobSummaryOk(_JobSummaryBase):
    """A processing or completed job."""

    status: Literal["processing", "completed"]


class JobSummaryFailed(_JobSummaryBase):
    """A failed job — the only summary that carries `error`."""

    status: Literal["failed"]
    error: str | None = None


# Discriminated for the same reason as JobStatus: `error` is absent (not null)
# on non-failed rows, and a single model with an optional field would emit it.
JobSummary = Annotated[
    JobSummaryOk | JobSummaryFailed,
    Field(discriminator="status"),
]


class JobList(BaseModel):
    """200 response from GET /jobs."""

    total: int = Field(description="Matches after filtering, before `limit`")
    returned: int = Field(description="How many are in this response")
    jobs: list[JobSummary]


class JobFlushResult(BaseModel):
    """200 response from POST /jobs/flush."""

    deleted: int = Field(description="Total job records removed")
    deleted_by_status: dict[str, int] = Field(
        default_factory=dict, description="Breakdown of what was removed, per status"
    )
    flushed_statuses: list[str] = Field(
        description="Which statuses this call actually targeted"
    )
    skipped_processing: int = Field(
        default=0,
        description=(
            "In-flight jobs left alone because `processing` wasn't targeted. "
            "Flushing those does not stop the pipeline — it only discards the "
            "result — so it requires asking for it explicitly."
        ),
    )
    remaining: int = Field(description="Job records still in Redis afterwards")


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    engine: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool = Field(description="True only if every downstream service responded 200")
    services: dict[str, bool] = Field(description="Per-service URL -> reachable")
