"""Redis-backed job registry for async /extract.

Only the supervisor talks to Redis — the other 3 services are unaffected.
Job records are stored as JSON strings under `job:{job_id}`, with a TTL so
old jobs expire automatically instead of accumulating forever.

Deliberately no fallback if Redis is unreachable (e.g. to an in-memory
dict) — that would risk split-brain state between two stores. If Redis is
down, job creation/lookup just fails loudly.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import redis.asyncio as redis

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")  # same convention as nodes.py

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
JOB_TTL_SEC = int(os.environ.get("JOB_TTL_SEC", str(24 * 3600)))

STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_redis = redis.from_url(REDIS_URL, decode_responses=True)


def _key(job_id: str) -> str:
    return f"job:{job_id}"


async def create_job(filename: str | None) -> str:
    """Create a job in STATUS_PROCESSING and return its job_id."""
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "status": STATUS_PROCESSING,
        "created_at": datetime.now(PACIFIC_TZ).isoformat(),
        "completed_at": None,
        "filename": filename,
        "result": None,
        "error": None,
    }
    await _redis.set(_key(job_id), json.dumps(record), ex=JOB_TTL_SEC)
    return job_id


async def get_job(job_id: str) -> dict[str, Any] | None:
    """Look up a job; None if unknown or expired."""
    raw = await _redis.get(_key(job_id))
    return json.loads(raw) if raw else None


async def list_jobs() -> list[dict[str, Any]]:
    """Return every unexpired job record, newest first.

    Uses SCAN rather than KEYS — KEYS blocks the whole Redis server while it
    walks the keyspace. The set is naturally bounded by JOB_TTL_SEC, so no
    pagination is needed at the Redis level; the caller caps how many it
    returns.
    """
    jobs: list[dict[str, Any]] = []
    async for key in _redis.scan_iter(match="job:*", count=100):
        raw = await _redis.get(key)
        if raw:
            jobs.append(json.loads(raw))
    # created_at is ISO-8601 with a fixed offset, so lexical sort == chronological
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return jobs


async def fail_orphaned_jobs() -> list[str]:
    """Mark every job still in `processing` as failed. Returns their ids.

    Called once at supervisor startup. Pipeline jobs run as in-process
    `asyncio.create_task` background tasks, so a restart kills them outright
    with no way to resume — any job still marked `processing` when the
    process comes up is therefore orphaned from a previous life. Without
    this, such a job sits in `processing` forever (well, until JOB_TTL_SEC),
    and a client polling it would wait out its full timeout for a result
    that is never coming.

    ASSUMES A SINGLE SUPERVISOR INSTANCE — true today (one container,
    uvicorn with no --workers; see the deployment note in the module
    docstring). If this is ever scaled to multiple replicas against one
    Redis, a starting instance would wrongly fail another instance's
    genuinely-in-flight jobs, and this needs an instance/owner tag per job.
    """
    orphaned = [j["job_id"] for j in await list_jobs() if j["status"] == STATUS_PROCESSING]
    for job_id in orphaned:
        await mark_failed(
            job_id,
            "Orphaned: the supervisor restarted while this job was running. "
            "Jobs do not resume across restarts — resubmit the PDF.",
        )
    return orphaned


async def mark_completed(job_id: str, result: dict[str, Any]) -> None:
    await _update(job_id, status=STATUS_COMPLETED, result=result)


async def mark_failed(job_id: str, error: str) -> None:
    await _update(job_id, status=STATUS_FAILED, error=error)


async def _update(job_id: str, **fields: Any) -> None:
    job = await get_job(job_id)
    if job is None:
        return  # job expired mid-run (only possible if JOB_TTL_SEC < pipeline duration)
    job.update(fields)
    job["completed_at"] = datetime.now(PACIFIC_TZ).isoformat()
    # Preserve remaining TTL rather than resetting the clock on every update.
    ttl = await _redis.ttl(_key(job_id))
    await _redis.set(_key(job_id), json.dumps(job), ex=ttl if ttl > 0 else JOB_TTL_SEC)
