"""Test configuration and shared fixtures."""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Sample files
SAMPLE_FILES = [
    ("samples/sample1.pdf", "tests/expected/sample1.json"),
    ("samples/sample2.pdf", "tests/expected/sample2.json"),
    ("samples/sample3..pdf", "tests/expected/sample3.json"),
    ("samples/sample4.pdf", "tests/expected/sample4.json"),
]

# Supervisor endpoints
SUPERVISOR_URL = os.environ.get("SUPERVISOR_URL", "http://localhost:8080/extract")
JOBS_URL = os.environ.get("JOBS_URL", "http://localhost:8080/jobs")

# Same API_KEY the live services were started with (see shared/auth.py) — a
# no-op empty dict when auth isn't configured, since the services skip the
# check in that case too.
_API_KEY = os.environ.get("API_KEY")
AUTH_HEADERS = {"Authorization": f"Bearer {_API_KEY}"} if _API_KEY else {}

POLL_INTERVAL_SEC = 3
DEFAULT_MAX_WAIT_SEC = 1080  # preserves the old sync-call timeout as the poll ceiling


async def poll_job(
    client,
    job_id: str,
    headers: dict | None = None,
    max_wait: float = DEFAULT_MAX_WAIT_SEC,
    interval: float = POLL_INTERVAL_SEC,
) -> dict:
    """Poll GET /jobs/{job_id} until status is completed/failed.

    Raises:
        TimeoutError: if the job is still processing after max_wait seconds.
    """
    deadline = time.monotonic() + max_wait
    while True:
        resp = await client.get(f"{JOBS_URL}/{job_id}", headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} still {body['status']} after {max_wait}s")
        await asyncio.sleep(interval)


@pytest.fixture(params=SAMPLE_FILES, ids=["sample1", "sample2", "sample3", "sample4"])
def sample_pair(request):
    """Yield (pdf_path, expected_path) for each sample."""
    pdf_rel, expected_rel = request.param
    return (PROJECT_ROOT / pdf_rel, PROJECT_ROOT / expected_rel)


@pytest.fixture
def sample_json(sample_pair):
    """Load expected JSON for a sample."""
    _, expected_path = sample_pair
    with open(expected_path) as f:
        return json.load(f)


@pytest.fixture
def all_expected():
    """Load all expected JSONs."""
    result = {}
    for pdf_rel, expected_rel in SAMPLE_FILES:
        expected_path = PROJECT_ROOT / expected_rel
        with open(expected_path) as f:
            result[expected_rel.split("/")[-1].replace(".json", "")] = json.load(f)
    return result
