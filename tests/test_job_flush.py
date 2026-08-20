"""Tests for job flushing, against a fake in-memory Redis.

`delete_jobs` is the destructive half of the job registry, so it's worth
covering without needing a live Redis: these assert it only ever removes the
statuses it was asked for.
"""

from __future__ import annotations

import json

import pytest

from services.supervisor.app import jobs as jobs_module
from services.supervisor.app.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    delete_jobs,
    list_jobs,
)


class FakeRedis:
    """Minimal stand-in for the async redis client used by jobs.py."""

    def __init__(self, records: dict[str, dict]):
        self.store = {f"job:{k}": json.dumps(v) for k, v in records.items()}

    async def scan_iter(self, match: str = "*", count: int = 100):
        for key in list(self.store):
            yield key

    async def get(self, key: str):
        return self.store.get(key)

    async def delete(self, key: str):
        self.store.pop(key, None)


def _job(job_id: str, status: str, created_at: str = "2026-08-19T00:00:00-07:00") -> dict:
    return {
        "job_id": job_id,
        "status": status,
        "created_at": created_at,
        "completed_at": None,
        "filename": f"{job_id}.pdf",
        "result": None,
        "error": None,
    }


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis(
        {
            "a": _job("a", STATUS_COMPLETED, "2026-08-19T03:00:00-07:00"),
            "b": _job("b", STATUS_FAILED, "2026-08-19T02:00:00-07:00"),
            "c": _job("c", STATUS_PROCESSING, "2026-08-19T01:00:00-07:00"),
            "d": _job("d", STATUS_COMPLETED, "2026-08-19T00:00:00-07:00"),
        }
    )
    monkeypatch.setattr(jobs_module, "_redis", fake)
    return fake


class TestDeleteJobs:
    async def test_terminal_only_leaves_processing(self, fake_redis):
        # The default flush must never disturb in-flight work.
        deleted = await delete_jobs({STATUS_COMPLETED, STATUS_FAILED})
        assert deleted == {STATUS_COMPLETED: 2, STATUS_FAILED: 1}
        remaining = await list_jobs()
        assert [j["status"] for j in remaining] == [STATUS_PROCESSING]

    async def test_single_status(self, fake_redis):
        assert await delete_jobs({STATUS_FAILED}) == {STATUS_FAILED: 1}
        assert len(await list_jobs()) == 3

    async def test_processing_only_when_asked(self, fake_redis):
        assert await delete_jobs({STATUS_PROCESSING}) == {STATUS_PROCESSING: 1}
        assert {j["status"] for j in await list_jobs()} == {
            STATUS_COMPLETED,
            STATUS_FAILED,
        }

    async def test_all_statuses(self, fake_redis):
        await delete_jobs({STATUS_COMPLETED, STATUS_FAILED, STATUS_PROCESSING})
        assert await list_jobs() == []

    async def test_empty_target_deletes_nothing(self, fake_redis):
        assert await delete_jobs(set()) == {}
        assert len(await list_jobs()) == 4

    async def test_unparseable_record_is_left_alone(self, monkeypatch):
        fake = FakeRedis({"a": _job("a", STATUS_COMPLETED)})
        fake.store["job:corrupt"] = "{not json"
        monkeypatch.setattr(jobs_module, "_redis", fake)

        assert await delete_jobs({STATUS_COMPLETED}) == {STATUS_COMPLETED: 1}
        # Better to leave something we can't read than to guess and drop it.
        assert "job:corrupt" in fake.store

    async def test_is_idempotent(self, fake_redis):
        await delete_jobs({STATUS_COMPLETED})
        assert await delete_jobs({STATUS_COMPLETED}) == {}
