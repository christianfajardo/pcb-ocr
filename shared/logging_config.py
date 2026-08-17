"""Structlog setup: normal console output, plus a per-PDF-filename file sink.

Every service in the pipeline (supervisor, tesseract, glm-ocr, qwen-vl) calls
`configure_logging()` once at startup. Whichever request handler received the
upload then does:

    with bind_pdf_filename(file.filename):
        ...

for the duration of that request. Any log event emitted inside that block —
from any module, in any of the four services, since they share this same
`/logs` volume — gets appended to `/logs/{filename}.log`, in addition to its
normal console output. This gives one consolidated log per PDF covering the
whole pipeline, not just whichever single service happens to log it.

Uses structlog's contextvars binding, so concurrent requests (including
concurrent requests for *different* PDFs handled by the same async process)
never leak their filename into each other's log lines.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import structlog

LOG_DIR = Path(os.environ.get("LOG_DIR", "/logs"))

_write_lock = threading.Lock()


def _safe_stem(filename: str) -> str:
    """Sanitize an uploaded filename into a safe log-file stem."""
    stem = Path(filename).stem or "upload"
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return stem or "upload"


def _pdf_file_sink(logger, method_name, event_dict):
    """Structlog processor: append this event to /logs/{pdf}.log when a
    pdf_filename is bound in the current context. Leaves the normal
    console-rendering pipeline untouched — this only adds a side effect."""
    filename = event_dict.get("pdf_filename")
    if filename:
        stem = _safe_stem(filename)
        payload = {k: v for k, v in event_dict.items() if k != "pdf_filename"}
        line = json.dumps(payload, default=str)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with _write_lock, open(LOG_DIR / f"{stem}.log", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass  # logging must never break the request it's logging
    return event_dict


def configure_logging() -> None:
    """Call once at service startup, before any logger is used."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            _pdf_file_sink,
            structlog.dev.ConsoleRenderer(),
        ],
        cache_logger_on_first_use=True,
    )


@contextmanager
def bind_pdf_filename(filename: str | None) -> Iterator[None]:
    """Bind the uploaded PDF's filename for the duration of a request, so
    every log event emitted while processing it — in this service or, via
    the shared /logs volume, any other service in the pipeline — lands in
    /logs/{filename}.log. No-ops if filename is falsy."""
    if not filename:
        yield
        return
    structlog.contextvars.bind_contextvars(pdf_filename=filename)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars("pdf_filename")
