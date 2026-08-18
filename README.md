# PCB OCR Agentic AI Pipeline

Multi-model OCR pipeline for extracting structured data from PCB fabrication drawings (PDF).

4 independent OCR/extraction engines run in parallel, then a LangGraph-supervised reconciliation step merges results into a clean `PCBData` Pydantic schema.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        Client (PDF upload)                        │
└──────────────────────────────┬────────────────────────────────────┘
                               │ POST /extract
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                Supervisor (LangGraph Orchestrator)                │
│                             Port 8080                             │
│                                                                   │
│ ┌──────────────┬──────────────┬──────────────┬────────────────┐   │
│ │Tesseract OCR │ GLM-OCR API  │   Qwen3-VL   │    PyMuPDF     │   │
│ │ (port 8001)  │ (port 8002)  │ (port 8003)  │  (in-process)  │   │
│ └──────────────┴──────────────┴──────────────┴────────────────┘   │
│         ▼              ▼              ▼               ▼           │
│     raw text       raw text      structured      text layer*      │
│    (+ parser)      (+ parser)       JSON                          │
│         ┴──────────────┴───────┬──────┴───────────────┴           │
│                                ▼                                  │
│            ┌────────────────────────────────────────┐             │
│            │        Reconciliation (merge +         │             │
│            │          confidence scoring +          │             │
│            │            majority voting)            │             │
│            └────────────────────────────────────────┘             │
│                                ▼                                  │
│                       ┌──────────────────┐                        │
│                       │  PCBData (JSON)  │              ← response│
│                       └──────────────────┘                        │
└───────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │ vLLM GLM-OCR │    │vLLM Qwen3-VL │
            │ (port 8010)  │    │ (port 8011)  │
            │     GPU      │    │     GPU      │
            └──────────────┘    └──────────────┘
```
\* PyMuPDF only contributes when the PDF has a substantial embedded text layer — see below.

### Rasterization

Every uploaded PDF is unconditionally rasterized to page images (`OCR_DPI`, default 300) before Tesseract, GLM-OCR, or Qwen-VL run — this happens regardless of whether the PDF has an extractable text layer, since none of those three read the PDF's text layer directly (Tesseract-the-tool is purely an image-OCR engine; it has no capability to read PDF text metadata at all). PyMuPDF is the one exception: it reads the PDF's embedded text objects directly, with no rasterization step, and no OCR-induced character errors when that text layer exists.

Mechanically, this is `shared/preprocessing.py::pdf_to_images` — `pdf2image.convert_from_path(pdf_path, dpi=OCR_DPI, fmt="png")`, wrapping Poppler's `pdftoppm`. It runs independently inside each of the three services (Tesseract, GLM-OCR, Qwen-VL), once per request — the same PDF gets rasterized three separate times at the same DPI, not shared across services, since each is a separate container/process. Each PDF page becomes one PIL `Image` in a list, one call producing all pages in one pass.

From there the three services diverge in what they do with the rasterized page:

- **Tesseract** runs an additional per-page adaptive preprocessing pass before OCR (`services/tesseract/app/ocr_engine.py`): classify the page as VECTOR / SCAN / LOW_CONTRAST / MIXED (via Laplacian sharpness, histogram bimodality, edge density), deskew if the page is rotated more than 0.5°, then apply a page-type-specific denoise + threshold (Otsu for scans, adaptive Gaussian threshold for vector/mixed) to binarize the image before handing it to `pytesseract`. This exists because Tesseract's LSTM engine is sensitive to noise and contrast in a way VLMs aren't.
- **GLM-OCR and Qwen-VL** send the plain rasterized PNG straight to their vLLM endpoint, base64-encoded, with no binarization — vision-language models are trained on natural images and generally perform *worse* on aggressively thresholded/binarized input, so Tesseract's enhancement pipeline is deliberately not applied here.
- `shared/preprocessing.py` also contains adaptive-DPI (`pdf_to_images_adaptive`) and ROI-detection (`detect_regions`) helpers that aren't wired into any current call path — dead code kept from an earlier design, not part of the live pipeline.

### Services

| Service | Port | Description |
|---------|------|-------------|
| **vllm-glm-ocr** | 8010 | vLLM server running `zai-org/GLM-OCR` (GPU) |
| **vllm-qwen-vl** | 8011 | vLLM server running `Qwen/Qwen3-VL-8B-Instruct` (GPU) |
| **tesseract-ocr** | 8001 | Tesseract 5 + heuristic PCB parser |
| **glm-ocr-api** | 8002 | FastAPI wrapper for GLM-OCR model |
| **qwen-vl-api** | 8003 | FastAPI wrapper for Qwen3-VL model |
| **supervisor** | 8080 | LangGraph orchestrator + reconciliation + PyMuPDF text-layer extraction (runs in-process, not a separate service) |

### OCR Engine Responsibilities

| Engine | Strengths | Weaknesses |
|--------|-----------|------------|
| **Tesseract** | Raw text extraction, note parsing | Can't read shape symbols, table borders |
| **GLM-OCR** | Dense text regions, small fonts | Slower, needs GPU; JSON reasoning over its own text is weak (see below) — used for OCR only |
| **Qwen3-VL** | Layout understanding, tables, shape symbols; reliable single-shot image→JSON | Slower, needs GPU |
| **PyMuPDF** | Exact-character extraction (no OCR errors) when the PDF has a real text layer; near-instant, no GPU | Only works on "born-digital" PDFs — contributes nothing for scans or CAD exports with text converted to vector outlines (common in production) |

Reconciliation uses confidence-weighted majority voting to resolve conflicts (see [Multi-engine reconciliation](#multi-engine-reconciliation) below).

**GLM-OCR's role is OCR only, not structured extraction.** GLM-OCR (`zai-org/GLM-OCR`) is a document-vision model — strong at raw text recognition, but per its own model card its "Information Extraction" mode is tuned for flat template documents (ID cards, invoices), not deeply nested engineering-drawing schemas. Verified directly against the model: given this project's full PCBData schema it returned empty strings for nearly every field; even with a minimal prompt copied from the model card's own example, it returned real values but misattributed to the wrong fields (e.g. a soldermask color reported as `surface_finish`). So this pipeline uses GLM-OCR only for its raw OCR pass (`extract_text`) — its text then goes through the same regex parser (`shared/pcb_parser.py`) that Tesseract and PyMuPDF use, rather than trusting the model's own JSON output. Qwen3-VL's single-shot image→JSON extraction was tested the same way and does not show this failure mode, so it keeps doing structured extraction directly.

## Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU with CUDA (for vLLM containers)
- NVIDIA Container Toolkit installed

Tested on an **NVIDIA DGX Spark** (GB10 Grace Blackwell Superchip, ~128GB unified CPU/GPU memory, ARM64). Being a unified-memory devkit rather than a discrete-VRAM datacenter GPU, it's noticeably slower for these model sizes than typical cloud GPUs — expect Qwen3-VL calls to take 90–360s per page. Both vLLM servers share the single GPU, so the `--gpu-memory-utilization` and `--max-model-len` values in `docker-compose.yml` are tuned for that constraint; adjust them if running on different hardware.

### Start the pipeline

```bash
# Build and start all services (polls health until ready)
make up

# Or manually:
bash scripts/start_services.sh
```

### Stop the pipeline

```bash
make down
# Or: bash scripts/stop_services.sh
```

### Check health

```bash
make health
```

### Extract a PCB drawing

```bash
curl -X POST http://localhost:8080/extract \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@samples/sample1.pdf" \
  | python3 -m json.tool
```

Every service's `POST /extract` requires this header when `API_KEY` is set (see [Authentication](#authentication) below); `/health` and `/ready` stay open for Docker healthchecks.

Response: `PCBData` JSON with all extracted fields + confidence scores.

### Run tests

```bash
# Full e2e tests (requires running services)
make test

# Local Tesseract-only tests (no Docker needed)
make test-local

# Unit tests
PYTHONPATH=. pytest tests/ -v
```

**What `make test` actually does** (`scripts/run_tests.sh`): checks `SUPERVISOR_HEALTH` (`http://localhost:8080/health`) is up — failing fast with a clear error if `make up` hasn't been run — then runs `PYTHONPATH=. pytest tests/ -v --timeout=1100`, then runs `scripts/local_test.py` as a final step. It's the union of the e2e path and the local path below, so a `make test` failure can come from either.

- **e2e extraction test** (`tests/test_e2e.py::test_extraction_accuracy`, parametrized once per `samples/*.pdf`): `POST`s the PDF to the live `supervisor` container's `/extract` endpoint — exercising the full real pipeline (rasterization → all 4 engines → reconciliation → validation), not a mock. The raw JSON response is written to `tests/output/{sample}.json` (this is what backs every "look at the live attribution/reconciliation output" check earlier in this session) before any assertions run, so a failing test still leaves the actual response on disk to inspect. `_check_accuracy` then deserializes it into `PCBData` and field-by-field compares it against the corresponding `tests/expected/{sample}.json` fixture, collecting every mismatch (not stopping at the first) into one combined failure message.
- **local extraction test** (`scripts/local_test.py`, also runnable standalone via `make test-local`): bypasses Docker and the supervisor entirely — imports `TesseractOCR` and `shared.pcb_parser.parse_pcb_text` directly in-process, so it only exercises Tesseract + the regex parser, never the VLMs or reconciliation. Same `tests/expected/*.json` fixtures, but with deliberately lenient checks on fields Tesseract genuinely cannot read from certain scans (e.g. `board_thickness`, `drill_table`, `layer_count`, `surface_finish` on specific samples) — it only validates the correctness of values Tesseract *did* find, never requires a value it has no way to detect. This is what you run for a fast (~seconds, no GPU, no Docker) sanity check while iterating on `shared/pcb_parser.py`.
- **unit tests** (everything else under `tests/`, e.g. `test_pcb_parser.py`, `test_schema_validation.py`, `test_reconciliation.py`, `test_pdf_text.py`): pure Python, no network calls, run in milliseconds.

## API Reference

### Supervisor: `POST /extract`

Upload a PCB fabrication PDF and receive structured `PCBData` JSON.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (PDF)

**Response:**
```json
{
  "layer_count": 4,
  "material": "FR4",
  "board_thickness": { "nominal": 0.062, "plus_tol": 0.005, "minus_tol": 0.005, "unit": "in" },
  "is_itar": true,
  "ipc_specs": ["IPC-2221", "IPC-6012"],
  "surface_finish": "ENIG",
  "drill_table": [ { "size_mils": 18.0, "qty": 270, "plated": true } ],
  "solder_mask": { "color": "GREEN", "type": "photo imageable" },
  "impedance_control": { "controlled": true, "single_ended": { "min": 90, "max": 90 } },
  "attribution": { "...": "per-field provenance — see below" },
  "reconciliation_log": [ "..." ],
  "errors": [],
  "engine_durations_sec": [
    { "engine": "tesseract", "duration_sec": 3.1, "start_time": "2026-01-01T00:00:00-08:00", "end_time": "2026-01-01T00:00:03-08:00" },
    { "engine": "glm-ocr", "duration_sec": 40.3, "start_time": "...", "end_time": "..." },
    { "engine": "qwen-vl", "duration_sec": 91.1, "start_time": "...", "end_time": "..." },
    { "engine": "pymupdf", "duration_sec": 0.04, "start_time": "...", "end_time": "..." }
  ],
  "total_duration_sec": 91.7
}
```

`engine_durations_sec[].start_time`/`end_time` are ISO-8601 timestamps in US Pacific time (`America/Los_Angeles`, so `-08:00`/PST or `-07:00`/PDT depending on the date) — not UTC.

### Per-field attribution

Every field in the response has a matching entry under `attribution`, answering *where this value came from* — including the literal drawing text behind it:

```json
"attribution": {
  "surface_finish": {
    "value": "Tin/Lead",
    "confidence": 0.56,
    "reason": "Only source (tesseract), confidence penalized to 0.8x",
    "engine": "tesseract",
    "source_text": "... SHALL BE TIN/LEAD PLATED .0003 TO .0005 THK.",
    "source_text_from": "tesseract",
    "contributors": [
      { "engine": "tesseract", "value": "Tin/Lead", "confidence": 0.7, "source_text": "... TIN/LEAD PLATED ..." }
    ]
  }
}
```

| Key | Meaning |
|---|---|
| `value` / `confidence` / `reason` | The merged value, its final confidence, and which voting rule selected it |
| `engine` | Which engine's value won |
| `source_text` | The line of drawing text that produced the value, when attributable |
| `source_text_from` | Which engine supplied that text (may differ from `engine` — see below) |
| `contributors` | Every engine that produced a value for this field, with its own value and evidence — lets you audit a disagreement without re-running the pipeline |

**When `source_text` is null:** the regex-based engines (Tesseract, PyMuPDF) can cite the exact line they matched. The vision models generally cannot — they emit structured JSON without reporting which pixels produced it. If the winning engine is a VLM but another engine *agreed on the same value*, that agreeing engine's evidence is used and `source_text_from` names it. Evidence is never borrowed from an engine that disagreed.

**Attribution is not available for values that weren't read directly** — e.g. a field recovered by the OCR-artifact post-processing rules. Those entries have a null `source_text` rather than a misleading one.

### Service Health Endpoints

| Service | Health URL |
|---------|-----------|
| Tesseract | `http://localhost:8001/health` |
| GLM-OCR | `http://localhost:8002/health` |
| Qwen-VL | `http://localhost:8003/health` |
| Supervisor | `http://localhost:8080/health` |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `API_KEY` | *(unset)* | Bearer token required on every service's `POST /extract` — see [Authentication](#authentication) |
| `OCR_DPI` | `300` | PDF rasterization DPI |
| `TESSERACT_URL` | `http://tesseract-ocr:8001/extract` | Tesseract service URL |
| `GLM_OCR_URL` | `http://glm-ocr-api:8002/extract` | GLM-OCR service URL |
| `QWEN_VL_URL` | `http://qwen-vl-api:8003/extract` | Qwen-VL service URL |
| `GLM_OCR_VLLM_URL` | `http://vllm-glm-ocr:8010/v1` | GLM-OCR vLLM endpoint |
| `QWEN_VL_VLLM_URL` | `http://vllm-qwen-vl:8011/v1` | Qwen-VL vLLM endpoint |
| `PDF_TEXT_LAYER_MIN_CHARS` | `200` | Min. alphanumeric characters in the PDF's embedded text layer before PyMuPDF contributes a result (below this, it's treated as a scan/no text layer and skipped) |
| `LOG_DIR` | `/logs` | Where per-PDF log files are written (see below) |

### Authentication

Set `API_KEY` in `.env` to require an `Authorization: Bearer <API_KEY>` header on every service's `POST /extract` (`shared/auth.py`). Each of the 4 services (`supervisor`, `tesseract-ocr`, `glm-ocr-api`, `qwen-vl-api`) checks it independently — there's no shared auth service or session. `/health` and `/ready` are never gated, since Docker's healthchecks and `scripts/health_check.py` don't send custom headers.

The supervisor forwards the same key on its own internal calls to the 3 downstream services, so the whole pipeline keeps working end-to-end once a key is set — no separate internal credential to manage.

Leaving `API_KEY` unset disables auth entirely (every request is accepted) — the default for local dev. All 3 host-tool entry points that call the live API directly source `.env` before running, so the key travels automatically once set: `scripts/run_tests.sh` (used by `make test`), the `benchmark` Makefile target, and `tests/conftest.py` (used by `pytest tests/test_e2e.py`).

## Per-PDF logs

Every request writes a consolidated log file to `/logs/{original filename}.log` (e.g. `/logs/sample1.pdf.log`), one JSON line per event, covering the *entire* pipeline for that file — not just whichever single service happens to log it. The supervisor's own orchestration events (ingest, fan-out, reconciliation, validation) and each downstream engine's internal events (Tesseract's OCR/parsing steps, GLM-OCR's/Qwen-VL's extraction calls, PyMuPDF's skip/extract decision) all land in the same file, in call order, timestamped.

This is distinct from `make logs` (`docker compose logs -f`), which streams raw, interleaved stdout from all containers — useful for watching everything live, but not grouped by PDF.

This works because:
- All four services (supervisor, tesseract-ocr, glm-ocr-api, qwen-vl-api) share a `./logs:/logs` bind mount — logs land directly in the project's `logs/` directory on the host, not inside a Docker-managed volume, so you can browse/tail them normally.
- The supervisor forwards the client's original filename to each downstream engine (not the temp file's random name), so every service groups its logs under the same key.
- `structlog.contextvars` binds the filename for the duration of each request — async-safe, so concurrent requests for different PDFs never cross-contaminate each other's log files.

```bash
# Tail a specific PDF's full-pipeline log directly on the host
tail -f logs/sample1.pdf.log
```

The `logs/` directory is created automatically (owned by root, since containers run as root) the first time you `docker compose up` after this change — the log files themselves are world-readable, so browsing/tailing from the host doesn't need sudo.

## Multi-engine reconciliation

Each PDF is run through up to four independent OCR/extraction engines — Tesseract, GLM-OCR, Qwen-VL, and (conditionally) PyMuPDF — and their outputs are merged field-by-field using a confidence-weighted voting scheme, not a simple average. PyMuPDF only participates when the PDF has a substantial embedded text layer (`PDF_TEXT_LAYER_MIN_CHARS`); otherwise it contributes nothing and reconciliation runs on the other three, exactly as before. The voting logic itself is written generically over however many engines actually produced a result for a given field:

1. **All engines that ran agree** on a value → use it, with confidence boosted to `max(confidences) × 1.1` (capped at 1.0).
2. **A strict majority agree** (more than half of the engines that ran — e.g. 2 of 3, or 3 of 4) → use the majority value, with confidence averaged across just the agreeing engines (the outlier(s) excluded).
3. **No group has a strict majority** (all engines disagree, or a tie — e.g. 2 of 4 with the other two also disagreeing) → use the single highest-confidence engine's value; the rest are discarded.
4. **Only 1 engine detected the field** → use it, with confidence penalized ×0.8.
5. **No engine detected it** → value is `null`, confidence `0.0`.

Each engine's own per-field confidence is computed independently beforehand (from a base score per engine/field type, adjusted by signals like Tesseract's OCR word-confidence or JSON-parse success) before this voting logic runs. PyMuPDF's base confidences run higher than any OCR engine's wherever it has data — it reads exact characters from the PDF's own text objects, with no OCR-induced misreads — except for symbol/table-heavy fields (`drill_table`, `layer_stackup`, `impedance_control`), where it shares Tesseract's blind spot: shape symbols and spatial table layout aren't recoverable from raw extracted text.

**Special cases:**

- `is_itar`: if any engine flags ITAR, the merged result is `True` regardless of the others, using the max confidence among the engines that flagged it.
- `ipc_specs`: unioned (deduplicated) across all engines that ran rather than voted on.
- `fabrication_notes`: the longest non-null value across engines is used.
- `drill_table` / `layer_stackup`: table fields go through the same voting logic as scalar fields — one engine's whole table is selected, not merged row-by-row across engines. A drill table with 3+ rows where every row reports the identical size is treated as a degenerate/hallucinated extraction and discarded before voting.


## Troubleshooting

**vLLM containers fail to start:**
- Check GPU memory: `nvidia-smi` (on unified-memory boards like DGX Spark, the `--query-gpu=memory.*` fields can report `[N/A]` — use plain `nvidia-smi` or `free -h` instead)
- Verify NVIDIA Container Toolkit is installed
- GLM-OCR and Qwen3-VL each need ~12GB VRAM

**Tesseract returns low-quality text:**
- Increase `OCR_DPI` to 400 or 600
- Check poppler-utils is installed: `pdftoppm -v`