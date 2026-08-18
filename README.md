# PCB OCR Pipeline

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
│     raw text      structured     structured      text layer*      │
│                      JSON           JSON                          │
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

Every uploaded PDF is unconditionally rasterized to page images (`OCR_DPI`, default 300) before Tesseract, GLM-OCR, or Qwen-VL run — this happens regardless of whether the PDF has an extractable text layer, since none of those three read the PDF's text layer directly (Tesseract-the-tool is purely an image-OCR engine; it has no capability to read PDF text metadata at all). PyMuPDF is the one exception: it reads the PDF's embedded text objects directly, with no rasterization step, and no OCR-induced character errors when that text layer exists.

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
| **GLM-OCR** | Dense text regions, small fonts | Slower, needs GPU |
| **Qwen3-VL** | Layout understanding, tables, shape symbols | Slower, needs GPU |
| **PyMuPDF** | Exact-character extraction (no OCR errors) when the PDF has a real text layer; near-instant, no GPU | Only works on "born-digital" PDFs — contributes nothing for scans or CAD exports with text converted to vector outlines (common in production) |

Reconciliation uses confidence-weighted majority voting to resolve conflicts (see [Multi-engine reconciliation](#multi-engine-reconciliation) below).

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
  -F "file=@samples/sample1.pdf" \
  | python3 -m json.tool
```

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

## Project Structure

```
pcb-ocr/
├── docker-compose.yml        # Full 6-service topology
├── .dockerignore             # Build exclusions
├── Makefile                  # Target shortcuts
├── prd/                      # Product requirements
│   ├── SYSTEM_PROMPT.md
│   ├── IMPLEMENTATION_SPEC.md
│   └── EXPECTED_OUTPUTS.md
├── samples/                  # Input PDFs
│   ├── sample1.pdf           # 4-layer flex, ITAR
│   ├── sample2.pdf           # 5-layer FR4
│   ├── sample3..pdf          # 8-layer 370HR (note: double-dot!)
│   └── sample4.pdf           # 10-layer PCB Prime
├── shared/                   # Common modules
│   ├── schemas.py            # Pydantic PCBData models
│   ├── preprocessing.py      # PDF→image, enhancement, ROI
│   ├── pdf_text.py           # PDF embedded text-layer extraction (PyMuPDF)
│   ├── pcb_parser.py         # Heuristic text→PCBData regex parser
│   ├── logging_config.py     # Per-PDF /logs file sink (structlog)
│   ├── confidence.py         # Confidence scoring
│   └── constants.py          # IPC standards, materials, regex
├── services/
│   ├── tesseract/            # Tesseract OCR service
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/              # FastAPI app
│   ├── glm_ocr/              # GLM-OCR service
│   ├── qwen_vl/              # Qwen3-VL service
│   └── supervisor/           # LangGraph orchestrator
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── main.py       # FastAPI app entrypoint
│           ├── router.py     # POST /extract handler
│           ├── graph.py      # LangGraph state graph
│           ├── nodes.py      # Node implementations
│           └── state.py      # Pipeline state definition
├── tests/
│   ├── expected/             # Ground truth JSON
│   ├── output/               # Actual response JSON from the last e2e run
│   ├── conftest.py
│   ├── test_e2e.py           # End-to-end tests
│   ├── test_reconciliation.py
│   ├── test_schema_validation.py
│   ├── test_pcb_parser.py
│   └── test_pdf_text.py
└── scripts/
    ├── start_services.sh     # Start + health polling
    ├── stop_services.sh      # Stop all
    ├── run_tests.sh          # Full test suite
    ├── local_test.py         # Tesseract-only validation
    ├── local_pipeline.py     # Tesseract + host vLLM, bypasses Docker
    ├── convert_expected.py   # txt→JSON converter
    ├── cache_ocr.py          # Cache Tesseract OCR output for faster iteration
    ├── health_check.py       # Service health checker
    └── benchmark.py          # Accuracy + timing benchmark
```

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
  "confidence": { ... },
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
| `OCR_DPI` | `300` | PDF rasterization DPI |
| `TESSERACT_URL` | `http://tesseract-ocr:8001/extract` | Tesseract service URL |
| `GLM_OCR_URL` | `http://glm-ocr-api:8002/extract` | GLM-OCR service URL |
| `QWEN_VL_URL` | `http://qwen-vl-api:8003/extract` | Qwen-VL service URL |
| `GLM_OCR_VLLM_URL` | `http://vllm-glm-ocr:8010/v1` | GLM-OCR vLLM endpoint |
| `QWEN_VL_VLLM_URL` | `http://vllm-qwen-vl:8011/v1` | Qwen-VL vLLM endpoint |
| `PDF_TEXT_LAYER_MIN_CHARS` | `200` | Min. alphanumeric characters in the PDF's embedded text layer before PyMuPDF contributes a result (below this, it's treated as a scan/no text layer and skipped) |
| `LOG_DIR` | `/logs` | Where per-PDF log files are written (see below) |

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