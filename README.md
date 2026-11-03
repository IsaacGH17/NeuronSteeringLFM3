# LFM3 Neuron Explorer

Interactive activation inspection and steering GUI for [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B).

A local-first research and educational tool: explore the model's internal activations and experimentally steer its behavior during inference — without ever modifying model weights.

## What it does

- **Architecture explorer** — hierarchical view of the loaded model's real modules and layers (decoder layers, attention, short-convolution operators, MLPs), discovered from the live model, not hardcoded.
- **Activation visualization** — per-layer heatmaps, distributions, top dimensions, and per-dimension series for captured activations, with token alignment to the prompt and generated text.
- **Steering and generation** — baseline vs. steered generation using forward hooks; editable prompt, steering targets, and strengths.
- **Multi-strength sweeps and controls** — run several strengths against one baseline, optionally adding a seeded random-feature control that matches the intervention magnitude.
- **Experiment history and export** — every run is saved as a JSONL metadata record; inspect, compare, reopen, and export selected or all runs as JSONL/CSV. Activation tensors stay ephemeral unless snapshot persistence is enabled.

Steering is inference-only via forward hooks. Hooks are installed per experiment and removed in `finally`; zero strength exactly reproduces the baseline. Experiment metadata survives backend restarts in `data/experiments.jsonl`; activation tensors are held in an in-memory store (oldest evicted) and are only written to `data/activations/` when persistence is explicitly enabled.

## Architecture

| Layer     | Stack                                   |
|-----------|-----------------------------------------|
| Backend   | Python 3.13, FastAPI, PyTorch, Transformers |
| Frontend  | React, TypeScript, Vite                 |
| Model     | `LiquidAI/LFM3-1.7B` via `trust_remote_code=True` |

The backend is the sole authority on tensor shapes, module names, and results; the frontend never fabricates model data. When `frontend/dist` exists, FastAPI serves the built frontend at `/`.

Development is phased (see `PLAN.md`); Phases 1–5 are implemented: model exploration, baseline generation, activation capture/visualization, steering, and experiment history/sweeps/export.

## Requirements

- Linux, Python 3.13 (`uv`-managed; system Python ≥ 3.14 is not supported by torch wheels)
- ~8 GB VRAM for GPU inference (developed on an AMD RX 7600 via ROCm 7.2 / PyTorch ROCm wheels); CPU fallback available
- The LFM3 checkpoint is **not** merged into `transformers`; the repo vendors the model code (`backend/app/vendor/`) and loads with `trust_remote_code=True`

## Getting started

### Backend

```bash
uv sync                            # create/update .venv
uv run pytest                      # tests pass without downloading model weights
uv run ruff check .                # lint
uv run uvicorn backend.app.main:app --reload
```

Host/port live in `backend/app/core/settings.py` and are env-overridable. Phase 5 settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LFM3_DATA_DIR` | `./data` | History JSONL and optional activation snapshots |
| `LFM3_PERSIST_ACTIVATIONS` | `0` | Server default for saving activation `.npz` snapshots |
| `LFM3_MAX_STEERING_STRENGTH` | `50` | Safety limit for steering/sweep strengths |

The history file is append-only JSONL. Requests may set `persist_activations` explicitly; the UI checkbox sends an explicit boolean, while API clients that omit it use the server default.

### Frontend

```bash
cd frontend
npm install
npm run dev       # Vite dev server, proxies /api to 127.0.0.1:8000
npm run build     # typecheck + build -> frontend/dist (served by FastAPI)
```

### Real-model integration (optional)

The default test suite uses small synthetic `Lfm3Config` models from vendored code and never downloads weights. To verify against the real checkpoint:

```bash
LFM3_RUN_INTEGRATION=1 uv run pytest backend/tests/test_integration_lfm3.py
```

Weights are cached in `~/.cache/huggingface`.

## API overview

Core endpoints (all under `/api`):

- `GET /health`, `GET /diagnostics`, `GET/POST /model/*`
- `POST /experiments/baseline`, `POST /experiments/steering`, `POST /experiments/sweep`
- `GET /experiments` (filters: `condition`, `sweep_id`, `module_id`, `limit`, `offset`)
- `GET /experiments/{id}`, `GET /experiments/{id}/activations[/{module_id}]`
- `GET /experiments/export?format=jsonl|csv&ids=...` (`ids` optional; empty exports all)

## Notes and limitations

- First generation includes ~5–6 s of ROCm/SDPA warmup; later tokens are faster.
- On ROCm, benign stderr noise (`(null): No such file or directory`) may appear at import.
- The ROCm torch stack is pinned in `pyproject.toml` via the `pytorch-rocm` uv index; keep those as declared direct deps.
- Sweeps run sequentially and can take minutes; there is no cancellation/streaming progress yet.
- Non-persisted activations disappear when evicted (8 experiments / 256 MB) or when the backend restarts. History keeps the metadata and marks the tensors unavailable; enable snapshot persistence to inspect them later.
- CSV export prefixes spreadsheet formula leaders (`=`, `+`, `@`) in text cells to prevent CSV injection. Numeric columns are exported raw.
- Don't report a steering result from a single changed response — behavioral claims need experimental evidence.

## Repository layout

```
backend/
  app/            FastAPI app, settings, schemas, model manager, vendors
  tests/          pytest suite (synthetic models + optional real integration)
frontend/
  src/            React/TypeScript three-panel workspace + history/sweep panels
  dist/           built assets served by FastAPI
data/             experiment JSONL and optional activation snapshots (gitignored)
PLAN.md           full design spec and phased plan
AGENTS.md         development conventions and verified environment facts
```
