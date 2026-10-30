# LFM3 Neuron Explorer

Interactive activation inspection and steering GUI for [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B).

A local-first research and educational tool: explore the model's internal activations and experimentally steer its behavior during inference — without ever modifying model weights.

## What it does

- **Architecture explorer** — hierarchical view of the loaded model's real modules and layers (decoder layers, attention, short-convolution operators, MLPs), discovered from the live model, not hardcoded.
- **Activation visualization** — per-layer heatmaps, distributions, top dimensions, and per-dimension series for captured activations, with token alignment to the prompt and generated text.
- **Steering and generation** — baseline vs. steered generation using forward hooks; editable prompt, steering targets, and strengths.

Steering is inference-only via forward hooks. Hooks are installed per experiment and removed in `finally`; zero strength exactly reproduces the baseline. Captured activations are held in an ephemeral in-memory store (oldest evicted, not persisted).

## Architecture

| Layer     | Stack                                   |
|-----------|-----------------------------------------|
| Backend   | Python 3.13, FastAPI, PyTorch, Transformers |
| Frontend  | React, TypeScript, Vite                 |
| Model     | `LiquidAI/LFM3-1.7B` via `trust_remote_code=True` |

The backend is the sole authority on tensor shapes, module names, and results; the frontend never fabricates model data. When `frontend/dist` exists, FastAPI serves the built frontend at `/`.

Development is phased (see `PLAN.md`); current status covers baseline generation and activation capture/inspection.

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

Host/port live in `backend/app/core/settings.py` and are env-overridable.

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

## Notes

- First generation includes ~5–6 s of ROCm/SDPA warmup; later tokens are faster.
- On ROCm, benign stderr noise (`(null): No such file or directory`) may appear at import.
- The ROCm torch stack is pinned in `pyproject.toml` via the `pytorch-rocm` uv index; keep those as declared direct deps.
- Don't report a steering result from a single changed response — behavioral claims need experimental evidence.

## Repository layout

```
backend/
  app/            FastAPI app, settings, schemas, model manager, vendors
  tests/          pytest suite (synthetic models + optional real integration)
frontend/
  src/            React/TypeScript three-panel workspace
  dist/           built assets served by FastAPI
PLAN.md           full design spec and phased plan
AGENTS.md         development conventions and verified environment facts
```
