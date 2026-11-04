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

Development is phased (see `PLAN.md`); Phases 1–7 are implemented: model exploration, baseline generation, activation capture/visualization, steering, experiment history/sweeps/export, validation, and documentation. Tested features are listed at the end of this file.

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
npm run typecheck # tsc --noEmit
npm test          # Vitest + Testing Library component/integration tests
npm run build     # typecheck + build -> frontend/dist (served by FastAPI)
```

### Example experiment

1. Start the backend (`uv run uvicorn backend.app.main:app`) and frontend (`npm run dev`), open `http://localhost:5173`, and click **Load model**.
2. In the architecture explorer, open `model.layers.22.feed_forward` and select it. The inspector shows its real class and parameter shapes, and offers **Set as steering target**.
3. In the generation panel, enable **Capture activations**, keep greedy decoding (`temperature = 0`), keep the **Baseline** mode, and click **Run baseline**. The center panel now shows the measured `[tokens, 2048]` tensor with a heatmap, distribution, and top dimensions.
4. Click **Set as steering target** in the inspector, switch to **Steered**, pick a hidden dimension (for example 500), set α to a negative value (for example −12), and click **Run baseline + steered**. Both runs share the same prompt, settings, and seed; the center panel shows the labeled `steered − baseline` difference.
5. Optionally switch to **Sweep** and run several strengths (for example `-20,-10,0,10,20`) with a seeded random-feature control. Every run appears in the experiment history, where you can inspect, compare, reopen persisted activations, and export JSONL/CSV.

Zero strength must reproduce the baseline exactly. A single changed response is not evidence that a dimension controls a behavior — use repeated prompts, multiple strengths, and controls before drawing conclusions.

### Real-model integration (optional)

The default test suite uses small synthetic `Lfm3Config` models from vendored code and never downloads weights. To verify against the real checkpoint:

```bash
LFM3_RUN_INTEGRATION=1 uv run pytest backend/tests/test_integration_lfm3.py
```

Weights are cached in `~/.cache/huggingface`.

### Performance measurement

```bash
uv run python -m backend.scripts.benchmark_model --max-new-tokens 32
uv run python -m backend.scripts.benchmark_model --device cpu --max-new-tokens 8
```

Measured on the reference machine (RX 7600, ROCm 7.2, BF16, greedy, `enable_thinking=False`):

| Run | Latency | Rate |
|-----|---------|------|
| baseline, 8 tokens | 0.30 s | ~27 tok/s |
| baseline, 32 requested (20 generated before EOS) | 0.73 s | ~27 tok/s |
| baseline + activation capture | 0.73 s | ~27 tok/s |
| steered α=−12 + capture | 0.77 s | ~27 tok/s |

Model load takes ~6.5–9 s once cached; peak allocated VRAM is 3.49 GB (~3.53 GB reserved). Enabling `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` gave no measurable gain (~28 vs ~27 tok/s), so it is not required.

## API overview

Core endpoints (all under `/api`):

- `GET /health`, `GET /diagnostics`, `GET/POST /model/*`
- `POST /experiments/baseline`, `POST /experiments/steering`, `POST /experiments/sweep`
- `GET /experiments` (filters: `condition`, `sweep_id`, `module_id`, `limit`, `offset`)
- `GET /experiments/{id}`, `GET /experiments/{id}/activations[/{module_id}]`
- `GET /experiments/export?format=jsonl|csv&ids=...` (`ids` optional; empty exports all)

## Notes and limitations

- First generation includes ~5–6 s of ROCm/SDPA warmup; steady-state decoding is ~27 tok/s on the reference RX 7600. The benchmark script warms up before timing.
- On ROCm, benign stderr noise (`(null): No such file or directory`) may appear at import.
- The ROCm torch stack is pinned in `pyproject.toml` via the `pytorch-rocm` uv index; keep those as declared direct deps.
- Sweeps run sequentially and can take minutes; there is no cancellation/streaming progress yet.
- Non-persisted activations disappear when evicted (8 experiments / 256 MB) or when the backend restarts. History keeps the metadata and marks the tensors unavailable; enable snapshot persistence to inspect them later.
- CSV export prefixes spreadsheet formula leaders (`=`, `+`, `@`) in text cells to prevent CSV injection. Numeric columns are exported raw.
- Don't report a steering result from a single changed response — behavioral claims need experimental evidence.

## Troubleshooting

- **`torch` or `transformers` import failures / wrong Python**: use `uv run …` from the repository root. The project pins Python 3.13; the system Python (≥ 3.14) has no compatible torch wheels.
- **Model load fails with a device error**: the default is `auto` (`cuda` when ROCm/CUDA is visible, otherwise CPU). Force CPU with `LFM3_DEVICE=cpu`. If the checkpoint cannot be fetched, confirm network/HF access and that `~/.cache/huggingface` is writable.
- **GPU out of memory**: lower `max_new_tokens`, shorten the prompt, and capture fewer modules. The reference 8 GB card uses 3.49 GB for weights/activations at the tested settings.
- **`(null): No such file or directory` on stderr**: benign ROCm import noise; it does not affect inference.
- **Activation route returns 404**: the run captured no tensors, the tensor was evicted (8 experiments / 256 MB), or the backend restarted without snapshot persistence. Re-run with capture enabled and optionally enable **Persist activation snapshot**.
- **Port already in use**: set `LFM3_PORT` for the backend and update the Vite proxy if needed; Vite uses 5173 by default.
- **Frontend shows API errors in the browser but the backend works**: build the frontend (`npm run build`) when serving through FastAPI, or run `npm run dev` so Vite proxies `/api` to the backend.
- **Tests cannot find weights**: the default suites are offline by design; only the optional integration test needs the cached checkpoint.

## Tested features

Verified on the reference machine unless noted:

- Real 24-layer LFM3-1.7B architecture discovery (16 attention/conv/MLP layer layout, 2048 hidden size).
- Baseline chat generation with the model's chat template and `enable_thinking=False`.
- Activation capture for selected hidden-state modules with token alignment, heatmap data, histogram, stats, top dimensions, and feature series.
- Single-dimension steering with positive/negative strengths, exact zero-strength baseline reproduction, and guaranteed hook removal.
- Baseline vs. steered pair runs, sweep runs with repeats, and seeded random-feature controls.
- JSONL experiment persistence across restarts, list/filter/pagination, detail/compare views, and JSONL/CSV export (including formula-injection protection).
- Optional activation snapshot persistence and restoration within the byte budget.
- Backend suite: 74 passed, 3 skipped offline; 3 passed against the real checkpoint. Frontend suite: 65 tests across 7 files; production build passes.
- End-to-end HTTP smoke test: FastAPI served the built frontend, loaded the real model, returned the architecture, ran a captured baseline, and read it back from history.

## Repository layout

```
backend/
  app/            FastAPI app, settings, schemas, model manager, vendors
  scripts/        benchmark_model.py (real-model latency/VRAM measurement)
  tests/          pytest suite (synthetic models + optional real integration)
frontend/
  src/            React/TypeScript three-panel workspace + history/sweep panels
  src/__tests__/  Vitest component/integration tests
  dist/           built assets served by FastAPI
data/             experiment JSONL and optional activation snapshots (gitignored)
PLAN.md           full design spec and phased plan
AGENTS.md         development conventions and verified environment facts
```
