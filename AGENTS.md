# AGENTS.md — NueronSteerLFM3

Interactive GUI to inspect and steer activations of `LiquidAI/LFM3-1.7B` (PyTorch + Transformers). Python/FastAPI backend + React/TypeScript/Vite frontend. Spec = `PLAN.md`; read it before working. Research (Phase 1, §18) is done — build with the facts below.

## Commands (verified)

- `uv sync` — create/update `.venv` (Python 3.13.15 via uv-managed CPython)
- `uv run pytest` — run tests (must pass without downloading LFM3 weights)
- `uv run ruff check .` / `uv run ruff format .` — lint/format
- Run anything: `uv run python -m ...` (never call system python; it has no torch)
- Backend dev: `uv run uvicorn backend.app.main:app --reload` (host/port from `backend/app/core/settings.py`, env-overridable)
- Frontend dev: `npm install` then `npm run dev` in `frontend/` (Vite proxies `/api` to `127.0.0.1:8000`)
- Frontend tests: `npm test` in `frontend/` (Vitest + Testing Library, jsdom); `npm run test:watch` for watch mode
- Frontend build: `npm run build` in `frontend/` (typecheck + Vite; output served by FastAPI at `/` when `frontend/dist` exists)
- Benchmark: `uv run python -m backend.scripts.benchmark_model [--device cpu] [--max-new-tokens N]`
- Real-model integration: `LFM3_RUN_INTEGRATION=1 uv run pytest backend/tests/test_integration_lfm3.py` (weights cached in `~/.cache/huggingface`)

## Environment (verified 2026-10-29)

- `.venv` in this repo: `torch==2.14.1+rocm7.2`, `triton-rocm==3.8.0` (from explicit `pytorch-rocm` uv index; see `pyproject.toml` `[tool.uv.sources]` — uvmapping transitive deps doesn't work, they must be declared as direct deps), `transformers==5.21.0`, FastAPI stack.
- HIP 7.2 sees the RX 7600 (8.57GB, `torch.cuda.is_available() == True`, matmul verified). Physical VRAM 8GB; model weights 3.46GB BF16 → keep context/generation length bounded.
- System Python is 3.14.8 — torch wheels target ≤3.13; the pinned venv Python is intentional.
- Repo has its own git (`.git` here). The parent `/home/isaac` is a separate zero-commit repo — don't touch it.
- Never modify the system/global PyTorch or ROCm (pacman) install; all torch deps live in this venv.

## Verified LFM3-1.7B architecture (from repo's own modeling code)

- NOT merged into transformers (PR #50036 open as of 2026-10-28). Loads via `trust_remote_code=True`; the HF repo ships `configuration_lfm3.py`, `modeling_lfm3.py`, `lfm3_conv.py`. Vendored copies for offline tests: `backend/app/vendor/` — keep in sync if the model repo updates.
- `Lfm3ForCausalLM` → `model` (`Lfm3Model`: `embed_tokens`, `layers[24]`, `embedding_norm` final norm).
- Each `Lfm3DecoderLayer`: `operator_norm` → operator → `post_operator_norm` → `+residual`; `ffn_norm` → `feed_forward` (Lfm3MLP, SwiGLU 2048→7680→2048, present in ALL layers) → `post_ffn_norm` → `+residual`.
- `config.layer_types`: idx 2,6,10,14,18,22 = `full_attention`; 4,8,12,16,20 = `sliding_attention` (win 256); other 13 = `Lfm3ShortConv` (in_proj h→3h, depthwise causal conv k=7, gate B·x, out_proj).
- `Lfm3Attention`: GQA 16:4 heads, head_dim 128, q/k RMSNorm, per-head sigmoid output gate (`gate_proj`), partial RoPE (0.5) in fp32.
- Hook points (validated in real code): decoder-layer output, `self_attn` / `conv` / `feed_forward` outputs (all `[B,T,2048]`), `feed_forward.w2` forward-pre-hook for exact MLP-neuron activations `[B,T,7680]`.
- Steering hooks must be out-of-place (`clone()` then modify): the conv path mutates its cache state in place, and the same cached tensor may flow elsewhere.
- Chat template: `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`; thinking is enabled by default — experiments default to greedy + `enable_thinking=False` (user decision).

## Rules easy to miss (still binding)

- The GUI is mandatory, not a CLI; every control must perform a real action. Layout: 3 panels — architecture explorer / activation viz / steering + generation (PLAN §3).
- Steering is inference-only via hooks; never change model weights. Hooks installed per-experiment, removed in `finally` (PLAN §6). Zero strength must reproduce baseline.
- `h[..., i] = h[..., i] + alpha` is conceptual — adapt to the observed shape. Don't claim an activation dimension controls a behavior without experimental evidence; never report one changed response as proof (PLAN §9).
- Backend is the sole authority on tensor shapes, module names, results; frontend must not fabricate model data (PLAN §11). API routes in PLAN §11 are proposed, not mandated — implement and document the final set consistently.
- Tests must run without downloading LFM3 weights (use small synthetic `Lfm3Config` models from vendored code); optional real-model integration test may be skipped offline. Also verify the baseline path installs no steering hooks (PLAN §15).
- Capture only selected modules; never load full activation histories into RAM; don't persist activation tensors by default (PLAN §10/§13).
- Anything displayed must come from real data of the loaded model; don't claim a feature works until tested (PLAN §18 final principle).

## Phase 2 status (implemented and verified)

- Backend: FastAPI app (`backend/app/main.py`), settings (`core/settings.py`), Pydantic schemas, `ModelManager` (lazy load, CPU/ROCm selection, diagnostics), architecture inspector, baseline chat generation, routes under `/api`.
- Routes: `GET /api/health`, `GET /api/diagnostics`, `GET /api/model/status`, `POST /api/model/load`, `GET /api/model/architecture`, `GET /api/model/modules/{module_id}`, `POST /api/experiments/baseline`.
- Frontend: React/Vite three-panel workspace (architecture tree + layer overview, module inspector, baseline generation) plus collapsible diagnostics; served by FastAPI from `frontend/dist`.
- Real checkpoint is cached in `~/.cache/huggingface`; `LFM3_RUN_INTEGRATION=1` verifies 24 layers, 2048 hidden, correct layer types, and baseline generation on the RX 7600 (BF16, `cuda` API through ROCm).

## Phase 3 status (implemented and verified)

- Baseline requests accept `capture_module_ids` (max 8, only hidden-state modules). Hooks are installed around `generate` and removed in `finally`; tensors are detached and copied to CPU float32.
- Captured activations live only in an ephemeral in-memory store (8 experiments / 256 MB, oldest evicted). Routes: `GET /api/experiments/{id}/activations` and `GET /api/experiments/{id}/activations/{module_id}?feature=&position=&max_positions=&max_features=`.
- Activation payloads include token alignment (prompt/generated), evenly sampled heatmap, 40-bin histogram, stats, top-20 dimensions, and a feature series. The final generated token usually has no activation because it is never forwarded.
- Frontend center panel shows the real tensor: token strip, canvas heatmap with tooltip/legend, clickable feature chart, distribution, top dimensions, and captured-module indicators in the architecture explorer.

## Phase 4 status (implemented and verified)

- `backend/app/services/steering_engine.py` — `SteeringSpec` (module, feature_index, strength), validation (class must be a hidden-state module, index bounded by `hidden_size`, strength finite and ≤ `settings.max_steering_strength`, default ±50, env `LFM3_MAX_STEERING_STRENGTH`), and `steer_modules()` contextmanager: forward hooks installed per experiment, removed in `finally`.
- Steering is out-of-place everywhere: the hook clones the output (`h[..., i] + alpha` only for zero-strength-safe identity elsewhere) and rebuilds tuples (`Lfm3Attention` returns `(attn_output, attn_weights)`; first tensor is steered).
- `POST /api/experiments/steering` runs a baseline + steered pair with identical prompt/settings/seed and stores both in the activation store; responses carry `condition` ("baseline"/"steered") and the applied `steering` spec.
- `GET /api/experiments/{id}/activations/{module_id}?compare_with={other_id}` returns a difference view (`steered − baseline`, direction labeled, truncated to shared positions with a warning).
- Hook ordering matters: capture hooks see steered tensors because steer contexts are entered before capture contexts in `generate_chat`.
- Real-model smoke verified on the RX 7600: zero strength reproduces baseline exactly; α = −12 on `model.layers.22.feed_forward` dim 500 changed the actual output text.
- Frontend right panel: explicit baseline/steered/sweep mode, a decoupled steering-target card (set from the module inspector; inspection changes do not silently retarget), steering controls (hidden dimension + α + slider + reset), one contextual Run button, and side-by-side comparison with per-condition metadata; center panel shows the labeled difference tensor.

## Phase 5 status (implemented and verified)

- `backend/app/services/experiment_store.py` persists every run (including runs with no captures) as append-only JSONL metadata under `settings.data_dir/experiments.jsonl`; reload tolerates a corrupt trailing line. Metadata includes the full request/result, tokens, captured tensor shapes/dtypes, environment (python/torch/transformers/fastapi versions, app version, git commit, platform), derived `condition_detail`, and sweep fields.
- Runs are recorded immediately after each generation completes, so a baseline survives a later steered/sweep failure. Completed sweep runs remain queryable if the sweep aborts.
- Activation snapshots are optional: `persist_activations` per request (or `LFM3_PERSIST_ACTIVATIONS`) writes `.npz` under `data/activations/<id>.npz` (id validated as 32 hex chars; path containment checked on restore) and restores lazily into the in-memory `ActivationStore` only if it fits the byte budget.
- Routes: `POST /api/experiments/sweep`, `GET /api/experiments` (filters `condition`/`sweep_id`/`module_id`, pagination), `GET /api/experiments/export?format=jsonl|csv&ids=` (export route declared before `/{id}`), `GET /api/experiments/{id}`. Activation GETs fall back to snapshots; a known run with no captures returns `[]` for the list route and 404 for the tensor route.
- Sweeps: strengths 1-9, repeats 1-3, total runs <=31; seeded random-feature control excludes the target feature and matches the max magnitude/sign of the last strength; control runs actually apply their hook (`condition="control"`). Greedy repeats warn that outputs are identical; unseeded sampling repeats warn they are not reproducible.
- Frontend: collapsible experiment-history panel (list/detail/compare/export, reopen available snapshots) and sweep controls/results in the generation panel; persist-snapshot checkbox sends an explicit boolean. The generation panel now groups mode → steering target → prompt/settings → run, with aria-live status and visible clamping notes. A topbar Help drawer explains the key concepts, and a persisted tooltip toggle disables native plus heatmap tooltips.
- Tests: `backend/tests/test_experiments.py` covers persistence/restart, corrupt tail, filters/pagination, export content + CSV injection/numeric integrity, route ordering, sweep ordering/control/seed/warnings, snapshot save/restore/containment/size guard, and no-capture behavior.

## Phase 6 status (implemented and verified)

- Backend suite: `uv run pytest` → 74 passed, 3 skipped (integration skipped without the env flag). `LFM3_RUN_INTEGRATION=1 uv run pytest backend/tests/test_integration_lfm3.py` → 3 passed on the real checkpoint (architecture, baseline generation, activation capture).
- Frontend suite: 8 Vitest files / 70 tests covering `api/client` request+error handling, architecture loading/search/collapse/layer view, module inspector states, activation token/feature/top-dimension interactions, generation/steering/sweep payloads and result rendering, history list/detail/compare/export/pagination, and full App load→select→baseline/steered/sweep/error flows. `npm run build` (tsc + Vite) passes.
- `backend/scripts/benchmark_model.py` measures steady-state latency/VRAM. Reference RX 7600 (BF16, greedy): load ~6.5-9 s, ~27 tok/s for 8-32 tokens, capture/steering add no measurable cost, peak allocated 3.49 GB (3.53 GB reserved).
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` was tested: no meaningful speedup (~28 vs ~27 tok/s), so it is not enabled by default.
- Frontend test env stubs `HTMLCanvasElement.getContext` (jsdom has no canvas backend); the heatmap drawing path is not asserted pixel-wise.

## Phase 7 status (implemented and verified)

- README now covers installation, launch (backend + frontend + production build), an example experiment workflow, real-model integration, benchmark numbers, API overview, known limitations, troubleshooting, and a tested-features summary. `PLAN.md` §17 Phase 7 is complete.

## Still open

- ROCm path notably emits benign stderr noise (`(null): No such file or directory`) at import.
- First real generation includes ~5-6 s of ROCm/SDPA warmup; later tokens are faster. Keep generation bounded (8 GB physical VRAM).
- Sweeps run sequentially with no cancellation/streaming progress; document wall-time before large sweeps.
- Canvas heatmap rendering is excluded from jsdom tests; verifying it requires a real browser.