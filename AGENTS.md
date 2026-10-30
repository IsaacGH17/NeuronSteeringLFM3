# AGENTS.md — NueronSteerLFM3

Interactive GUI to inspect and steer activations of `LiquidAI/LFM3-1.7B` (PyTorch + Transformers). Python/FastAPI backend + React/TypeScript/Vite frontend. Spec = `PLAN.md`; read it before working. Research (Phase 1, §18) is done — build with the facts below.

## Commands (verified)

- `uv sync` — create/update `.venv` (Python 3.13.15 via uv-managed CPython)
- `uv run pytest` — run tests (must pass without downloading LFM3 weights)
- `uv run ruff check .` / `uv run ruff format .` — lint/format
- Run anything: `uv run python -m ...` (never call system python; it has no torch)
- Backend dev: `uv run uvicorn backend.app.main:app --reload` (host/port from `backend/app/core/settings.py`, env-overridable)
- Frontend dev: `npm install` then `npm run dev` in `frontend/` (Vite proxies `/api` to `127.0.0.1:8000`)
- Frontend build: `npm run build` in `frontend/` (typecheck + Vite; output served by FastAPI at `/` when `frontend/dist` exists)
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

## Still open

- ROCm path notably emits benign stderr noise (`(null): No such file or directory`) at import.
- First real generation includes ~5–6 s of ROCm/SDPA warmup; later tokens are faster. Keep generation bounded (8 GB physical VRAM).
- Activation tensors are not persisted; reopening old experiments after a backend restart is Phase 5 work.
