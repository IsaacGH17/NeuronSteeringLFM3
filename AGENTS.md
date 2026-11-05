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

## Session handoff

See `docs/handoff-2026-11-04.md` for the closing state: shipped features, the σ bug fix, seven controlled experiments with their numbers, closed vs open hypotheses, environment notes (GPU sandbox, `UV_CACHE_DIR`), and suggested commit splits.

## Still open

- ROCm path notably emits benign stderr noise (`(null): No such file or directory`) at import.
- First real generation includes ~5-6 s of ROCm/SDPA warmup; later tokens are faster. Keep generation bounded (8 GB physical VRAM).
- Synchronous sweeps still block the request and cannot be cancelled; for long runs use a batch, which reports progress and cancels between runs.
- Canvas heatmap rendering is excluded from jsdom tests; verifying it requires a real browser.
## Phase 8 status (implemented and verified)

Phase 8 supersedes the caps quoted in Phases 3–5 (capture modules, strength limit, sweep sizes, activation-store budget).

- **Vector steering.** `SteeringSpec` is now module + `components` (weighted hidden dimensions) + `strength` + `normalization` (`raw` | `sigma`). Legacy single-dimension requests (`feature_index`) are normalized into one component, so old records and clients keep working. `sigma` scales α by that dimension's standard deviation calibrated on the prompt prefill (decode steps carry a single token, so a per-step spread would be zero) and reused for the rest of the generation (portable across layers); `raw` keeps the old units. Arithmetic runs in float32 and is cast back, so α = 0 is numerically identical to the baseline; the input tensor is always cloned first.
- **Per-dimension statistics.** Every activation payload carries `feature_stats` (mean, σ, min, max per hidden dimension, NaN-safe, rounded).
- **Direction extraction.** `POST /api/experiments/direction` averages two captured states (`mean_prompt` | `mean_all` | `last_prompt`), subtracts them, and returns the top-k weighted dimensions plus σ, separation, and norm. The history panel exposes it as "Build direction" for two selected runs (first selected = positive side) and targets that module.
- **Text metrics.** Every run gets deterministic metrics: words, sentences, distinct-3, 4-gram repetition, refusal flag, tone lexicons (warmth/technical/hedging/assertive, markers per 100 words) and suite checks. They are persisted in JSONL, exported as CSV columns, and aggregated per batch cell.
- **Prompt suites.** `trait-neutral` (12 neutral requests) and `capability-core` (10 tasks with deterministic checks); `GET /api/suites`.
- **Batch runner.** `POST /api/experiments/batch` → `202` + job; `GET /api/experiments/batch/{id}?include_runs=`; `POST /api/experiments/batch/{id}/cancel`. One batch runs at a time, cancellation happens between runs (the in-flight generation finishes), and every completed run is persisted immediately with `sweep_id = job_id` so history/export keep working. Cells aggregate words, distinct-3, refusal rate, check pass rate, and lexicon means.
- **Limits are configurable instead of hardcoded.** `LFM3_MAX_STEERING_STRENGTH` (default ±1000), `LFM3_MAX_BATCH_RUNS` (2000), `LFM3_ACTIVATION_STORE_EXPERIMENTS` (32), `LFM3_ACTIVATION_STORE_BYTES` (512 MB); capture modules 8 → 128, sweep strengths 9 → 32, repeats 3 → 10 (all validated server-side). `MAX_CAPTURE_POSITIONS = 2048` remains as the memory guard.
- **Verification.** Backend `uv run pytest` → 113 passed, 3 skipped; `uv run ruff check .` clean. Frontend 80 tests across 8 files; `npm run build` passes; the batch panel, direction chips, and progress table were rendered in Firefox at 1680/1100/900/420 px.
- **Pilot result (`backend/scripts/personality_pilot.py`, `docs/personality-pilot.md`).** System prompts move persona strongly; single-layer mean-difference directions did not install or cancel it, and the multi-layer follow-up (5 residual layers, full 2048-d vectors, α_total up to 8 rms, coherent outputs, shuffled/random controls at 0%) also failed: install 0/6, cancel 2/6 with opposite-sign cases. Residual steering above ~40σ collapsed into repetition. The teacher-forced follow-up (identical assistant text under both systems, layers 8–22) also failed: install 0/6, and its apparent cancellation (4/6, mean +29%) matched shuffled/random control directions (+25%), so it is generic perturbation, not a persona vector. The probed axis is largely formatting/instruction behaviour, which this model seems not to store as a linear residual direction of this kind. Individual-neuron follow-up (`backend/scripts/neuron_probe.py`, `neuron_generation.py`): with the w2 column gain including the post-FFN norm weight, 2/80 neurons beat random controls on next-token style mass and L16 n=3177 showed a clean dose-response (+0.010→+0.114 from α=0.1→0.7), but generation moved the style score on only 1/4 prompts — the same as a random neuron — so the effect does not reach the text at these alphas. Group-of-neurons follow-up (`neuron_group_probe.py`) also failed: L16 top-8 at α=0.7 gave Δstyle +0.139 in the next-token screen but the best arm was the *negative* sign (the screen metric has a sign bias) and generation moved 0/4 prompts, same as a random group. Next-token style mass is not a valid proxy for text-level persona. Treat high separation as a hypothesis, not causation. The σ-calibration bug above was found by this pilot.
- **Interpret drawer (Phase 8 addendum).** Topbar → Interpret: `POST /experiments/lens` returns per-layer top-k predictions at a selected token (logit lens) plus a per-layer ablation KL (each decoder layer’s contribution skipped on the prompt prefill, returning the layer input so the residual stream is preserved); `POST /experiments/patch` replaces one module's state at the recipient's last prompt token with the donor's for the first forward only. Services: `services/lens.py`, `services/patching.py`; UI: `components/InterpretDrawer.tsx`.
- **Lens contrast (`lens_contrast.py`).** Warm vs clinical at the last prompt token: layers 0–6 agree on the chat-template token, 7–14 are uncommitted (≤0.03), and the condition split consolidates from layer ~15 (final layer: 'Oh' 0.78 vs 'The' 0.81 for the rainy prompt; mean min-probability from L15 = 0.219). Persona choice at the first token is late-forming, which is consistent with the earlier nulls at layers 8–14. A targeted late-layer run (extract at the last prompt token, steer only L19–22, full vectors, α up to 4 rms/layer) also failed: install 0/6, cancel 2/6 vs shuffled/random controls +31 %, first-token flips 1/6 and not toward warm. The lens says when the preference is readable, not which direction causes it.
- **Format-steering pilot.** `backend/scripts/format_pilot.py` extracted a bullet-vs-prose direction from generated states; the ladder moved one bullet line on one prompt, held-out moved 0/2 (controls 0/2), capability intact. Consistent with the other instruction-bound nulls.
- **Caveats.** Metrics are lexical/structural heuristics, not judgments about quality. A batch cannot be interrupted mid-generation. `steering_summary` in experiment summaries is the canonical label for multi-dimension runs (`feature_index` stays null for them).

## Phase 9 status — live flow (implemented and verified)

- **Backend.** `POST /api/experiments/flow` (`FlowRequest` → `FlowResponse`) runs one real greedy generation with every decoder layer captured, then `services/flow.py` computes per-layer residual write (`mean ‖h_l − h_{l−1}‖ / ‖h_{l−1}‖`), consecutive-layer cosine, gain, per-token energy/delta, and deterministic spherical k-means clusters of hidden dimensions (z-scored profiles, ≤128 sampled positions, k configurable: UI 3–10, API 2–12). Clusters carry size, mass, cohesion, peak token, top dimensions with σ, and ready-to-use steering components. The trace is capped at `MAX_FLOW_POSITIONS = 512` (~100 MB at 24×2048); the run is recorded in history and captures live in the activation store.
- **Frontend.** Topbar → **Flow** opens `components/FlowDrawer.tsx`: prompt/settings → trace; animated SVG residual stream with measured ribbon widths/opacity, per-token playback, token strip, clickable layer nodes and cluster strands; inspector with layer metrics, cluster profile, top dimensions, σ-unit α, inline baseline+steered comparison, and "Use these dimensions in the steering panel" (feeds the existing `direction` state).
- **Honesty rules encoded in the UI.** Widths are measured and need not grow with depth; strand colors are per-layer ranks, not matched clusters; clusters are descriptive, not causal; a null steering result is shown as a real result; the flow is a playback of a completed generation, not streaming.
- **Verification.** Backend `uv run pytest` → 125 passed, 3 skipped; `ruff check` clean. Frontend 92 tests across 10 files; `npm run build` passes. Real RX 7600 smoke: 24 layers, 23 edges, 52 positions, 6 clusters/layer (236–416 dims), trace ~2.6 s for 28 positions, 3.25 GB allocated; α = 2 σ on layer 23 cluster 0 changed the output, α = 1 σ on layer 12 cluster 0 did not (one prompt, not a controlled experiment). Headless Chromium rendered the drawer, traced, selected a cluster, and ran the real baseline/steered pair; screenshots in `docs/screenshots/flow*.png`.
- **Docs.** `docs/live-flow.md` defines every visual, the clustering, the API, the reference numbers, and the limits. README updated.

## Phase 10 status — Apple-inspired redesign (implemented and verified)

- **Visual system.** The warm editorial theme was replaced by an Apple-inspired light system. One neutral sans: the `-apple-system / SF Pro` stack first, bundled **Inter** (`@fontsource/inter`) on Linux; IBM Plex Mono remains for module paths, shapes, and raw values. White and `#f5f5f7` surfaces, hairline separators, soft shadows, 6–20 px radii, and system pigments (blue, teal, indigo, orange, pink, purple). The activation heatmap now paints orange for positive, blue for negative, and `#f5f5f7` for zero. Theme tokens live at the top of `frontend/src/styles.css`; a late override block retunes component-level styles without markup changes.
- **Flow graph.** Rebuilt as a line-based stream: 24 layer markers (hairline + node), curved gradient ribbons whose width/opacity are the measured write/similarity, animated dashes for token playback, and colored cluster strands that fan from each layer. The selected layer's fan is full opacity and the rest are dimmed; the trace form collapses to a compact card after a trace so playback and the token strip stay visible. All data, interactions, and aria labels are unchanged.
- **Verification.** Frontend `npm test` → 92 tests across 10 files; `npm run build` passes. Backend unchanged: 125 passed, 3 skipped; `ruff check` clean. Headless Chromium resolved Inter, reported no 404s or console errors, and rendered the workspace, Flow (trace + cluster + steer), Help, and Interpret at 1680×1120 / 760×1120 / 700×1500. Screenshots refreshed in `docs/screenshots/`; `README.md` and `docs/live-flow.md` updated.

## Phase 11 status — next-token probe (implemented and verified)

- **Backend.** `POST /api/experiments/next-token` (`NextTokenRequest` → `NextTokenResponse`) renders the chat prompt, runs one forward with `logits_to_keep=1`, and reads the raw softmax at the last prompt token. With a steering spec it validates it, forwards the same prompt a second time under `steer_modules`, and returns both top-k distributions, the merged candidate list with exact probabilities/deltas, KL divergence (nats), entropies, duration, and warnings. No generation, no capture, no experiment-history entry. Service: `services/next_token.py`; route next to the flow route; schemas added after `FlowResponse`.
- **Frontend.** Topbar → **Probe** opens `components/NextTokenProbe.tsx`: prompt/system/top-k, steering card (module, dims, normalization, α number + slider, clear), and a baseline/steered result view with top-1 cards, KL/entropy/duration, and a candidate table (blue baseline bars, orange steered bars, Δ in percentage points). α changes re-probe after a 350 ms debounce; stale responses are ignored by request id. The Flow cluster inspector has **Probe next token**, which closes Flow and opens the probe with the traced prompt and cluster dimensions.
- **Verification.** Backend `uv run pytest` → 134 passed, 3 skipped; `ruff check` clean. Frontend `npm test` → 101 tests across 11 files; `npm run build` passes. Real RX 7600 smoke: `"The capital of France is"` top-1 `The` 96.5% (`Paris` 3.3%); L12 dim 100 at α = 5 σ moved `Paris` to 6.7% (KL 0.014969); α = 0 gave KL 0; a Flow L23 cluster at α = 2 σ moved `In` 9.4% → 11.6% (KL 0.0054, 83 ms). Screenshot: `docs/screenshots/next-token-probe.png`; docs: `docs/next-token-probe.md`, README, Help drawer.
- **Limits.** Last prompt token only; raw temperature-1 softmax (not the greedy/sampled decision); steered probes cost two forwards; probes never enter history; a changed top-1 is one observation, not causal proof.

## Phase 12 status — flow diff + ablation (implemented and verified)

- **Backend.** `FlowRequest.include_ablation` (default false) and `FlowLayer.ablation_kl` (nullable). `services/ablation.py` computes one extra forward per decoder layer and returns the next-token KL at the last prompt token. The hook is **residual-preserving**: it returns the layer's input (`args[0].clone()`) instead of the layer output, skipping only that layer's contribution. The Interpret lens ablation was fixed to the same intervention.
- **Bug found and fixed.** The first ablation zeroed the layer output, which cuts the residual stream for every later layer; all 24 layers produced the same ablated state and the same KL (11.5973 on the France prompt; 0.028069 on the tiny model). Returning the layer input makes the values vary meaningfully (France trace: L0 5.139, L1 0.183, L6 1.616, L12 0.072, L23 0.177). The lens had the same bug; both now use the residual-preserving version.
- **Frontend.** Flow has a **Single / Compare** toggle. Compare takes A/B prompts (shared system/settings), runs both traces sequentially, and renders `FlowDiffGraph` (width = |Δ write|, teal B > A, orange A > B, rings = |Δ ablation KL|) plus `DiffLayerInspector` (A/B/Δ table for norm, write, cosine, gain, ablation, clusters, top mass). Single mode shows ablation rings and an "ablation KL" metric. Both traces stay in history; token playback and cluster strands remain Single-only because positions differ.
- **Verification.** Backend `uv run pytest` → 135 passed, 3 skipped; `ruff check` clean. Frontend `npm test` → 102 tests across 11 files; `npm run build` passes. Real RX 7600: France vs Italy, 47 positions each, trace with ablation ~3 s; L12 write 0.749 → 0.672, cosine 0.780 → 0.814, ablation 0.072 → 0.029. Screenshot `docs/screenshots/flow-diff.png`; docs `docs/live-flow.md`, README, Help drawer.
- **Limits.** Compare runs two traces; clusters are not matched across A/B; ablation is one intervention at the last prompt token and adds one forward per layer; the diff aggregates positions.

## Phase 13 status — patch sweep (implemented and verified)

- **Backend.** `POST /api/experiments/patch-sweep` (`PatchSweepRequest` → `PatchSweepResponse`); `services/patch_sweep.py`. One baseline generation and one donor prefill with every decoder layer captured are shared by the grid; each cell runs one recipient generation with an out-of-place hook that replaces the donor state at one layer/position for the first forward only. Metrics: `changed`, `first_token_changed`, `common_prefix_tokens`, `generated_tokens`, `patched_output`, `donor_position`. Cap: 48 cells; invalid positions are skipped with a warning; donor positions clamp to the donor's last token. The sweep is synchronous and does not enter experiment history.
- **Frontend.** Third Interpret tab **Patch sweep** with recipient/donor prompts, donor system, max tokens, position mode (last / last 4 / first 4 / custom index), layer range, and a compact grid (`=` unchanged, `≠` output changed, `⇄` first token flipped) plus a per-cell patched-text detail. The cell count and 48-cell cap are shown before running.
- **Real finding.** France vs Italy, greedy, 8 new tokens, all 24 layers: patching the **last prompt token** (15, template tail) changed 0/24 outputs; patching the **country token** (7, `" France"`) changed 20/24 outputs and flipped the first token in layers 5–7 to `"Rome."`; layers 0–4 and 8–19 kept the first three tokens and changed the answer; layers 20–23 did not change. 24 cells took 13.1 s. One prompt pair, not a controlled experiment, but it shows why the position matters.
- **Verification.** Backend `uv run pytest` → 143 passed, 3 skipped; `ruff check` clean. Frontend `npm test` → 103 tests across 11 files; `npm run build` passes. Screenshot `docs/screenshots/patch-sweep.png`; docs `docs/patch-sweep.md`, README, Help drawer.
- **Limits.** Synchronous (no cancellation/background job); max 48 cells; ephemeral (no history); donor position clamps; one intervention per cell.

## Phase 14 status — neuron explorer (w2) (implemented and verified)

- **Backend.** `POST /api/experiments/neurons` (`NeuronTraceRequest` → `NeuronTraceResponse`) runs one real generation with a forward pre-hook on `model.layers.<i>.feed_forward.w2`, captures the `[T, 7680]` intermediate, and clusters neurons by z-scored activation profiles (deterministic spherical k-means, ≤128 sampled positions). Each cluster carries size/mass/cohesion/peak/profile/top neurons; the response includes global top neurons and `assignments` (cluster id per neuron). The trace run is recorded in history (metadata only; tensors are ephemeral). `POST /api/experiments/neuron-steering` (`NeuronSteeringRequest` → `NeuronSteeringResponse`) computes `direction = Σ w_i·(w2[:,i]·post_ffn_norm.weight)`, `δ_i = α·residual_rms/‖direction‖·w_i`, and runs a baseline + steered pair with an out-of-place w2 pre-hook. The steered pair is ephemeral.
- **Frontend.** Topbar → **Neurons** opens `components/NeuronDrawer.tsx`: prompt/system, layer (default 16), clusters, max tokens; trace; a canvas cluster map grouped by cluster, cluster list, top-neuron chips, profile sparkline, α slider in residual-RMS units, and a baseline/steered comparison. Fixed a default-layer bug: the drawer initialized to layer 0 when the layer list was empty; it now starts at 16 and falls back to the middle layer only when 16 is unavailable.
- **Real result.** Layer 16, `"Describe a rainy morning."`, 6 clusters (1643/1504/1403/1181/1011/938), residual RMS 0.1193; cluster 0 at α = 0.7 (direction norm 0.1147) changed *"The world is wrapped in a soft, steady drizzle…"* to *"The sky is a heavy, slate-grey blanket…"*. One prompt/group; the pilot's nulls mean this is a hypothesis, not causal proof.
- **Verification.** Backend `uv run pytest` → 151 passed, 3 skipped; `ruff check` clean. Frontend `npm test` → 107 tests across 12 files; `npm run build` passes. Screenshots `docs/screenshots/neuron-explorer.png` and `neuron-steering.png`; docs `docs/neurons.md`, README, Help drawer.
- **Limits.** One layer at a time; clustering descriptive; group steering at every token; steered pair ephemeral; the map groups by cluster, not by neuron index.
