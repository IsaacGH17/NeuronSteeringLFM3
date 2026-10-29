# LFM3 Neuron Explorer — Interactive Activation Steering Lab

## 1. Project objective

Build an interactive, local-first GUI inspired by the visual exploration and neuron-steering workflow of `lyramakesmusic/neuron-explorer`.

**Target model:** `LiquidAI/LFM3-1.7B`

**Primary goal:** Explore the model's internal activations and experimentally change its behavior during inference without modifying its original weights.

This is a research and educational tool. It should help the user discover how interventions in internal representations affect generated text.

The application must include:
- An interactive visualization of the model's actual architecture and layers.
- Layer and module selection.
- Activation inspection and visualization.
- Selectable activation dimensions (often casually called neurons).
- Positive and negative steering strengths.
- Baseline versus steered generation.
- Interactive comparison of outputs.
- Experiment history, reproducibility, and export.
- Local inference with a CPU fallback and optional AMD ROCm acceleration.

Do not build only a command-line interface. The GUI is a central requirement.

## 2. Development instructions for OpenCode

Act as an ML systems engineer, Python developer, and frontend engineer.

Before implementing anything:

1. Inspect the official LFM3 model card, configuration, architecture, and current Transformers integration.
2. Inspect the reference repository: https://github.com/lyramakesmusic/neuron-explorer
3. Study its approach to model inspection, activation intervention, and interactive visualization. Reuse ideas, not incompatible assumptions.
4. Determine the actual modules, tensor shapes, and execution behavior of LFM3-1.7B.
5. Verify compatibility among Python, PyTorch, Transformers, and the available hardware.
6. Present a short architecture and implementation plan before making major changes.
7. Build the project incrementally, running tests at each stage.

Do not assume LFM3 uses the same module names, layer layout, neuron indices, or hook locations as Qwen3.5.

Do not claim that an activation dimension corresponds to a specific semantic concept without experimental evidence.

Keep the original model weights unchanged throughout the initial implementation.

## 3. User experience and GUI

The interface should feel like a small interactive neural-network laboratory.

Use a dark, modern developer-tool aesthetic with a clear visual hierarchy. The interface should remain usable on a normal desktop display.

### Main layout

Create a three-panel workspace:

**Left panel — Architecture explorer**
- Model name and loading status.
- Hierarchical tree of modules and layers.
- Expandable module groups.
- Search by module name or type.
- Clickable layer/module entries.
- Distinct visual indicators for selected modules and modules with captured activations.
- Actual layer and module counts discovered from the loaded model.

**Center panel — Activation visualization**
- Interactive layer/module view.
- Activation heatmap when tensor data is available.
- Dimension/neuron selection.
- Activation magnitude and distribution plots.
- Selected module metadata and tensor shape.
- Baseline versus intervened activation comparison.
- Color scale with a visible legend.
- Tooltips displaying indices and measured values.

**Right panel — Steering and generation**
- Prompt editor.
- System instruction editor if supported by the model interface.
- Baseline generation button.
- Steered generation button.
- Steering mode selector.
- Target dimension/index.
- Steering strength slider.
- Maximum generated tokens.
- Temperature and other supported sampling parameters.
- Generated output and execution status.

Provide a collapsible bottom panel for experiment logs, comparisons, and technical diagnostics.

### Important UI behavior

Selecting a layer in the architecture explorer must update the center panel.

Selecting a dimension in the activation view must update the steering controls.

Changing a steering parameter must update the pending experiment configuration, but must not silently modify the model's persistent weights.

The user must explicitly run an experiment to generate a new result.

All controls must perform real actions. Do not create decorative buttons that do nothing.

## 4. Architecture visualization

Implement a dynamic architecture explorer.

The application must discover modules from the actual loaded model rather than using a hardcoded list of Transformer layers.

Display:
- Module name.
- Module class/type.
- Parent and child relationships when discoverable.
- Input/output tensor shapes when observed.
- Parameter counts where available.
- Whether the module supports the selected instrumentation method.
- Whether activations have been captured.

Provide at least two visualization modes:

### A. Hierarchical tree

A navigable tree of the model's real module structure.

### B. Layer overview

A visual representation of the model's detected computational blocks.

Use distinct labels for different block types when they can be identified reliably. For LFM3, investigate the actual implementation of its convolutional, sliding-window attention, and global attention components.

Do not force all components into a conventional Transformer diagram if the architecture differs.

Clicking a block should select it and expose its inspectable activation data.

If a module cannot be instrumented safely, display the reason instead of pretending it works.

## 5. Activation explorer

The activation explorer is one of the core features.

After a user runs a prompt, allow them to select a supported module and inspect its captured activation tensors.

### Visualization modes

Implement the following when supported by the tensor shapes:

1. **Heatmap:** activation values across token positions and feature dimensions.
2. **Feature view:** values for a selected dimension across token positions.
3. **Distribution:** histogram or distribution summary of activation values.
4. **Top dimensions:** dimensions with the largest absolute activations for the selected observation.
5. **Difference view:** baseline activations minus steered activations, or the reverse, with the direction explicitly labeled.

Display the tensor shape, dtype, token position, and aggregation method.

Allow selection of a feature index from the visualization or a numeric input.

When a tensor is too large, aggregate or downsample it for display. Keep the original measurements where practical, and make the aggregation method clear.

Do not load every layer's complete activation history into RAM or the browser.

### Token alignment

When token-level data is available, show token positions and decoded tokens.

Handle prompt tokens and generated tokens separately if the model's inference implementation permits reliable identification.

Do not assume that a tensor position maps directly to a readable word. Document tokenization and alignment limitations.

## 6. Activation steering engine

Implement activation steering as a reversible inference-time intervention.

The first version must not change persistent model parameters.

### Steering modes

**Mode A — Single-dimension steering**

Allow the user to choose a supported module, a feature dimension, and a strength alpha.

Conceptually:

`h[..., i] = h[..., i] + alpha`

Here, `h` is the activation tensor at the selected module, and `i` is a validated feature index.

The actual operation must be adapted to the observed tensor shape and execution behavior.

**Mode B — Vector steering**

Allow the user to load or create a compatible steering direction `v` and apply:

`h' = h + alpha * v`

Validate the shape, dtype, device, and normalization of the direction.

Do not implement semantic direction discovery until basic steering and evaluation work reliably.

**Mode C — Multiple dimensions**

Allow a small configurable set of dimensions and strengths.

Show each selected dimension in a table with its index, strength, and enabled status.

Validate every index before generation.

### Strength control

Provide:
- A numeric input.
- A slider.
- A zero/reset option.
- Positive and negative values.
- A configurable safety range.
- The ability to test multiple strengths in separate runs.

The initial strength range is experimental, not a recommendation about optimal values.

Zero strength must reproduce the baseline intervention behavior as closely as the implementation allows.

Do not assume that increasing alpha monotonically increases a particular behavioral trait.

### Hook lifecycle

Hooks must:
- Be installed only for the requested experiment.
- Be removed in a `finally` block or equivalent cleanup mechanism.
- Avoid in-place mutation where it can cause unintended side effects.
- Validate tensor shape, dimension, dtype, and device.
- Record errors and intervention metadata.
- Avoid retaining computational graphs unnecessarily.
- Avoid changing model weights.
- Handle exceptions without leaving the model in a modified execution state.

Investigate how LFM3 performs autoregressive generation and whether the chosen hook executes during prompt processing, token generation, or both.

Only expose temporal controls when their behavior has been verified.

## 7. Baseline and steered generation

The GUI must support a direct comparison between the unmodified model and the intervened model.

For each experiment, provide:

- Baseline output.
- Steered output.
- Side-by-side display.
- Optional synchronized scrolling.
- Text difference highlighting.
- Generation duration.
- Token count when available.
- Tokens per second when measurable.
- Selected module and feature index.
- Steering strength.
- Generation settings.
- Errors and warnings.

Use the same prompt and generation configuration across conditions.

Where possible, control random seeds and sampling settings. Explain that identical seeds do not always guarantee identical outputs when different interventions change the generation trajectory.

Provide a multi-run mode for testing several strengths against the same baseline.

## 8. Experiment workflow

The typical workflow should be:

1. Load LFM3-1.7B.
2. Open the architecture explorer.
3. Select a module or computational block.
4. Run a baseline prompt.
5. Inspect captured activations.
6. Select a supported feature dimension.
7. Set a steering strength.
8. Generate the steered response.
9. Compare both outputs.
10. Save the experiment.
11. Repeat with other strengths or dimensions.

Provide a quick experiment mode that executes a predefined list of strengths and presents the outputs together.

Also provide an explicit reset button that clears the pending steering configuration without modifying the underlying model.

## 9. Controlled experiments

Implement controls to help distinguish genuine effects from random variation.

Each comparison should support:

- Baseline: no intervention.
- Positive intervention.
- Negative intervention.
- Random-feature control.
- Optional repeated runs.

Use fixed experiment configurations and prompts.

When testing a selected feature against a random control, document whether the control matches the number of modified dimensions and intervention magnitude.

Do not report a single changed response as proof that a feature has a stable semantic meaning.

### Initial behavior test suite

Include prompts that test:

- Response length and verbosity.
- Formal versus informal writing.
- Structured output and valid JSON.
- Instruction following.
- Expression of uncertainty.
- Simple Python and Linux troubleshooting.

These are test categories, not claims that specific dimensions control those behaviors.

Allow users to add and edit their own prompts.

## 10. Experiment history and data export

Every run should be saved as a structured record.

Store:
- Experiment ID.
- Timestamp.
- Model ID and revision, when available.
- Prompt and relevant generation settings.
- Condition: baseline, positive, negative, or control.
- Target module.
- Target dimension or direction.
- Steering strength.
- Tensor shape and dtype.
- Random seed.
- Output text.
- Runtime measurements when available.
- Warnings and errors.
- Application and dependency versions.

Support:
- JSONL export.
- CSV export for tabular experiment summaries.
- Export of experiment configuration.
- Reopening saved experiments for inspection.
- Comparison of selected historical runs.

Do not persist complete activation tensors by default. Make activation snapshots optional because they may consume significant disk space.

Store large snapshots separately from experiment metadata.

Do not save access tokens, private credentials, or unrelated personal data.

## 11. Technical architecture

Use a Python backend and a modern web frontend.

### Backend

Recommended:
- Python.
- PyTorch.
- Transformers.
- FastAPI.
- Pydantic.
- Uvicorn.
- pytest.
- Ruff.

Responsibilities:
- Model loading.
- Architecture inspection.
- Inference.
- Hook management.
- Activation capture.
- Steering.
- Evaluation.
- Experiment storage and export.
- Streaming generation progress or status to the frontend.

Use a single controlled model worker initially. Avoid concurrent requests modifying the same model's hooks or steering state.

Implement explicit job states such as queued, running, completed, cancelled, and failed.

### Frontend

Recommended:
- React.
- TypeScript.
- Vite.
- A graph visualization library such as React Flow for the architecture view.
- A plotting library such as Plotly or Apache ECharts for heatmaps and distributions.

Use a clean component architecture.

Suggested components:
- `ArchitectureExplorer`
- `LayerGraph`
- `ModuleInspector`
- `ActivationHeatmap`
- `FeatureSelector`
- `SteeringControls`
- `PromptEditor`
- `GenerationComparison`
- `ExperimentHistory`
- `DiagnosticsPanel`

The backend is the authority for tensor shapes, module names, and experiment results. The frontend must not fabricate model data.

Use WebSockets or server-sent events for progress updates if useful. Handle disconnects and failed jobs.

### API design

Create a documented API with endpoints equivalent to:

- `GET /api/health`
- `GET /api/model/status`
- `POST /api/model/load`
- `GET /api/model/architecture`
- `GET /api/model/modules/{module_id}`
- `POST /api/experiments/baseline`
- `POST /api/experiments/steering`
- `GET /api/experiments/{experiment_id}`
- `GET /api/experiments`
- `GET /api/experiments/{experiment_id}/activations`
- `GET /api/experiments/{experiment_id}/export`

These routes are a proposed API, not existing interfaces. Implement and document the final routes consistently.

Validate requests and avoid exposing arbitrary filesystem access through the API.

## 12. Hardware and ROCm support

The target development system may use:
- AMD Radeon RX 7600.
- 32 GB system RAM.
- Arch Linux.
- PyTorch with ROCm.
- Optional CPU fallback.

Do not assume ROCm is correctly installed.

Create a diagnostics screen showing:
- Python version.
- PyTorch version.
- `torch.cuda.is_available()`.
- `torch.version.hip`, when available.
- Visible device name.
- Model loading status.
- Device used for inference.
- Memory statistics when supported.

PyTorch's `torch.cuda` API may expose AMD GPUs through ROCm. Do not interpret the API name alone as proof that an NVIDIA GPU is present.

Use an isolated virtual environment. Never replace the user's global PyTorch or ROCm installation automatically.

If GPU loading fails, explain the error and allow CPU inference when practical.

## 13. Performance and memory

- Load the model once and reuse it.
- Limit generation length and batch size.
- Capture only selected modules.
- Capture activations only when requested.
- Detach captured tensors from autograd.
- Release temporary tensors after experiments.
- Avoid storing entire activation histories unnecessarily.
- Provide cancellation for long-running jobs where feasible.
- Report memory or device errors clearly.
- Use quantization only when verified compatible with the model and backend.

The interface should remain responsive while inference is running.

## 14. Suggested project structure

```text
lfm3-neuron-explorer/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── services/
│   │   │   ├── model_loader.py
│   │   │   ├── architecture_inspector.py
│   │   │   ├── activation_capture.py
│   │   │   ├── steering_engine.py
│   │   │   ├── generation.py
│   │   │   ├── evaluation.py
│   │   │   └── experiment_store.py
│   │   ├── schemas/
│   │   └── core/
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── api/
│   │   ├── hooks/
│   │   └── types/
├── experiments/
│   ├── prompts.jsonl
│   └── examples/
└── data/
    └── .gitkeep
```

Adapt this structure if the implementation benefits from a simpler arrangement. Do not create empty placeholder modules without a purpose.

## 15. Testing requirements

Create automated tests for:

- Configuration validation.
- Architecture discovery using a synthetic test model.
- Tensor-shape validation.
- Steering arithmetic.
- Positive and negative intervention.
- Invalid feature indices.
- Hook cleanup after success.
- Hook cleanup after exceptions.
- Baseline behavior with no intervention.
- Experiment metadata persistence.
- API validation and error handling.
- Frontend loading, selection, and experiment states.

Tests must run without downloading LFM3-1.7B.

Add an optional integration test for the actual model. It may be skipped if the model, network, memory, or supported hardware is unavailable.

Verify that the baseline path does not install active steering hooks.

## 16. Definition of done

The project is not complete until:

1. The frontend launches and communicates with the backend.
2. The actual model architecture can be inspected.
3. The user can select a real, supported module in the GUI.
4. Activations can be captured and visualized after a generation.
5. The user can select a valid dimension and change steering strength.
6. Baseline and steered outputs appear side by side.
7. The intervention is applied during inference and removed afterward.
8. The original model weights remain unchanged.
9. Experiments can be saved and exported.
10. Invalid operations produce understandable errors.
11. Automated tests pass.
12. The README documents installation, launch, usage, limitations, and troubleshooting.

Do not claim a feature works until it has been tested.

## 17. Implementation roadmap

### Phase 1 — Research and environment
Inspect the reference repository and official LFM3 implementation. Verify architecture, dependencies, model loading, and hardware support. Report any incompatibilities before proceeding.

### Phase 2 — Minimal end-to-end application
Build the backend, frontend, model-loading flow, diagnostics, architecture explorer, and baseline generation.

### Phase 3 — Activation visualization
Capture activations for selected modules and implement the heatmap, feature selector, and module inspector.

### Phase 4 — Steering
Implement single-dimension steering, strength controls, hook lifecycle management, and baseline-versus-steered comparisons.

### Phase 5 — Experiments and history
Add randomized controls, multi-strength runs, experiment persistence, export, and reproducibility metadata.

### Phase 6 — Validation and optimization
Run tests, verify actual LFM3 integration, measure performance, and investigate ROCm acceleration without changing the user's global environment.

### Phase 7 — Final documentation
Provide installation commands, launch commands, example experiments, known limitations, and a summary of tested features.

## 18. First task for OpenCode

Start by inspecting the official model implementation and the reference repository.

Then report:
1. The actual LFM3 module hierarchy and which tensors can be instrumented.
2. Whether the reference repository's approach transfers directly or needs adaptation.
3. The verified software dependencies.
4. The proposed frontend and backend architecture.
5. Any technical blockers.

After that, implement Phase 1 and Phase 2. Do not attempt to discover semantic neurons or build a complicated steering system until the GUI can load the model, inspect its architecture, and generate a baseline response.

**Final principle:** This is an interactive research instrument, not a fake visualization. Every displayed layer, activation, feature index, and intervention must correspond to real data from the loaded model.

