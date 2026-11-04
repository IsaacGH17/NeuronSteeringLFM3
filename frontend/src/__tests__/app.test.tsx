import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import {
  architectureFixture,
  baselineFixture,
  diagnosticsFixture,
  loadedStatus,
  moduleDetailFixture,
  unloadedStatus,
} from "../test/fixtures";

const api = vi.hoisted(() => {
  class ApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }
  return {
    ApiError,
    getStatus: vi.fn(),
    getDiagnostics: vi.fn(),
    loadModel: vi.fn(),
    getArchitecture: vi.fn(),
    getModule: vi.fn(),
    runBaseline: vi.fn(),
    runSteering: vi.fn(),
    runSweep: vi.fn(),
    listExperiments: vi.fn(),
    getExperiment: vi.fn(),
    listActivations: vi.fn(),
    getActivation: vi.fn(),
    downloadExport: vi.fn(),
  };
});

vi.mock("../api/client", () => api);

const baselineId = "a".repeat(32);
const steeredId = "b".repeat(32);

beforeEach(() => {
  api.getStatus.mockReset().mockResolvedValue(unloadedStatus);
  api.getDiagnostics.mockReset().mockResolvedValue(diagnosticsFixture);
  api.loadModel.mockReset().mockResolvedValue(loadedStatus);
  api.getArchitecture.mockReset().mockResolvedValue(architectureFixture());
  api.getModule.mockReset().mockResolvedValue(moduleDetailFixture({ id: "model.layers.0" }));
  api.runBaseline.mockReset().mockResolvedValue(baselineFixture({ id: baselineId }));
  api.runSteering.mockReset().mockResolvedValue({
    baseline: baselineFixture({ id: baselineId }),
    steered: baselineFixture({
      id: steeredId,
      condition: "steered",
      output: "Steered output",
      steering: { module_id: "model.layers.0", feature_index: 0, strength: -12 },
    }),
  });
  api.runSweep.mockReset().mockResolvedValue({
    sweep_id: "c".repeat(32),
    runs: [],
    warnings: [],
  });
  api.listExperiments.mockReset().mockResolvedValue({ total: 0, offset: 0, limit: 20, items: [] });
  api.getExperiment.mockReset();
  api.listActivations.mockReset().mockResolvedValue([]);
  api.getActivation.mockReset().mockResolvedValue(null);
  api.downloadExport.mockReset();
});

async function loadModelAndSelectLayer(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Load model" }));
  await screen.findByText("Lfm3ForCausalLM · 24 layers · hidden 2048");
  await user.click(screen.getByText("Lfm3DecoderLayer"));
  await screen.findByRole("heading", { name: "model.layers.0" });
}

describe("App", () => {
  it("starts unloaded and disables generation", async () => {
    render(<App />);
    expect(await screen.findByText("unloaded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load model" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Run baseline" })).toBeDisabled();
    expect(api.getArchitecture).not.toHaveBeenCalled();
  });

  it("loads the model, discovers the architecture, and selects a real module", async () => {
    const user = userEvent.setup();
    render(<App />);
    await loadModelAndSelectLayer(user);

    expect(api.loadModel).toHaveBeenCalled();
    expect(api.getArchitecture).toHaveBeenCalled();
    expect(api.getModule).toHaveBeenCalledWith("model.layers.0");
    expect(screen.getByText("Lfm3ForCausalLM · 24 layers · hidden 2048")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run baseline" })).toBeEnabled();
  });

  it("automatically restores the architecture when the model is already loaded", async () => {
    api.getStatus.mockResolvedValue(loadedStatus);
    render(<App />);
    expect(
      await screen.findByText("Lfm3ForCausalLM · 24 layers · hidden 2048"),
    ).toBeInTheDocument();
    expect(api.getArchitecture).toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Load model" })).not.toBeInTheDocument();
  });

  it("runs a baseline and displays the measured output", async () => {
    const user = userEvent.setup();
    render(<App />);
    await loadModelAndSelectLayer(user);

    await user.click(screen.getByRole("button", { name: "Run baseline" }));
    expect(
      await screen.findByText("A transformer activation is a hidden-state vector."),
    ).toBeInTheDocument();

    expect(api.runBaseline).toHaveBeenCalledWith(
      "Explain what a transformer activation is in two sentences.",
      null,
      expect.objectContaining({ max_new_tokens: 128, temperature: 0, top_p: 1 }),
      [],
      false,
    );
  });

  it("runs the baseline + steered pair with the selected intervention", async () => {
    const user = userEvent.setup();
    render(<App />);
    await loadModelAndSelectLayer(user);

    await user.click(screen.getByRole("button", { name: "Set as steering target" }));
    await user.click(screen.getByRole("button", { name: "Steered" }));
    fireEvent.change(screen.getByLabelText("Strength α"), { target: { value: "-12" } });
    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    expect(await screen.findByText("Steered output")).toBeInTheDocument();
    expect(
      screen.getByText("Baseline vs steered · hook: model.layers.0 · dim 0 · α -12"),
    ).toBeInTheDocument();
    expect(api.runSteering).toHaveBeenCalledWith(
      expect.any(String),
      null,
      expect.any(Object),
      { module_id: "model.layers.0", feature_index: 0, strength: -12 },
      [],
      false,
    );
  });

  it("runs a strength sweep and renders its results", async () => {
    const user = userEvent.setup();
    render(<App />);
    await loadModelAndSelectLayer(user);

    await user.click(screen.getByRole("button", { name: "Set as steering target" }));
    await user.click(screen.getByRole("button", { name: "Sweep" }));
    await user.click(screen.getByRole("button", { name: "Run sweep" }));
    expect(await screen.findByText(/Sweep cccccccc · 0 runs/)).toBeInTheDocument();
    expect(api.runSweep).toHaveBeenCalledWith(
      expect.objectContaining({
        module_id: "model.layers.0",
        strengths: [-20, -10, 0, 10, 20],
      }),
    );
  });

  it("loads the captured tensor after a run that requested activations", async () => {
    api.runBaseline.mockResolvedValue(
      baselineFixture({
        id: baselineId,
        captured_modules: ["model.layers.0"],
      }),
    );
    api.getActivation.mockResolvedValue({
      experiment_id: baselineId,
      module_id: "model.layers.0",
      shape: [3, 2048],
      dtype: "float32",
      aggregation: "All positions",
      prompt_token_count: 2,
      tokens: [],
      heatmap: { positions: [], features: [], values: [], sampling: "none" },
      histogram: { bin_edges: [], counts: [] },
      stats: { minimum: 0, maximum: 0, mean: 0, std: 0, abs_mean: 0 },
      position: 0,
      top_dimensions: [],
      feature_series: { feature: 0, positions: [], values: [] },
      comparison: null,
      warnings: [],
    });

    const user = userEvent.setup();
    render(<App />);
    await loadModelAndSelectLayer(user);

    await user.click(screen.getByLabelText("Capture activations for model.layers.0"));
    await user.click(screen.getByRole("button", { name: "Run baseline" }));

    await waitFor(() =>
      expect(api.getActivation).toHaveBeenCalledWith(baselineId, "model.layers.0", {}),
    );
    expect(await screen.findByText(/shape \[3, 2048\]/)).toBeInTheDocument();
  });

  it("reports load errors without pretending the model is ready", async () => {
    api.loadModel.mockRejectedValue(new api.ApiError("insufficient VRAM", 507));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Load model" }));
    expect(await screen.findByText("insufficient VRAM")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load model" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run baseline" })).toBeDisabled();
  });

  it("reports generation errors in the steering panel", async () => {
    api.runBaseline.mockRejectedValue(new api.ApiError("model is not loaded", 409));
    const user = userEvent.setup();
    render(<App />);
    await loadModelAndSelectLayer(user);

    await user.click(screen.getByRole("button", { name: "Run baseline" }));
    expect(await screen.findByText("model is not loaded")).toBeInTheDocument();
  });
});
