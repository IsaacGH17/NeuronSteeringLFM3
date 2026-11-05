import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { InterpretDrawer } from "../components/InterpretDrawer";
import type { LensResponse, PatchResponse } from "../types";
import { patchSweepFixture } from "../test/fixtures";

const api = vi.hoisted(() => ({
  runLens: vi.fn(),
  runPatch: vi.fn(),
  runPatchSweep: vi.fn(),
}));

vi.mock("../api/client", () => api);

const layers = [
  { layer_index: 0, module_id: "model.layers.0" },
  { layer_index: 1, module_id: "model.layers.1" },
];

const lensResponse: LensResponse = {
  prompt: "Hello",
  system: null,
  output: "A transformer activation is a hidden-state vector.",
  prompt_tokens: 3,
  position: 2,
  tokens: [
    { index: 0, token_id: 1, text: "Hello", phase: "prompt" },
    { index: 1, token_id: 2, text: " world", phase: "prompt" },
    { index: 2, token_id: 3, text: " activated", phase: "generated" },
  ],
  layers: [
    {
      layer_index: 0,
      module_id: "embeddings",
      top1_ids: [1, 2, 3],
      top1_probs: [0.5, 0.4, 0.3],
      top_tokens: [{ token_id: 1, text: "Hello", probability: 0.5 }],
      ablation_kl: null,
      ablation_top1: null,
    },
    {
      layer_index: 1,
      module_id: "model.layers.0",
      top1_ids: [4],
      top1_probs: [0.9],
      top_tokens: [{ token_id: 4, text: "activation", probability: 0.9 }],
      ablation_kl: 0.123,
      ablation_top1: { token_id: 4, text: "activation", probability: 0.8 },
    },
  ],
  vocab: { "1": "Hello", "4": "activation" },
  warnings: [],
};

const patchResponse: PatchResponse = {
  module_id: "model.layers.1",
  position: 2,
  donor_output: "Donor output text.",
  baseline_output: "Baseline output text.",
  patched_output: "Patched output text.",
  changed: true,
  prompt_tokens: 3,
  warnings: [],
};

beforeEach(() => {
  api.runLens.mockReset().mockResolvedValue(lensResponse);
  api.runPatch.mockReset().mockResolvedValue(patchResponse);
  api.runPatchSweep.mockReset().mockResolvedValue(patchSweepFixture());
});

function renderDrawer() {
  return render(
    <InterpretDrawer
      open
      onClose={() => undefined}
      layers={layers}
      defaultModuleId="model.layers.1"
    />,
  );
}

describe("InterpretDrawer", () => {
  it("runs a patch sweep and inspects a cell", async () => {
    const user = userEvent.setup();
    renderDrawer();

    await user.click(screen.getByRole("button", { name: "Patch sweep" }));
    await user.click(screen.getByRole("button", { name: "Run patch sweep" }));

    expect(api.runPatchSweep).toHaveBeenCalledWith(
      expect.objectContaining({
        recipient_prompt: expect.any(String),
        donor_prompt: expect.any(String),
        settings: expect.objectContaining({ max_new_tokens: 16, temperature: 0 }),
        layer_indices: [0, 1],
        positions: [-1],
      }),
    );
    expect(await screen.findByText("4 cells")).toBeInTheDocument();
    expect(screen.getByText("2 outputs changed")).toBeInTheDocument();
    expect(screen.getByText("1 first tokens flipped")).toBeInTheDocument();

    const cell = screen.getByRole("button", {
      name: "Layer 1 position 10: first token changed",
    });
    await user.click(cell);
    expect(screen.getByText("Layer 1 · position 10")).toBeInTheDocument();
    expect(screen.getByText("Another answer.")).toBeInTheDocument();
  });

  it("runs the logit lens and renders layer predictions and ablation", async () => {
    const user = userEvent.setup();
    renderDrawer();

    await user.click(screen.getByRole("button", { name: "Run logit lens" }));

    expect(api.runLens).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: expect.any(String),
        settings: expect.objectContaining({ max_new_tokens: 16, temperature: 0 }),
        top_k: 5,
        position: null,
      }),
    );
    expect(await screen.findByText("90.0%")).toBeInTheDocument();
    expect(screen.getByText("0.1230")).toBeInTheDocument();
    expect(screen.getByText("embeddings")).toBeInTheDocument();
  });

  it("re-runs the lens at the clicked token position", async () => {
    const user = userEvent.setup();
    renderDrawer();
    await user.click(screen.getByRole("button", { name: "Run logit lens" }));
    await screen.findByText("90.0%");

    await user.click(screen.getByRole("button", { name: "Position 0" }));
    expect(api.runLens).toHaveBeenLastCalledWith(expect.objectContaining({ position: 0 }));
  });

  it("runs an activation patch and shows the three outputs", async () => {
    const user = userEvent.setup();
    renderDrawer();
    await user.click(screen.getByRole("button", { name: "Activation patching" }));
    await user.type(screen.getByLabelText("Donor"), " Describe a rainy morning.");

    await user.click(screen.getByRole("button", { name: "Run patch" }));

    expect(api.runPatch).toHaveBeenCalledWith(
      expect.objectContaining({
        recipient_prompt: expect.any(String),
        donor_prompt: expect.stringContaining("Describe a rainy morning."),
        module_id: "model.layers.1",
      }),
    );
    expect(await screen.findByText("Donor output text.")).toBeInTheDocument();
    expect(screen.getByText("Baseline output text.")).toBeInTheDocument();
    expect(screen.getByText("Patched output text.")).toBeInTheDocument();
    expect(screen.getByText("output changed")).toBeInTheDocument();
  });

  it("reports lens failures without clearing the form", async () => {
    const user = userEvent.setup();
    api.runLens.mockRejectedValue(new Error("model is not loaded"));
    renderDrawer();
    await user.click(screen.getByRole("button", { name: "Run logit lens" }));
    expect(await screen.findByText("model is not loaded")).toBeInTheDocument();
  });
});
