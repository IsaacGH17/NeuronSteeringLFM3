import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { GenerationPanel } from "../components/GenerationPanel";
import { baselineFixture, DEFAULT_SETTINGS } from "../test/fixtures";
import type { SteeringResponse, SweepResponse } from "../types";

const noop = () => undefined;

function renderPanel(overrides: Partial<ComponentProps<typeof GenerationPanel>> = {}) {
  return render(
    <GenerationPanel
      disabled={false}
      running={false}
      result={null}
      pair={null}
      sweepResult={null}
      sweepError={null}
      error={null}
      selectedSteerableId={null}
      steeringTargetId={null}
      onSetSteeringTarget={noop}
      hiddenSize={2048}
      onRun={noop}
      onRunSteered={noop}
      onRunSweep={noop}
      {...overrides}
    />,
  );
}

async function openMode(user: ReturnType<typeof userEvent.setup>, label: string) {
  await user.click(screen.getByRole("button", { name: label }));
}

describe("GenerationPanel", () => {
  it("disables generation and explains why when the model is not loaded", async () => {
    const user = userEvent.setup();
    renderPanel({ disabled: true });
    expect(screen.getByText("Load the model to enable generation.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run baseline" })).toBeDisabled();

    await openMode(user, "Steered");
    expect(screen.getByRole("button", { name: "Run baseline + steered" })).toBeDisabled();
  });

  it("does not run steered or sweep modes until a target is set", async () => {
    const onRunSteered = vi.fn();
    const onRunSweep = vi.fn();
    const user = userEvent.setup();
    renderPanel({ onRunSteered, onRunSweep, selectedSteerableId: "model.layers.0.feed_forward" });

    expect(screen.getByText("No module selected")).toBeInTheDocument();
    await openMode(user, "Steered");
    expect(screen.getByRole("button", { name: "Run baseline + steered" })).toBeDisabled();
    expect(screen.getByText("Set a steering target to run this mode.")).toBeInTheDocument();

    await openMode(user, "Sweep");
    expect(screen.getByRole("button", { name: "Run sweep" })).toBeDisabled();
  });

  it("sets the inspected module as the steering target from the target card", async () => {
    const onSetSteeringTarget = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onSetSteeringTarget,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.22.feed_forward",
    });

    expect(screen.getByText("model.layers.22.feed_forward")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use inspected module" }));
    expect(onSetSteeringTarget).toHaveBeenCalledWith("model.layers.0.feed_forward");
  });

  it("submits the prompt, system instruction, and generation settings", async () => {
    const onRun = vi.fn();
    const user = userEvent.setup();
    renderPanel({ onRun });

    await user.clear(screen.getByLabelText("Prompt"));
    await user.type(screen.getByLabelText("Prompt"), "Say ready");
    await user.type(screen.getByLabelText("System instruction"), "Be terse");
    await user.clear(screen.getByLabelText("Max new tokens"));
    await user.type(screen.getByLabelText("Max new tokens"), "16");
    await user.clear(screen.getByLabelText("Temperature"));
    await user.type(screen.getByLabelText("Temperature"), "0.7");
    await user.clear(screen.getByLabelText("Top-p"));
    await user.type(screen.getByLabelText("Top-p"), "0.9");
    await user.type(screen.getByLabelText("Seed"), "42");
    await user.click(screen.getByLabelText("Enable thinking (off by default)"));

    await user.click(screen.getByRole("button", { name: "Run baseline" }));

    expect(onRun).toHaveBeenCalledWith(
      "Say ready",
      "Be terse",
      { max_new_tokens: 16, temperature: 0.7, top_p: 0.9, seed: 42, enable_thinking: true },
      [],
      false,
    );
  });

  it("captures the inspected module only when the capture checkbox is enabled", async () => {
    const onRun = vi.fn();
    const user = userEvent.setup();
    renderPanel({ onRun, selectedSteerableId: "model.layers.0.feed_forward" });

    await user.click(screen.getByLabelText("Capture activations for model.layers.0.feed_forward"));
    await user.click(screen.getByRole("button", { name: "Run baseline" }));

    expect(onRun).toHaveBeenCalledWith(
      expect.any(String),
      null,
      DEFAULT_SETTINGS,
      ["model.layers.0.feed_forward"],
      false,
    );
  });

  it("sends an explicit persist flag when snapshot persistence is enabled", async () => {
    const onRun = vi.fn();
    const user = userEvent.setup();
    renderPanel({ onRun, selectedSteerableId: "model.layers.0.feed_forward" });

    await user.click(screen.getByLabelText("Persist activation snapshot (saves tensor data to disk)"));
    await user.click(screen.getByRole("button", { name: "Run baseline" }));

    expect(onRun).toHaveBeenCalledWith(expect.any(String), null, DEFAULT_SETTINGS, [], true);
  });

  it("submits a steered run with the selected dimension and strength", async () => {
    const onRunSteered = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onRunSteered,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Steered");

    const featureInput = screen.getByLabelText("Hidden dimension (0–2047)");
    await user.clear(featureInput);
    await user.type(featureInput, "500");
    fireEvent.change(screen.getByLabelText("Strength α"), { target: { value: "-12" } });

    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    expect(onRunSteered).toHaveBeenCalledWith(
      expect.any(String),
      null,
      DEFAULT_SETTINGS,
      { module_id: "model.layers.0.feed_forward", feature_index: 500, strength: -12 },
      [],
      false,
    );
  });

  it("clamps out-of-range dimensions and strengths before submission", async () => {
    const onRunSteered = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onRunSteered,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
      hiddenSize: 2048,
    });
    await openMode(user, "Steered");

    const featureInput = screen.getByLabelText("Hidden dimension (0–2047)");
    await user.clear(featureInput);
    await user.type(featureInput, "9999");
    await user.clear(screen.getByLabelText("Strength α"));
    await user.type(screen.getByLabelText("Strength α"), "100");

    expect(screen.getByText("Hidden dimension clamped to 2047.")).toBeInTheDocument();
    expect(screen.getByText("Strength clamped to 49.5.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    expect(onRunSteered).toHaveBeenCalledWith(
      expect.any(String),
      null,
      DEFAULT_SETTINGS,
      { module_id: "model.layers.0.feed_forward", feature_index: 2047, strength: 49.5 },
      [],
      false,
    );
  });

  it("resets the strength slider to zero", async () => {
    const user = userEvent.setup();
    renderPanel({
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Steered");

    const slider = screen.getByLabelText("Strength slider: 0");
    await user.clear(screen.getByLabelText("Strength α"));
    await user.type(screen.getByLabelText("Strength α"), "5");

    const reset = screen.getByRole("button", { name: "Reset strength to 0" });
    expect(reset).toBeEnabled();
    await user.click(reset);
    expect(screen.getByLabelText("Strength slider: 0")).toBe(slider);
  });

  it("rejects a malformed sweep before calling the API", async () => {
    const onRunSweep = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onRunSweep,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Sweep");

    const strengths = screen.getByLabelText("Strengths α (comma-separated, 1–9 values)");
    await user.clear(strengths);
    await user.type(strengths, "1,oops");
    await user.click(screen.getByRole("button", { name: "Run sweep" }));

    expect(screen.getByText('Invalid strength: "oops"')).toBeInTheDocument();
    expect(onRunSweep).not.toHaveBeenCalled();
  });

  it("submits a valid sweep with parsed strengths and a seeded control", async () => {
    const onRunSweep = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onRunSweep,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Sweep");

    const strengths = screen.getByLabelText("Strengths α (comma-separated, 1–9 values)");
    await user.clear(strengths);
    await user.type(strengths, "-10, 10");
    await user.click(screen.getByLabelText("Include random-feature control"));
    await user.type(screen.getByLabelText("Control seed (optional)"), "7");
    await user.click(screen.getByRole("button", { name: "Run sweep" }));

    expect(onRunSweep).toHaveBeenCalledWith({
      prompt: expect.any(String),
      system: null,
      settings: DEFAULT_SETTINGS,
      module_id: "model.layers.0.feed_forward",
      feature_index: 0,
      strengths: [-10, 10],
      include_control: true,
      control_seed: 7,
      repeats: 1,
      capture_module_ids: [],
      persist_activations: false,
    });
  });

  it("renders a baseline result with measured metadata", () => {
    renderPanel({ result: baselineFixture({ warnings: ["greedy repeat warning"] }) });
    expect(screen.getByText("A transformer activation is a hidden-state vector.")).toBeInTheDocument();
    expect(screen.getByText("12 tokens")).toBeInTheDocument();
    expect(screen.getByText("9.7 tok/s")).toBeInTheDocument();
    expect(screen.getByText("greedy repeat warning")).toBeInTheDocument();
  });

  it("renders baseline and steered outputs side by side", () => {
    const baseline = baselineFixture();
    const steered = baselineFixture({
      id: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      condition: "steered",
      output: "The steered output.",
      steering: { module_id: "model.layers.0.feed_forward", feature_index: 500, strength: -12 },
    });
    const pair: SteeringResponse = { baseline, steered };
    renderPanel({ pair });
    expect(
      screen.getByText("Baseline vs steered · hook: model.layers.0.feed_forward · dim 500 · α -12"),
    ).toBeInTheDocument();
    expect(screen.getByText("The steered output.")).toBeInTheDocument();
    expect(screen.getByText(baseline.output)).toBeInTheDocument();
  });

  it("renders sweep rows and toggles a stored output", async () => {
    const user = userEvent.setup();
    const baseline = baselineFixture();
    const steered = baselineFixture({
      id: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      condition: "steered",
      output: "Swept output text",
      steering: { module_id: "model.layers.0.feed_forward", feature_index: 3, strength: 10 },
    });
    const sweep: SweepResponse = {
      sweep_id: "cccccccccccccccccccccccccccccccc",
      warnings: [],
      runs: [
        { kind: "baseline", sweep_index: 0, repeat_index: 0, run: baseline },
        { kind: "strength", sweep_index: 1, repeat_index: 0, run: steered },
      ],
    };
    renderPanel({ sweepResult: sweep });

    expect(screen.getByText(/Sweep cccccccc · 2 runs/)).toBeInTheDocument();
    expect(screen.getAllByText("Show output")).toHaveLength(2);
    await user.click(screen.getAllByText("Show output")[1]);
    expect(screen.getByText("Swept output text")).toBeInTheDocument();
  });

  it("shows API errors", () => {
    renderPanel({ error: "backend exploded", sweepError: "sweep exploded" });
    expect(screen.getByText("backend exploded")).toBeInTheDocument();
    expect(screen.getByText("sweep exploded")).toBeInTheDocument();
  });
});
