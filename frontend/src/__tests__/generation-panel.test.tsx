import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { GenerationPanel } from "../components/GenerationPanel";
import { baselineFixture, DEFAULT_SETTINGS } from "../test/fixtures";
import type {
  BatchJob,
  PromptSuite,
  SteeringResponse,
  SweepResponse,
  TextMetrics,
} from "../types";

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
    expect(screen.getByText("Load the model first and generation switches on.")).toBeInTheDocument();
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
    expect(screen.getByText("Choose a steering target for this mode.")).toBeInTheDocument();

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
    await user.type(screen.getByLabelText("Strength α"), "5000");

    expect(screen.getByText("Hidden dimension clamped to 2047.")).toBeInTheDocument();
    expect(screen.getByText(/Strength clamped to 1000/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    expect(onRunSteered).toHaveBeenCalledWith(
      expect.any(String),
      null,
      DEFAULT_SETTINGS,
      { module_id: "model.layers.0.feed_forward", feature_index: 2047, strength: 1000 },
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

    const strengths = screen.getByLabelText("Strengths α (comma-separated, 1–32 values)");
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

    const strengths = screen.getByLabelText("Strengths α (comma-separated, 1–32 values)");
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


  it("submits a multi-dimension direction with weights", async () => {
    const onRunSteered = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onRunSteered,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Steered");

    fireEvent.change(screen.getByLabelText("New dimension (0–2047)"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Weight"), { target: { value: "0.5" } });
    await user.click(screen.getByRole("button", { name: "Add dimension" }));
    fireEvent.change(screen.getByLabelText("New dimension (0–2047)"), { target: { value: "9" } });
    fireEvent.change(screen.getByLabelText("Weight"), { target: { value: "-1" } });
    await user.click(screen.getByRole("button", { name: "Add dimension" }));
    fireEvent.change(screen.getByLabelText("Strength α"), { target: { value: "2" } });

    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    expect(onRunSteered).toHaveBeenCalledWith(
      expect.any(String),
      null,
      DEFAULT_SETTINGS,
      {
        module_id: "model.layers.0.feed_forward",
        strength: 2,
        feature_index: null,
        components: [
          { feature_index: 5, weight: 0.5 },
          { feature_index: 9, weight: -1 },
        ],
        normalization: "raw",
      },
      [],
      false,
    );
  });

  it("can steer a single dimension in sigma units", async () => {
    const onRunSteered = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      onRunSteered,
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Steered");

    await user.selectOptions(screen.getByLabelText("Normalization"), "sigma");
    fireEvent.change(screen.getByLabelText("Strength α"), { target: { value: "1.5" } });

    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    expect(onRunSteered).toHaveBeenCalledWith(
      expect.any(String),
      null,
      DEFAULT_SETTINGS,
      {
        module_id: "model.layers.0.feed_forward",
        feature_index: 0,
        strength: 1.5,
        normalization: "sigma",
      },
      [],
      false,
    );
  });

  it("adopts a direction built from two runs", async () => {
    const onClearDirection = vi.fn();
    const user = userEvent.setup();
    renderPanel({
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
      onClearDirection,
      direction: {
        module_id: "model.layers.0.feed_forward",
        components: [
          { feature_index: 3, weight: 1 },
          { feature_index: 7, weight: -0.4 },
        ],
        sigma: [0.1, 0.2],
        norm: 1.2,
        separation: 2.5,
        positions_compared: 6,
        warnings: [],
      },
    });

    expect(screen.getByText("direction · 2 dims")).toBeInTheDocument();
    await openMode(user, "Steered");
    expect(screen.getByRole("button", { name: "Remove dimension 3" })).toHaveTextContent("#3 ×1");
    expect(screen.getByRole("button", { name: "Remove dimension 7" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear direction" }));
    expect(onClearDirection).toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Remove dimension 3" })).not.toBeInTheDocument();
  });

  it("starts a batch from a suite and a module preset", async () => {
    const onStartBatch = vi.fn();
    const user = userEvent.setup();
    const suite: PromptSuite = {
      id: "capability-core",
      name: "Capability checks",
      description: "Ten verifiable tasks.",
      kind: "capability",
      prompts: [{ id: "c1", text: "Do the thing.", tags: [], checks: [] }],
    };
    renderPanel({
      onStartBatch,
      suites: [suite],
      modulePresets: [
        {
          id: "ffn",
          label: "MLP outputs",
          modules: [
            { module_id: "model.layers.0.feed_forward", layer_index: 0 },
            { module_id: "model.layers.1.feed_forward", layer_index: 1 },
          ],
        },
      ],
      selectedSteerableId: "model.layers.0.feed_forward",
      steeringTargetId: "model.layers.0.feed_forward",
    });
    await openMode(user, "Batch");

    await user.selectOptions(screen.getByLabelText("Prompt source"), "capability-core");
    fireEvent.change(screen.getByLabelText("Layers to"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Strengths α (comma-separated)"), {
      target: { value: "1,-1" },
    });
    fireEvent.change(screen.getByLabelText("Repeats (1–10)"), { target: { value: "2" } });

    await user.click(screen.getByRole("button", { name: "Start batch" }));

    expect(onStartBatch).toHaveBeenCalledWith({
      suite_id: "capability-core",
      prompts: [],
      system: null,
      settings: DEFAULT_SETTINGS,
      plan: {
        module_ids: ["model.layers.0.feed_forward"],
        components: [{ feature_index: 0, weight: 1 }],
        normalization: "raw",
        strengths: [1, -1],
        include_control: false,
        control_seed: null,
        repeats: 2,
        baseline_per_prompt: true,
      },
      capture_module_ids: [],
      persist_activations: false,
    });
  });

  it("shows batch progress and cancels it", async () => {
    const onCancelBatch = vi.fn();
    const user = userEvent.setup();
    const job: BatchJob = {
      job_id: "abcd1234" + "0".repeat(24),
      state: "running",
      total: 10,
      completed: 3,
      current: "c1 · model.layers.0 · α 1",
      started_at: null,
      finished_at: null,
      error: null,
      plan: "capability · 1 module × 2 strengths = 10 runs",
      warnings: [],
      runs: [],
      cells: [
        {
          kind: "steered",
          module_id: "model.layers.0.feed_forward",
          strength: 1,
          runs: 2,
          words_mean: 12.5,
          distinct_3_mean: 0.9,
          refusal_rate: 0,
          checks_passed_rate: 0.5,
          lexicon_mean: { warmth: 1.2 },
        },
      ],
    };
    renderPanel({ batchJob: job, onCancelBatch });

    expect(screen.getByText(/Batch abcd1234/)).toBeInTheDocument();
    expect(screen.getByText(/3\/10 runs \(30%\)/)).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel batch" }));
    expect(onCancelBatch).toHaveBeenCalled();
  });

  it("shows measured text metrics next to a result", () => {
    const metrics: TextMetrics = {
      words: 9,
      characters: 48,
      sentences: 2,
      avg_words_per_sentence: 4.5,
      distinct_3: 0.85,
      repeated_4gram_ratio: 0.05,
      refusal: false,
      lexicon: { warmth: 3.2 },
      checks: [],
      checks_passed: null,
      checks_total: null,
    };
    renderPanel({ result: baselineFixture({ metrics }) });
    expect(screen.getByText(/9 words · distinct-3 0.85/)).toBeInTheDocument();
    expect(screen.getByText(/warmth 3.2/)).toBeInTheDocument();
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
