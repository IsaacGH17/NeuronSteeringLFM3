import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ExperimentHistory } from "../components/ExperimentHistory";
import { detailFixture, summaryFixture } from "../test/fixtures";

const api = vi.hoisted(() => ({
  listExperiments: vi.fn(),
  getExperiment: vi.fn(),
  downloadExport: vi.fn(),
  computeDirection: vi.fn(),
}));

vi.mock("../api/client", () => api);

const noop = () => undefined;

const first = summaryFixture({ id: "a".repeat(32), prompt_preview: "First prompt", output_preview: "First output" });
const second = summaryFixture({
  id: "b".repeat(32),
  prompt_preview: "Second prompt",
  output_preview: "Second output",
  condition: "steered",
  condition_detail: "negative",
  module_id: "model.layers.0.feed_forward",
  feature_index: 5,
  strength: -10,
});

beforeEach(() => {
  api.listExperiments.mockReset().mockResolvedValue({ total: 0, offset: 0, limit: 20, items: [] });
  api.getExperiment.mockReset().mockImplementation((id: string) =>
    Promise.resolve(
      detailFixture({
        ...(id === first.id ? first : second),
        output: `Full output ${id.slice(0, 1)}`,
        steering:
          id === second.id
            ? { module_id: "model.layers.0.feed_forward", feature_index: 5, strength: -10 }
            : null,
      }),
    ),
  );
  api.downloadExport.mockReset();
  api.computeDirection.mockReset();
});

function renderHistory(onOpenActivations = noop) {
  return render(<ExperimentHistory refreshTrigger={0} onOpenActivations={onOpenActivations} />);
}

describe("ExperimentHistory", () => {
  it("loads the saved-run count even while collapsed", async () => {
    api.listExperiments.mockResolvedValue({ total: 3, offset: 0, limit: 20, items: [] });
    renderHistory();
    expect(await screen.findByText("3 saved runs")).toBeInTheDocument();
  });

  it("shows an empty state when no experiments exist", async () => {
    const user = userEvent.setup();
    renderHistory();
    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    expect(
      await screen.findByText(
        "Nothing saved yet. Run a baseline, steered, or sweep experiment and it will land here.",
      ),
    ).toBeInTheDocument();
  });

  it("lists saved runs and opens their stored detail", async () => {
    api.listExperiments.mockResolvedValue({ total: 2, offset: 0, limit: 20, items: [first, second] });
    const user = userEvent.setup();
    renderHistory();

    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    const firstRow = (await screen.findByText("First prompt")).closest("tr");
    expect(firstRow).not.toBeNull();
    expect(within(firstRow as HTMLElement).getByText("First output")).toBeInTheDocument();
    expect(within(firstRow as HTMLElement).getByText("baseline")).toBeInTheDocument();

    await user.click(firstRow as HTMLElement);
    expect(await screen.findByText("Full output a")).toBeInTheDocument();
    expect(screen.getByText("Run ID")).toBeInTheDocument();
    expect(screen.getByText(first.id)).toBeInTheDocument();
    expect(api.getExperiment).toHaveBeenCalledWith(first.id);
  });

  it("renders steering metadata for a steered run", async () => {
    api.listExperiments.mockResolvedValue({ total: 1, offset: 0, limit: 20, items: [second] });
    const user = userEvent.setup();
    renderHistory();

    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    const row = (await screen.findByText("Second prompt")).closest("tr") as HTMLElement;
    expect(within(row).getByText("steered (negative)")).toBeInTheDocument();
    expect(within(row).getByText("model.layers.0.feed_forward · dim 5 · α -10")).toBeInTheDocument();

    await user.click(row);
    expect(await screen.findByText("Steering")).toBeInTheDocument();
    expect(screen.getAllByText("model.layers.0.feed_forward").length).toBeGreaterThan(0);
  });

  it("compares up to two selected runs", async () => {
    api.listExperiments.mockResolvedValue({ total: 2, offset: 0, limit: 20, items: [first, second] });
    const user = userEvent.setup();
    renderHistory();

    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    await user.click(await screen.findByLabelText(`Select ${first.id} for comparison`));
    await user.click(screen.getByLabelText(`Select ${second.id} for comparison`));

    expect(await screen.findByText("Compare outputs")).toBeInTheDocument();
    await waitFor(() => expect(api.getExperiment).toHaveBeenCalledWith(second.id));
    expect(screen.getAllByText("Full output a").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Full output b").length).toBeGreaterThan(0);
  });

  it("exports only the selected runs", async () => {
    api.listExperiments.mockResolvedValue({ total: 2, offset: 0, limit: 20, items: [first, second] });
    const user = userEvent.setup();
    renderHistory();

    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    await user.click(await screen.findByLabelText(`Select ${first.id} for comparison`));
    await user.click(screen.getByRole("button", { name: "Export selected JSONL" }));
    expect(api.downloadExport).toHaveBeenCalledWith({ format: "jsonl", ids: [first.id] });

    await user.click(screen.getByRole("button", { name: "Export all CSV" }));
    expect(api.downloadExport).toHaveBeenCalledWith({ format: "csv" });
  });

  it("opens a restored activation for the first captured module", async () => {
    const withActivations = summaryFixture({
      id: first.id,
      prompt_preview: "First prompt",
      has_activations: true,
      activation_source: "snapshot",
    });
    api.listExperiments.mockResolvedValue({ total: 1, offset: 0, limit: 20, items: [withActivations] });
    api.getExperiment.mockResolvedValue(
      detailFixture({
        ...withActivations,
        captured_modules: ["model.layers.0.feed_forward"],
        captures: [{ module_id: "model.layers.0.feed_forward", shape: [4, 2048], dtype: "float32" }],
      }),
    );
    const onOpenActivations = vi.fn();
    const user = userEvent.setup();
    renderHistory(onOpenActivations);

    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    await user.click(await screen.findByText("First prompt"));
    await user.click(await screen.findByRole("button", { name: "Open activations" }));

    expect(onOpenActivations).toHaveBeenCalledWith(first.id, "model.layers.0.feed_forward");
  });

  it("paginates using the backend offset", async () => {
    api.listExperiments.mockResolvedValue({ total: 30, offset: 0, limit: 20, items: [first] });
    const user = userEvent.setup();
    renderHistory();

    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    await user.click(await screen.findByRole("button", { name: "Next" }));

    await waitFor(() =>
      expect(api.listExperiments).toHaveBeenLastCalledWith({ limit: 20, offset: 20 }),
    );
  });

  it("builds a steering direction from the two selected runs", async () => {
    const user = userEvent.setup();
    const onDirectionBuilt = vi.fn();
    const withCaptures = summaryFixture({
      id: "c".repeat(32),
      prompt_preview: "Warm run",
      has_activations: true,
      activation_source: "live",
    });
    const alsoCaptured = summaryFixture({
      id: "d".repeat(32),
      prompt_preview: "Clinical run",
      condition: "steered",
      has_activations: true,
      activation_source: "live",
    });
    api.listExperiments.mockResolvedValue({
      total: 2,
      offset: 0,
      limit: 20,
      items: [withCaptures, alsoCaptured],
    });
    api.computeDirection.mockResolvedValue({
      module_id: "model.layers.0.feed_forward",
      components: [{ feature_index: 3, weight: 1 }],
      sigma: [0.25],
      norm: 1.1,
      separation: 2.4,
      positions_compared: 6,
      warnings: [],
    });

    render(
      <ExperimentHistory
        refreshTrigger={0}
        onOpenActivations={noop}
        onDirectionBuilt={onDirectionBuilt}
      />,
    );
    await user.click(screen.getByRole("button", { name: /Experiment history/ }));

    await user.click(await screen.findByLabelText(`Select ${withCaptures.id} for comparison`));
    await user.click(screen.getByLabelText(`Select ${alsoCaptured.id} for comparison`));
    await user.click(screen.getByRole("button", { name: "Build direction" }));

    await waitFor(() => expect(onDirectionBuilt).toHaveBeenCalled());
    expect(api.computeDirection).toHaveBeenCalledWith({
      positive_experiment_id: withCaptures.id,
      negative_experiment_id: alsoCaptured.id,
      top_k: 8,
    });
  });

  it("reports a failed direction build without clearing the comparison", async () => {
    const user = userEvent.setup();
    const onDirectionBuilt = vi.fn();
    const withCaptures = summaryFixture({
      id: "e".repeat(32),
      has_activations: true,
      activation_source: "live",
    });
    const other = summaryFixture({
      id: "f".repeat(32),
      has_activations: true,
      activation_source: "live",
    });
    api.listExperiments.mockResolvedValue({
      total: 2,
      offset: 0,
      limit: 20,
      items: [withCaptures, other],
    });
    api.computeDirection.mockRejectedValue(new Error("The two captured states are identical"));

    render(
      <ExperimentHistory
        refreshTrigger={0}
        onOpenActivations={noop}
        onDirectionBuilt={onDirectionBuilt}
      />,
    );
    await user.click(screen.getByRole("button", { name: /Experiment history/ }));
    await user.click(await screen.findByLabelText(`Select ${withCaptures.id} for comparison`));
    await user.click(screen.getByLabelText(`Select ${other.id} for comparison`));
    await user.click(screen.getByRole("button", { name: "Build direction" }));

    expect(await screen.findByText("The two captured states are identical")).toBeInTheDocument();
    expect(onDirectionBuilt).not.toHaveBeenCalled();
  });
});
