import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModuleInspector } from "../components/ModuleInspector";
import { activationFixture, moduleDetailFixture } from "../test/fixtures";

const noop = () => undefined;

function renderInspector(overrides: Partial<React.ComponentProps<typeof ModuleInspector>> = {}) {
  return render(
    <ModuleInspector
      detail={null}
      selectedId={null}
      loading={false}
      error={null}
      onSelect={noop}
      steeringTargetId={null}
      onSetSteeringTarget={noop}
      activation={null}
      activationLoading={false}
      activationError={null}
      onActivationQuery={noop}
      {...overrides}
    />,
  );
}

describe("ModuleInspector", () => {
  it("prompts for a selection when none is made", () => {
    renderInspector();
    expect(screen.getByText("Pick a module on the left to look inside it.")).toBeInTheDocument();
  });

  it("shows a loading state for the selected module", () => {
    renderInspector({ selectedId: "model.layers.0.feed_forward", loading: true });
    expect(screen.getByText("Loading model.layers.0.feed_forward…")).toBeInTheDocument();
  });

  it("renders backend errors", () => {
    renderInspector({ selectedId: "bad", error: "Unknown module" });
    expect(screen.getByText("Unknown module")).toBeInTheDocument();
  });

  it("renders real module metadata, parameters, and navigable children", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    renderInspector({ selectedId: "model.layers.0.feed_forward", detail: moduleDetailFixture(), onSelect });

    expect(screen.getByRole("heading", { name: "model.layers.0.feed_forward" })).toBeInTheDocument();
    expect(screen.getByText("Lfm3MLP")).toBeInTheDocument();
    expect(screen.getByText("steerable")).toBeInTheDocument();
    expect(screen.getAllByText("31,457,280")).toHaveLength(2);
    expect(screen.getByText("w2.weight")).toBeInTheDocument();
    expect(screen.getByText("[2048, 7680]")).toBeInTheDocument();
    expect(screen.getByText("torch.bfloat16")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "w1" }));
    expect(onSelect).toHaveBeenCalledWith("model.layers.0.feed_forward.w1");
  });

  it("sets a steerable module as the experiment steering target", async () => {
    const onSetSteeringTarget = vi.fn();
    const user = userEvent.setup();
    renderInspector({
      selectedId: "model.layers.0.feed_forward",
      detail: moduleDetailFixture(),
      onSetSteeringTarget,
      steeringTargetId: "model.layers.22.feed_forward",
    });

    await user.click(screen.getByRole("button", { name: "Set as steering target" }));
    expect(onSetSteeringTarget).toHaveBeenCalledWith("model.layers.0.feed_forward");
  });

  it("marks non-steerable modules and explains why", () => {
    renderInspector({
      selectedId: "model.embed_tokens",
      detail: moduleDetailFixture({
        id: "model.embed_tokens",
        supports_steering: false,
        steering_reason: "Embedding output is an embedding lookup, not a hidden state.",
        child_ids: [],
        parameters: [],
      }),
    });
    expect(screen.getByText("not steerable")).toBeInTheDocument();
    expect(
      screen.getByText("Embedding output is an embedding lookup, not a hidden state."),
    ).toBeInTheDocument();
    expect(screen.getByText("This module holds no parameters of its own.")).toBeInTheDocument();
  });

  it("falls back to a saved activation when live metadata is unavailable", () => {
    renderInspector({
      selectedId: "model.layers.0.feed_forward",
      activation: activationFixture(),
    });
    expect(screen.getByText("saved run")).toBeInTheDocument();
    expect(
      screen.getByText("The model isn't loaded right now, so only the tensor saved with this run is shown below."),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/model\.layers\.0\.feed_forward/).length).toBeGreaterThan(0);
  });
});
