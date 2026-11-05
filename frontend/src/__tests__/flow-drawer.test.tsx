import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FlowDrawer } from "../components/FlowDrawer";
import { baselineFixture, flowFixture } from "../test/fixtures";

const api = vi.hoisted(() => ({
  runFlow: vi.fn(),
  runSteering: vi.fn(),
}));

vi.mock("../api/client", () => api);

const flow = flowFixture();

beforeEach(() => {
  api.runFlow.mockReset().mockResolvedValue(flow);
  api.runSteering.mockReset().mockResolvedValue({
    baseline: baselineFixture({
      id: "a".repeat(32),
      output: "An activation is a hidden-state vector.",
    }),
    steered: baselineFixture({
      id: "b".repeat(32),
      condition: "steered",
      output: "An activation is a steered vector.",
      steering: {
        module_id: "model.layers.2",
        strength: 2,
        feature_index: null,
        components: [
          { feature_index: 20, weight: 1 },
          { feature_index: 25, weight: 0.5 },
        ],
        normalization: "sigma",
      },
    }),
  });
});

function renderDrawer(overrides: Partial<Parameters<typeof FlowDrawer>[0]> = {}) {
  return render(
    <FlowDrawer
      open
      onClose={vi.fn()}
      onUseDirection={vi.fn()}
      onExperimentRecorded={vi.fn()}
      {...overrides}
    />,
  );
}

async function trace(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Trace flow" }));
  await screen.findByText("An activation is a hidden-state vector.");
}

describe("FlowDrawer", () => {
  it("renders nothing while closed", () => {
    renderDrawer({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("traces one real generation and draws every captured layer", async () => {
    const user = userEvent.setup();
    renderDrawer();

    await trace(user);

    expect(api.runFlow).toHaveBeenCalledTimes(1);
    const payload = api.runFlow.mock.calls[0][0];
    expect(payload.prompt).toContain("transformer activation");
    expect(payload.settings.temperature).toBe(0);
    expect(payload.settings.enable_thinking).toBe(false);
    expect(payload.cluster_k).toBe(6);
    expect(payload.include_ablation).toBe(true);

    expect(
      screen.getByRole("group", { name: /Layer flow graph with 3 layers and 2 measured connections/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Layer 2 cluster 0/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Token 3: blue/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Layer 1" })).toBeInTheDocument();
    expect(screen.getByText("ablation KL")).toBeInTheDocument();
  });

  it("selects a cluster, shows its dimensions, and steers the group", async () => {
    const user = userEvent.setup();
    const onExperimentRecorded = vi.fn();
    renderDrawer({ onExperimentRecorded });

    await trace(user);
    await user.click(screen.getByRole("button", { name: /Layer 2 cluster 0/ }));

    expect(screen.getByRole("heading", { name: "Layer 2 · cluster 0" })).toBeInTheDocument();
    expect(screen.getByText("#20")).toBeInTheDocument();
    expect(screen.getByText("#25")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Strength α/), { target: { value: "2" } });
    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));

    await waitFor(() => expect(api.runSteering).toHaveBeenCalledTimes(1));
    const [prompt, system, settings, spec] = api.runSteering.mock.calls[0];
    expect(prompt).toBe(flow.prompt);
    expect(system).toBeNull();
    expect(settings).toEqual(flow.settings);
    expect(spec).toMatchObject({
      module_id: "model.layers.2",
      strength: 2,
      normalization: "sigma",
      feature_index: null,
    });
    expect(spec.components).toEqual(flow.layers[2].clusters[0].components);

    expect(await screen.findByText("output changed")).toBeInTheDocument();
    expect(screen.getByText("An activation is a steered vector.")).toBeInTheDocument();
    expect(onExperimentRecorded).toHaveBeenCalled();
  });

  it("compares two traces and shows the ablation delta", async () => {
    const user = userEvent.setup();
    const left = flowFixture();
    const right = flowFixture({
      prompt: "Describe a rainy morning.",
      output: "Rain on the window.",
    });
    right.layers = right.layers.map((layer, index) => ({
      ...layer,
      write: layer.write + 0.05 * (index + 1),
      ablation_kl: (layer.ablation_kl ?? 0) + 0.1,
    }));
    api.runFlow.mockReset().mockResolvedValueOnce(left).mockResolvedValueOnce(right);
    renderDrawer();

    await user.click(screen.getByRole("button", { name: "Compare" }));
    await user.click(screen.getByRole("button", { name: "Trace both" }));

    await waitFor(() => expect(api.runFlow).toHaveBeenCalledTimes(2));
    const first = api.runFlow.mock.calls[0][0];
    const second = api.runFlow.mock.calls[1][0];
    expect(first.prompt).toContain("transformer activation");
    expect(second.prompt).toBe("Describe a rainy morning.");
    expect(first.include_ablation).toBe(true);
    expect(second.include_ablation).toBe(true);

    expect(
      await screen.findByRole("group", { name: /Flow diff graph with 3 layers/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("Δ stream · B − A")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Layer 2, A attention/ }));
    expect(screen.getByRole("heading", { name: "Layer 2 · B − A" })).toBeInTheDocument();
    expect(screen.getByText("ablation KL")).toBeInTheDocument();
  });

  it("hands the selected cluster to the next-token probe", async () => {
    const user = userEvent.setup();
    const onProbeCluster = vi.fn();
    renderDrawer({ onProbeCluster });

    await trace(user);
    await user.click(screen.getByRole("button", { name: /Layer 2 cluster 0/ }));
    await user.click(screen.getByRole("button", { name: "Probe next token" }));

    expect(onProbeCluster).toHaveBeenCalledWith({
      prompt: flow.prompt,
      system: null,
      steering: expect.objectContaining({
        module_id: "model.layers.2",
        strength: 1,
        normalization: "sigma",
        components: flow.layers[2].clusters[0].components,
      }),
    });
  });

  it("sends the selected cluster to the steering panel", async () => {
    const user = userEvent.setup();
    const onUseDirection = vi.fn();
    const onClose = vi.fn();
    renderDrawer({ onUseDirection, onClose });

    await trace(user);
    await user.click(screen.getByRole("button", { name: /Layer 2 cluster 0/ }));
    await user.click(
      screen.getByRole("button", { name: "Use these dimensions in the steering panel" }),
    );

    expect(onUseDirection).toHaveBeenCalledWith(
      expect.objectContaining({
        module_id: "model.layers.2",
        components: flow.layers[2].clusters[0].components,
        separation: flow.layers[2].clusters[0].cohesion,
        positions_compared: flow.cluster_positions,
      }),
    );
    expect(onClose).toHaveBeenCalled();
  });

  it("reports a trace failure and keeps the drawer usable", async () => {
    const user = userEvent.setup();
    api.runFlow.mockRejectedValueOnce(new Error("GPU out of memory"));
    renderDrawer();

    await user.click(screen.getByRole("button", { name: "Trace flow" }));
    expect(await screen.findByText("GPU out of memory")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Trace flow" })).toBeEnabled();
  });
});
