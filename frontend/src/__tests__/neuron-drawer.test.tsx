import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NeuronDrawer } from "../components/NeuronDrawer";
import { neuronSteeringFixture, neuronTraceFixture } from "../test/fixtures";

const api = vi.hoisted(() => ({
  runNeuronTrace: vi.fn(),
  runNeuronSteering: vi.fn(),
}));

vi.mock("../api/client", () => api);

const layers = [
  { layer_index: 0, module_id: "model.layers.0" },
  { layer_index: 1, module_id: "model.layers.1" },
];

beforeEach(() => {
  api.runNeuronTrace.mockReset().mockResolvedValue(neuronTraceFixture());
  api.runNeuronSteering.mockReset().mockResolvedValue(neuronSteeringFixture());
});

function renderDrawer(overrides: Partial<Parameters<typeof NeuronDrawer>[0]> = {}) {
  return render(
    <NeuronDrawer open onClose={vi.fn()} layers={layers} {...overrides} />,
  );
}

describe("NeuronDrawer", () => {
  it("traces a layer, shows clusters, and steers a group", async () => {
    const user = userEvent.setup();
    renderDrawer();

    await user.click(screen.getByRole("button", { name: "Trace neurons" }));
    await waitFor(() => expect(api.runNeuronTrace).toHaveBeenCalledTimes(1));
    const payload = api.runNeuronTrace.mock.calls[0][0];
    expect(payload).toMatchObject({
      prompt: expect.any(String),
      layer_index: 1,
      cluster_k: 6,
      cluster_max_positions: 128,
      settings: expect.objectContaining({ max_new_tokens: 16, temperature: 0 }),
    });

    expect(await screen.findByText("Neuron map")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cluster 1/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /cluster 1/ }));
    expect(screen.getByText("n=200")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Run baseline + steered" }));
    await waitFor(() => expect(api.runNeuronSteering).toHaveBeenCalledTimes(1));
    const steer = api.runNeuronSteering.mock.calls[0][0];
    expect(steer).toMatchObject({
      layer_index: 1,
      alpha: 0.7,
      neurons: [{ neuron: 200, weight: -0.8 }],
      prompt: "Hello",
    });
    expect(await screen.findByText("output changed")).toBeInTheDocument();
    expect(screen.getByText("A steered answer.")).toBeInTheDocument();
  });

  it("does not trace while disabled", async () => {
    renderDrawer({ disabled: true });
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(api.runNeuronTrace).not.toHaveBeenCalled();
    expect(screen.getByText(/Load the model first/)).toBeInTheDocument();
  });

  it("renders nothing while closed", () => {
    renderDrawer({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
