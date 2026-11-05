import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextTokenProbe } from "../components/NextTokenProbe";
import { nextTokenFixture } from "../test/fixtures";
import type { ProbeSeed } from "../types";

const api = vi.hoisted(() => ({
  runNextToken: vi.fn(),
}));

vi.mock("../api/client", () => api);

const seed: ProbeSeed = {
  prompt: "Hello",
  system: null,
  steering: {
    module_id: "model.layers.12",
    strength: 1,
    feature_index: null,
    components: [{ feature_index: 1, weight: 1 }],
    normalization: "sigma",
  },
};

beforeEach(() => {
  api.runNextToken.mockReset().mockResolvedValue(nextTokenFixture());
});

function renderProbe(overrides: Partial<Parameters<typeof NextTokenProbe>[0]> = {}) {
  return render(
    <NextTokenProbe open seed={seed} onClose={vi.fn()} {...overrides} />,
  );
}

describe("NextTokenProbe", () => {
  it("runs one seeded probe and shows baseline and steered distributions", async () => {
    renderProbe();

    await waitFor(() => expect(api.runNextToken).toHaveBeenCalledTimes(1));
    const payload = api.runNextToken.mock.calls[0][0];
    expect(payload).toMatchObject({
      prompt: "Hello",
      system: null,
      enable_thinking: false,
      top_k: 10,
      steering: expect.objectContaining({
        module_id: "model.layers.12",
        strength: 1,
        normalization: "sigma",
      }),
    });

    expect(await screen.findByText("0.2300")).toBeInTheDocument();
    expect(screen.getByText("1.200 → 1.100")).toBeInTheDocument();
    expect(screen.getAllByText("A").length).toBeGreaterThan(0);
    expect(screen.getAllByText("The").length).toBeGreaterThan(0);
    expect(screen.getAllByText("-11.0 pp").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+17.0 pp").length).toBeGreaterThan(0);
  });

  it("re-probes when alpha changes", async () => {
    renderProbe();
    await waitFor(() => expect(api.runNextToken).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("Probe steering strength"), {
      target: { value: "2" },
    });

    await waitFor(() => expect(api.runNextToken).toHaveBeenCalledTimes(2), {
      timeout: 2500,
    });
    expect(api.runNextToken.mock.calls[1][0].steering).toMatchObject({
      module_id: "model.layers.12",
      strength: 2,
    });
  });

  it("clears steering and probes the baseline only", async () => {
    const user = userEvent.setup();
    renderProbe();
    await waitFor(() => expect(api.runNextToken).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "Clear steering" }));

    await waitFor(() => expect(api.runNextToken).toHaveBeenCalledTimes(2), {
      timeout: 2500,
    });
    expect(api.runNextToken.mock.calls[1][0].steering).toBeNull();
  });

  it("renders a baseline-only seed without a steered distribution", async () => {
    api.runNextToken.mockResolvedValue(
      nextTokenFixture({
        steered: null,
        kl_divergence: null,
        steering: null,
        candidates: [
          {
            token_id: 100,
            text: "A",
            baseline_probability: 0.42,
            steered_probability: null,
            delta: null,
          },
        ],
      }),
    );
    renderProbe({ seed: { prompt: "Hello", system: null, steering: null } });

    await waitFor(() => expect(api.runNextToken).toHaveBeenCalledTimes(1));
    expect(api.runNextToken.mock.calls[0][0].steering).toBeNull();
    expect(await screen.findByText(/Baseline only/)).toBeInTheDocument();
    expect(screen.getByText("Steered")).toBeInTheDocument();
  });

  it("shows probe errors", async () => {
    api.runNextToken.mockRejectedValueOnce(new Error("GPU out of memory"));
    renderProbe();
    expect(await screen.findByText("GPU out of memory")).toBeInTheDocument();
  });

  it("does not call the API while the model is disabled", async () => {
    renderProbe({ disabled: true });
    await new Promise((resolve) => setTimeout(resolve, 500));
    expect(api.runNextToken).not.toHaveBeenCalled();
    expect(screen.getByText(/Load the model first/)).toBeInTheDocument();
  });

  it("renders nothing while closed", () => {
    renderProbe({ open: false });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
