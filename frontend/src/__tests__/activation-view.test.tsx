import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ActivationView } from "../components/ActivationView";
import { activationFixture } from "../test/fixtures";

const noop = () => undefined;

describe("ActivationView", () => {
  it("prompts for a captured run when no tensor is available", () => {
    render(<ActivationView activation={null} loading={false} error={null} onQuery={noop} />);
    expect(screen.getByText("Nothing captured yet")).toBeInTheDocument();
  });

  it("renders capture errors instead of stale tensors", () => {
    render(<ActivationView activation={null} loading={false} error="No capture for module" onQuery={noop} />);
    expect(screen.getByText("No capture for module")).toBeInTheDocument();
    expect(screen.queryByText("Nothing captured yet")).not.toBeInTheDocument();
  });

  it("renders the real tensor shape, tokens, stats, and top dimensions", () => {
    render(<ActivationView activation={activationFixture()} loading={false} error={null} onQuery={noop} />);
    expect(screen.getByText(/shape \[4, 2048\] · float32/)).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("world")).toBeInTheDocument();
    expect(screen.getByText("#7")).toBeInTheDocument();
    expect(screen.getByText("#42")).toBeInTheDocument();
    expect(screen.getByText("mean")).toBeInTheDocument();
  });

  it("queries the backend when a token position is picked", async () => {
    const onQuery = vi.fn();
    const user = userEvent.setup();
    render(<ActivationView activation={activationFixture()} loading={false} error={null} onQuery={onQuery} />);

    await user.click(screen.getByTitle("#0 · token 1 · prompt"));
    expect(onQuery).toHaveBeenCalledWith({ position: 0, feature: 7 });
  });

  it("queries the backend when a top dimension is picked", async () => {
    const onQuery = vi.fn();
    const user = userEvent.setup();
    render(<ActivationView activation={activationFixture()} loading={false} error={null} onQuery={onQuery} />);

    await user.click(screen.getByRole("button", { name: /#42/ }));
    expect(onQuery).toHaveBeenCalledWith({ feature: 42, position: 2 });
  });

  it("validates feature indices before querying", async () => {
    const onQuery = vi.fn();
    const user = userEvent.setup();
    render(<ActivationView activation={activationFixture()} loading={false} error={null} onQuery={onQuery} />);

    const input = screen.getByLabelText("Feature index");
    await user.clear(input);
    await user.type(input, "2048");
    await user.click(screen.getByRole("button", { name: "View feature" }));
    expect(onQuery).not.toHaveBeenCalled();

    await user.clear(input);
    await user.type(input, "12");
    await user.click(screen.getByRole("button", { name: "View feature" }));
    expect(onQuery).toHaveBeenCalledWith({ feature: 12, position: 2 });
  });

  it("renders the comparison label and warnings for difference views", () => {
    render(
      <ActivationView
        activation={activationFixture({
          comparison: "steered − baseline",
          warnings: ["Truncated to shared positions"],
        })}
        loading={false}
        error={null}
        onQuery={noop}
      />,
    );
    expect(screen.getByText("steered − baseline")).toBeInTheDocument();
    expect(screen.getByText("Truncated to shared positions")).toBeInTheDocument();
  });

  it("shows the per-dimension sigma for the selected feature", () => {
    const stats = {
      mean: Array(8).fill(0),
      std: Array.from({ length: 8 }, (_, index) => (index === 7 ? 0.25 : 0.1)),
      minimum: Array(8).fill(0),
      maximum: Array(8).fill(0),
    };
    render(
      <ActivationView
        activation={activationFixture({ feature_stats: stats })}
        loading={false}
        error={null}
        onQuery={noop}
      />,
    );
    expect(screen.getByText(/σ\(dim 7\) = 0\.2500/)).toBeInTheDocument();
  });
});
