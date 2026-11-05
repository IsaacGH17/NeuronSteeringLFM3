import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { HelpPanel } from "../components/HelpPanel";

describe("HelpPanel", () => {
  it("renders nothing while closed", () => {
    render(<HelpPanel open={false} onClose={() => undefined} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("explains the core concepts and closes from the button", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<HelpPanel open onClose={onClose} />);

    expect(screen.getByRole("dialog", { name: "Help & key concepts" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Hidden dimensions" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Steering α" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Controls and sweeps" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Reproducibility" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Live flow" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("closes with Escape", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<HelpPanel open onClose={onClose} />);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });
});
