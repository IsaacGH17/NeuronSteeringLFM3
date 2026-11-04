import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ArchitectureExplorer } from "../components/ArchitectureExplorer";
import { architectureFixture } from "../test/fixtures";

const noop = () => undefined;

describe("ArchitectureExplorer", () => {
  it("shows an empty state before the model is loaded", () => {
    render(
      <ArchitectureExplorer architecture={null} selectedId={null} onSelect={noop} loading={false} capturedIds={new Set()} />,
    );
    expect(screen.getByText("Load the model to discover its real module hierarchy.")).toBeInTheDocument();
  });

  it("shows a loading state while discovery is in flight", () => {
    render(
      <ArchitectureExplorer architecture={null} selectedId={null} onSelect={noop} loading capturedIds={new Set()} />,
    );
    expect(screen.getByText("Loading architecture…")).toBeInTheDocument();
  });

  it("renders real model counts and expandable module rows", async () => {
    const user = userEvent.setup();
    render(
      <ArchitectureExplorer
        architecture={architectureFixture()}
        selectedId={null}
        onSelect={noop}
        loading={false}
        capturedIds={new Set()}
      />,
    );
    expect(screen.getByText("Lfm3ForCausalLM · 24 layers · hidden 2048")).toBeInTheDocument();
    expect(screen.getByText("Lfm3DecoderLayer")).toBeInTheDocument();
    expect(screen.queryByText("Lfm3ShortConv")).not.toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Expand" })[0]);
    expect(screen.getByText("Lfm3ShortConv")).toBeInTheDocument();
    expect(screen.getByText("Lfm3MLP")).toBeInTheDocument();
    expect(screen.getByText("feed_forward")).toBeInTheDocument();
  });

  it("selects a module when a tree row is clicked", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <ArchitectureExplorer
        architecture={architectureFixture()}
        selectedId={null}
        onSelect={onSelect}
        loading={false}
        capturedIds={new Set()}
      />,
    );
    await user.click(screen.getAllByRole("button", { name: "Expand" })[0]);
    await user.click(screen.getByText("Lfm3ShortConv"));
    expect(onSelect).toHaveBeenCalledWith("model.layers.0.conv");
  });

  it("filters the tree by class name and reports no matches", async () => {
    const user = userEvent.setup();
    render(
      <ArchitectureExplorer
        architecture={architectureFixture()}
        selectedId={null}
        onSelect={noop}
        loading={false}
        capturedIds={new Set()}
      />,
    );
    const search = screen.getByPlaceholderText("Search module, class, or path");
    await user.type(search, "Lfm3MLP");
    expect(screen.getByText("Lfm3MLP")).toBeInTheDocument();
    expect(screen.queryByText("Lfm3ShortConv")).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "does-not-exist");
    expect(screen.getByText("No modules match “does-not-exist”.")).toBeInTheDocument();
  });

  it("collapses and expands module children", async () => {
    const user = userEvent.setup();
    render(
      <ArchitectureExplorer
        architecture={architectureFixture()}
        selectedId={null}
        onSelect={noop}
        loading={false}
        capturedIds={new Set()}
      />,
    );
    await user.click(screen.getAllByRole("button", { name: "Expand" })[0]);
    expect(screen.getByText("Lfm3ShortConv")).toBeInTheDocument();
    const collapses = screen.getAllByRole("button", { name: "Collapse" });
    await user.click(collapses[collapses.length - 1]);
    expect(screen.queryByText("Lfm3ShortConv")).not.toBeInTheDocument();
  });

  it("marks modules with captured activations", () => {
    render(
      <ArchitectureExplorer
        architecture={architectureFixture()}
        selectedId={null}
        onSelect={noop}
        loading={false}
        capturedIds={new Set(["model.layers"])}
      />,
    );
    expect(screen.getByTitle("Activations captured in the latest run")).toBeInTheDocument();
  });

  it("selects modules from the layer overview", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <ArchitectureExplorer
        architecture={architectureFixture()}
        selectedId={null}
        onSelect={onSelect}
        loading={false}
        capturedIds={new Set(["model.layers.2"])}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Layers" }));
    expect(screen.getByText("full_attention")).toBeInTheDocument();
    await user.click(screen.getByText("Lfm3Attention"));
    expect(onSelect).toHaveBeenCalledWith("model.layers.2");
  });
});
