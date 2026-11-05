import { useEffect, useMemo, useRef, useState } from "react";
import { runLens, runPatch, runPatchSweep } from "../api/client";
import type {
  GenerationSettings,
  LensResponse,
  PatchResponse,
  PatchSweepCell,
  PatchSweepResponse,
} from "../types";

interface LayerOption {
  layer_index: number;
  module_id: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  layers: LayerOption[];
  defaultModuleId: string | null;
}

type Tab = "lens" | "patch" | "sweep";

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";
const DEFAULT_DONOR = "Describe a rainy morning.";
const GREEDY: Omit<GenerationSettings, "max_new_tokens"> = {
  temperature: 0,
  top_p: 1,
  seed: null,
  enable_thinking: false,
};

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function InterpretDrawer({ open, onClose, layers, defaultModuleId }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [tab, setTab] = useState<Tab>("lens");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [maxTokens, setMaxTokens] = useState(16);
  const [topK, setTopK] = useState(5);
  const [position, setPosition] = useState<number | null>(null);
  const [lens, setLens] = useState<LensResponse | null>(null);
  const [lensLoading, setLensLoading] = useState(false);
  const [lensError, setLensError] = useState<string | null>(null);
  const [donorPrompt, setDonorPrompt] = useState(DEFAULT_DONOR);
  const [moduleId, setModuleId] = useState(defaultModuleId ?? layers[0]?.module_id ?? "");
  const [patch, setPatch] = useState<PatchResponse | null>(null);
  const [patchLoading, setPatchLoading] = useState(false);
  const [patchError, setPatchError] = useState<string | null>(null);
  const [donorSystem, setDonorSystem] = useState("");
  const [sweepLayerFrom, setSweepLayerFrom] = useState(0);
  const [sweepLayerTo, setSweepLayerTo] = useState(999);
  const [sweepPositionMode, setSweepPositionMode] = useState<
    "last" | "last4" | "first4" | "custom"
  >("last");
  const [sweepCustomPosition, setSweepCustomPosition] = useState(7);
  const [sweep, setSweep] = useState<PatchSweepResponse | null>(null);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [sweepError, setSweepError] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<PatchSweepCell | null>(null);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, onClose]);

  useEffect(() => {
    if (!moduleId && defaultModuleId) setModuleId(defaultModuleId);
  }, [defaultModuleId, moduleId]);

  const maxAblation = useMemo(
    () => Math.max(1e-9, ...(lens?.layers.map((layer) => layer.ablation_kl ?? 0) ?? [0])),
    [lens],
  );

  const layerMin = layers.length > 0 ? layers[0].layer_index : 0;
  const layerMax = layers.length > 0 ? layers[layers.length - 1].layer_index : 0;
  const resolvedFrom = clamp(sweepLayerFrom, layerMin, layerMax);
  const resolvedTo = clamp(sweepLayerTo, resolvedFrom, layerMax);
  const sweepLayerIndices = useMemo(
    () =>
      layers
        .filter((layer) => layer.layer_index >= resolvedFrom && layer.layer_index <= resolvedTo)
        .map((layer) => layer.layer_index),
    [layers, resolvedFrom, resolvedTo],
  );
  const sweepPositions = useMemo(
    () =>
      sweepPositionMode === "last"
        ? [-1]
        : sweepPositionMode === "last4"
          ? [-4, -3, -2, -1]
          : sweepPositionMode === "first4"
            ? [0, 1, 2, 3]
            : [sweepCustomPosition],
    [sweepPositionMode, sweepCustomPosition],
  );
  const sweepCellCount = sweepLayerIndices.length * sweepPositions.length;
  const changedCount = sweep ? sweep.cells.filter((cell) => cell.changed).length : 0;
  const flipCount = sweep ? sweep.cells.filter((cell) => cell.first_token_changed).length : 0;

  if (!open) return null;

  const settings = (tokens: number): GenerationSettings => ({
    ...GREEDY,
    max_new_tokens: Math.max(1, Math.min(256, Math.round(tokens))),
  });

  const submitLens = async (nextPosition: number | null = position) => {
    if (!prompt.trim()) return;
    setLensLoading(true);
    setLensError(null);
    try {
      const result = await runLens({
        prompt,
        system: system.trim() === "" ? null : system,
        settings: settings(maxTokens),
        top_k: topK,
        position: nextPosition,
      });
      setLens(result);
      setPosition(nextPosition);
    } catch (error) {
      setLensError(errorMessage(error));
    } finally {
      setLensLoading(false);
    }
  };

  const submitPatch = async () => {
    if (!prompt.trim() || !donorPrompt.trim() || !moduleId) return;
    setPatchLoading(true);
    setPatchError(null);
    try {
      const result = await runPatch({
        recipient_prompt: prompt,
        donor_prompt: donorPrompt,
        module_id: moduleId,
        settings: settings(maxTokens),
      });
      setPatch(result);
    } catch (error) {
      setPatchError(errorMessage(error));
    } finally {
      setPatchLoading(false);
    }
  };

  const submitSweep = async () => {
    if (
      !prompt.trim() ||
      !donorPrompt.trim() ||
      sweepCellCount === 0 ||
      sweepCellCount > 48
    ) {
      return;
    }
    setSweepLoading(true);
    setSweepError(null);
    setSweep(null);
    setSelectedCell(null);
    try {
      const result = await runPatchSweep({
        recipient_prompt: prompt,
        recipient_system: system.trim() === "" ? null : system,
        donor_prompt: donorPrompt,
        donor_system: donorSystem.trim() === "" ? null : donorSystem,
        settings: settings(maxTokens),
        layer_indices: sweepLayerIndices,
        positions: sweepPositions,
      });
      setSweep(result);
      setSelectedCell(result.cells[0] ?? null);
    } catch (error) {
      setSweepError(errorMessage(error));
    } finally {
      setSweepLoading(false);
    }
  };

  return (
    <div className="help-backdrop" onClick={onClose}>
      <aside
        className="help-drawer interpret-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="interpret-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="help-header">
          <div>
            <p className="help-kicker">Internal mechanics</p>
            <h2 id="interpret-title">Interpret</h2>
          </div>
          <button ref={closeRef} className="secondary" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="interpret-tabs" role="group" aria-label="Interpret mode">
          <button
            type="button"
            className={tab === "lens" ? "active" : ""}
            aria-pressed={tab === "lens"}
            onClick={() => setTab("lens")}
          >
            Logit lens
          </button>
          <button
            type="button"
            className={tab === "patch" ? "active" : ""}
            aria-pressed={tab === "patch"}
            onClick={() => setTab("patch")}
          >
            Activation patching
          </button>
          <button
            type="button"
            className={tab === "sweep" ? "active" : ""}
            aria-pressed={tab === "sweep"}
            onClick={() => setTab("sweep")}
          >
            Patch sweep
          </button>
        </div>

        <div className="help-body interpret-body">
          <p className="subtle">
            These tools read the model&apos;s own forward pass; they never change weights.
          </p>

          <fieldset className="control-card">
            <legend>{tab === "lens" ? "Prompt" : "Recipient prompt"}</legend>
            {tab !== "patch" && (
              <label>
                <span>System instruction (optional)</span>
                <textarea
                  rows={2}
                  value={system}
                  placeholder="Optional"
                  onChange={(event) => setSystem(event.target.value)}
                />
              </label>
            )}
            <label>
              <span>Prompt</span>
              <textarea rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </label>
          </fieldset>

          {tab === "sweep" ? (
            <>
              <fieldset className="control-card">
                <legend>Donor prompt</legend>
                <label>
                  <span>Donor</span>
                  <textarea
                    rows={2}
                    value={donorPrompt}
                    onChange={(event) => setDonorPrompt(event.target.value)}
                  />
                </label>
                <label>
                  <span>Donor system (optional)</span>
                  <textarea
                    rows={2}
                    value={donorSystem}
                    placeholder="Optional"
                    onChange={(event) => setDonorSystem(event.target.value)}
                  />
                </label>
                <p className="subtle">
                  The donor&apos;s activation at each layer and position replaces the recipient&apos;s
                  for the first forward only.
                </p>
              </fieldset>

              <div className="form-grid">
                <label>
                  <span>Max new tokens</span>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={maxTokens}
                    onChange={(event) => setMaxTokens(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Positions</span>
                  <select
                    value={sweepPositionMode}
                    onChange={(event) =>
                      setSweepPositionMode(
                        event.target.value as "last" | "last4" | "first4" | "custom",
                      )
                    }
                  >
                    <option value="last">Last prompt token</option>
                    <option value="last4">Last 4 prompt tokens</option>
                    <option value="first4">First 4 prompt tokens</option>
                    <option value="custom">Custom position</option>
                  </select>
                </label>
                {sweepPositionMode === "custom" && (
                  <label>
                    <span>Prompt position (negative from end)</span>
                    <input
                      type="number"
                      value={sweepCustomPosition}
                      onChange={(event) => setSweepCustomPosition(Number(event.target.value))}
                    />
                  </label>
                )}
                <label>
                  <span>Layers from</span>
                  <input
                    type="number"
                    min={layerMin}
                    max={layerMax}
                    value={resolvedFrom}
                    onChange={(event) => setSweepLayerFrom(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Layers to</span>
                  <input
                    type="number"
                    min={layerMin}
                    max={layerMax}
                    value={resolvedTo}
                    onChange={(event) => setSweepLayerTo(Number(event.target.value))}
                  />
                </label>
              </div>

              <p className="subtle">
                {sweepCellCount} cells (max 48). Each cell is one recipient generation; start with
                the last prompt token and a layer range.
              </p>
              {sweepCellCount > 48 && (
                <p className="field-note">Narrow the layer range or use fewer positions.</p>
              )}
              <button
                className="primary"
                onClick={() => void submitSweep()}
                disabled={
                  sweepLoading ||
                  !prompt.trim() ||
                  !donorPrompt.trim() ||
                  sweepCellCount === 0 ||
                  sweepCellCount > 48
                }
              >
                {sweepLoading ? `Running ${sweepCellCount} cells…` : "Run patch sweep"}
              </button>
              {sweepError && <div className="error-banner">{sweepError}</div>}

              {sweep && (
                <div className="patch-sweep-results">
                  <div className="patch-sweep-summary">
                    <span className="badge">{sweep.layers.length} layers</span>
                    <span className="badge">{sweep.positions.length} positions</span>
                    <span className="badge">{sweep.cells.length} cells</span>
                    <span
                      className={`badge ${changedCount > 0 ? "condition-steered" : "muted-badge"}`}
                    >
                      {changedCount} outputs changed
                    </span>
                    <span className={`badge ${flipCount > 0 ? "condition-steered" : "muted-badge"}`}>
                      {flipCount} first tokens flipped
                    </span>
                    <span className="subtle">{sweep.duration_ms.toFixed(0)} ms</span>
                  </div>

                  <h4>Recipient baseline</h4>
                  <pre className="output small">{sweep.baseline_output || "(no text generated)"}</pre>
                  <h4>Donor</h4>
                  <pre className="output small">{sweep.donor_output || "(no text generated)"}</pre>

                  <div className="patch-sweep-legend">
                    <span>
                      <i className="patch-sweep-swatch same" /> unchanged
                    </span>
                    <span>
                      <i className="patch-sweep-swatch changed" /> output changed
                    </span>
                    <span>
                      <i className="patch-sweep-swatch flip" /> first token flipped
                    </span>
                  </div>

                  <div className="patch-sweep-table-wrap">
                    <table className="data-table patch-sweep-table">
                      <thead>
                        <tr>
                          <th>layer</th>
                          {sweep.positions.map((position) => (
                            <th key={position}>
                              {position === sweep.prompt_tokens - 1
                                ? `pos ${position} · last`
                                : `pos ${position}`}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sweep.layers.map((layer) => (
                          <tr key={layer}>
                            <th scope="row">L{layer}</th>
                            {sweep.positions.map((position) => {
                              const cell = sweep.cells.find(
                                (candidate) =>
                                  candidate.layer_index === layer &&
                                  candidate.position === position,
                              );
                              if (!cell) return <td key={position}>—</td>;
                              return (
                                <td key={position}>
                                  <button
                                    type="button"
                                    className={`patch-cell ${
                                      cell.first_token_changed
                                        ? "flip"
                                        : cell.changed
                                          ? "changed"
                                          : "same"
                                    } ${selectedCell === cell ? "selected" : ""}`}
                                    aria-label={`Layer ${layer} position ${position}: ${
                                      cell.first_token_changed
                                        ? "first token changed"
                                        : cell.changed
                                          ? "output changed"
                                          : "unchanged"
                                    }`}
                                    title={`L${layer} pos ${position}: ${
                                      cell.first_token_changed
                                        ? "first token changed"
                                        : cell.changed
                                          ? "output changed"
                                          : "unchanged"
                                    } · prefix ${cell.common_prefix_tokens}/${cell.generated_tokens}`}
                                    onClick={() => setSelectedCell(cell)}
                                  >
                                    {cell.first_token_changed ? "⇄" : cell.changed ? "≠" : "="}
                                  </button>
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {selectedCell && (
                    <div className="patch-sweep-detail">
                      <h4>
                        Layer {selectedCell.layer_index} · position {selectedCell.position}
                      </h4>
                      <p className="subtle">
                        donor position {selectedCell.donor_position} · common prefix{" "}
                        {selectedCell.common_prefix_tokens}/{selectedCell.generated_tokens} generated
                        tokens ·{" "}
                        {selectedCell.first_token_changed
                          ? "first token changed"
                          : selectedCell.changed
                            ? "output changed"
                            : "unchanged"}
                      </p>
                      <h4>Patched</h4>
                      <pre className="output small">
                        {selectedCell.patched_output || "(no text generated)"}
                      </pre>
                      {selectedCell.warnings.map((warning) => (
                        <div key={warning} className="warning-banner inline">
                          {warning}
                        </div>
                      ))}
                    </div>
                  )}

                  {changedCount === 0 && (
                    <p className="subtle">
                      No output changed in this grid. That is a real result; try another layer range
                      or positions.
                    </p>
                  )}
                  {sweep.warnings.map((warning) => (
                    <div key={warning} className="warning-banner inline">
                      {warning}
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : tab === "lens" ? (
            <>
              <div className="form-grid">
                <label>
                  <span>Max new tokens</span>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={maxTokens}
                    onChange={(event) => setMaxTokens(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Top-k tokens</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                  />
                </label>
              </div>
              <button className="primary" onClick={() => void submitLens()} disabled={lensLoading || !prompt.trim()}>
                {lensLoading ? "Running lens…" : "Run logit lens"}
              </button>
              {lensError && <div className="error-banner">{lensError}</div>}

              {lens && (
                <div className="lens-results">
                  <p className="subtle">
                    {lens.prompt_tokens} prompt tokens · selected position {lens.position} ·{" "}
                    {lens.layers.length} layers
                  </p>
                  <div className="token-strip">
                    {lens.tokens.map((token) => (
                      <button
                        key={token.index}
                        type="button"
                        className={`token-chip ${token.phase} ${token.index === lens.position ? "selected" : ""}`}
                        aria-label={`Position ${token.index}`}
                        aria-pressed={token.index === lens.position}
                        onClick={() => void submitLens(token.index)}
                      >
                        <span className="token-index">{token.index}</span>
                        <span className="token-text">
                          {token.text === "" ? "∅" : token.text.replace(/\n/g, "\\n")}
                        </span>
                      </button>
                    ))}
                  </div>
                  <pre className="output small">{lens.output || "(no text generated)"}</pre>
                  <div className="lens-table-wrap">
                  <table className="data-table lens-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Module</th>
                        <th>Top-1 token</th>
                        <th>p</th>
                        <th>Ablation ΔKL</th>
                        <th>relative</th>
                      </tr>
                    </thead>
                    <tbody>
                      {lens.layers.map((layer) => (
                        <tr key={layer.layer_index}>
                          <td>{layer.layer_index}</td>
                          <td>{layer.module_id}</td>
                          <td>
                            <code>{layer.top_tokens[0]?.text || "—"}</code>
                          </td>
                          <td>{layer.top_tokens[0] ? percent(layer.top_tokens[0].probability) : "—"}</td>
                          <td>{layer.ablation_kl === null ? "—" : layer.ablation_kl.toFixed(4)}</td>
                          <td>
                            {layer.ablation_kl !== null && (
                              <span
                                className="lens-bar"
                                style={{
                                  width: `${Math.max(2, Math.round((layer.ablation_kl / maxAblation) * 100))}%`,
                                }}
                              />
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  </div>
                  {lens.warnings.map((warning) => (
                    <div key={warning} className="warning-banner inline">
                      {warning}
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <fieldset className="control-card">
                <legend>Donor prompt</legend>
                <label>
                  <span>Donor</span>
                  <textarea
                    rows={2}
                    value={donorPrompt}
                    onChange={(event) => setDonorPrompt(event.target.value)}
                  />
                </label>
                <label>
                  <span>Patch module</span>
                  <select value={moduleId} onChange={(event) => setModuleId(event.target.value)}>
                    {layers.map((layer) => (
                      <option key={layer.module_id} value={layer.module_id}>
                        layer {layer.layer_index}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="subtle">
                  The donor&apos;s activation at the last prompt token replaces the recipient&apos;s at
                  the same layer for the first forward only.
                </p>
              </fieldset>
              <div className="form-grid">
                <label>
                  <span>Max new tokens</span>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={maxTokens}
                    onChange={(event) => setMaxTokens(Number(event.target.value))}
                  />
                </label>
              </div>
              <button className="primary" onClick={() => void submitPatch()} disabled={patchLoading || !moduleId}>
                {patchLoading ? "Patching…" : "Run patch"}
              </button>
              {patchError && <div className="error-banner">{patchError}</div>}
              {patch && (
                <div className="patch-results">
                  <p className="subtle">
                    {patch.module_id} · position {patch.position} ·{" "}
                    <span className={`badge ${patch.changed ? "condition-steered" : "muted-badge"}`}>
                      {patch.changed ? "output changed" : "no change"}
                    </span>
                  </p>
                  <h4>Donor</h4>
                  <pre className="output small">{patch.donor_output || "(no text generated)"}</pre>
                  <h4>Recipient baseline</h4>
                  <pre className="output small">{patch.baseline_output || "(no text generated)"}</pre>
                  <h4>Recipient patched</h4>
                  <pre className="output small">{patch.patched_output || "(no text generated)"}</pre>
                  {patch.warnings.map((warning) => (
                    <div key={warning} className="warning-banner inline">
                      {warning}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
