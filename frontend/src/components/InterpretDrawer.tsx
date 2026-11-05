import { useEffect, useMemo, useRef, useState } from "react";
import { runLens, runPatch } from "../api/client";
import type {
  GenerationSettings,
  LensResponse,
  PatchResponse,
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

type Tab = "lens" | "patch";

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
        </div>

        <div className="help-body interpret-body">
          <p className="subtle">
            These tools read the model&apos;s own forward pass; they never change weights.
          </p>

          <fieldset className="control-card">
            <legend>{tab === "lens" ? "Prompt" : "Recipient prompt"}</legend>
            {tab === "lens" && (
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

          {tab === "lens" ? (
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
