import { useEffect, useMemo, useRef, useState } from "react";
import { runNextToken } from "../api/client";
import type {
  NextTokenResponse,
  ProbeSeed,
  SteeringComponent,
  SteeringNormalization,
  SteeringSpec,
} from "../types";

interface Props {
  open: boolean;
  disabled?: boolean;
  seed: ProbeSeed | null;
  onClose: () => void;
}

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";
const SIGMA_LIMIT = 10;
const RAW_LIMIT = 100;
const DEFAULT_TOP_K = 10;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function displayToken(text: string): string {
  if (text === "") return "∅";
  return text.replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

function percent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

function seedComponents(seed: ProbeSeed | null): SteeringComponent[] {
  const spec = seed?.steering;
  if (!spec) return [];
  if (spec.components && spec.components.length > 0) return spec.components;
  if (spec.feature_index !== null && spec.feature_index !== undefined) {
    return [{ feature_index: spec.feature_index, weight: 1 }];
  }
  return [];
}

export function NextTokenProbe({ open, disabled = false, seed, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const requestRef = useRef(0);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [moduleId, setModuleId] = useState<string | null>(null);
  const [components, setComponents] = useState<SteeringComponent[]>([]);
  const [alpha, setAlpha] = useState(1);
  const [normalization, setNormalization] = useState<SteeringNormalization>("sigma");
  const [result, setResult] = useState<NextTokenResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    if (!open) return;
    if (!seed) {
      setPrompt(DEFAULT_PROMPT);
      setSystem("");
      setModuleId(null);
      setComponents([]);
      setAlpha(1);
      setNormalization("sigma");
      return;
    }
    setPrompt(seed.prompt);
    setSystem(seed.system ?? "");
    const spec = seed.steering;
    setModuleId(spec?.module_id ?? null);
    setComponents(seedComponents(seed));
    setAlpha(spec && Number.isFinite(spec.strength) ? spec.strength : 1);
    setNormalization(spec?.normalization ?? "sigma");
  }, [open, seed]);

  const limit = normalization === "raw" ? RAW_LIMIT : SIGMA_LIMIT;

  useEffect(() => {
    if (!open || disabled || !prompt.trim()) return;
    const timer = setTimeout(() => {
      const requestId = ++requestRef.current;
      const steering: SteeringSpec | null =
        moduleId && components.length > 0
          ? {
              module_id: moduleId,
              strength: Number(clamp(alpha, -limit, limit).toFixed(6)),
              feature_index: components.length === 1 ? components[0].feature_index : null,
              components,
              normalization,
            }
          : null;
      setLoading(true);
      setError(null);
      runNextToken({
        prompt,
        system: system.trim() === "" ? null : system,
        enable_thinking: false,
        top_k: clamp(Math.round(topK), 1, 50),
        steering,
      })
        .then((response) => {
          if (requestId === requestRef.current) setResult(response);
        })
        .catch((probeError) => {
          if (requestId === requestRef.current) setError(errorMessage(probeError));
        })
        .finally(() => {
          if (requestId === requestRef.current) setLoading(false);
        });
    }, 350);
    return () => clearTimeout(timer);
  }, [open, disabled, prompt, system, topK, alpha, normalization, moduleId, components, limit]);

  const baselineTop = result?.baseline.tokens[0] ?? null;
  const steeredTop = result?.steered?.tokens[0] ?? null;
  const changed = Boolean(
    steeredTop && baselineTop && steeredTop.token_id !== baselineTop.token_id,
  );
  const maxProbability = useMemo(
    () =>
      Math.max(
        1e-9,
        ...(result?.candidates.map((candidate) =>
          Math.max(candidate.baseline_probability, candidate.steered_probability ?? 0),
        ) ?? []),
      ),
    [result],
  );

  if (!open) return null;

  return (
    <div className="probe-backdrop" onClick={onClose}>
      <section
        className="probe-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="probe-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="probe-header">
          <div>
            <p className="help-kicker">Single forward</p>
            <h2 id="probe-title">Next-token probe</h2>
            <p className="subtle">
              Raw softmax at the last prompt token. No text is generated and nothing is saved.
            </p>
          </div>
          <button ref={closeRef} className="secondary" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="probe-body">
          <div className="probe-controls">
            <fieldset className="control-card">
              <legend>Prompt</legend>
              <label>
                <span>System instruction (optional)</span>
                <textarea
                  rows={2}
                  value={system}
                  placeholder="Optional"
                  onChange={(event) => setSystem(event.target.value)}
                />
              </label>
              <label>
                <span>Prompt</span>
                <textarea
                  rows={3}
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                />
              </label>
            </fieldset>

            <fieldset className="control-card">
              <legend>Probe</legend>
              <label>
                <span>Top-k tokens</span>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
              <p className="subtle">
                One forward over the templated prompt. The bar chart shows the probability the model
                assigns to the next token, before any sampling.
              </p>
            </fieldset>

            <fieldset className={`control-card probe-steering-card ${moduleId ? "active" : ""}`}>
              <legend>Steering</legend>
              {moduleId && components.length > 0 ? (
                <>
                  <div className="probe-target-head">
                    <code>{moduleId}</code>
                    <span className="badge">{components.length} dims</span>
                  </div>
                  <label>
                    <span>Normalization</span>
                    <select
                      value={normalization}
                      onChange={(event) =>
                        setNormalization(event.target.value as SteeringNormalization)
                      }
                    >
                      <option value="sigma">σ units</option>
                      <option value="raw">raw activation units</option>
                    </select>
                  </label>
                  <label>
                    <span>Strength α ({normalization === "sigma" ? "σ" : "raw"})</span>
                    <input
                      type="number"
                      step={0.1}
                      value={alpha}
                      onChange={(event) => setAlpha(Number(event.target.value))}
                    />
                  </label>
                  <input
                    type="range"
                    min={-limit}
                    max={limit}
                    step={normalization === "sigma" ? 0.1 : 1}
                    value={clamp(alpha, -limit, limit)}
                    aria-label="Probe steering strength"
                    onChange={(event) => setAlpha(Number(event.target.value))}
                  />
                  <p className="subtle">
                    The probe re-forwards the prompt with this intervention, then compares both
                    distributions. α = 0 reproduces the baseline.
                  </p>
                  <button
                    type="button"
                    className="secondary small"
                    onClick={() => {
                      setModuleId(null);
                      setComponents([]);
                    }}
                  >
                    Clear steering
                  </button>
                </>
              ) : (
                <p className="subtle">
                  Baseline only. Select a cluster in <strong>Flow</strong> and press{" "}
                  <strong>Probe next token</strong>, or build a direction in the main panel, to see a
                  steered distribution here.
                </p>
              )}
            </fieldset>

            {disabled && (
              <div className="warning-banner inline">
                Load the model first. The probe needs one real forward pass.
              </div>
            )}
          </div>

          <div className="probe-results" aria-live="polite">
            {error && <div className="error-banner inline">{error}</div>}

            {!result && !error && (
              <div className="empty probe-empty">
                {loading ? "Probing the next token…" : "Run a probe to see the distribution."}
              </div>
            )}

            {result && (
              <>
                <div className="probe-summary">
                  <div className={`probe-top1 ${changed ? "changed" : ""}`}>
                    <span className="probe-top1-label">Baseline</span>
                    <strong>{displayToken(baselineTop?.text ?? "—")}</strong>
                    <span>{baselineTop ? percent(baselineTop.probability) : "—"}</span>
                  </div>
                  <div className={`probe-top1 ${changed ? "changed" : ""}`}>
                    <span className="probe-top1-label">Steered</span>
                    <strong>{displayToken(steeredTop?.text ?? "—")}</strong>
                    <span>{steeredTop ? percent(steeredTop.probability) : "—"}</span>
                  </div>
                  <div className="probe-metrics">
                    <span>
                      KL <b>{result.kl_divergence === null ? "—" : result.kl_divergence.toFixed(4)}</b>
                    </span>
                    <span>
                      entropy{" "}
                      <b>
                        {result.baseline.entropy.toFixed(3)}
                        {result.steered ? ` → ${result.steered.entropy.toFixed(3)}` : ""}
                      </b>
                    </span>
                    <span>
                      <b>{result.duration_ms.toFixed(0)} ms</b>
                    </span>
                  </div>
                </div>

                <div className="probe-legend">
                  <span>
                    <i className="probe-swatch baseline" /> baseline
                  </span>
                  <span>
                    <i className="probe-swatch steered" /> steered
                  </span>
                  <span>
                    position {result.position} · {result.prompt_tokens} prompt tokens ·{" "}
                    {result.device}
                  </span>
                </div>

                <div className="probe-table">
                  <div className="probe-row probe-row-head">
                    <span>token</span>
                    <span>distribution</span>
                    <span>baseline</span>
                    {result.steered && <span>steered</span>}
                    {result.steered && <span>Δ</span>}
                  </div>
                  {result.candidates.map((candidate) => (
                    <div className="probe-row" key={candidate.token_id}>
                      <code className="probe-token">{displayToken(candidate.text)}</code>
                      <div className="probe-bars">
                        <span
                          className="probe-bar baseline"
                          style={{
                            width: `${Math.max(
                              1,
                              (candidate.baseline_probability / maxProbability) * 100,
                            )}%`,
                          }}
                        />
                        {result.steered && (
                          <span
                            className="probe-bar steered"
                            style={{
                              width: `${Math.max(
                                1,
                                ((candidate.steered_probability ?? 0) / maxProbability) * 100,
                              )}%`,
                            }}
                          />
                        )}
                      </div>
                      <span className="probe-number">{percent(candidate.baseline_probability)}</span>
                      {result.steered && (
                        <span className="probe-number">
                          {percent(candidate.steered_probability ?? 0)}
                        </span>
                      )}
                      {result.steered && (
                        <span
                          className={`probe-delta ${
                            (candidate.delta ?? 0) > 0
                              ? "up"
                              : (candidate.delta ?? 0) < 0
                                ? "down"
                                : ""
                          }`}
                        >
                          {(candidate.delta ?? 0) >= 0 ? "+" : ""}
                          {((candidate.delta ?? 0) * 100).toFixed(1)} pp
                        </span>
                      )}
                    </div>
                  ))}
                </div>

                {result.warnings.map((warning) => (
                  <div key={warning} className="warning-banner inline">
                    {warning}
                  </div>
                ))}

                <p className="subtle">
                  KL is in nats between the full baseline and steered distributions. A change in
                  top-1 is one observation, not proof of a behaviour.
                </p>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
