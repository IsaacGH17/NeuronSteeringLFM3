import { useState } from "react";
import type { BaselineResponse, GenerationSettings, SteeringResponse, SteeringSpec } from "../types";

interface Props {
  disabled: boolean;
  running: boolean;
  result: BaselineResponse | null;
  pair: SteeringResponse | null;
  error: string | null;
  captureModuleId: string | null;
  hiddenSize: number | null;
  onRun: (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    captureModuleIds: string[],
  ) => void;
  onRunSteered: (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    steering: SteeringSpec,
    captureModuleIds: string[],
  ) => void;
}

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";
const STRENGTH_LIMIT = 49.5;

export function GenerationPanel({
  disabled,
  running,
  result,
  pair,
  error,
  captureModuleId,
  hiddenSize,
  onRun,
  onRunSteered,
}: Props) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [seed, setSeed] = useState("");
  const [enableThinking, setEnableThinking] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [featureIndex, setFeatureIndex] = useState(0);
  const [strength, setStrength] = useState(0);

  const buildSettings = (): GenerationSettings => ({
    max_new_tokens: Math.max(1, Math.min(4096, Math.round(maxNewTokens))),
    temperature,
    top_p: topP,
    seed: Number.isFinite(Number(seed)) && seed.trim() !== "" ? Number(seed) : null,
    enable_thinking: enableThinking,
  });

  const sharedArgs = () => ({
    promptText: prompt,
    systemText: system.trim() === "" ? null : system,
    settings: buildSettings(),
    captures: captureEnabled && captureModuleId ? [captureModuleId] : [],
  });

  const submit = () => {
    if (!prompt.trim() || disabled || running) return;
    const { promptText, systemText, settings, captures } = sharedArgs();
    onRun(promptText, systemText, settings, captures);
  };

  const submitSteered = () => {
    if (!prompt.trim() || disabled || running) return;
    if (!captureModuleId) return;
    const clampedFeature = Math.max(
      0,
      Math.min(hiddenSize !== null ? hiddenSize - 1 : Number.MAX_SAFE_INTEGER, Math.round(featureIndex)),
    );
    const clampedStrength = Math.max(-STRENGTH_LIMIT, Math.min(STRENGTH_LIMIT, strength));
    const { promptText, systemText, settings, captures } = sharedArgs();
    onRunSteered(
      promptText,
      systemText,
      settings,
      {
        module_id: captureModuleId,
        feature_index: clampedFeature,
        strength: Number(clampedStrength.toFixed(6)),
      },
      captures,
    );
  };

  const featureMax = hiddenSize !== null ? hiddenSize - 1 : 2047;
  const steeringReady = Boolean(captureModuleId) && !disabled && !running;

  return (
    <section className="panel generation">
      <header className="panel-header">
        <div>
          <h2>Steering and generation</h2>
          <p className="subtle">Baseline and steered runs reuse the same prompt and settings.</p>
        </div>
      </header>

      <div className="form">
        <label>
          <span>System instruction</span>
          <textarea
            rows={2}
            value={system}
            placeholder="Optional"
            onChange={(event) => setSystem(event.target.value)}
          />
        </label>
        <label>
          <span>Prompt</span>
          <textarea rows={5} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
        </label>

        <div className="form-grid">
          <label>
            <span>Max new tokens</span>
            <input
              type="number"
              min={1}
              max={4096}
              value={maxNewTokens}
              onChange={(event) => setMaxNewTokens(Number(event.target.value))}
            />
          </label>
          <label>
            <span>Temperature</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
          </label>
          <label>
            <span>Top-p</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={topP}
              onChange={(event) => setTopP(Number(event.target.value))}
            />
          </label>
          <label>
            <span>Seed</span>
            <input
              type="number"
              placeholder="random"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
            />
          </label>
        </div>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={enableThinking}
            onChange={(event) => setEnableThinking(event.target.checked)}
          />
          <span>Enable thinking (off by default)</span>
        </label>

        <div className="steering-box">
          <h4>Steering (inference-only hook)</h4>
          <p className="subtle">
            Target module:{" "}
            {captureModuleId ? (
              <code>{captureModuleId}</code>
            ) : (
              "select a steerable module in the architecture explorer"
            )}
          </p>
          <div className="form-grid">
            <label>
              <span>Feature index (max {featureMax})</span>
              <input
                type="number"
                min={0}
                max={featureMax}
                value={featureIndex}
                onChange={(event) => setFeatureIndex(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Strength (numeric)</span>
              <input
                type="number"
                min={-STRENGTH_LIMIT}
                max={STRENGTH_LIMIT}
                step={0.5}
                value={strength}
                onChange={(event) => setStrength(Number(event.target.value))}
              />
            </label>
          </div>
          <label className="strength-slider">
            <span>Strength slider: {strength}</span>
            <input
              type="range"
              min={-STRENGTH_LIMIT}
              max={STRENGTH_LIMIT}
              step={0.5}
              value={strength}
              onChange={(event) => setStrength(Number(event.target.value))}
            />
          </label>
          <button
            className="secondary"
            onClick={() => setStrength(0)}
            disabled={strength === 0}
          >
            Reset strength to 0
          </button>
        </div>

        <label className="checkbox">
          <input
            type="checkbox"
            checked={captureEnabled}
            disabled={!captureModuleId}
            onChange={(event) => setCaptureEnabled(event.target.checked)}
          />
          <span>
            Capture activations{captureModuleId ? ` for ${captureModuleId}` : " (select a steerable module)"}
          </span>
        </label>

        <div className="button-row">
          <button className="primary" onClick={submit} disabled={disabled || running || !prompt.trim()}>
            {running ? "Generating…" : "Run baseline"}
          </button>
          <button
            className="primary"
            onClick={submitSteered}
            disabled={!steeringReady || !prompt.trim()}
            title={captureModuleId ? "Run baseline + steered pair" : "Select a steerable module first"}
          >
            {running ? "Generating…" : "Run baseline + steered"}
          </button>
        </div>
        {disabled && <p className="subtle">Load the model to enable generation.</p>}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {pair && (
        <div className="comparison">
          <h4>Baseline vs steered · hook: {pair.steered.steering?.module_id} · dim{" "}
            {pair.steered.steering?.feature_index} · α {pair.steered.steering?.strength}</h4>
          <div className="comparison-columns">
            {(["baseline", "steered"] as const).map((condition) => {
              const run = pair[condition];
              return (
                <div key={run.id} className="result">
                  <h5 className="condition-label">{condition}</h5>
                  <div className="result-meta">
                    <span>{run.generated_tokens} tokens</span>
                    <span>{run.duration_ms.toFixed(0)} ms</span>
                    <span>{run.tokens_per_second?.toFixed(1) ?? "—"} tok/s</span>
                    <span>{run.device}</span>
                  </div>
                  {run.warnings.map((warning) => (
                    <div key={warning} className="warning-banner inline">
                      {warning}
                    </div>
                  ))}
                  <pre className="output">{run.output || "(empty output)"}</pre>
                  <p className="subtle">
                    run {run.id.slice(0, 8)} · prompt tokens: {run.prompt_tokens}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {!pair && result && (
        <div className="result">
          <div className="result-meta">
            <span>{result.generated_tokens} tokens</span>
            <span>{result.duration_ms.toFixed(0)} ms</span>
            <span>{result.tokens_per_second?.toFixed(1) ?? "—"} tok/s</span>
            <span>{result.device}</span>
          </div>
          {result.warnings.map((warning) => (
            <div key={warning} className="warning-banner">
              {warning}
            </div>
          ))}
          <pre className="output">{result.output || "(empty output)"}</pre>
          <p className="subtle">
            prompt tokens: {result.prompt_tokens} · run {result.id.slice(0, 8)} · {result.model_id}
          </p>
        </div>
      )}
    </section>
  );
}
