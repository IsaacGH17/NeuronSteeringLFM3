import { useEffect, useState } from "react";
import type {
  BaselineResponse,
  GenerationSettings,
  SteeringResponse,
  SteeringSpec,
  SweepResponse,
} from "../types";

interface Props {
  disabled: boolean;
  running: boolean;
  result: BaselineResponse | null;
  pair: SteeringResponse | null;
  sweepResult: SweepResponse | null;
  sweepError: string | null;
  error: string | null;
  selectedSteerableId: string | null;
  steeringTargetId: string | null;
  onSetSteeringTarget: (moduleId: string) => void;
  hiddenSize: number | null;
  onRun: (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    captureModuleIds: string[],
    persistActivations: boolean | null,
  ) => void;
  onRunSteered: (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    steering: SteeringSpec,
    captureModuleIds: string[],
    persistActivations: boolean | null,
  ) => void;
  onRunSweep: (payload: {
    prompt: string;
    system: string | null;
    settings: GenerationSettings;
    module_id: string;
    feature_index: number;
    strengths: number[];
    include_control: boolean;
    control_seed: number | null;
    repeats: number;
    capture_module_ids: string[];
    persist_activations: boolean | null;
  }) => void;
}

type Mode = "baseline" | "steered" | "sweep";

const MODES: Record<Mode, { label: string; runLabel: string; hint: string }> = {
  baseline: {
    label: "Baseline",
    runLabel: "Run baseline",
    hint: "One unmodified generation. Use it as the reference for everything else.",
  },
  steered: {
    label: "Steered",
    runLabel: "Run baseline + steered",
    hint: "Runs a baseline and a steered copy with the same prompt, settings, and seed.",
  },
  sweep: {
    label: "Sweep",
    runLabel: "Run sweep",
    hint: "Runs one shared baseline plus a steered run for each strength and an optional random control.",
  },
};

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";
const DEFAULT_STRENGTHS = "-20,-10,0,10,20";
const STRENGTH_LIMIT = 49.5;

export function GenerationPanel({
  disabled,
  running,
  result,
  pair,
  sweepResult,
  sweepError,
  error,
  selectedSteerableId,
  steeringTargetId,
  onSetSteeringTarget,
  hiddenSize,
  onRun,
  onRunSteered,
  onRunSweep,
}: Props) {
  const [mode, setMode] = useState<Mode>("baseline");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [seed, setSeed] = useState("");
  const [enableThinking, setEnableThinking] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [persistActivations, setPersistActivations] = useState(false);
  const [featureIndex, setFeatureIndex] = useState(0);
  const [strength, setStrength] = useState(0);

  const [sweepFeatureIndex, setSweepFeatureIndex] = useState(0);
  const [strengthsText, setStrengthsText] = useState(DEFAULT_STRENGTHS);
  const [includeControl, setIncludeControl] = useState(false);
  const [controlSeed, setControlSeed] = useState("");
  const [repeats, setRepeats] = useState(1);
  const [sweepRunning, setSweepRunning] = useState(false);
  const [sweepLocalError, setSweepLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (sweepResult || sweepError) {
      setSweepRunning(false);
    }
  }, [sweepResult, sweepError]);

  const featureMax = hiddenSize !== null ? hiddenSize - 1 : 2047;
  const clampFeature = (value: number) =>
    Math.max(0, Math.min(featureMax, Math.round(Number.isFinite(value) ? value : 0)));
  const clampStrength = (value: number) =>
    Math.max(-STRENGTH_LIMIT, Math.min(STRENGTH_LIMIT, Number.isFinite(value) ? value : 0));

  const clampedFeature = clampFeature(featureIndex);
  const clampedSweepFeature = clampFeature(sweepFeatureIndex);
  const clampedStrength = clampStrength(strength);
  const featureClamped = clampedFeature !== Math.round(Number.isFinite(featureIndex) ? featureIndex : 0);
  const sweepFeatureClamped =
    clampedSweepFeature !== Math.round(Number.isFinite(sweepFeatureIndex) ? sweepFeatureIndex : 0);
  const strengthClamped = clampedStrength !== strength;

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
    captures: captureEnabled && selectedSteerableId ? [selectedSteerableId] : [],
    persistActivations,
  });

  const submit = () => {
    if (!prompt.trim() || disabled || running) return;
    const { promptText, systemText, settings, captures, persistActivations: persist } = sharedArgs();
    onRun(promptText, systemText, settings, captures, persist);
  };

  const submitSteered = () => {
    if (!prompt.trim() || disabled || running || !steeringTargetId) return;
    const { promptText, systemText, settings, captures, persistActivations: persist } = sharedArgs();
    onRunSteered(
      promptText,
      systemText,
      settings,
      {
        module_id: steeringTargetId,
        feature_index: clampedFeature,
        strength: Number(clampedStrength.toFixed(6)),
      },
      captures,
      persist,
    );
  };

  const parseStrengths = (): { strengths: number[]; error: string | null } => {
    const parts = strengthsText
      .split(",")
      .map((part) => part.trim())
      .filter((part) => part !== "");
    const strengths: number[] = [];
    for (const part of parts) {
      const value = Number(part);
      if (!Number.isFinite(value)) {
        return { strengths: [], error: `Invalid strength: "${part}"` };
      }
      strengths.push(value);
    }
    if (strengths.length === 0) return { strengths: [], error: "Enter at least one strength." };
    if (strengths.length > 9) return { strengths: [], error: "At most 9 strengths allowed." };
    return { strengths, error: null };
  };

  const submitSweep = () => {
    if (!prompt.trim() || disabled || sweepRunning || !steeringTargetId) return;
    const { strengths, error: parseError } = parseStrengths();
    if (parseError) {
      setSweepLocalError(parseError);
      return;
    }
    setSweepLocalError(null);
    const clampedRepeats = Math.max(1, Math.min(3, Math.round(repeats)));
    const controlSeedValue =
      Number.isFinite(Number(controlSeed)) && controlSeed.trim() !== "" ? Number(controlSeed) : null;
    setSweepRunning(true);
    onRunSweep({
      prompt,
      system: system.trim() === "" ? null : system,
      settings: buildSettings(),
      module_id: steeringTargetId,
      feature_index: clampedSweepFeature,
      strengths,
      include_control: includeControl,
      control_seed: controlSeedValue,
      repeats: clampedRepeats,
      capture_module_ids: captureEnabled && selectedSteerableId ? [selectedSteerableId] : [],
      persist_activations: persistActivations,
    });
  };

  const runCurrent = () => {
    if (mode === "baseline") submit();
    else if (mode === "steered") submitSteered();
    else submitSweep();
  };

  const promptReady = prompt.trim().length > 0;
  const targetReady = Boolean(steeringTargetId);
  const busy = running || (mode === "sweep" && sweepRunning);
  const canRun = !disabled && promptReady && !busy && (mode === "baseline" || targetReady);
  const runLabel = busy ? (mode === "sweep" ? "Running sweep…" : "Generating…") : MODES[mode].runLabel;
  const statusText = disabled
    ? "Load the model to enable generation."
    : !promptReady
      ? "Enter a prompt to run."
      : mode !== "baseline" && !targetReady
        ? "Set a steering target to run this mode."
        : busy
          ? mode === "sweep"
            ? "Sweeping sequentially. Larger sweeps can take several minutes."
            : "Generation in progress."
          : "Ready.";

  return (
    <section className="panel generation">
      <header className="panel-header">
        <div>
          <h2>Steering and generation</h2>
          <p className="subtle">Pick a mode, confirm the target, then run one explicit experiment.</p>
        </div>
      </header>

      <div className="form">
        <div className="mode-bar" role="group" aria-label="Experiment mode">
          {(Object.keys(MODES) as Mode[]).map((key) => (
            <button
              key={key}
              type="button"
              className={mode === key ? "active" : ""}
              aria-pressed={mode === key}
              onClick={() => setMode(key)}
            >
              {MODES[key].label}
            </button>
          ))}
        </div>
        <p className="mode-hint" aria-live="polite">
          {MODES[mode].hint}
        </p>

        <fieldset className={`target-card ${targetReady ? "ready" : "missing"}`}>
          <legend>Steering target</legend>
          <div className="target-card-head">
            <code className="target-module">{steeringTargetId ?? "No module selected"}</code>
            <span className={`badge ${targetReady ? "steerable-badge" : "muted-badge"}`}>
              {targetReady ? "set" : "not set"}
            </span>
          </div>
          <p className="subtle">
            {mode === "baseline"
              ? "Baseline runs ignore the steering target."
              : targetReady
                ? "Steered and sweep runs shift this module's output while you inspect any module you like."
                : "Select a steerable module in the inspector, then set it as the target. Inspecting another module will not change it."}
          </p>
          {selectedSteerableId && selectedSteerableId !== steeringTargetId && (
            <button
              type="button"
              className="secondary small"
              onClick={() => onSetSteeringTarget(selectedSteerableId)}
            >
              Use inspected module
            </button>
          )}
        </fieldset>

        <fieldset className="control-card">
          <legend>Prompt</legend>
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
        </fieldset>

        <fieldset className="control-card">
          <legend>Generation settings</legend>
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
        </fieldset>

        {mode === "steered" && (
          <fieldset className="control-card steering-card">
            <legend>Steering settings</legend>
            <p className="subtle">
              Shift one hidden dimension by α at every token. α=0 must reproduce the baseline exactly.
            </p>
            <div className="form-grid">
              <label>
                <span>Hidden dimension (0–{featureMax})</span>
                <input
                  type="number"
                  min={0}
                  max={featureMax}
                  value={featureIndex}
                  onChange={(event) => setFeatureIndex(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Strength α</span>
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
                value={clampedStrength}
                onChange={(event) => setStrength(Number(event.target.value))}
              />
            </label>
            {featureClamped && (
              <p className="field-note">Hidden dimension clamped to {clampedFeature}.</p>
            )}
            {strengthClamped && <p className="field-note">Strength clamped to {clampedStrength}.</p>}
            <button type="button" className="secondary" onClick={() => setStrength(0)} disabled={strength === 0}>
              Reset strength to 0
            </button>
          </fieldset>
        )}

        {mode === "sweep" && (
          <fieldset className="control-card steering-card">
            <legend>Sweep settings</legend>
            <p className="subtle">
              Each strength and optional random control is compared against one shared baseline.
            </p>
            <div className="form-grid">
              <label>
                <span>Hidden dimension (0–{featureMax})</span>
                <input
                  type="number"
                  min={0}
                  max={featureMax}
                  value={sweepFeatureIndex}
                  onChange={(event) => setSweepFeatureIndex(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Repeats (1–3)</span>
                <input
                  type="number"
                  min={1}
                  max={3}
                  value={repeats}
                  onChange={(event) => setRepeats(Number(event.target.value))}
                />
              </label>
            </div>
            <label>
              <span>Strengths α (comma-separated, 1–9 values)</span>
              <input
                type="text"
                value={strengthsText}
                placeholder="-20,-10,0,10,20"
                onChange={(event) => setStrengthsText(event.target.value)}
              />
            </label>
            {sweepFeatureClamped && (
              <p className="field-note">Hidden dimension clamped to {clampedSweepFeature}.</p>
            )}
            <label className="checkbox">
              <input
                type="checkbox"
                checked={includeControl}
                onChange={(event) => setIncludeControl(event.target.checked)}
              />
              <span>Include random-feature control</span>
            </label>
            {includeControl && (
              <label>
                <span>Control seed (optional)</span>
                <input
                  type="number"
                  placeholder="random"
                  value={controlSeed}
                  onChange={(event) => setControlSeed(event.target.value)}
                />
              </label>
            )}
            <p className="subtle">Sweeps run sequentially; larger sweeps can take several minutes.</p>
          </fieldset>
        )}

        <fieldset className="control-card">
          <legend>Data options</legend>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={captureEnabled}
              disabled={!selectedSteerableId}
              onChange={(event) => setCaptureEnabled(event.target.checked)}
            />
            <span>
              Capture activations
              {selectedSteerableId ? ` for ${selectedSteerableId}` : " (inspect a hidden-state module)"}
            </span>
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={persistActivations}
              onChange={(event) => setPersistActivations(event.target.checked)}
            />
            <span>Persist activation snapshot (saves tensor data to disk)</span>
          </label>
        </fieldset>
      </div>

      <div className="generation-footer">
        <div className="run-bar">
          <button className="primary run-button" onClick={runCurrent} disabled={!canRun}>
            {runLabel}
          </button>
          <p className="run-status" aria-live="polite">
            {statusText}
          </p>
        </div>

        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}
        {(sweepLocalError || sweepError) && (
          <div className="error-banner" role="alert">
            {sweepLocalError ?? sweepError}
          </div>
        )}
      </div>

      <div className="results" aria-live="polite">
        {pair && (
          <div className="comparison">
            <h4>
              Baseline vs steered · hook: {pair.steered.steering?.module_id} · dim{" "}
              {pair.steered.steering?.feature_index} · α {pair.steered.steering?.strength}
            </h4>
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

        {sweepResult && (
          <div className="sweep-results">
            <h4>
              Sweep {sweepResult.sweep_id.slice(0, 8)} · {sweepResult.runs.length} run
              {sweepResult.runs.length === 1 ? "" : "s"}
            </h4>
            {sweepResult.warnings.length > 0 && (
              <div className="warning-banner">{sweepResult.warnings.join("; ")}</div>
            )}
            <table className="data-table sweep-table">
              <thead>
                <tr>
                  <th>Kind</th>
                  <th>Index</th>
                  <th>Repeat</th>
                  <th>Feature</th>
                  <th>Strength</th>
                  <th>Duration</th>
                  <th>Tok/s</th>
                  <th>Output</th>
                </tr>
              </thead>
              <tbody>
                {sweepResult.runs.map((run) => (
                  <tr key={`${run.kind}-${run.sweep_index}-${run.repeat_index}`}>
                    <td>
                      <span className={`badge condition-${run.kind}`}>{run.kind}</span>
                    </td>
                    <td>{run.sweep_index}</td>
                    <td>{run.repeat_index}</td>
                    <td>{run.run.steering?.feature_index ?? "—"}</td>
                    <td>{run.run.steering?.strength ?? "—"}</td>
                    <td>{run.run.duration_ms.toFixed(0)} ms</td>
                    <td>{run.run.tokens_per_second?.toFixed(1) ?? "—"}</td>
                    <td>
                      <SweepOutputCell run={run.run} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

function SweepOutputCell({ run }: { run: BaselineResponse }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="sweep-output-cell">
      <button className="secondary small" onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Collapse" : "Show output"}
      </button>
      {expanded && <pre className="output small">{run.output || "(empty output)"}</pre>}
    </div>
  );
}
