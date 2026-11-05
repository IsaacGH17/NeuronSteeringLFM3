import { useEffect, useMemo, useState } from "react";
import type {
  BaselineResponse,
  BatchJob,
  BatchRequest,
  DirectionResponse,
  GenerationSettings,
  ModulePreset,
  PromptSuite,
  SteeringComponent,
  SteeringNormalization,
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
    feature_index: number | null;
    components?: SteeringComponent[];
    normalization?: SteeringNormalization;
    strengths: number[];
    include_control: boolean;
    control_seed: number | null;
    repeats: number;
    capture_module_ids: string[];
    persist_activations: boolean | null;
  }) => void;
  direction?: DirectionResponse | null;
  onClearDirection?: () => void;
  modulePresets?: ModulePreset[];
  suites?: PromptSuite[];
  batchJob?: BatchJob | null;
  batchError?: string | null;
  onStartBatch?: (payload: BatchRequest) => void;
  onCancelBatch?: () => void;
}

type Mode = "baseline" | "steered" | "sweep" | "batch";

const MODES: Record<Mode, { label: string; runLabel: string; hint: string }> = {
  baseline: {
    label: "Baseline",
    runLabel: "Run baseline",
    hint: "A plain generation with nothing touched. It is the reference for everything else.",
  },
  steered: {
    label: "Steered",
    runLabel: "Run baseline + steered",
    hint: "Two generations, plain and steered, with the same prompt, settings, and seed.",
  },
  sweep: {
    label: "Sweep",
    runLabel: "Run sweep",
    hint: "One shared baseline, then one steered run per strength, plus an optional random control.",
  },
  batch: {
    label: "Batch",
    runLabel: "Start batch",
    hint: "A whole matrix — prompt suite × modules × strengths — with live progress and cancellation.",
  },
};

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";
const DEFAULT_STRENGTHS = "-20,-10,0,10,20";
const DEFAULT_BATCH_STRENGTHS = "-2,-1,0,1,2";
const RAW_LIMIT = 1000;
const SIGMA_LIMIT = 20;
const SLIDER_LIMIT = 50;
const SIGMA_SLIDER_LIMIT = 10;

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
  direction = null,
  onClearDirection,
  modulePresets = [],
  suites = [],
  batchJob = null,
  batchError = null,
  onStartBatch,
  onCancelBatch,
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
  const [normalization, setNormalization] = useState<SteeringNormalization>("raw");
  const [components, setComponents] = useState<SteeringComponent[]>([]);
  const [newDimension, setNewDimension] = useState(0);
  const [newWeight, setNewWeight] = useState(1);

  const [sweepFeatureIndex, setSweepFeatureIndex] = useState(0);
  const [strengthsText, setStrengthsText] = useState(DEFAULT_STRENGTHS);
  const [includeControl, setIncludeControl] = useState(false);
  const [controlSeed, setControlSeed] = useState("");
  const [repeats, setRepeats] = useState(1);
  const [sweepRunning, setSweepRunning] = useState(false);
  const [sweepLocalError, setSweepLocalError] = useState<string | null>(null);

  const [suiteId, setSuiteId] = useState("");
  const [presetId, setPresetId] = useState("");
  const [layerFrom, setLayerFrom] = useState(0);
  const [layerTo, setLayerTo] = useState(999);
  const [batchStrengthsText, setBatchStrengthsText] = useState(DEFAULT_BATCH_STRENGTHS);
  const [batchRepeats, setBatchRepeats] = useState(1);
  const [batchControl, setBatchControl] = useState(false);
  const [batchBaseline, setBatchBaseline] = useState(true);
  const [batchLocalError, setBatchLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (sweepResult || sweepError) {
      setSweepRunning(false);
    }
  }, [sweepResult, sweepError]);

  useEffect(() => {
    if (direction) {
      setComponents(direction.components);
      setNormalization("sigma");
    }
  }, [direction]);

  useEffect(() => {
    if (!presetId && modulePresets.length > 0) {
      setPresetId(modulePresets[0].id);
    }
  }, [modulePresets, presetId]);

  const featureMax = hiddenSize !== null ? hiddenSize - 1 : 2047;
  const clampFeature = (value: number) =>
    Math.max(0, Math.min(featureMax, Math.round(Number.isFinite(value) ? value : 0)));
  const strengthLimit = normalization === "raw" ? RAW_LIMIT : SIGMA_LIMIT;
  const clampStrength = (value: number) =>
    Math.max(-strengthLimit, Math.min(strengthLimit, Number.isFinite(value) ? value : 0));

  const clampedFeature = clampFeature(featureIndex);
  const clampedSweepFeature = clampFeature(sweepFeatureIndex);
  const clampedStrength = clampStrength(strength);
  const clampedNewDimension = clampFeature(newDimension);
  const featureClamped = clampedFeature !== Math.round(Number.isFinite(featureIndex) ? featureIndex : 0);
  const sweepFeatureClamped =
    clampedSweepFeature !== Math.round(Number.isFinite(sweepFeatureIndex) ? sweepFeatureIndex : 0);
  const strengthClamped = clampedStrength !== strength;
  const multiDim = components.length > 0;

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

  const steeringPayload = (moduleId: string): SteeringSpec => {
    if (multiDim) {
      return {
        module_id: moduleId,
        strength: Number(clampedStrength.toFixed(6)),
        feature_index: components.length === 1 ? components[0].feature_index : null,
        components,
        normalization,
      };
    }
    const single: SteeringSpec = {
      module_id: moduleId,
      feature_index: clampedFeature,
      strength: Number(clampedStrength.toFixed(6)),
    };
    if (normalization !== "raw") single.normalization = normalization;
    return single;
  };

  const submit = () => {
    if (!prompt.trim() || disabled || running) return;
    const { promptText, systemText, settings, captures, persistActivations: persist } = sharedArgs();
    onRun(promptText, systemText, settings, captures, persist);
  };

  const submitSteered = () => {
    if (!prompt.trim() || disabled || running || !steeringTargetId || !targetReadyNow()) return;
    const { promptText, systemText, settings, captures, persistActivations: persist } = sharedArgs();
    onRunSteered(
      promptText,
      systemText,
      settings,
      steeringPayload(steeringTargetId),
      captures,
      persist,
    );
  };

  const parseStrengths = (text: string): { strengths: number[]; error: string | null } => {
    const parts = text
      .split(",")
      .map((part) => part.trim())
      .filter((part) => part !== "");
    const values: number[] = [];
    for (const part of parts) {
      const value = Number(part);
      if (!Number.isFinite(value)) {
        return { strengths: [], error: `Invalid strength: "${part}"` };
      }
      values.push(value);
    }
    if (values.length === 0) return { strengths: [], error: "Enter at least one strength." };
    if (values.length > 32) return { strengths: [], error: "At most 32 strengths allowed." };
    return { strengths: values, error: null };
  };

  const submitSweep = () => {
    if (!prompt.trim() || disabled || sweepRunning || !steeringTargetId || !targetReadyNow()) return;
    const { strengths, error: parseError } = parseStrengths(strengthsText);
    if (parseError) {
      setSweepLocalError(parseError);
      return;
    }
    setSweepLocalError(null);
    const clampedRepeats = Math.max(1, Math.min(10, Math.round(repeats)));
    const controlSeedValue =
      Number.isFinite(Number(controlSeed)) && controlSeed.trim() !== "" ? Number(controlSeed) : null;
    setSweepRunning(true);
    const { promptText, systemText, settings, captures, persistActivations: persist } = sharedArgs();
    onRunSweep({
      prompt: promptText,
      system: systemText,
      settings,
      module_id: steeringTargetId,
      feature_index: multiDim ? null : clampedFeature,
      ...(multiDim
        ? { components, normalization }
        : normalization === "raw"
          ? {}
          : { normalization }),
      strengths,
      include_control: includeControl,
      control_seed: controlSeedValue,
      repeats: clampedRepeats,
      capture_module_ids: captures,
      persist_activations: persist,
    });
  };

  const selectedPreset = useMemo(
    () => modulePresets.find((preset) => preset.id === presetId) ?? modulePresets[0] ?? null,
    [modulePresets, presetId],
  );
  const selectedSuite = useMemo(
    () => suites.find((suite) => suite.id === suiteId) ?? null,
    [suites, suiteId],
  );
  const batchModules = useMemo(
    () =>
      (selectedPreset?.modules ?? []).filter(
        (module) => module.layer_index >= layerFrom && module.layer_index <= layerTo,
      ),
    [selectedPreset, layerFrom, layerTo],
  );

  const batchPlan = useMemo(() => {
    const parsed = parseStrengths(batchStrengthsText);
    const strengths = parsed.strengths;
    const promptCount = selectedSuite ? selectedSuite.prompts.length : 1;
    const perPrompt = strengths.length + (batchControl ? 1 : 0);
    const total =
      promptCount * (batchBaseline ? 1 : 0) +
      promptCount * batchModules.length * perPrompt * Math.max(1, Math.round(batchRepeats));
    return { strengths, error: parsed.error, promptCount, total };
  }, [batchStrengthsText, batchModules.length, batchBaseline, batchControl, batchRepeats, selectedSuite]);

  const submitBatch = () => {
    if (!steeringTargetId || disabled || !onStartBatch) return;
    if (!selectedSuite && !prompt.trim()) return;
    const { strengths, error: parseError } = batchPlan;
    if (parseError) {
      setBatchLocalError(parseError);
      return;
    }
    if (batchModules.length === 0) {
      setBatchLocalError("No modules in that layer range.");
      return;
    }
    if (batchPlan.total === 0) {
      setBatchLocalError("This plan would run nothing. Enable the baseline or add strengths.");
      return;
    }
    setBatchLocalError(null);
    const { systemText, settings, captures, persistActivations: persist } = sharedArgs();
    const planComponents: SteeringComponent[] = multiDim
      ? components
      : [{ feature_index: clampedFeature, weight: 1 }];
    onStartBatch({
      suite_id: selectedSuite ? selectedSuite.id : null,
      prompts: selectedSuite ? [] : [prompt],
      system: systemText,
      settings,
      plan: {
        module_ids: batchModules.map((module) => module.module_id),
        components: planComponents,
        normalization,
        strengths,
        include_control: batchControl,
        control_seed: null,
        repeats: Math.max(1, Math.min(10, Math.round(batchRepeats))),
        baseline_per_prompt: batchBaseline,
      },
      capture_module_ids: captures,
      persist_activations: persist,
    });
  };

  function targetReadyNow(): boolean {
    return Boolean(steeringTargetId) && (multiDim ? components.length > 0 : true);
  }

  const addComponent = () => {
    const index = clampedNewDimension;
    const weight = Number.isFinite(newWeight) ? newWeight : 1;
    setComponents((current) => {
      const existing = current.find((component) => component.feature_index === index);
      if (existing) {
        return current.map((component) =>
          component.feature_index === index
            ? { ...component, weight: Number((component.weight + weight).toFixed(6)) }
            : component,
        );
      }
      return [...current, { feature_index: index, weight }];
    });
  };

  const removeComponent = (index: number) => {
    setComponents((current) => current.filter((component) => component.feature_index !== index));
    if (components.length <= 1 && onClearDirection) onClearDirection();
  };

  const clearDirection = () => {
    setComponents([]);
    if (onClearDirection) onClearDirection();
  };

  const runCurrent = () => {
    if (mode === "baseline") submit();
    else if (mode === "steered") submitSteered();
    else if (mode === "sweep") submitSweep();
    else submitBatch();
  };

  const batchBusy = Boolean(batchJob && (batchJob.state === "queued" || batchJob.state === "running"));
  const promptReady = prompt.trim().length > 0;
  const targetReady = targetReadyNow();
  const busy = running || (mode === "sweep" && sweepRunning) || (mode === "batch" && batchBusy);
  const promptSourceReady = mode === "batch" ? Boolean(selectedSuite) || promptReady : promptReady;
  const canRun =
    !disabled && promptSourceReady && !busy && (mode === "baseline" || targetReady);
  const runLabel = busy
    ? mode === "sweep"
      ? "Running sweep…"
      : mode === "batch"
        ? "Batch running…"
        : "Generating…"
    : MODES[mode].runLabel;
  const statusText = disabled
    ? "Load the model first and generation switches on."
    : mode === "batch" && !promptSourceReady
      ? "Write a prompt, or pick a suite to run the matrix across."
      : !promptSourceReady
        ? "Write a prompt to run."
        : mode !== "baseline" && !targetReady
          ? "Choose a steering target for this mode."
          : busy
            ? mode === "sweep"
              ? "Working through the sweep one run at a time. A big one can take a few minutes."
              : mode === "batch"
                ? "Running the matrix one generation at a time."
                : "The model is writing."
            : "Ready when you are.";

  return (
    <section className="panel generation">
      <header className="panel-header">
        <div>
          <h2>Steering and generation</h2>
          <p className="subtle">Choose a mode, check the target, then run one experiment at a time.</p>
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
              ? "Baseline runs do not use the target at all."
              : targetReady
                ? "Steered, sweep, and batch runs nudge this module's output while you inspect whatever you like."
                : "Pick a steerable module in the inspector and set it as the target. Browsing other modules will not move it."}
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
          {direction && (
            <div className="direction-row">
              <span className="badge steerable-badge">direction · {direction.components.length} dims</span>
              <span className="subtle">
                from two runs · separation {direction.separation.toFixed(2)} · {direction.positions_compared}{" "}
                positions
              </span>
              <button type="button" className="secondary small" onClick={clearDirection}>
                Clear direction
              </button>
            </div>
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

        {mode !== "baseline" && (
          <fieldset className="control-card steering-card">
            <legend>Steering</legend>
            <p className="subtle">
              Shift hidden dimensions by α at every token. α=0 must reproduce the baseline exactly.
              In σ mode, α is measured in units of each dimension&apos;s standard deviation, so the same
              α means the same displacement across layers.
            </p>
            <div className="form-grid">
              <label>
                <span>Normalization</span>
                <select
                  value={normalization}
                  onChange={(event) =>
                    setNormalization(event.target.value as SteeringNormalization)
                  }
                >
                  <option value="raw">raw activation units</option>
                  <option value="sigma">σ units (per dimension)</option>
                </select>
              </label>
              <label>
                <span>Strength α</span>
                <input
                  type="number"
                  step={0.5}
                  value={strength}
                  onChange={(event) => setStrength(Number(event.target.value))}
                />
              </label>
            </div>

            {!multiDim ? (
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
            ) : (
              <div className="component-list" aria-label="Steering direction components">
                {components.map((component) => (
                  <button
                    key={component.feature_index}
                    type="button"
                    className="component-chip"
                    aria-label={`Remove dimension ${component.feature_index}`}
                    onClick={() => removeComponent(component.feature_index)}
                  >
                    <span>
                      #{component.feature_index} ×{component.weight}
                    </span>
                    <span aria-hidden="true">✕</span>
                  </button>
                ))}
              </div>
            )}

            <div className="dimension-add">
              {!multiDim && (
                <p className="subtle dimension-add-hint">
                  Combine dimensions: add them with weights to steer a whole direction instead of a
                  single slot.
                </p>
              )}
              <label>
                <span>New dimension (0–{featureMax})</span>
                <input
                  type="number"
                  min={0}
                  max={featureMax}
                  value={newDimension}
                  onChange={(event) => setNewDimension(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Weight</span>
                <input
                  type="number"
                  step={0.1}
                  value={newWeight}
                  onChange={(event) => setNewWeight(Number(event.target.value))}
                />
              </label>
              <button type="button" className="secondary small" onClick={addComponent}>
                Add dimension
              </button>
            </div>

            <label className="strength-slider">
              <span>Strength slider: {strength}</span>
              <input
                type="range"
                min={normalization === "raw" ? -SLIDER_LIMIT : -SIGMA_SLIDER_LIMIT}
                max={normalization === "raw" ? SLIDER_LIMIT : SIGMA_SLIDER_LIMIT}
                step={normalization === "raw" ? 0.5 : 0.1}
                value={Math.max(
                  normalization === "raw" ? -SLIDER_LIMIT : -SIGMA_SLIDER_LIMIT,
                  Math.min(
                    normalization === "raw" ? SLIDER_LIMIT : SIGMA_SLIDER_LIMIT,
                    Number.isFinite(strength) ? strength : 0,
                  ),
                )}
                onChange={(event) => setStrength(Number(event.target.value))}
              />
            </label>
            {featureClamped && !multiDim && (
              <p className="field-note">Hidden dimension clamped to {clampedFeature}.</p>
            )}
            {strengthClamped && (
              <p className="field-note">
                Strength clamped to {clampedStrength} (server limit ±{strengthLimit}).
              </p>
            )}
            <div className="button-row">
              <button
                type="button"
                className="secondary"
                onClick={() => setStrength(0)}
                disabled={strength === 0}
              >
                Reset strength to 0
              </button>
              {multiDim && !direction && (
                <button type="button" className="secondary" onClick={clearDirection}>
                  Remove all dimensions
                </button>
              )}
            </div>
          </fieldset>
        )}

        {mode === "sweep" && (
          <fieldset className="control-card">
            <legend>Sweep settings</legend>
            <p className="subtle">
              Every strength, and the optional random control, is measured against the same baseline.
            </p>
            <div className="form-grid">
              {!multiDim && (
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
              )}
              <label>
                <span>Repeats (1–10)</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={repeats}
                  onChange={(event) => setRepeats(Number(event.target.value))}
                />
              </label>
            </div>
            <label>
              <span>Strengths α (comma-separated, 1–32 values)</span>
              <input
                type="text"
                value={strengthsText}
                placeholder="-20,-10,0,10,20"
                onChange={(event) => setStrengthsText(event.target.value)}
              />
            </label>
            {sweepFeatureClamped && !multiDim && (
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
            <p className="subtle">Runs happen one after another, so a long sweep can take a few minutes.</p>
          </fieldset>
        )}

        {mode === "batch" && (
          <fieldset className="control-card steering-card">
            <legend>Batch plan</legend>
            <div className="form-grid">
              <label>
                <span>Prompt source</span>
                <select value={suiteId} onChange={(event) => setSuiteId(event.target.value)}>
                  <option value="">Current prompt only</option>
                  {suites.map((suite) => (
                    <option key={suite.id} value={suite.id}>
                      {suite.name} ({suite.prompts.length})
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Module preset</span>
                <select value={presetId} onChange={(event) => setPresetId(event.target.value)}>
                  {modulePresets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.label} ({preset.modules.length})
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {selectedSuite && <p className="subtle">{selectedSuite.description}</p>}
            <div className="form-grid">
              <label>
                <span>Strengths α (comma-separated)</span>
                <input
                  type="text"
                  value={batchStrengthsText}
                  onChange={(event) => setBatchStrengthsText(event.target.value)}
                />
              </label>
              <label>
                <span>Repeats (1–10)</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={batchRepeats}
                  onChange={(event) => setBatchRepeats(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Layers from</span>
                <input
                  type="number"
                  min={0}
                  value={layerFrom}
                  onChange={(event) => setLayerFrom(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Layers to</span>
                <input
                  type="number"
                  min={0}
                  value={layerTo}
                  onChange={(event) => setLayerTo(Number(event.target.value))}
                />
              </label>
            </div>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={batchControl}
                onChange={(event) => setBatchControl(event.target.checked)}
              />
              <span>Include random-feature control</span>
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={batchBaseline}
                onChange={(event) => setBatchBaseline(event.target.checked)}
              />
              <span>Baseline per prompt</span>
            </label>
            <p className="subtle">
              {batchPlan.error
                ? batchPlan.error
                : `${batchModules.length} module(s) × ${batchPlan.strengths.length} strength(s) × ${
                    batchPlan.promptCount
                  } prompt(s) × ${Math.max(1, Math.round(batchRepeats))} repeat(s)${
                    batchControl ? " + control" : ""
                  }${batchBaseline ? ` + ${batchPlan.promptCount} baseline` : ""} = ${
                    batchPlan.total
                  } generations.`}
            </p>
            <p className="subtle">
              A batch runs in the background: you can cancel it, and completed runs are saved as you go.
            </p>
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
        {batchLocalError && (
          <div className="error-banner" role="alert">
            {batchLocalError}
          </div>
        )}
        {batchError && (
          <div className="error-banner" role="alert">
            {batchError}
          </div>
        )}
      </div>

      <div className="results" aria-live="polite">
        {batchJob && <BatchProgress job={batchJob} onCancel={onCancelBatch} />}

        {pair && (
          <div className="comparison">
            <h4>
              Baseline vs steered · hook: {pair.steered.steering?.module_id} ·{" "}
              {pair.steered.steering?.components?.length
                ? `${pair.steered.steering.components.length} dims`
                : `dim ${pair.steered.steering?.feature_index}`}{" "}
              · α {pair.steered.steering?.strength}
              {pair.steered.steering?.normalization === "sigma" ? " (σ)" : ""}
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
                    <pre className="output">{run.output || "(no text generated)"}</pre>
                    <MetricsLine run={run} />
                    <p className="subtle">
                      run {run.id.slice(0, 8)} · prompt tokens: {run.prompt_tokens}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {!pair && !batchJob && result && (
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
            <pre className="output">{result.output || "(no text generated)"}</pre>
            <MetricsLine run={result} />
            <p className="subtle">
              prompt tokens: {result.prompt_tokens} · run {result.id.slice(0, 8)} · {result.model_id}
            </p>
          </div>
        )}

        {!batchJob && sweepResult && (
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
                  <th>Checks</th>
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
                      {run.run.metrics?.checks_total
                        ? `${run.run.metrics.checks_passed}/${run.run.metrics.checks_total}`
                        : "—"}
                    </td>
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

function MetricsLine({ run }: { run: BaselineResponse }) {
  const metrics = run.metrics;
  if (!metrics) return null;
  const markers = Object.entries(metrics.lexicon)
    .filter(([, value]) => value > 0)
    .map(([name, value]) => `${name} ${value.toFixed(1)}`)
    .join(" · ");
  return (
    <p className="subtle metrics-line">
      {metrics.words} words · distinct-3 {metrics.distinct_3.toFixed(2)} · repetition{" "}
      {metrics.repeated_4gram_ratio.toFixed(2)}
      {metrics.checks_total ? ` · checks ${metrics.checks_passed}/${metrics.checks_total}` : ""}
      {markers ? ` · ${markers}` : ""}
      {metrics.refusal ? " · refusal" : ""}
    </p>
  );
}

function BatchProgress({ job, onCancel }: { job: BatchJob; onCancel?: () => void }) {
  const percent = job.total > 0 ? Math.round((job.completed / job.total) * 100) : 0;
  const running = job.state === "queued" || job.state === "running";
  const lexiconKeys = Array.from(
    new Set(job.cells.flatMap((cell) => Object.keys(cell.lexicon_mean))),
  ).sort();
  return (
    <div className="batch-progress">
      <div className="batch-head">
        <h4>
          Batch {job.job_id.slice(0, 8)}
          <span className={`badge condition-${job.state === "completed" ? "baseline" : "control"}`}>
            {job.state}
          </span>
        </h4>
        {running && onCancel && (
          <button type="button" className="secondary small" onClick={onCancel}>
            Cancel batch
          </button>
        )}
      </div>
      <p className="subtle">{job.plan}</p>
      <progress
        className="batch-bar"
        max={Math.max(1, job.total)}
        value={job.completed}
        aria-label={`Batch progress: ${job.completed} of ${job.total} runs`}
      />
      <p className="subtle">
        {job.completed}/{job.total} runs ({percent}%){job.current ? ` · ${job.current}` : ""}
        {job.error ? ` · ${job.error}` : ""}
      </p>
      {job.warnings.map((warning) => (
        <div key={warning} className="warning-banner inline">
          {warning}
        </div>
      ))}
      {job.cells.length > 0 && (
        <table className="data-table sweep-table batch-table">
          <thead>
            <tr>
              <th>Kind</th>
              <th>Module</th>
              <th>α</th>
              <th>Runs</th>
              <th>Words</th>
              <th>Distinct-3</th>
              <th>Checks</th>
              <th>Refusals</th>
              {lexiconKeys.map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {job.cells.map((cell) => (
              <tr key={`${cell.kind}-${cell.module_id}-${cell.strength}`}>
                <td>
                  <span className={`badge condition-${cell.kind}`}>{cell.kind}</span>
                </td>
                <td>{cell.module_id ?? "—"}</td>
                <td>{cell.strength ?? "—"}</td>
                <td>{cell.runs}</td>
                <td>{cell.words_mean.toFixed(1)}</td>
                <td>{cell.distinct_3_mean.toFixed(2)}</td>
                <td>
                  {cell.checks_passed_rate === null
                    ? "—"
                    : `${Math.round(cell.checks_passed_rate * 100)}%`}
                </td>
                <td>{Math.round(cell.refusal_rate * 100)}%</td>
                {lexiconKeys.map((key) => (
                  <td key={key}>{(cell.lexicon_mean[key] ?? 0).toFixed(1)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function SweepOutputCell({ run }: { run: BaselineResponse }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="sweep-output-cell">
      <button className="secondary small" onClick={() => setExpanded((value) => !value)}>
        {expanded ? "Collapse" : "Show output"}
      </button>
      {expanded && <pre className="output small">{run.output || "(no text generated)"}</pre>}
    </div>
  );
}
