import { useEffect, useRef, useState } from "react";
import { runNeuronSteering, runNeuronTrace } from "../api/client";
import type {
  GenerationSettings,
  NeuronSteeringResponse,
  NeuronTraceResponse,
} from "../types";

interface LayerOption {
  layer_index: number;
  module_id: string;
}

interface Props {
  open: boolean;
  disabled?: boolean;
  onClose: () => void;
  layers: LayerOption[];
}

const DEFAULT_PROMPT = "Describe a rainy morning.";
const GREEDY: Omit<GenerationSettings, "max_new_tokens"> = {
  temperature: 0,
  top_p: 1,
  seed: null,
  enable_thinking: false,
};
const NEURON_COLORS = [
  "#ff9500",
  "#34c759",
  "#ff2d55",
  "#af52de",
  "#5e5ce6",
  "#00c7be",
  "#ffcc00",
  "#8e8e93",
];

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function metric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function displayToken(text: string): string {
  if (text === "") return "∅";
  return text.replace(/\n/g, "\\n");
}

function clusterColor(id: number): string {
  return NEURON_COLORS[id % NEURON_COLORS.length];
}

export function NeuronDrawer({ open, disabled = false, onClose, layers }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const mapRef = useRef<HTMLCanvasElement>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [layerIndex, setLayerIndex] = useState(16);
  const [clusterK, setClusterK] = useState(6);
  const [maxTokens, setMaxTokens] = useState(16);
  const [trace, setTrace] = useState<NeuronTraceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [alpha, setAlpha] = useState(0.7);
  const [steering, setSteering] = useState<NeuronSteeringResponse | null>(null);
  const [steeringLoading, setSteeringLoading] = useState(false);
  const [steeringError, setSteeringError] = useState<string | null>(null);

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
    if (layers.length === 0) return;
    if (!layers.some((layer) => layer.layer_index === layerIndex)) {
      setLayerIndex(
        layers.find((layer) => layer.layer_index === 16)?.layer_index ??
          layers[Math.floor(layers.length / 2)].layer_index,
      );
    }
  }, [layers, layerIndex]);

  useEffect(() => {
    const canvas = mapRef.current;
    if (!canvas || !trace) return;
    const count = trace.intermediate_size;
    const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
    const rows = Math.max(1, Math.ceil(count / columns));
    const cell = clamp(Math.floor(280 / columns), 3, 10);
    canvas.width = columns * cell;
    canvas.height = rows * cell;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = "#f5f5f7";
    context.fillRect(0, 0, canvas.width, canvas.height);
    const order = Array.from({ length: count }, (_, index) => index).sort((left, right) => {
      const leftCluster = trace.assignments[left] ?? -1;
      const rightCluster = trace.assignments[right] ?? -1;
      return leftCluster - rightCluster || left - right;
    });
    order.forEach((index, position) => {
      const clusterId = trace.assignments[index] ?? -1;
      context.fillStyle = clusterId >= 0 ? clusterColor(clusterId) : "#c7c7cc";
      context.fillRect(
        (position % columns) * cell,
        Math.floor(position / columns) * cell,
        Math.max(1, cell - 0.5),
        Math.max(1, cell - 0.5),
      );
    });
  }, [trace]);

  const selectedCluster = trace?.clusters.find((cluster) => cluster.id === selectedClusterId) ?? null;
  const selectedPeak =
    selectedCluster && trace ? trace.tokens[selectedCluster.peak_position] ?? null : null;
  const changed = Boolean(steering && steering.steered.output !== steering.baseline.output);

  if (!open) return null;

  const settingsForTrace = (): GenerationSettings => ({
    ...GREEDY,
    max_new_tokens: clamp(Math.round(maxTokens), 1, 256),
  });

  const submitTrace = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setSteering(null);
    setSteeringError(null);
    try {
      const result = await runNeuronTrace({
        prompt,
        system: system.trim() === "" ? null : system,
        settings: settingsForTrace(),
        layer_index: layerIndex,
        cluster_k: clusterK,
        cluster_max_positions: 128,
      });
      setTrace(result);
      setSelectedClusterId(result.clusters[0]?.id ?? null);
      setAlpha(0.7);
    } catch (traceError) {
      setError(errorMessage(traceError));
    } finally {
      setLoading(false);
    }
  };

  const submitSteering = async () => {
    if (!trace || !selectedCluster || steeringLoading) return;
    setSteeringLoading(true);
    setSteeringError(null);
    try {
      const result = await runNeuronSteering({
        prompt: trace.prompt,
        system: trace.system,
        settings: settingsForTrace(),
        layer_index: trace.layer_index,
        neurons: selectedCluster.top_neurons.map((top) => ({
          neuron: top.neuron,
          weight: top.weight,
        })),
        alpha: Number(clamp(alpha, -1.5, 1.5).toFixed(4)),
      });
      setSteering(result);
    } catch (steerError) {
      setSteeringError(errorMessage(steerError));
    } finally {
      setSteeringLoading(false);
    }
  };

  return (
    <div className="neuron-backdrop" onClick={onClose}>
      <section
        className="neuron-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="neuron-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="neuron-header">
          <div>
            <p className="help-kicker">MLP w2 intermediate</p>
            <h2 id="neuron-title">Neuron explorer</h2>
            <p className="subtle">
              One real generation, the selected layer&apos;s 7680 MLP neurons captured from the w2
              input and clustered by activation profile.
            </p>
          </div>
          <button ref={closeRef} className="secondary" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="neuron-body">
          <div className="neuron-controls">
            <fieldset className="control-card">
              <legend>Trace</legend>
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
              <div className="form-grid">
                <label>
                  <span>Layer</span>
                  <select
                    value={layerIndex}
                    onChange={(event) => setLayerIndex(Number(event.target.value))}
                  >
                    {layers.map((layer) => (
                      <option key={layer.module_id} value={layer.layer_index}>
                        layer {layer.layer_index}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Clusters</span>
                  <select
                    value={clusterK}
                    onChange={(event) => setClusterK(Number(event.target.value))}
                  >
                    {[3, 4, 5, 6, 8, 10].map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
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
              <button
                className="primary"
                onClick={() => void submitTrace()}
                disabled={disabled || loading || !prompt.trim() || layers.length === 0}
              >
                {loading ? "Tracing neurons…" : "Trace neurons"}
              </button>
              <p className="subtle">
                The trace captures the w2 input for one layer; clusters use up to 128 sampled
                positions. The run is saved to history without its tensors.
              </p>
            </fieldset>

            {disabled && (
              <div className="warning-banner inline">
                Load the model first. The trace needs a real forward pass.
              </div>
            )}
            {error && <div className="error-banner inline">{error}</div>}
          </div>

          <div className="neuron-results" aria-live="polite">
            {!trace && !error && (
              <div className="empty neuron-empty">
                Run a trace to cluster the selected layer&apos;s MLP neurons. Each cluster is a group
                of neurons whose z-scored activation profiles move together; the steering below
                perturbs the w2 intermediate.
              </div>
            )}

            {trace && (
              <>
                <div className="neuron-summary">
                  <span className="badge steerable-badge">layer {trace.layer_index}</span>
                  <span className="badge">{trace.layer_type}</span>
                  <span className="badge">{trace.intermediate_size} neurons</span>
                  <span className="badge">residual RMS {metric(trace.residual_rms)}</span>
                  <span className="badge">k = {trace.clusters.length}</span>
                </div>
                <pre className="output small neuron-output">
                  {trace.output || "(no text generated)"}
                </pre>

                <div className="neuron-overview">
                  <div className="neuron-map-card">
                    <h4>Neuron map</h4>
                    <canvas
                      ref={mapRef}
                      className="neuron-map"
                      role="img"
                      aria-label="Neuron cluster map over the w2 intermediate"
                    />
                    <p className="subtle">Each cell is one w2 neuron, grouped by cluster; color = cluster.</p>
                  </div>
                  <div className="neuron-cluster-list">
                    {trace.clusters.map((cluster) => (
                      <button
                        key={cluster.id}
                        type="button"
                        className={`neuron-cluster-row ${
                          selectedClusterId === cluster.id ? "selected" : ""
                        }`}
                        onClick={() => {
                          setSelectedClusterId(cluster.id);
                          setSteering(null);
                          setSteeringError(null);
                        }}
                      >
                        <span
                          className="neuron-swatch"
                          style={{ background: clusterColor(cluster.id) }}
                        />
                        <span className="neuron-cluster-name">cluster {cluster.id}</span>
                        <span className="neuron-cluster-meta">
                          {cluster.size} · {percent(cluster.mass)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {selectedCluster && (
                  <div className="neuron-detail">
                    <h4>Cluster {selectedCluster.id}</h4>
                    <p className="subtle">
                      {selectedCluster.size} neurons · {percent(selectedCluster.mass)} of activation
                      · cohesion {selectedCluster.cohesion.toFixed(2)}
                      {selectedPeak
                        ? ` · peak at token ${selectedCluster.peak_position} (“${displayToken(
                            selectedPeak.text,
                          )}”)`
                        : ""}
                    </p>
                    <Sparkline profile={selectedCluster.profile} />
                    <h4>Top neurons</h4>
                    <div className="neuron-chips">
                      {selectedCluster.top_neurons.map((top) => (
                        <span
                          key={top.neuron}
                          className="neuron-chip"
                          title={`mean |a| ${top.mean_abs.toFixed(4)} · σ ${top.sigma.toFixed(4)}`}
                        >
                          n={top.neuron}{" "}
                          <b>
                            {top.weight >= 0 ? "+" : ""}
                            {top.weight.toFixed(2)}
                          </b>
                        </span>
                      ))}
                    </div>

                    <fieldset className="control-card neuron-steer-card">
                      <legend>Steer this group</legend>
                      <label>
                        <span>Strength α (residual RMS units)</span>
                        <input
                          type="number"
                          step={0.05}
                          value={alpha}
                          onChange={(event) => setAlpha(Number(event.target.value))}
                        />
                      </label>
                      <input
                        type="range"
                        min={-1.5}
                        max={1.5}
                        step={0.05}
                        value={clamp(alpha, -1.5, 1.5)}
                        aria-label="Neuron steering alpha"
                        onChange={(event) => setAlpha(Number(event.target.value))}
                      />
                      <p className="subtle">
                        The backend computes δᵢ = α · residual_rms / ‖direction‖ · wᵢ on the w2
                        intermediate. α = 0 reproduces the baseline exactly.
                      </p>
                      <button
                        className="primary"
                        onClick={() => void submitSteering()}
                        disabled={steeringLoading}
                      >
                        {steeringLoading ? "Running baseline + steered…" : "Run baseline + steered"}
                      </button>
                      {steeringError && <div className="error-banner inline">{steeringError}</div>}
                      {steering && (
                        <div className="neuron-steer-result">
                          <span
                            className={`badge ${changed ? "steerable-badge" : "muted-badge"}`}
                          >
                            {changed ? "output changed" : "output unchanged"}
                          </span>
                          <p className="subtle">
                            direction norm {metric(steering.direction_norm)} · residual RMS{" "}
                            {metric(steering.residual_rms)} · {steering.duration_ms.toFixed(0)} ms
                          </p>
                          <div className="neuron-compare">
                            <div>
                              <h4>Baseline</h4>
                              <pre className="output small">
                                {steering.baseline.output || "(no text)"}
                              </pre>
                            </div>
                            <div>
                              <h4>Steered</h4>
                              <pre className="output small">
                                {steering.steered.output || "(no text)"}
                              </pre>
                            </div>
                          </div>
                          {!changed && (
                            <p className="subtle">
                              No change is a real result; try a larger |α| or another cluster.
                            </p>
                          )}
                        </div>
                      )}
                    </fieldset>

                    <p className="subtle">
                      This is the pilot&apos;s w2-column group intervention: high separation is a
                      hypothesis, not evidence that the group controls a behaviour.
                    </p>
                  </div>
                )}

                {trace.warnings.map((warning) => (
                  <div key={warning} className="warning-banner inline">
                    {warning}
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function Sparkline({ profile }: { profile: number[] }) {
  if (profile.length < 2) return null;
  const width = 280;
  const height = 54;
  const pad = 5;
  const max = Math.max(1e-9, ...profile.map((value) => Math.abs(value)));
  const points = profile
    .map((value, index) => {
      const x = pad + (index / (profile.length - 1)) * (width - pad * 2);
      const y = height / 2 - (value / max) * (height / 2 - pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      className="neuron-sparkline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Cluster activation profile across the trace"
    >
      <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} stroke="var(--rule)" />
      <polyline points={points} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
    </svg>
  );
}
