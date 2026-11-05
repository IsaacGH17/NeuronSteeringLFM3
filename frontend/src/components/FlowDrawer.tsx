import { useEffect, useMemo, useRef, useState } from "react";
import { runFlow, runSteering } from "../api/client";
import type {
  DirectionResponse,
  FlowCluster,
  FlowLayer,
  FlowResponse,
  GenerationSettings,
  SteeringResponse,
  SteeringSpec,
  TokenInfo,
} from "../types";

interface Props {
  open: boolean;
  disabled?: boolean;
  onClose: () => void;
  onUseDirection: (direction: DirectionResponse) => void;
  onExperimentRecorded?: () => void;
}

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";
const GREEDY: Omit<GenerationSettings, "max_new_tokens"> = {
  temperature: 0,
  top_p: 1,
  seed: null,
  enable_thinking: false,
};
const CLUSTER_COLORS = [
  "var(--flow-1)",
  "var(--flow-2)",
  "var(--flow-3)",
  "var(--flow-4)",
  "var(--flow-5)",
  "var(--flow-6)",
];

const MIN_STEP = 36;
const MAX_STEP = 64;
const GRAPH_HEIGHT = 300;
const CENTER_Y = 150;
const LEFT = 44;
const SIGMA_LIMIT = 10;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function clusterColor(id: number): string {
  return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function displayToken(text: string): string {
  if (text === "") return "∅";
  return text.replace(/\n/g, "\\n");
}

function metric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

function tokenAt(tokens: TokenInfo[], index: number): TokenInfo | null {
  return tokens.find((token) => token.index === index) ?? null;
}

export function FlowDrawer({
  open,
  disabled = false,
  onClose,
  onUseDirection,
  onExperimentRecorded,
}: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [maxTokens, setMaxTokens] = useState(32);
  const [clusterCount, setClusterCount] = useState(6);
  const [flow, setFlow] = useState<FlowResponse | null>(null);
  const [traceFormOpen, setTraceFormOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [tokenIndex, setTokenIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [alpha, setAlpha] = useState(1);
  const [steerResult, setSteerResult] = useState<SteeringResponse | null>(null);
  const [steerLoading, setSteerLoading] = useState(false);
  const [steerError, setSteerError] = useState<string | null>(null);

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
    if (!playing || !flow) return;
    if (tokenIndex >= flow.tokens.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setTokenIndex((current) => current + 1), 520);
    return () => clearTimeout(timer);
  }, [playing, tokenIndex, flow]);

  const selectedLayerData = useMemo(
    () => flow?.layers.find((layer) => layer.layer_index === selectedLayer) ?? null,
    [flow, selectedLayer],
  );
  const selectedClusterData = useMemo(
    () => selectedLayerData?.clusters.find((cluster) => cluster.id === selectedCluster) ?? null,
    [selectedLayerData, selectedCluster],
  );
  const currentToken = flow ? tokenAt(flow.tokens, tokenIndex) : null;

  if (!open) return null;

  const settingsForTrace = (): GenerationSettings => ({
    ...GREEDY,
    max_new_tokens: clamp(Math.round(maxTokens), 1, 128),
  });

  const submitTrace = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setSteerResult(null);
    setSteerError(null);
    try {
      const result = await runFlow({
        prompt,
        system: system.trim() === "" ? null : system,
        settings: settingsForTrace(),
        cluster_k: clusterCount,
        cluster_max_positions: 128,
      });
      setFlow(result);
      setTokenIndex(
        clamp(result.prompt_tokens - 1, 0, Math.max(0, result.tokens.length - 1)),
      );
      setSelectedLayer(
        result.layers.length > 0
          ? result.layers[Math.floor(result.layers.length / 2)].layer_index
          : null,
      );
      setSelectedCluster(null);
      setPlaying(false);
      setTraceFormOpen(false);
    } catch (traceError) {
      setError(errorMessage(traceError));
    } finally {
      setLoading(false);
    }
  };

  const selectLayer = (layerIndex: number) => {
    setSelectedLayer(layerIndex);
    setSelectedCluster(null);
    setSteerResult(null);
    setSteerError(null);
    setPlaying(false);
  };

  const selectCluster = (layerIndex: number, clusterId: number) => {
    setSelectedLayer(layerIndex);
    setSelectedCluster(clusterId);
    setSteerResult(null);
    setSteerError(null);
    setAlpha(1);
    setPlaying(false);
  };

  const submitSteer = async () => {
    if (!flow || !selectedLayerData || !selectedClusterData) return;
    const components = selectedClusterData.components;
    const spec: SteeringSpec = {
      module_id: selectedLayerData.module_id,
      strength: Number(clamp(alpha, -SIGMA_LIMIT, SIGMA_LIMIT).toFixed(6)),
      feature_index: components.length === 1 ? components[0].feature_index : null,
      components,
      normalization: "sigma",
    };
    setSteerLoading(true);
    setSteerError(null);
    try {
      const pair = await runSteering(flow.prompt, flow.system, flow.settings, spec);
      setSteerResult(pair);
      onExperimentRecorded?.();
    } catch (steerFailure) {
      setSteerError(errorMessage(steerFailure));
    } finally {
      setSteerLoading(false);
    }
  };

  const useInSteeringPanel = () => {
    if (!flow || !selectedLayerData || !selectedClusterData) return;
    onUseDirection({
      module_id: selectedLayerData.module_id,
      components: selectedClusterData.components,
      sigma: selectedClusterData.top_dimensions.map((dimension) => dimension.sigma),
      norm: selectedClusterData.direction_norm,
      separation: selectedClusterData.cohesion,
      positions_compared: flow.cluster_positions,
      warnings: [],
    });
    onClose();
  };

  return (
    <div className="flow-backdrop" onClick={onClose}>
      <section
        className="flow-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="flow-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flow-header">
          <div>
            <p className="help-kicker">Live trace</p>
            <h2 id="flow-title">Transformer flow</h2>
            <p className="subtle">
              One real generation, every decoder layer captured, drawn from the tensors themselves.
            </p>
          </div>
          <button ref={closeRef} className="secondary" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="flow-body">
          {flow && !traceFormOpen ? (
            <div className="flow-trace-compact">
              <div className="flow-trace-compact-copy">
                <span className="flow-trace-compact-label">Trace</span>
                <span className="flow-trace-compact-prompt">{flow.prompt}</span>
                <span className="subtle">
                  max {flow.settings.max_new_tokens} tokens · k = {flow.cluster_k} ·{" "}
                  {flow.layers.length} layers
                </span>
              </div>
              <button
                type="button"
                className="secondary small"
                onClick={() => setTraceFormOpen(true)}
              >
                New trace
              </button>
            </div>
          ) : (
          <fieldset className="control-card flow-trace-card">
            <legend>Trace a real generation</legend>
            <div className="flow-trace-fields">
              <label>
                <span>Prompt</span>
                <textarea rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
              </label>
              <label>
                <span>System instruction (optional)</span>
                <textarea
                  rows={2}
                  value={system}
                  placeholder="Optional"
                  onChange={(event) => setSystem(event.target.value)}
                />
              </label>
            </div>
            <div className="flow-trace-row">
              <label>
                <span>Max new tokens</span>
                <input
                  type="number"
                  min={1}
                  max={128}
                  value={maxTokens}
                  onChange={(event) => setMaxTokens(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Clusters per layer</span>
                <select
                  value={clusterCount}
                  onChange={(event) => setClusterCount(Number(event.target.value))}
                >
                  {[3, 4, 5, 6, 8, 10].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              {flow && (
                <button
                  type="button"
                  className="secondary small"
                  onClick={() => setTraceFormOpen(false)}
                >
                  Cancel
                </button>
              )}
              <button
                className="primary"
                onClick={() => void submitTrace()}
                disabled={disabled || loading || !prompt.trim()}
              >
                {loading ? "Tracing every layer…" : "Trace flow"}
              </button>
            </div>
            <p className="subtle">
              A trace captures all decoder layers, so it is capped at 512 positions. Generation is greedy
              (temperature 0) and thinking is off, matching the rest of the app.
            </p>
          </fieldset>
          )}

          {disabled && (
            <div className="warning-banner inline">
              Load the model first. Tracing needs a real forward pass; nothing here is simulated.
            </div>
          )}

          {error && <div className="error-banner">{error}</div>}

          {!flow && (
            <div className="empty flow-empty">
              <p>
                Run a trace to see the residual stream as it actually is: each layer writes into the
                stream, and the ribbons between layers are measured, not decorative.
              </p>
              <ul>
                <li>Ribbon width is the residual write: mean ‖h<sub>l</sub> − h<sub>l−1</sub>‖ / ‖h<sub>l−1</sub>‖.</li>
                <li>Ribbon opacity is the mean cosine between consecutive layer states.</li>
                <li>Colored strands are clusters of hidden dimensions whose activation profiles move together.</li>
              </ul>
            </div>
          )}

          {flow && (
            <>
              <div className="flow-summary">
                <div>
                  <span className="badge steerable-badge">{flow.layers.length} layers</span>{" "}
                  <span className="badge">{flow.tokens.length} positions</span>{" "}
                  <span className="badge">{flow.edges.length} connections</span>{" "}
                  <span className="badge">k = {flow.cluster_k}</span>
                </div>
                <pre className="output small flow-output">{flow.output || "(no text generated)"}</pre>
              </div>

              {flow.warnings.map((warning) => (
                <div key={warning} className="warning-banner inline">
                  {warning}
                </div>
              ))}

              <div className="flow-main">
                <div className="flow-graph-panel">
                  <div className="flow-graph-head">
                    <h3>Residual stream</h3>
                    <span className="subtle">left to right: layer 0 → layer {flow.layers.length - 1}</span>
                  </div>
                  <FlowGraph
                    flow={flow}
                    selectedLayer={selectedLayer}
                    selectedCluster={selectedCluster}
                    tokenIndex={tokenIndex}
                    onSelectLayer={selectLayer}
                    onSelectCluster={selectCluster}
                  />
                  <div className="flow-legend">
                    <span>
                      <i className="flow-legend-swatch attention" /> attention
                    </span>
                    <span>
                      <i className="flow-legend-swatch conv" /> short conv
                    </span>
                    <span>ribbon width = residual write</span>
                    <span>opacity = cosine similarity</span>
                    <span>strands = dimension clusters · width = mass</span>
                    <span>colors = per-layer rank, not matched across layers</span>
                  </div>
                  <div className="flow-playback">
                    <button
                      className="secondary"
                      aria-label={playing ? "Pause token playback" : "Play token playback"}
                      onClick={() => setPlaying((current) => !current)}
                      disabled={flow.tokens.length < 2}
                    >
                      {playing ? "Pause" : "Play"}
                    </button>
                    <input
                      type="range"
                      min={0}
                      max={Math.max(0, flow.tokens.length - 1)}
                      value={tokenIndex}
                      aria-label="Token position"
                      onChange={(event) => {
                        setPlaying(false);
                        setTokenIndex(Number(event.target.value));
                      }}
                    />
                    <span className="subtle">
                      token {Math.min(tokenIndex + 1, flow.tokens.length)}/{flow.tokens.length}
                      {currentToken
                        ? ` · ${currentToken.phase} · “${displayToken(currentToken.text)}”`
                        : ""}
                    </span>
                  </div>
                  <div className="token-strip flow-token-strip">
                    {flow.tokens.map((token) => (
                      <button
                        key={token.index}
                        type="button"
                        className={`token-chip ${token.phase} ${token.index === tokenIndex ? "selected" : ""}`}
                        aria-label={`Token ${token.index}: ${displayToken(token.text)}`}
                        aria-pressed={token.index === tokenIndex}
                        onClick={() => {
                          setPlaying(false);
                          setTokenIndex(token.index);
                        }}
                      >
                        <span className="token-index">{token.index}</span>
                        <span className="token-text">{displayToken(token.text)}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <aside className="flow-inspector" aria-live="polite">
                  {selectedClusterData && selectedLayerData ? (
                    <ClusterInspector
                      flow={flow}
                      layer={selectedLayerData}
                      cluster={selectedClusterData}
                      alpha={alpha}
                      onAlpha={setAlpha}
                      onBack={() => selectLayer(selectedLayerData.layer_index)}
                      onSteer={() => void submitSteer()}
                      steerLoading={steerLoading}
                      steerResult={steerResult}
                      steerError={steerError}
                      onUseDirection={useInSteeringPanel}
                    />
                  ) : selectedLayerData ? (
                    <LayerInspector
                      layer={selectedLayerData}
                      onSelectCluster={(clusterId) =>
                        selectCluster(selectedLayerData.layer_index, clusterId)
                      }
                    />
                  ) : (
                    <div className="flow-inspector-inner">
                      <h3>Pick a layer or a strand</h3>
                      <p className="subtle">
                        Click any layer to read its measured write, similarity, and gain. Click a
                        colored strand to see the dimensions in that cluster and steer the group.
                      </p>
                      <p className="subtle">
                        Clusters are descriptive. A cluster says these dimensions moved together on this
                        trace; it is not evidence that the group controls a behaviour.
                      </p>
                    </div>
                  )}
                </aside>
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}

const ATTENTION_COLOR = "#0071e3";
const CONV_COLOR = "#30b0c7";

function operatorColor(layer: FlowLayer): string {
  return layer.operator_type === "attention" ? ATTENTION_COLOR : CONV_COLOR;
}

function FlowGraph({
  flow,
  selectedLayer,
  selectedCluster,
  tokenIndex,
  onSelectLayer,
  onSelectCluster,
}: {
  flow: FlowResponse;
  selectedLayer: number | null;
  selectedCluster: number | null;
  tokenIndex: number;
  onSelectLayer: (layerIndex: number) => void;
  onSelectCluster: (layerIndex: number, clusterId: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState(0);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const update = () => setAvailable(element.clientWidth);
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const step =
    available > 0
      ? clamp((available - LEFT * 2) / Math.max(1, flow.layers.length - 1), MIN_STEP, MAX_STEP)
      : MAX_STEP;
  const width = LEFT * 2 + Math.max(0, flow.layers.length - 1) * step;
  const maxWrite = Math.max(1e-9, ...flow.edges.map((edge) => edge.write));
  const token = clamp(tokenIndex, 0, Math.max(0, flow.tokens.length - 1));
  const xFor = (index: number) => LEFT + index * step;

  useEffect(() => {
    const container = scrollRef.current;
    if (!container || selectedLayer === null) return;
    const node = container.querySelector<SVGGElement>('[data-selected="true"]');
    if (!node) return;
    const nodeRect = node.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    if (nodeRect.left < containerRect.left || nodeRect.right > containerRect.right) {
      node.scrollIntoView?.({ block: "nearest", inline: "center" });
    }
  }, [selectedLayer, step]);

  const curve = (x1: number, x2: number, offset = 0) =>
    `M ${x1} ${CENTER_Y + offset} C ${x1 + step * 0.38} ${CENTER_Y + offset}, ${
      x2 - step * 0.38
    } ${CENTER_Y + offset}, ${x2} ${CENTER_Y + offset}`;

  const ribbon = (x1: number, x2: number, widthStart: number, widthEnd: number) => {
    const top = `M ${x1} ${CENTER_Y - widthStart / 2} C ${x1 + step * 0.38} ${
      CENTER_Y - widthStart / 2
    }, ${x2 - step * 0.38} ${CENTER_Y - widthEnd / 2}, ${x2} ${CENTER_Y - widthEnd / 2}`;
    const bottom = `L ${x2} ${CENTER_Y + widthEnd / 2} C ${x2 - step * 0.38} ${
      CENTER_Y + widthEnd / 2
    }, ${x1 + step * 0.38} ${CENTER_Y + widthStart / 2}, ${x1} ${
      CENTER_Y + widthStart / 2
    } Z`;
    return `${top} ${bottom}`;
  };

  const fanOffset = (index: number) => {
    const side = index % 2 === 0 ? -1 : 1;
    const rank = Math.floor(index / 2);
    return side * (20 + rank * 16);
  };

  return (
    <div className="flow-graph-scroll" ref={scrollRef}>
      <svg
        className="flow-svg"
        width={width}
        height={GRAPH_HEIGHT}
        role="group"
        aria-label={`Layer flow graph with ${flow.layers.length} layers and ${flow.edges.length} measured connections`}
      >
        <defs>
          {flow.edges.map((edge, edgeIndex) => {
            const source = flow.layers[edge.source_layer];
            const target = flow.layers[edge.target_layer];
            return (
              <linearGradient
                key={`gradient-${edgeIndex}`}
                id={`flow-gradient-${edgeIndex}`}
                x1="0"
                y1="0"
                x2="1"
                y2="0"
              >
                <stop
                  offset="0%"
                  stopColor={source ? operatorColor(source) : ATTENTION_COLOR}
                />
                <stop
                  offset="100%"
                  stopColor={target ? operatorColor(target) : CONV_COLOR}
                />
              </linearGradient>
            );
          })}
        </defs>

        <rect
          x={LEFT - 8}
          y={CENTER_Y - 32}
          width={Math.max(0, width - LEFT * 2 + 16)}
          height={64}
          rx={32}
          fill="rgba(0, 0, 0, 0.018)"
        />

        {flow.edges.map((edge, edgeIndex) => {
          const target = flow.layers[edge.target_layer];
          const source = flow.layers[edge.source_layer];
          if (!target || !source) return null;
          const x1 = xFor(source.layer_index) + 4;
          const x2 = xFor(target.layer_index) - 4;
          const widthEnd = 1 + 12 * (edge.write / maxWrite);
          const widthStart =
            edgeIndex === 0 ? widthEnd : 1 + 12 * (flow.edges[edgeIndex - 1].write / maxWrite);
          const deltaMax = Math.max(1e-9, ...target.delta);
          const pulse = 0.2 + 0.7 * clamp(target.delta[token] / deltaMax, 0, 1);
          const targetColor = operatorColor(target);
          return (
            <g key={`${edge.source_layer}-${edge.target_layer}`}>
              <path
                d={ribbon(x1, x2, widthStart, widthEnd)}
                fill={`url(#flow-gradient-${edgeIndex})`}
                opacity={0.2 + 0.32 * clamp(edge.similarity, 0, 1)}
              />
              <path
                d={curve(x1, x2)}
                fill="none"
                stroke={targetColor}
                strokeWidth={1}
                opacity={0.35}
              />
              <path
                className="flow-dash"
                d={curve(x1, x2)}
                fill="none"
                stroke={targetColor}
                strokeWidth={1.7}
                strokeLinecap="round"
                strokeDasharray="2 12"
                opacity={pulse}
                style={{ animationDelay: `${-(edgeIndex % 7) * 0.18}s` }}
              />
            </g>
          );
        })}

        {flow.layers.map((layer) => {
          const x = xFor(layer.layer_index);
          const energyMax = Math.max(1e-9, ...layer.energy);
          const energyRatio = layer.energy[token] / energyMax;
          const color = operatorColor(layer);
          const selected = selectedLayer === layer.layer_index;
          return (
            <g
              key={layer.layer_index}
              className={`flow-node ${selected ? "selected" : ""}`}
              data-selected={selected ? "true" : undefined}
              role="button"
              tabIndex={0}
              aria-label={`Layer ${layer.layer_index}, ${layer.operator_type}, mean norm ${metric(
                layer.mean_norm,
              )}`}
              onClick={() => onSelectLayer(layer.layer_index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelectLayer(layer.layer_index);
                }
              }}
            >
              <title>
                {`Layer ${layer.layer_index} · ${layer.operator_type} · mean norm ${metric(
                  layer.mean_norm,
                )} · write ${layer.layer_index === 0 ? "input" : metric(layer.write)} · cosine ${metric(
                  layer.similarity,
                )}`}
              </title>
              <circle
                cx={x}
                cy={CENTER_Y}
                r={8 + 16 * clamp(energyRatio, 0, 1)}
                fill={color}
                opacity={0.06 + 0.12 * clamp(energyRatio, 0, 1)}
              />
              <line
                x1={x}
                y1={CENTER_Y - 40}
                x2={x}
                y2={CENTER_Y + 40}
                stroke={color}
                strokeWidth={1.2}
                strokeLinecap="round"
                opacity={selected ? 0.9 : 0.42}
              />
              <circle
                cx={x}
                cy={CENTER_Y}
                r={selected ? 5 : 3.6}
                fill={color}
                stroke="var(--surface)"
                strokeWidth={1.5}
              />
              {selected && (
                <circle
                  cx={x}
                  cy={CENTER_Y}
                  r={9}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.2}
                  opacity={0.7}
                />
              )}
              <rect
                x={x - 11}
                y={CENTER_Y - 54}
                width={22}
                height={108}
                fill="transparent"
              />
              <text x={x} y={CENTER_Y - 74} textAnchor="middle" className="flow-node-type">
                {layer.operator_type === "attention" ? "A" : "C"}
              </text>
              <text x={x} y={CENTER_Y + 82} textAnchor="middle" className="flow-node-index">
                {layer.layer_index}
              </text>
            </g>
          );
        })}

        {flow.layers.map((layer) => {
          const x = xFor(layer.layer_index);
          return (
            <g
              key={`clusters-${layer.layer_index}`}
              className="flow-clusters"
              style={{ opacity: selectedLayer === layer.layer_index ? 1 : undefined }}
            >
              {layer.clusters.map((cluster, index) => {
                const offset = fanOffset(index);
                const endX = x + 12;
                const endY = CENTER_Y + offset;
                const selected =
                  selectedLayer === layer.layer_index && selectedCluster === cluster.id;
                const strandWidth = 1 + 3.2 * Math.max(0, cluster.mass);
                const radius = 2.6 + 3.4 * Math.sqrt(Math.max(0, cluster.mass));
                return (
                  <g
                    key={cluster.id}
                    className={`flow-cluster ${selected ? "selected" : ""}`}
                    role="button"
                    tabIndex={selectedLayer === layer.layer_index ? 0 : -1}
                    aria-label={`Layer ${layer.layer_index} cluster ${cluster.id}: ${cluster.size} dimensions, ${percent(
                      cluster.mass,
                    )} mass`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectCluster(layer.layer_index, cluster.id);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onSelectCluster(layer.layer_index, cluster.id);
                      }
                    }}
                  >
                    <title>
                      {`Layer ${layer.layer_index} · cluster ${cluster.id} · ${cluster.size} dims · ${percent(
                        cluster.mass,
                      )} of activation · cohesion ${cluster.cohesion.toFixed(2)}`}
                    </title>
                    <path
                      d={`M ${x} ${CENTER_Y} Q ${x + 7} ${CENTER_Y + offset * 0.55} ${endX} ${endY}`}
                      fill="none"
                      stroke={clusterColor(cluster.id)}
                      strokeWidth={strandWidth}
                      strokeLinecap="round"
                      opacity={selected ? 0.95 : 0.55 + 0.35 * clamp(cluster.cohesion, 0, 1)}
                    />
                    <circle
                      cx={endX}
                      cy={endY}
                      r={selected ? radius * 1.2 : radius}
                      fill={clusterColor(cluster.id)}
                      stroke={selected ? "var(--ink)" : "var(--surface)"}
                      strokeWidth={selected ? 2 : 1}
                    />
                    <circle cx={endX} cy={endY} r={10} fill="transparent" />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function LayerInspector({
  layer,
  onSelectCluster,
}: {
  layer: FlowLayer;
  onSelectCluster: (clusterId: number) => void;
}) {
  return (
    <div className="flow-inspector-inner">
      <h3>Layer {layer.layer_index}</h3>
      <code className="flow-module">{layer.module_id}</code>
      <dl className="flow-metrics">
        <div>
          <dt>operator</dt>
          <dd>
            {layer.operator_type} · {layer.layer_type}
          </dd>
        </div>
        <div>
          <dt>mean norm</dt>
          <dd>{metric(layer.mean_norm)}</dd>
        </div>
        <div>
          <dt>residual write</dt>
          <dd>{layer.layer_index === 0 ? "input" : metric(layer.write)}</dd>
        </div>
        <div>
          <dt>cosine to prev</dt>
          <dd>{metric(layer.similarity)}</dd>
        </div>
        <div>
          <dt>gain</dt>
          <dd>{metric(layer.gain)}</dd>
        </div>
        <div>
          <dt>clusters</dt>
          <dd>{layer.clusters.length}</dd>
        </div>
      </dl>
      <h4>Dimension clusters</h4>
      <div className="flow-cluster-list">
        {layer.clusters.map((cluster) => (
          <button
            key={cluster.id}
            type="button"
            className="flow-cluster-row"
            onClick={() => onSelectCluster(cluster.id)}
          >
            <span className="flow-swatch" style={{ background: clusterColor(cluster.id) }} />
            <span className="flow-cluster-name">cluster {cluster.id}</span>
            <span className="flow-cluster-meta">
              {cluster.size} dims · {percent(cluster.mass)}
            </span>
          </button>
        ))}
      </div>
      <p className="subtle">
        Clusters group hidden dimensions whose z-scored activation profiles move together across the
        captured positions. Click a row or a strand on the graph to inspect and steer the group.
      </p>
    </div>
  );
}

function ClusterInspector({
  flow,
  layer,
  cluster,
  alpha,
  onAlpha,
  onBack,
  onSteer,
  steerLoading,
  steerResult,
  steerError,
  onUseDirection,
}: {
  flow: FlowResponse;
  layer: FlowLayer;
  cluster: FlowCluster;
  alpha: number;
  onAlpha: (value: number) => void;
  onBack: () => void;
  onSteer: () => void;
  steerLoading: boolean;
  steerResult: SteeringResponse | null;
  steerError: string | null;
  onUseDirection: () => void;
}) {
  const peak = tokenAt(flow.tokens, cluster.peak_position);
  const changed = Boolean(
    steerResult && steerResult.steered.output !== steerResult.baseline.output,
  );
  return (
    <div className="flow-inspector-inner">
      <button type="button" className="secondary small" onClick={onBack}>
        ← Layer {layer.layer_index}
      </button>
      <h3>
        Layer {layer.layer_index} · cluster {cluster.id}
      </h3>
      <div className="flow-cluster-badges">
        <span className="badge">{cluster.size} dims</span>
        <span className="badge">{percent(cluster.mass)} mass</span>
        <span className="badge">cohesion {cluster.cohesion.toFixed(2)}</span>
      </div>
      <p className="subtle">
        Peak at token {cluster.peak_position}
        {peak ? ` (“${displayToken(peak.text)}”)` : ""}.
      </p>
      <Sparkline profile={cluster.profile} positions={cluster.profile_positions} />
      <h4>Top dimensions</h4>
      <div className="flow-dimension-chips">
        {cluster.top_dimensions.map((dimension) => (
          <span
            key={dimension.index}
            className="flow-dimension-chip"
            title={`mean |a| ${dimension.mean_abs.toFixed(4)} · trace σ ${dimension.sigma.toFixed(4)}`}
          >
            #{dimension.index}{" "}
            <b>
              {dimension.weight >= 0 ? "+" : ""}
              {dimension.weight.toFixed(2)}
            </b>
          </span>
        ))}
      </div>
      <p className="subtle">
        Weights are each dimension&apos;s mean activation over the trace, normalized so the largest is
        ±1. σ mode scales each dimension by its prompt standard deviation.
      </p>

      <fieldset className="control-card flow-steer-card">
        <legend>Steer this cluster</legend>
        <label>
          <span>Strength α (σ units)</span>
          <input
            type="number"
            step={0.1}
            min={-SIGMA_LIMIT}
            max={SIGMA_LIMIT}
            value={alpha}
            onChange={(event) => onAlpha(Number(event.target.value))}
          />
        </label>
        <input
          type="range"
          min={-6}
          max={6}
          step={0.1}
          value={clamp(alpha, -6, 6)}
          aria-label="Cluster steering strength"
          onChange={(event) => onAlpha(Number(event.target.value))}
        />
        <p className="subtle">
          A fresh baseline + steered pair with the traced prompt. α = 0 reproduces the baseline exactly.
        </p>
        {Math.abs(alpha) > SIGMA_LIMIT && (
          <p className="field-note">α will be clamped to ±{SIGMA_LIMIT} σ.</p>
        )}
        <button className="primary" onClick={onSteer} disabled={steerLoading}>
          {steerLoading ? "Running baseline + steered…" : "Run baseline + steered"}
        </button>
        {steerError && <div className="error-banner inline">{steerError}</div>}
        {steerResult && (
          <div className="flow-steer-result">
            <span className={`badge ${changed ? "steerable-badge" : "muted-badge"}`}>
              {changed ? "output changed" : "output unchanged"}
            </span>
            <div className="flow-compare">
              <div>
                <h4>Baseline</h4>
                <pre className="output small">{steerResult.baseline.output || "(no text)"}</pre>
              </div>
              <div>
                <h4>Steered</h4>
                <pre className="output small">{steerResult.steered.output || "(no text)"}</pre>
              </div>
            </div>
            {!changed && (
              <p className="subtle">
                No change is a real result here. Try a larger |α| or a different cluster; a null
                effect is evidence too.
              </p>
            )}
          </div>
        )}
        <button type="button" className="secondary" onClick={onUseDirection}>
          Use these dimensions in the steering panel
        </button>
      </fieldset>
    </div>
  );
}

function Sparkline({ profile, positions }: { profile: number[]; positions: number[] }) {
  if (profile.length < 2) return null;
  const width = 280;
  const height = 64;
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
      className="flow-sparkline"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Cluster activation profile across the trace"
    >
      <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} stroke="var(--rule)" />
      <polyline points={points} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
      <title>{`Mean activation over ${positions.length} sampled positions`}</title>
    </svg>
  );
}
