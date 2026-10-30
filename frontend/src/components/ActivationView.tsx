import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent } from "react";
import type { ActivationResponse, HeatmapData, TokenInfo } from "../types";

interface Props {
  activation: ActivationResponse | null;
  loading: boolean;
  error: string | null;
  onQuery: (query: { feature?: number; position?: number }) => void;
}

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) >= 1000) return value.toExponential(2);
  return value.toFixed(4);
}

function tokenLabel(token: TokenInfo): string {
  if (token.text === "") return "∅";
  return token.text.replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

function colorFor(value: number, maxAbs: number): string {
  const normalized = maxAbs === 0 ? 0 : Math.max(-1, Math.min(1, value / maxAbs));
  const intensity = Math.sqrt(Math.abs(normalized));
  const base = [11, 17, 28];
  const target = normalized >= 0 ? [251, 113, 133] : [56, 189, 248];
  const rgb = base.map((channel, index) => Math.round(channel + (target[index] - channel) * intensity));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function Heatmap({
  data,
  onPick,
}: {
  data: HeatmapData;
  onPick: (position: number, feature: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{
    position: number;
    feature: number;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  const maxAbs = useMemo(() => {
    let maximum = 0;
    for (const row of data.values) {
      for (const value of row) maximum = Math.max(maximum, Math.abs(value));
    }
    return maximum;
  }, [data.values]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const rows = data.values.length;
    const columns = data.values[0]?.length ?? 0;
    if (rows === 0 || columns === 0) return;
    const cellWidth = canvas.width / columns;
    const cellHeight = canvas.height / rows;
    context.fillStyle = "#0b111c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        context.fillStyle = colorFor(data.values[row][column], maxAbs);
        context.fillRect(
          Math.floor(column * cellWidth),
          Math.floor(row * cellHeight),
          Math.ceil(cellWidth),
          Math.ceil(cellHeight),
        );
      }
    }
  }, [data, maxAbs]);

  const handleMove = (event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const rows = data.values.length;
    const columns = data.values[0]?.length ?? 0;
    const column = Math.min(columns - 1, Math.floor((x / rect.width) * columns));
    const row = Math.min(rows - 1, Math.floor((y / rect.height) * rows));
    if (column < 0 || row < 0) return;
    setHover({
      position: data.positions[row],
      feature: data.features[column],
      value: data.values[row][column],
      x,
      y,
    });
  };

  const handleClick = () => {
    if (hover) onPick(hover.position, hover.feature);
  };

  return (
    <div className="heatmap-wrap">
      <canvas
        ref={canvasRef}
        className="heatmap-canvas"
        width={900}
        height={340}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
        onClick={handleClick}
      />
      {hover && (
        <div className="heatmap-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          position {hover.position} · feature {hover.feature} · {formatValue(hover.value)}
        </div>
      )}
    </div>
  );
}

function FeatureChart({
  activation,
  onPickPosition,
}: {
  activation: ActivationResponse;
  onPickPosition: (position: number) => void;
}) {
  const values = activation.feature_series.values;
  const width = 720;
  const height = 150;
  const padding = 10;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const span = maximum - minimum || 1;
  const points = values
    .map((value, index) => {
      const x = padding + (index / Math.max(1, values.length - 1)) * (width - padding * 2);
      const y = height - padding - ((value - minimum) / span) * (height - padding * 2);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  const zeroY = height - padding - ((0 - minimum) / span) * (height - padding * 2);
  const selectedX =
    padding + (activation.position / Math.max(1, values.length - 1)) * (width - padding * 2);

  return (
    <svg
      className="feature-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Feature ${activation.feature_series.feature} across token positions`}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = (event.clientX - rect.left) / rect.width;
        const index = Math.round(ratio * Math.max(1, values.length - 1));
        onPickPosition(Math.max(0, Math.min(values.length - 1, index)));
      }}
    >
      <line x1={0} y1={zeroY} x2={width} y2={zeroY} className="chart-zero" />
      <line x1={selectedX} y1={0} x2={selectedX} y2={height} className="chart-cursor" />
      <polyline points={points} className="chart-line" />
      <text x={padding} y={14} className="chart-label">
        max {formatValue(maximum)}
      </text>
      <text x={padding} y={height - 2} className="chart-label">
        min {formatValue(minimum)}
      </text>
    </svg>
  );
}

function Distribution({ activation }: { activation: ActivationResponse }) {
  const { counts, bin_edges } = activation.histogram;
  const width = 720;
  const height = 120;
  const maximum = Math.max(...counts, 1);
  const barWidth = width / Math.max(1, counts.length);
  return (
    <svg className="distribution" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Activation distribution">
      {counts.map((count, index) => {
        const barHeight = (count / maximum) * (height - 6);
        return (
          <rect
            key={index}
            x={index * barWidth + 1}
            y={height - barHeight}
            width={Math.max(1, barWidth - 2)}
            height={barHeight}
            className="distribution-bar"
          >
            <title>
              [{formatValue(bin_edges[index])}, {formatValue(bin_edges[index + 1])}]: {count}
            </title>
          </rect>
        );
      })}
    </svg>
  );
}

export function ActivationView({ activation, loading, error, onQuery }: Props) {
  const [featureInput, setFeatureInput] = useState("");

  useEffect(() => {
    if (activation) setFeatureInput(String(activation.feature_series.feature));
  }, [activation]);

  if (error) return <div className="error-banner">{error}</div>;
  if (!activation) {
    return (
      <div className="placeholder-card">
        <strong>No activations captured yet</strong>
        <p>
          Select a steerable module, tick “Capture activations” in the generation panel, and run a
          baseline. Only the selected module is captured; nothing is written to disk.
        </p>
      </div>
    );
  }

  const submitFeature = () => {
    const parsed = Number(featureInput);
    if (!Number.isInteger(parsed) || parsed < 0 || parsed >= activation.shape[1]) return;
    onQuery({ feature: parsed, position: activation.position });
  };

  return (
    <div className="activation-view">
      <div className="activation-head">
        <div>
          <h4>Activation tensor</h4>
          <p className="subtle">
            {activation.module_id} · shape [{activation.shape.join(", ")}] · {activation.dtype}
          </p>
        </div>
        {loading && <span className="subtle">updating…</span>}
      </div>
      <p className="subtle">{activation.aggregation}</p>
      {activation.warnings.map((warning) => (
        <div key={warning} className="warning-banner inline">
          {warning}
        </div>
      ))}

      <h4>Token positions</h4>
      <div className="token-strip">
        {activation.tokens.map((token) => (
          <button
            key={token.index}
            className={`token-chip ${token.phase} ${token.index === activation.position ? "selected" : ""}`}
            title={`#${token.index} · token ${token.token_id} · ${token.phase}`}
            onClick={() => onQuery({ position: token.index, feature: activation.feature_series.feature })}
          >
            <span className="token-index">{token.index}</span>
            <span className="token-text">{tokenLabel(token)}</span>
          </button>
        ))}
      </div>

      <h4>Heatmap</h4>
      <Heatmap
        data={activation.heatmap}
        onPick={(position, feature) => onQuery({ position, feature })}
      />
      <div className="heatmap-footer">
        <span className="subtle">{activation.heatmap.sampling}</span>
        <span className="legend-scale">
          <span>−{formatValue(Math.abs(activation.stats.minimum))}</span>
          <span className="gradient-bar" />
          <span>+{formatValue(activation.stats.maximum)}</span>
        </span>
      </div>
      <p className="subtle">Click a cell to select its position and feature.</p>

      <h4>
        Feature {activation.feature_series.feature} across positions (selected position{" "}
        {activation.position})
      </h4>
      <FeatureChart
        activation={activation}
        onPickPosition={(position) =>
          onQuery({ position, feature: activation.feature_series.feature })
        }
      />
      <div className="feature-controls">
        <label>
          <span>Feature index</span>
          <input
            type="number"
            min={0}
            max={activation.shape[1] - 1}
            value={featureInput}
            onChange={(event) => setFeatureInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitFeature();
            }}
          />
        </label>
        <button className="secondary" onClick={submitFeature}>
          View feature
        </button>
        <span className="subtle">
          value at position {activation.position}:{" "}
          {formatValue(
            activation.feature_series.values[activation.position] ?? Number.NaN,
          )}
        </span>
      </div>

      <div className="activation-columns">
        <div>
          <h4>Top dimensions at position {activation.position}</h4>
          <div className="top-dims">
            {activation.top_dimensions.map((dimension) => (
              <button
                key={dimension.index}
                className={`top-dim ${
                  dimension.index === activation.feature_series.feature ? "selected" : ""
                }`}
                onClick={() =>
                  onQuery({ feature: dimension.index, position: activation.position })
                }
              >
                <span>#{dimension.index}</span>
                <span>{formatValue(dimension.value)}</span>
              </button>
            ))}
          </div>
        </div>
        <div>
          <h4>Distribution</h4>
          <Distribution activation={activation} />
          <dl className="stats-list">
            <div>
              <dt>mean</dt>
              <dd>{formatValue(activation.stats.mean)}</dd>
            </div>
            <div>
              <dt>std</dt>
              <dd>{formatValue(activation.stats.std)}</dd>
            </div>
            <div>
              <dt>abs mean</dt>
              <dd>{formatValue(activation.stats.abs_mean)}</dd>
            </div>
            <div>
              <dt>min / max</dt>
              <dd>
                {formatValue(activation.stats.minimum)} / {formatValue(activation.stats.maximum)}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}
