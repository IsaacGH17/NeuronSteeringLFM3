import { useState } from "react";
import type { BaselineResponse, GenerationSettings } from "../types";

interface Props {
  disabled: boolean;
  running: boolean;
  result: BaselineResponse | null;
  error: string | null;
  onRun: (prompt: string, system: string | null, settings: GenerationSettings) => void;
}

const DEFAULT_PROMPT = "Explain what a transformer activation is in two sentences.";

export function GenerationPanel({ disabled, running, result, error, onRun }: Props) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [system, setSystem] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [seed, setSeed] = useState("");
  const [enableThinking, setEnableThinking] = useState(false);

  const submit = () => {
    if (!prompt.trim() || disabled || running) return;
    const parsedSeed = seed.trim() === "" ? null : Number(seed);
    onRun(prompt, system.trim() === "" ? null : system, {
      max_new_tokens: Math.max(1, Math.min(4096, Math.round(maxNewTokens))),
      temperature,
      top_p: topP,
      seed: parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
      enable_thinking: enableThinking,
    });
  };

  return (
    <section className="panel generation">
      <header className="panel-header">
        <div>
          <h2>Baseline generation</h2>
          <p className="subtle">Same prompt and settings will be reused for steered runs.</p>
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

        <button className="primary" onClick={submit} disabled={disabled || running || !prompt.trim()}>
          {running ? "Generating…" : "Run baseline"}
        </button>
        {disabled && <p className="subtle">Load the model to enable generation.</p>}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
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
