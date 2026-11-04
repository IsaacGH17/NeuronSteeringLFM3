import { useCallback, useEffect, useState } from "react";
import { downloadExport, getExperiment, listExperiments } from "../api/client";
import type { ExperimentDetail, ExperimentSummary } from "../types";
import { tooltipTitle, useTooltips } from "../ui/tooltips";

interface Props {
  refreshTrigger: number;
  onOpenActivations: (experimentId: string, moduleId: string) => void;
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function preview(text: string, max = 90): string {
  const trimmed = text.replace(/\s+/g, " ").trim();
  return trimmed.length > max ? `${trimmed.slice(0, max)}…` : trimmed;
}

function conditionLabel(summary: ExperimentSummary): string {
  const parts: string[] = [summary.condition];
  if (summary.condition_detail && summary.condition_detail !== "none") {
    parts.push(`(${summary.condition_detail})`);
  }
  return parts.join(" ");
}

function steeringLabel(summary: ExperimentSummary): string {
  if (!summary.module_id) return "—";
  const feature = summary.feature_index ?? "—";
  const strength = summary.strength ?? "—";
  return `${summary.module_id} · dim ${feature} · α ${strength}`;
}

export function ExperimentHistory({ refreshTrigger, onOpenActivations }: Props) {
  const { enabled: tooltipsEnabled } = useTooltips();
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<ExperimentSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<Set<string>>(() => new Set());
  const [compareDetails, setCompareDetails] = useState<Record<string, ExperimentDetail>>({});
  const [limit, setLimit] = useState(20);
  const [offset, setOffset] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listExperiments({ limit, offset });
      setItems(response.items);
      setTotal(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setItems([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [limit, offset]);

  useEffect(() => {
    void load();
  }, [load, refreshTrigger]);

  useEffect(() => {
    if (!selectedId || !open) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    setDetailError(null);
    getExperiment(selectedId)
      .then((result) => {
        if (!cancelled) setDetail(result);
      })
      .catch((err) => {
        if (!cancelled) setDetailError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, open]);

  useEffect(() => {
    const ids = [...compareIds];
    if (ids.length === 0) {
      setCompareDetails({});
      return;
    }
    let cancelled = false;
    Promise.all(ids.map((id) => getExperiment(id)))
      .then((details) => {
        if (cancelled) return;
        setCompareDetails(Object.fromEntries(details.map((detail) => [detail.id, detail])));
      })
      .catch(() => {
        if (!cancelled) setCompareDetails({});
      });
    return () => {
      cancelled = true;
    };
  }, [compareIds]);

  const toggleCompare = (id: string) => {
    setCompareIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 2) {
        next.add(id);
      }
      return next;
    });
  };

  const selectedForCompare = items.filter((item) => compareIds.has(item.id));
  const canExportSelected = selectedForCompare.length > 0;

  return (
    <section className={`history ${open ? "open" : ""}`}>
      <button className="history-toggle" onClick={() => setOpen((value) => !value)}>
        <span>{open ? "▾" : "▸"} Experiment history</span>
        <span className="subtle">{total} saved run{total === 1 ? "" : "s"}</span>
      </button>

      {open && (
        <div className="history-body">
          <div className="history-toolbar">
            <div className="history-pagination">
              <label>
                <span>Show</span>
                <select
                  value={limit}
                  onChange={(event) => {
                    setLimit(Number(event.target.value));
                    setOffset(0);
                  }}
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                </select>
              </label>
              <button className="secondary" onClick={load} disabled={loading}>
                Refresh
              </button>
            </div>
            <div className="history-export">
              <button
                className="secondary"
                onClick={() => downloadExport({ format: "jsonl", ids: [...compareIds] })}
                disabled={!canExportSelected}
                title={tooltipTitle(
                  tooltipsEnabled,
                  canExportSelected ? "Export selected as JSONL" : "Select one or two rows to export",
                )}
              >
                Export selected JSONL
              </button>
              <button
                className="secondary"
                onClick={() => downloadExport({ format: "csv", ids: [...compareIds] })}
                disabled={!canExportSelected}
                title={tooltipTitle(
                  tooltipsEnabled,
                  canExportSelected ? "Export selected as CSV" : "Select one or two rows to export",
                )}
              >
                Export selected CSV
              </button>
              <button
                className="secondary"
                onClick={() => downloadExport({ format: "jsonl" })}
              >
                Export all JSONL
              </button>
              <button
                className="secondary"
                onClick={() => downloadExport({ format: "csv" })}
              >
                Export all CSV
              </button>
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}
          {loading && items.length === 0 && <div className="empty">Loading history…</div>}
          {!loading && items.length === 0 && !error && (
            <div className="empty">No saved experiments yet. Run a baseline, steered, or sweep experiment to build history.</div>
          )}

          {items.length > 0 && (
            <div className="history-layout">
              <div className="history-table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Compare</th>
                      <th>Time</th>
                      <th>Condition</th>
                      <th>Target</th>
                      <th>Prompt</th>
                      <th>Output</th>
                      <th>Activations</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item) => {
                      const isSelected = selectedId === item.id;
                      const isCompared = compareIds.has(item.id);
                      return (
                        <tr
                          key={item.id}
                          className={`${isSelected ? "selected" : ""} ${isCompared ? "compared" : ""}`}
                          onClick={() => setSelectedId(item.id)}
                        >
                          <td onClick={(event) => event.stopPropagation()}>
                            <input
                              type="checkbox"
                              checked={isCompared}
                              onChange={() => toggleCompare(item.id)}
                              disabled={!isCompared && compareIds.size >= 2}
                              aria-label={`Select ${item.id} for comparison`}
                            />
                          </td>
                          <td>{formatTime(item.timestamp)}</td>
                          <td>
                            <span className={`badge condition-${item.condition}`}>
                              {conditionLabel(item)}
                            </span>
                          </td>
                          <td>{steeringLabel(item)}</td>
                          <td>{preview(item.prompt_preview)}</td>
                          <td>{preview(item.output_preview)}</td>
                          <td>
                            {item.has_activations && item.activation_source ? (
                              <span className="badge activation-live">{item.activation_source}</span>
                            ) : item.has_activations ? (
                              <span
                                className="badge activation-none"
                                title={tooltipTitle(
                                  tooltipsEnabled,
                                  "Captured at run time, but no live or saved tensor remains.",
                                )}
                              >
                                unavailable
                              </span>
                            ) : (
                              <span className="badge activation-none">none</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>

                <div className="history-page-controls">
                  <button
                    className="secondary"
                    disabled={offset === 0}
                    onClick={() => setOffset((value) => Math.max(0, value - limit))}
                  >
                    Previous
                  </button>
                  <span className="subtle">
                    {offset + 1}–{Math.min(offset + limit, total)} of {total}
                  </span>
                  <button
                    className="secondary"
                    disabled={offset + limit >= total}
                    onClick={() => setOffset((value) => value + limit)}
                  >
                    Next
                  </button>
                </div>
              </div>

              <div className="history-detail">
                {detailLoading && <div className="empty">Loading detail…</div>}
                {detailError && <div className="error-banner">{detailError}</div>}
                {!detailLoading && !detail && !detailError && (
                  <div className="empty">Select a row to view experiment detail.</div>
                )}
                {detail && (
                  <>
                    <div className="history-detail-head">
                      <h4>{conditionLabel(detail)}</h4>
                      <span className="subtle">{formatTime(detail.timestamp)}</span>
                    </div>

                    <dl className="meta-grid">
                      <div>
                        <dt>Run ID</dt>
                        <dd>{detail.id}</dd>
                      </div>
                      <div>
                        <dt>Model</dt>
                        <dd>{detail.model_id}</dd>
                      </div>
                      <div>
                        <dt>Revision</dt>
                        <dd>{detail.revision}</dd>
                      </div>
                      <div>
                        <dt>Resolved revision</dt>
                        <dd>{detail.resolved_revision}</dd>
                      </div>
                      <div>
                        <dt>Device</dt>
                        <dd>{detail.device}</dd>
                      </div>
                      <div>
                        <dt>Duration</dt>
                        <dd>{formatDuration(detail.duration_ms)}</dd>
                      </div>
                      <div>
                        <dt>Tokens</dt>
                        <dd>
                          {detail.generated_tokens} generated / {detail.prompt_tokens} prompt
                        </dd>
                      </div>
                      <div>
                        <dt>Speed</dt>
                        <dd>{detail.tokens_per_second?.toFixed(1) ?? "—"} tok/s</dd>
                      </div>
                    </dl>

                    {detail.steering && (
                      <div className="history-steering">
                        <strong>Steering</strong>
                        <span>{detail.steering.module_id}</span>
                        <span>dim {detail.steering.feature_index}</span>
                        <span>α {detail.steering.strength}</span>
                      </div>
                    )}

                    <div className="history-section">
                      <h5>Settings</h5>
                      <p className="subtle mono">
                        max_new_tokens={detail.settings.max_new_tokens} · temperature={detail.settings.temperature} · top_p={detail.settings.top_p} · seed={detail.settings.seed ?? "random"} · thinking={String(detail.settings.enable_thinking)}
                      </p>
                    </div>

                    <div className="history-section">
                      <h5>Prompt</h5>
                      <pre className="output small">{detail.prompt}</pre>
                    </div>

                    <div className="history-section">
                      <h5>Output</h5>
                      <pre className="output small">{detail.output || "(empty output)"}</pre>
                    </div>

                    {detail.captures.length > 0 && (
                      <div className="history-section">
                        <h5>Captured tensors</h5>
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Module</th>
                              <th>Shape</th>
                              <th>dtype</th>
                            </tr>
                          </thead>
                          <tbody>
                            {detail.captures.map((capture) => (
                              <tr key={capture.module_id}>
                                <td>{capture.module_id}</td>
                                <td>[{capture.shape.join(", ")}]</td>
                                <td>{capture.dtype}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {detail.warnings.length > 0 && (
                      <div className="history-section">
                        {detail.warnings.map((warning) => (
                          <div key={warning} className="warning-banner inline">
                            {warning}
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="history-section">
                      <h5>Environment</h5>
                      <dl className="meta-grid">
                        <div>
                          <dt>Python</dt>
                          <dd>{detail.environment.python_version}</dd>
                        </div>
                        <div>
                          <dt>PyTorch</dt>
                          <dd>{detail.environment.torch_version}</dd>
                        </div>
                        <div>
                          <dt>Transformers</dt>
                          <dd>{detail.environment.transformers_version}</dd>
                        </div>
                        <div>
                          <dt>Git commit</dt>
                          <dd>{detail.environment.git_commit}</dd>
                        </div>
                      </dl>
                    </div>

                    <div className="history-actions">
                      <button
                        className="secondary"
                        onClick={() => {
                          const moduleId = detail.captured_modules[0] ?? detail.module_id ?? null;
                          if (moduleId) {
                            onOpenActivations(detail.id, moduleId);
                          }
                        }}
                        disabled={!detail.activation_source || !detail.captured_modules.length}
                        title={tooltipTitle(
                          tooltipsEnabled,
                          !detail.captured_modules.length
                            ? "No captured modules recorded."
                            : !detail.activation_source
                              ? "Activations were captured but are no longer available; enable snapshot persistence on future runs."
                              : "Load the first captured module into the inspector",
                        )}
                      >
                        Open activations
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {selectedForCompare.length > 0 && (
            <div className="history-compare">
              <h4>Compare outputs</h4>
              <div className={`history-compare-columns count-${selectedForCompare.length}`}>
                {selectedForCompare.map((item) => {
                  const full = compareDetails[item.id];
                  return (
                    <div key={item.id} className="result">
                      <div className="result-meta">
                        <span>{conditionLabel(item)}</span>
                        <span>{item.tokens_per_second?.toFixed(1) ?? "—"} tok/s</span>
                      </div>
                      <pre className="output small">{full?.output ?? item.output_preview}</pre>
                      <p className="subtle">
                        {item.id.slice(0, 8)}
                        {!full ? " · loading full output…" : ""}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
