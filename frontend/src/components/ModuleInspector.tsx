import type { ActivationResponse, ModuleDetail } from "../types";
import { ActivationView } from "./ActivationView";

interface Props {
  detail: ModuleDetail | null;
  selectedId: string | null;
  loading: boolean;
  error: string | null;
  onSelect: (moduleId: string) => void;
  activation: ActivationResponse | null;
  activationLoading: boolean;
  activationError: string | null;
  onActivationQuery: (query: { feature?: number; position?: number }) => void;
}

function formatNumber(value: number): string {
  return value.toLocaleString();
}

export function ModuleInspector({
  detail,
  selectedId,
  loading,
  error,
  onSelect,
  activation,
  activationLoading,
  activationError,
  onActivationQuery,
}: Props) {
  return (
    <section className="panel inspector">
      <header className="panel-header">
        <div>
          <h2>Module inspector</h2>
          <p className="subtle">Metadata comes from the loaded model; activations arrive in Phase 3.</p>
        </div>
      </header>

      {!selectedId && <div className="empty">Select a module in the architecture explorer.</div>}
      {selectedId && loading && <div className="empty">Loading {selectedId}…</div>}
      {error && <div className="error-banner">{error}</div>}

      {detail && !loading && (
        <div className="inspector-body">
          <div className="module-title">
            <h3>{detail.id}</h3>
            <span className={`badge ${detail.supports_steering ? "steerable-badge" : "muted-badge"}`}>
              {detail.supports_steering ? "steerable" : "not steerable"}
            </span>
          </div>

          <dl className="meta-grid">
            <div>
              <dt>Class</dt>
              <dd>{detail.class_name}</dd>
            </div>
            <div>
              <dt>Layer type</dt>
              <dd>{detail.layer_type ?? "—"}</dd>
            </div>
            <div>
              <dt>Parent</dt>
              <dd>{detail.parent_id ?? "—"}</dd>
            </div>
            <div>
              <dt>Parameters</dt>
              <dd>{formatNumber(detail.parameter_count)}</dd>
            </div>
            <div>
              <dt>Direct parameters</dt>
              <dd>{formatNumber(detail.direct_parameter_count)}</dd>
            </div>
            <div>
              <dt>Children</dt>
              <dd>{detail.child_ids.length}</dd>
            </div>
          </dl>

          <div className={`callout ${detail.supports_steering ? "ok" : "muted"}`}>{detail.steering_reason}</div>

          <h4>Activations</h4>
          <ActivationView
            activation={activation}
            loading={activationLoading}
            error={activationError}
            onQuery={onActivationQuery}
          />

          <h4>Direct parameters</h4>
          {detail.parameters.length === 0 ? (
            <p className="subtle">No parameters registered directly on this module.</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Shape</th>
                  <th>dtype</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {detail.parameters.map((param) => (
                  <tr key={param.name}>
                    <td>{param.name}</td>
                    <td>[{param.shape.join(", ")}]</td>
                    <td>{param.dtype}</td>
                    <td>{formatNumber(param.num_parameters)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {detail.child_ids.length > 0 && (
            <>
              <h4>Children</h4>
              <div className="chip-list">
                {detail.child_ids.map((childId) => (
                  <button key={childId} className="chip" onClick={() => onSelect(childId)}>
                    {childId.split(".").slice(-1)[0]}
                  </button>
                ))}
              </div>
            </>
          )}

        </div>
      )}
    </section>
  );
}
