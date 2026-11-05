import type { ActivationResponse, ModuleDetail } from "../types";
import { tooltipTitle, useTooltips } from "../ui/tooltips";
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
  steeringTargetId: string | null;
  onSetSteeringTarget: (moduleId: string) => void;
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
  steeringTargetId,
  onSetSteeringTarget,
  onActivationQuery,
}: Props) {
  const { enabled: tooltipsEnabled } = useTooltips();

  return (
    <section className="panel inspector">
      <header className="panel-header">
        <div>
          <h2>Module inspector</h2>
          <p className="subtle">Everything here is read from the loaded model. Activations can come from a live run or a saved snapshot.</p>
        </div>
      </header>

      {!selectedId && <div className="empty">Pick a module on the left to look inside it.</div>}
      {selectedId && loading && <div className="empty">Loading {selectedId}…</div>}
      {error && <div className="error-banner">{error}</div>}

      {!detail && !loading && activation && (
        <div className="inspector-body">
          <div className="module-title">
            <h3>{activation.module_id}</h3>
            <span className="badge muted-badge">saved run</span>
          </div>
          <p className="subtle">
            The model isn't loaded right now, so only the tensor saved with this run is shown below.
          </p>
          <h4 className="eyebrow">Activations</h4>
          <ActivationView
            activation={activation}
            loading={activationLoading}
            error={activationError}
            onQuery={onActivationQuery}
          />
        </div>
      )}

      {detail && !loading && (
        <div className="inspector-body">
          <div className="module-title">
            <h3>{detail.id}</h3>
            <span className={`badge ${detail.supports_steering ? "steerable-badge" : "muted-badge"}`}>
              {detail.supports_steering ? "steerable" : "not steerable"}
            </span>
            {detail.supports_steering && (
              <button
                className="secondary small"
                onClick={() => onSetSteeringTarget(detail.id)}
                disabled={steeringTargetId === detail.id}
                title={tooltipTitle(
                  tooltipsEnabled,
                  steeringTargetId === detail.id
                    ? "This module is already the steering target"
                    : "Steer this module without changing what you are inspecting",
                )}
              >
                {steeringTargetId === detail.id ? "Steering target" : "Set as steering target"}
              </button>
            )}
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

          <h4 className="eyebrow">Activations</h4>
          <ActivationView
            activation={activation}
            loading={activationLoading}
            error={activationError}
            onQuery={onActivationQuery}
          />

          <h4>Direct parameters</h4>
          {detail.parameters.length === 0 ? (
            <p className="subtle">This module holds no parameters of its own.</p>
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
