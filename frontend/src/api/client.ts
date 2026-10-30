import type {
  ActivationResponse,
  ActivationSummary,
  Architecture,
  BaselineResponse,
  Diagnostics,
  GenerationSettings,
  ModelStatus,
  ModuleDetail,
} from "../types";

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (body.detail !== undefined) {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // keep the status text
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export function getStatus(): Promise<ModelStatus> {
  return request<ModelStatus>("/model/status");
}

export function getDiagnostics(): Promise<Diagnostics> {
  return request<Diagnostics>("/diagnostics");
}

export function loadModel(): Promise<ModelStatus> {
  return request<ModelStatus>("/model/load", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getArchitecture(): Promise<Architecture> {
  return request<Architecture>("/model/architecture");
}

export function getModule(moduleId: string): Promise<ModuleDetail> {
  return request<ModuleDetail>(`/model/modules/${moduleId}`);
}

export function runBaseline(
  prompt: string,
  system: string | null,
  settings: GenerationSettings,
  captureModuleIds: string[] = [],
): Promise<BaselineResponse> {
  return request<BaselineResponse>("/experiments/baseline", {
    method: "POST",
    body: JSON.stringify({ prompt, system, settings, capture_module_ids: captureModuleIds }),
  });
}

export function listActivations(experimentId: string): Promise<ActivationSummary[]> {
  return request<ActivationSummary[]>(`/experiments/${experimentId}/activations`);
}

export interface ActivationQuery {
  feature?: number;
  position?: number;
  maxPositions?: number;
  maxFeatures?: number;
}

export function getActivation(
  experimentId: string,
  moduleId: string,
  query: ActivationQuery = {},
): Promise<ActivationResponse> {
  const params = new URLSearchParams();
  if (query.feature !== undefined) params.set("feature", String(query.feature));
  if (query.position !== undefined) params.set("position", String(query.position));
  if (query.maxPositions !== undefined) params.set("max_positions", String(query.maxPositions));
  if (query.maxFeatures !== undefined) params.set("max_features", String(query.maxFeatures));
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return request<ActivationResponse>(`/experiments/${experimentId}/activations/${moduleId}${suffix}`);
}
