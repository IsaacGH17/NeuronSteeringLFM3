import type {
  ActivationResponse,
  ActivationSummary,
  Architecture,
  BaselineResponse,
  Diagnostics,
  ExperimentDetail,
  ExperimentListResponse,
  GenerationSettings,
  ModelStatus,
  ModuleDetail,
  SteeringResponse,
  SteeringSpec,
  SweepResponse,
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
  persistActivations: boolean | null = null,
): Promise<BaselineResponse> {
  return request<BaselineResponse>("/experiments/baseline", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      system,
      settings,
      capture_module_ids: captureModuleIds,
      persist_activations: persistActivations,
    }),
  });
}

export function runSteering(
  prompt: string,
  system: string | null,
  settings: GenerationSettings,
  steering: SteeringSpec,
  captureModuleIds: string[] = [],
  persistActivations: boolean | null = null,
): Promise<SteeringResponse> {
  return request<SteeringResponse>("/experiments/steering", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      system,
      settings,
      steering,
      capture_module_ids: captureModuleIds,
      persist_activations: persistActivations,
    }),
  });
}

export interface SweepPayload {
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
}

export function runSweep(payload: SweepPayload): Promise<SweepResponse> {
  return request<SweepResponse>("/experiments/sweep", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface ExperimentListParams {
  condition?: "baseline" | "steered" | "control";
  sweep_id?: string;
  module_id?: string;
  limit?: number;
  offset?: number;
}

export function listExperiments(params: ExperimentListParams = {}): Promise<ExperimentListResponse> {
  const search = new URLSearchParams();
  if (params.condition) search.set("condition", params.condition);
  if (params.sweep_id) search.set("sweep_id", params.sweep_id);
  if (params.module_id) search.set("module_id", params.module_id);
  if (params.limit !== undefined) search.set("limit", String(params.limit));
  if (params.offset !== undefined) search.set("offset", String(params.offset));
  const suffix = search.size > 0 ? `?${search.toString()}` : "";
  return request<ExperimentListResponse>(`/experiments${suffix}`);
}

export function getExperiment(experimentId: string): Promise<ExperimentDetail> {
  return request<ExperimentDetail>(`/experiments/${experimentId}`);
}

export interface ExportParams {
  format: "jsonl" | "csv";
  ids?: string[];
}

export function buildExportUrl(params: ExportParams): string {
  const search = new URLSearchParams();
  search.set("format", params.format);
  if (params.ids && params.ids.length > 0) search.set("ids", params.ids.join(","));
  return `${API_BASE}/experiments/export?${search.toString()}`;
}

export function downloadExport(params: ExportParams): void {
  const link = document.createElement("a");
  link.href = buildExportUrl(params);
  link.download = params.format === "csv" ? "experiments.csv" : "experiments.jsonl";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function listActivations(experimentId: string): Promise<ActivationSummary[]> {
  return request<ActivationSummary[]>(`/experiments/${experimentId}/activations`);
}

export interface ActivationQuery {
  feature?: number;
  position?: number;
  maxPositions?: number;
  maxFeatures?: number;
  compareWith?: string | null;
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
  if (query.compareWith) params.set("compare_with", query.compareWith);
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return request<ActivationResponse>(`/experiments/${experimentId}/activations/${moduleId}${suffix}`);
}
