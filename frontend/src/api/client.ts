import type {
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
): Promise<BaselineResponse> {
  return request<BaselineResponse>("/experiments/baseline", {
    method: "POST",
    body: JSON.stringify({ prompt, system, settings }),
  });
}
