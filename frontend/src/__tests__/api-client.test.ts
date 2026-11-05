import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  buildExportUrl,
  downloadExport,
  getActivation,
  getModule,
  getStatus,
  listExperiments,
  runBaseline,
  runFlow,
  runSteering,
} from "../api/client";
import { DEFAULT_SETTINGS } from "../test/fixtures";

function jsonResponse(data: unknown, init: { ok?: boolean; status?: number; statusText?: string } = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    json: async () => data,
  };
}

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("requests the model status endpoint", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ state: "unloaded" }));
    await expect(getStatus()).resolves.toEqual({ state: "unloaded" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/model/status",
      expect.objectContaining({ headers: { "Content-Type": "application/json" } }),
    );
  });

  it("encodes module ids in detail requests", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: "model.layers.0" }));
    await getModule("model.layers.0");
    expect(fetchMock).toHaveBeenCalledWith("/api/model/modules/model.layers.0", expect.any(Object));
  });

  it("sends the baseline payload with explicit capture and persistence fields", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: "run" }));
    await runBaseline("Hello", "Be brief", DEFAULT_SETTINGS, ["model.layers.0"], true);
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      prompt: "Hello",
      system: "Be brief",
      settings: DEFAULT_SETTINGS,
      capture_module_ids: ["model.layers.0"],
      persist_activations: true,
    });
  });

  it("sends a flow trace request to the flow endpoint", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ layers: [] }));
    await runFlow({
      prompt: "Hello",
      system: null,
      settings: DEFAULT_SETTINGS,
      cluster_k: 6,
      cluster_max_positions: 128,
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/experiments/flow");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      prompt: "Hello",
      system: null,
      settings: DEFAULT_SETTINGS,
      cluster_k: 6,
      cluster_max_positions: 128,
    });
  });

  it("sends the steering spec alongside the shared settings", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ baseline: {}, steered: {} }));
    await runSteering("Hello", null, DEFAULT_SETTINGS, {
      module_id: "model.layers.0.feed_forward",
      feature_index: 500,
      strength: -12,
    });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      prompt: "Hello",
      system: null,
      settings: DEFAULT_SETTINGS,
      steering: { module_id: "model.layers.0.feed_forward", feature_index: 500, strength: -12 },
      capture_module_ids: [],
      persist_activations: null,
    });
  });

  it("builds activation queries from only the provided options", async () => {
    fetchMock.mockResolvedValue(jsonResponse({}));
    await getActivation("a".repeat(32), "model.layers.0", {
      feature: 7,
      compareWith: "b".repeat(32),
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/experiments/${"a".repeat(32)}/activations/model.layers.0?feature=7&compare_with=${"b".repeat(32)}`,
      expect.any(Object),
    );
  });

  it("builds experiment list filters and pagination", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ total: 0, offset: 10, limit: 10, items: [] }));
    await listExperiments({
      condition: "steered",
      sweep_id: "c".repeat(32),
      module_id: "model.layers.0",
      limit: 10,
      offset: 10,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/experiments?condition=steered&sweep_id=${"c".repeat(32)}&module_id=model.layers.0&limit=10&offset=10`,
      expect.any(Object),
    );
  });

  it("surfaces a string error detail from the backend", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ detail: "feature_index must be < 2048" }, { ok: false, status: 422, statusText: "Unprocessable Entity" }),
    );
    const error = await listExperiments().catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toBe("feature_index must be < 2048");
    expect((error as ApiError).status).toBe(422);
  });

  it("serializes structured error details", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ detail: [{ loc: ["body", "m"], msg: "required" }] }, { ok: false, status: 422 }),
    );
    const error = await listExperiments().catch((err: unknown) => err);
    expect((error as ApiError).message).toBe('[{"loc":["body","m"],"msg":"required"}]');
  });

  it("falls back to the HTTP status when the body is not JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    });
    const error = await listExperiments().catch((err: unknown) => err);
    expect((error as ApiError).message).toBe("500 Internal Server Error");
  });

  it("builds export URLs with selected ids", () => {
    expect(buildExportUrl({ format: "csv", ids: ["a", "b"] })).toBe(
      "/api/experiments/export?format=csv&ids=a%2Cb",
    );
    expect(buildExportUrl({ format: "jsonl" })).toBe("/api/experiments/export?format=jsonl");
  });

  it("triggers a browser download for exports", () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    downloadExport({ format: "csv", ids: ["a"] });
    expect(click).toHaveBeenCalledTimes(1);
  });
});
