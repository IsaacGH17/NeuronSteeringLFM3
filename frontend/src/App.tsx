import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  getActivation,
  getArchitecture,
  getDiagnostics,
  getModule,
  getStatus,
  loadModel as apiLoadModel,
  runBaseline,
} from "./api/client";
import { ArchitectureExplorer } from "./components/ArchitectureExplorer";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { GenerationPanel } from "./components/GenerationPanel";
import { ModuleInspector } from "./components/ModuleInspector";
import type {
  ActivationResponse,
  Architecture,
  BaselineResponse,
  Diagnostics,
  GenerationSettings,
  ModelStatus,
  ModuleDetail,
} from "./types";

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

const INITIAL_STATUS: ModelStatus = {
  state: "unloaded",
  model_id: "LiquidAI/LFM3-1.7B",
  revision: "main",
  device: null,
  dtype: null,
  loaded: false,
  error: null,
};

export default function App() {
  const [status, setStatus] = useState<ModelStatus>(INITIAL_STATUS);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [architecture, setArchitecture] = useState<Architecture | null>(null);
  const [architectureLoading, setArchitectureLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ModuleDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [baseline, setBaseline] = useState<BaselineResponse | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [activation, setActivation] = useState<ActivationResponse | null>(null);
  const [activationLoading, setActivationLoading] = useState(false);
  const [activationError, setActivationError] = useState<string | null>(null);
  const [capturedIds, setCapturedIds] = useState<Set<string>>(() => new Set());
  const selectionRequest = useRef(0);
  const activationRequest = useRef(0);

  const refreshStatus = useCallback(async () => {
    const next = await getStatus();
    setStatus(next);
    return next;
  }, []);

  const refreshDiagnostics = useCallback(async () => {
    try {
      setDiagnostics(await getDiagnostics());
    } catch {
      setDiagnostics(null);
    }
  }, []);

  const loadArchitecture = useCallback(async () => {
    setArchitectureLoading(true);
    try {
      setArchitecture(await getArchitecture());
      setLoadError(null);
    } catch (error) {
      setArchitecture(null);
      setLoadError(errorMessage(error));
    } finally {
      setArchitectureLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const next = await getStatus();
        if (cancelled) return;
        setStatus(next);
        if (next.loaded) await loadArchitecture();
      } catch (error) {
        if (!cancelled) setLoadError(errorMessage(error));
      }
      if (!cancelled) await refreshDiagnostics();
    })();
    return () => {
      cancelled = true;
    };
  }, [loadArchitecture, refreshDiagnostics]);

  const handleLoad = async () => {
    setLoadingModel(true);
    setLoadError(null);
    try {
      const next = await apiLoadModel();
      setStatus(next);
      await Promise.all([loadArchitecture(), refreshDiagnostics()]);
    } catch (error) {
      setLoadError(errorMessage(error));
      await refreshStatus().catch(() => undefined);
      await refreshDiagnostics();
    } finally {
      setLoadingModel(false);
    }
  };

  const loadActivation = useCallback(
    async (experimentId: string, moduleId: string, query: { feature?: number; position?: number }) => {
      const requestId = ++activationRequest.current;
      setActivationLoading(true);
      setActivationError(null);
      try {
        const next = await getActivation(experimentId, moduleId, query);
        if (requestId === activationRequest.current) setActivation(next);
      } catch (error) {
        if (requestId === activationRequest.current) setActivationError(errorMessage(error));
      } finally {
        if (requestId === activationRequest.current) setActivationLoading(false);
      }
    },
    [],
  );

  const handleSelect = async (moduleId: string) => {
    setSelectedId(moduleId);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    setActivation((current) => (current && current.module_id === moduleId ? current : null));
    setActivationError(null);
    const requestId = ++selectionRequest.current;
    try {
      const next = await getModule(moduleId);
      if (requestId === selectionRequest.current) setDetail(next);
    } catch (error) {
      if (requestId === selectionRequest.current) setDetailError(errorMessage(error));
    } finally {
      if (requestId === selectionRequest.current) setDetailLoading(false);
    }
  };

  const handleRunBaseline = async (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    captureModuleIds: string[],
  ) => {
    setRunning(true);
    setGenerationError(null);
    try {
      const run = await runBaseline(prompt, system, settings, captureModuleIds);
      setBaseline(run);
      setCapturedIds(new Set(run.captured_modules));
      if (run.captured_modules.length > 0) {
        await loadActivation(run.id, run.captured_modules[0], {});
      } else {
        setActivation(null);
        setActivationError(null);
      }
      await refreshDiagnostics();
    } catch (error) {
      setGenerationError(errorMessage(error));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          LFM3 <span>Neuron Explorer</span>
        </div>
        <div className="status-group">
          <span className={`status-pill ${status.state}`}>{status.state}</span>
          <span className="model-id">{status.model_id}</span>
          {status.device && (
            <span className="subtle">
              {status.device} · {status.dtype}
            </span>
          )}
          <button className="primary" onClick={handleLoad} disabled={loadingModel || status.loaded}>
            {loadingModel || status.state === "loading" ? "Loading…" : status.loaded ? "Model loaded" : "Load model"}
          </button>
        </div>
      </header>

      {(loadError || status.error) && <div className="error-banner global">{loadError ?? status.error}</div>}

      <main className="workspace">
        <ArchitectureExplorer
          architecture={architecture}
          selectedId={selectedId}
          onSelect={handleSelect}
          loading={architectureLoading}
          capturedIds={capturedIds}
        />
        <ModuleInspector
          detail={detail}
          selectedId={selectedId}
          loading={detailLoading}
          error={detailError}
          onSelect={handleSelect}
          activation={activation}
          activationLoading={activationLoading}
          activationError={activationError}
          onActivationQuery={(query) => {
            if (activation) {
              void loadActivation(activation.experiment_id, activation.module_id, query);
            }
          }}
        />
        <GenerationPanel
          disabled={!status.loaded}
          running={running}
          result={baseline}
          error={generationError}
          captureModuleId={selectedId && detail?.supports_steering ? selectedId : null}
          onRun={handleRunBaseline}
        />
      </main>

      <DiagnosticsPanel diagnostics={diagnostics} />
    </div>
  );
}
