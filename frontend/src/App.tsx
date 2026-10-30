import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
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
  const selectionRequest = useRef(0);

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

  const handleSelect = async (moduleId: string) => {
    setSelectedId(moduleId);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
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
  ) => {
    setRunning(true);
    setGenerationError(null);
    try {
      setBaseline(await runBaseline(prompt, system, settings));
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
        />
        <ModuleInspector
          detail={detail}
          selectedId={selectedId}
          loading={detailLoading}
          error={detailError}
          onSelect={handleSelect}
        />
        <GenerationPanel
          disabled={!status.loaded}
          running={running}
          result={baseline}
          error={generationError}
          onRun={handleRunBaseline}
        />
      </main>

      <DiagnosticsPanel diagnostics={diagnostics} />
    </div>
  );
}
