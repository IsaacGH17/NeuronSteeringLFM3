import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  cancelBatchJob,
  getActivation,
  getArchitecture,
  getBatchJob,
  getDiagnostics,
  getModule,
  getStatus,
  listActivations,
  listSuites,
  loadModel as apiLoadModel,
  runBaseline,
  runSteering,
  runSweep,
  startBatch,
} from "./api/client";
import { ArchitectureExplorer } from "./components/ArchitectureExplorer";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { ExperimentHistory } from "./components/ExperimentHistory";
import { FlowDrawer } from "./components/FlowDrawer";
import { GenerationPanel } from "./components/GenerationPanel";
import { InterpretDrawer } from "./components/InterpretDrawer";
import { HelpPanel } from "./components/HelpPanel";
import { ModuleInspector } from "./components/ModuleInspector";
import { useTooltips } from "./ui/tooltips";
import type {
  ActivationResponse,
  Architecture,
  BaselineResponse,
  BatchJob,
  BatchRequest,
  Diagnostics,
  DirectionResponse,
  GenerationSettings,
  ModelStatus,
  ModuleDetail,
  ModulePreset,
  PromptSuite,
  SteeringResponse,
  SteeringSpec,
  SweepResponse,
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
  const [pair, setPair] = useState<SteeringResponse | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [sweepResult, setSweepResult] = useState<SweepResponse | null>(null);
  const [sweepError, setSweepError] = useState<string | null>(null);
  const [activation, setActivation] = useState<ActivationResponse | null>(null);
  const [activationLoading, setActivationLoading] = useState(false);
  const [activationError, setActivationError] = useState<string | null>(null);
  const [capturedIds, setCapturedIds] = useState<Set<string>>(() => new Set());
  const [steeringTargetId, setSteeringTargetId] = useState<string | null>(null);
  const [compareWith, setCompareWith] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [interpretOpen, setInterpretOpen] = useState(false);
  const [flowOpen, setFlowOpen] = useState(false);
  const [direction, setDirection] = useState<DirectionResponse | null>(null);
  const [suites, setSuites] = useState<PromptSuite[]>([]);
  const [batchJob, setBatchJob] = useState<BatchJob | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const { enabled: tooltipsEnabled, setEnabled: setTooltipsEnabled } = useTooltips();
  const [historyRefresh, setHistoryRefresh] = useState(0);
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

  useEffect(() => {
    let cancelled = false;
    listSuites()
      .then((response) => {
        if (!cancelled) setSuites(response.suites);
      })
      .catch(() => {
        if (!cancelled) setSuites([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!batchJob) return;
    if (batchJob.state !== "queued" && batchJob.state !== "running") return;
    const timer = setTimeout(() => {
      getBatchJob(batchJob.job_id)
        .then((next) => setBatchJob(next))
        .catch((error) => setBatchError(errorMessage(error)));
    }, 1000);
    return () => clearTimeout(timer);
  }, [batchJob]);

  useEffect(() => {
    if (!batchJob) return;
    if (batchJob.state === "completed" || batchJob.state === "cancelled" || batchJob.state === "failed") {
      setHistoryRefresh((value) => value + 1);
      void refreshDiagnostics();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [batchJob?.state]);

  const modulePresets = useMemo<ModulePreset[]>(() => {
    if (!architecture) return [];
    return [
      {
        id: "ffn",
        label: "MLP outputs",
        modules: architecture.layer_overview.map((block) => ({
          module_id: block.ffn_module_id,
          layer_index: block.layer_index,
        })),
      },
      {
        id: "layers",
        label: "Decoder layer outputs",
        modules: architecture.layer_overview.map((block) => ({
          module_id: block.module_id,
          layer_index: block.layer_index,
        })),
      },
    ];
  }, [architecture]);

  const handleLoad = async () => {
    setLoadingModel(true);
    setLoadError(null);
    try {
      const next = await apiLoadModel();
      setStatus(next);
      setSteeringTargetId(null);
      setDirection(null);
      setBatchJob(null);
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
    async (
      experimentId: string,
      moduleId: string,
      query: { feature?: number; position?: number; compareWith?: string | null },
    ) => {
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

  const setBaselineResult = (run: BaselineResponse) => {
    setBaseline(run);
    setPair(null);
    setSweepResult(null);
    setSweepError(null);
    setCompareWith(null);
    setCapturedIds((current) => new Set([...current, ...run.captured_modules]));
    setHistoryRefresh((value) => value + 1);
    if (run.captured_modules.length > 0) {
      void loadActivation(run.id, run.captured_modules[0], {});
    } else {
      setActivation(null);
      setActivationError(null);
    }
  };

  const handleRunBaseline = async (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    captureModuleIds: string[],
    persistActivations: boolean | null = null,
  ) => {
    setRunning(true);
    setGenerationError(null);
    try {
      const run = await runBaseline(prompt, system, settings, captureModuleIds, persistActivations);
      setBaselineResult(run);
      await refreshDiagnostics();
    } catch (error) {
      setGenerationError(errorMessage(error));
    } finally {
      setRunning(false);
    }
  };

  const handleRunSteered = async (
    prompt: string,
    system: string | null,
    settings: GenerationSettings,
    steering: SteeringSpec,
    captureModuleIds: string[],
    persistActivations: boolean | null = null,
  ) => {
    setRunning(true);
    setGenerationError(null);
    try {
      const response = await runSteering(prompt, system, settings, steering, captureModuleIds, persistActivations);
      setBaseline(null);
      setSweepResult(null);
      setSweepError(null);
      setPair(response);
      const captured = new Set([
        ...response.baseline.captured_modules,
        ...response.steered.captured_modules,
      ]);
      setCapturedIds((current) => new Set([...current, ...captured]));
      setHistoryRefresh((value) => value + 1);
      // Show steered − baseline difference for the first captured module.
      if (response.steered.captured_modules.length > 0) {
        const moduleId = response.steered.captured_modules[0];
        setCompareWith(response.baseline.id);
        await loadActivation(response.steered.id, moduleId, {
          compareWith: response.baseline.id,
        });
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

  const handleRunSweep = async (payload: {
    prompt: string;
    system: string | null;
    settings: GenerationSettings;
    module_id: string;
    feature_index: number | null;
    components?: { feature_index: number; weight: number }[];
    normalization?: "raw" | "sigma";
    strengths: number[];
    include_control: boolean;
    control_seed: number | null;
    repeats: number;
    capture_module_ids: string[];
    persist_activations: boolean | null;
  }) => {
    setRunning(true);
    setSweepError(null);
    setSweepResult(null);
    setGenerationError(null);
    try {
      const response = await runSweep(payload);
      setBaseline(null);
      setPair(null);
      setSweepResult(response);
      const captured = new Set<string>();
      for (const run of response.runs) {
        for (const moduleId of run.run.captured_modules) {
          captured.add(moduleId);
        }
      }
      setCapturedIds((current) => new Set([...current, ...captured]));
      setHistoryRefresh((value) => value + 1);
      await refreshDiagnostics();
    } catch (error) {
      setSweepError(errorMessage(error));
    } finally {
      setRunning(false);
    }
  };

  const handleOpenHistoricalActivations = async (experimentId: string, moduleId: string) => {
    setActivationError(null);
    try {
      const summaries = await listActivations(experimentId);
      if (summaries.length === 0) {
        throw new Error("No activation data available for this experiment.");
      }
      const target = summaries.find((summary) => summary.module_id === moduleId) ?? summaries[0];
      setCompareWith(null);
      if (status.loaded) {
        await handleSelect(target.module_id);
      } else {
        setSelectedId(target.module_id);
        setDetail(null);
        setDetailError(null);
      }
      await loadActivation(experimentId, target.module_id, {});
    } catch (error) {
      setActivationError(errorMessage(error));
    }
  };

  const handleStartBatch = async (payload: BatchRequest) => {
    setBatchError(null);
    try {
      const job = await startBatch(payload);
      setBatchJob(job);
    } catch (error) {
      setBatchError(errorMessage(error));
    }
  };

  const handleCancelBatch = async () => {
    if (!batchJob) return;
    setBatchError(null);
    try {
      setBatchJob(await cancelBatchJob(batchJob.job_id));
    } catch (error) {
      setBatchError(errorMessage(error));
    }
  };

  const handleDirectionBuilt = (built: DirectionResponse) => {
    setDirection(built);
    setSteeringTargetId(built.module_id);
    setGenerationError(null);
  };

  const layerModules = useMemo(
    () =>
      (architecture?.layer_overview ?? []).map((block) => ({
        layer_index: block.layer_index,
        module_id: block.module_id,
      })),
    [architecture],
  );

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
          <label className="tooltip-toggle">
            <input
              type="checkbox"
              checked={tooltipsEnabled}
              onChange={(event) => setTooltipsEnabled(event.target.checked)}
            />
            <span>Tooltips</span>
          </label>
          <button
            className="secondary topbar-button"
            onClick={() => setFlowOpen(true)}
            aria-haspopup="dialog"
          >
            Flow
          </button>
          <button
            className="secondary topbar-button"
            onClick={() => setInterpretOpen(true)}
            aria-haspopup="dialog"
          >
            Interpret
          </button>
          <button
            className="secondary topbar-button"
            onClick={() => setHelpOpen(true)}
            aria-haspopup="dialog"
          >
            Help
          </button>
          {!status.loaded && (
            <button className="primary" onClick={handleLoad} disabled={loadingModel || status.state === "loading"}>
              {loadingModel || status.state === "loading" ? "Loading…" : "Load model"}
            </button>
          )}
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
          steeringTargetId={steeringTargetId}
          onSetSteeringTarget={setSteeringTargetId}
          onActivationQuery={(query) => {
            if (activation) {
              void loadActivation(activation.experiment_id, activation.module_id, {
                ...query,
                compareWith,
              });
            }
          }}
        />
        <GenerationPanel
          disabled={!status.loaded}
          running={running}
          result={baseline}
          pair={pair}
          sweepResult={sweepResult}
          sweepError={sweepError}
          error={generationError}
          selectedSteerableId={selectedId && detail?.supports_steering ? selectedId : null}
          steeringTargetId={steeringTargetId}
          onSetSteeringTarget={setSteeringTargetId}
          hiddenSize={architecture?.hidden_size ?? null}
          onRun={handleRunBaseline}
          onRunSteered={handleRunSteered}
          onRunSweep={handleRunSweep}
          direction={direction}
          onClearDirection={() => setDirection(null)}
          modulePresets={modulePresets}
          suites={suites}
          batchJob={batchJob}
          batchError={batchError}
          onStartBatch={handleStartBatch}
          onCancelBatch={handleCancelBatch}
        />
      </main>

      <FlowDrawer
        open={flowOpen}
        disabled={!status.loaded}
        onClose={() => setFlowOpen(false)}
        onUseDirection={(built) => {
          handleDirectionBuilt(built);
          setFlowOpen(false);
        }}
        onExperimentRecorded={() => setHistoryRefresh((value) => value + 1)}
      />
      <InterpretDrawer
        open={interpretOpen}
        onClose={() => setInterpretOpen(false)}
        layers={layerModules}
        defaultModuleId={steeringTargetId ?? layerModules[Math.floor(layerModules.length / 2)]?.module_id ?? null}
      />
      <HelpPanel open={helpOpen} onClose={() => setHelpOpen(false)} />
      <ExperimentHistory
        refreshTrigger={historyRefresh}
        onOpenActivations={handleOpenHistoricalActivations}
        onDirectionBuilt={handleDirectionBuilt}
      />
      <DiagnosticsPanel diagnostics={diagnostics} />
    </div>
  );
}
