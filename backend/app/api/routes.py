"""HTTP routes for the LFM3 Neuron Explorer."""

from __future__ import annotations

import uuid
from typing import Literal

import torch
from fastapi import APIRouter, HTTPException, Query, Request, Response

from backend.app.schemas.api import (
    ActivationResponse,
    ActivationSummary,
    ArchitectureResponse,
    BaselineRequest,
    BaselineResponse,
    BatchJobResponse,
    BatchRequest,
    DiagnosticsResponse,
    DirectionRequest,
    DirectionResponse,
    ExperimentDetail,
    ExperimentListResponse,
    FlowRequest,
    FlowResponse,
    HealthResponse,
    LensRequest,
    LensResponse,
    LoadModelRequest,
    ModelStatus,
    ModuleDetail,
    PatchRequest,
    PatchResponse,
    SteeringRequest,
    SteeringResponse,
    SuiteListResponse,
    SweepRequest,
    SweepResponse,
    SweepRun,
)
from backend.app.services.activation_store import (
    StoredExperiment,
    build_activation_response,
)
from backend.app.services.architecture_inspector import build_architecture, describe_module
from backend.app.services.direction import compute_direction
from backend.app.services.experiment_store import ExperimentStore
from backend.app.services.flow import trace_flow
from backend.app.services.generation import generate_baseline, generate_steered
from backend.app.services.jobs import BatchJobRegistry
from backend.app.services.lens import compute_lens
from backend.app.services.model_loader import (
    ModelLoadError,
    ModelManager,
    ModelNotLoadedError,
)
from backend.app.services.patching import run_patch
from backend.app.services.steering_engine import validate_steering_spec
from backend.app.services.suites import list_suites
from backend.app.services.sweep import run_sweep

router = APIRouter(prefix="/api")


def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


def get_store(request: Request) -> ExperimentStore:
    return request.app.state.experiment_store


def get_registry(request: Request) -> BatchJobRegistry:
    return request.app.state.batch_jobs


def _require_loaded(manager: ModelManager) -> None:
    if not manager.loaded:
        raise HTTPException(status_code=409, detail="No model is loaded. POST /api/model/load first.")


def _record(
    result,
    request: Request,
    *,
    persist_activations: bool | None = None,
    sweep_id: str | None = None,
    sweep_index: int | None = None,
    repeat_index: int | None = None,
) -> None:
    store = get_store(request)
    manager = get_manager(request)
    config = getattr(manager.model, "config", None)
    resolved_revision = getattr(config, "_commit_hash", None) or result.response.revision
    store.record_run(
        result,
        persist_activations=persist_activations,
        resolved_revision=resolved_revision,
        sweep_id=sweep_id,
        sweep_index=sweep_index,
        repeat_index=repeat_index,
    )


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, ModelNotLoadedError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return HTTPException(
            status_code=507,
            detail=f"GPU out of memory: {exc}. Reduce max_new_tokens or prompt length.",
        )
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


def _get_experiment(store: ExperimentStore, experiment_id: str) -> StoredExperiment:
    """Return a live StoredExperiment, restoring from snapshot if necessary."""
    try:
        return store.activation_store.get(experiment_id)
    except KeyError:
        pass
    try:
        store.restore_activations(experiment_id)
    except KeyError:
        raise KeyError(f"Experiment {experiment_id!r} has no captured activations.") from None
    return store.activation_store.get(experiment_id)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(request: Request) -> DiagnosticsResponse:
    return get_manager(request).diagnostics()


@router.get("/model/status", response_model=ModelStatus)
def model_status(request: Request) -> ModelStatus:
    return get_manager(request).status()


@router.post("/model/load", response_model=ModelStatus)
def load_model(payload: LoadModelRequest, request: Request) -> ModelStatus:
    manager = get_manager(request)
    try:
        return manager.load(
            model_id=payload.model_id,
            revision=payload.revision,
            device=payload.device,
        )
    except ModelLoadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/model/architecture", response_model=ArchitectureResponse)
def model_architecture(request: Request) -> ArchitectureResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    status = manager.status()
    return build_architecture(manager.model, model_id=status.model_id, revision=status.revision)


@router.get("/model/modules/{module_id:path}", response_model=ModuleDetail)
def model_module(module_id: str, request: Request) -> ModuleDetail:
    manager = get_manager(request)
    _require_loaded(manager)
    try:
        return describe_module(manager.model, module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/experiments/baseline", response_model=BaselineResponse)
def baseline(payload: BaselineRequest, request: Request) -> BaselineResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    try:
        result = generate_baseline(manager, payload, capture_ids=payload.capture_module_ids)
        _record(result, request, persist_activations=payload.persist_activations)
        return result.response
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        raise _fail(exc) from exc


@router.post("/experiments/steering", response_model=SteeringResponse)
def steering(payload: SteeringRequest, request: Request) -> SteeringResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    payload.capture_module_ids = list(dict.fromkeys(payload.capture_module_ids))
    try:
        # Validate the steering spec up front so 400s hit before any generation.
        validate_steering_spec(
            manager.model, payload.steering, strength_limit=manager.settings.max_steering_strength
        )
        baseline_result = generate_baseline(manager, payload, capture_ids=payload.capture_module_ids)
        _record(baseline_result, request, persist_activations=payload.persist_activations)
        steered_result = generate_steered(
            manager,
            payload,
            payload.steering,
            capture_ids=payload.capture_module_ids,
        )
        _record(steered_result, request, persist_activations=payload.persist_activations)
        return SteeringResponse(baseline=baseline_result.response, steered=steered_result.response)
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        raise _fail(exc) from exc


@router.post("/experiments/flow", response_model=FlowResponse)
def flow(payload: FlowRequest, request: Request) -> FlowResponse:
    """Trace one real generation across every decoder layer."""
    manager = get_manager(request)
    _require_loaded(manager)
    try:
        _, flow_response = trace_flow(
            manager,
            payload,
            on_generation=lambda generated: _record(
                generated, request, persist_activations=payload.persist_activations
            ),
        )
        return flow_response
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        raise _fail(exc) from exc


@router.post("/experiments/sweep", response_model=SweepResponse)
def sweep(payload: SweepRequest, request: Request) -> SweepResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    payload.capture_module_ids = list(dict.fromkeys(payload.capture_module_ids))
    sweep_id = uuid.uuid4().hex

    def _on_run(run) -> None:
        _record(
            run.result,
            request,
            persist_activations=payload.persist_activations,
            sweep_id=sweep_id,
            sweep_index=run.sweep_index,
            repeat_index=run.repeat_index,
        )

    try:
        result = run_sweep(
            manager,
            payload,
            capture_ids=payload.capture_module_ids,
            on_run=_on_run,
            sweep_id=sweep_id,
        )
        return SweepResponse(
            sweep_id=result.sweep_id,
            runs=[
                SweepRun(
                    kind=run.kind,
                    sweep_index=run.sweep_index,
                    repeat_index=run.repeat_index,
                    run=run.result.response,
                )
                for run in result.runs
            ],
            warnings=result.warnings,
        )
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        failure = _fail(exc)
        failure.detail = (
            f"{failure.detail} (sweep {sweep_id}; completed runs were saved and remain queryable.)"
        )
        raise failure from exc


@router.get("/experiments", response_model=ExperimentListResponse)
def list_experiments(
    request: Request,
    condition: Literal["baseline", "steered", "control"] | None = None,
    sweep_id: str | None = None,
    module_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ExperimentListResponse:
    store = get_store(request)
    return store.list_summaries(
        condition=condition,
        sweep_id=sweep_id,
        module_id=module_id,
        limit=limit,
        offset=offset,
    )


@router.get("/experiments/export")
def export_experiments(
    request: Request,
    format: str = Query(..., pattern="^(jsonl|csv)$"),
    ids: str = Query(default=""),
) -> Response:
    store = get_store(request)
    requested_ids = [eid.strip() for eid in ids.split(",") if eid.strip()]
    # Empty ids means "export all records".
    experiment_ids = requested_ids or store.list_ids()
    try:
        if format == "jsonl":
            body = store.export_jsonl(experiment_ids)
            media_type = "application/jsonl"
            filename = "experiments.jsonl"
        else:
            body = store.export_csv(experiment_ids)
            media_type = "text/csv"
            filename = "experiments.csv"
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/suites", response_model=SuiteListResponse)
def suites() -> SuiteListResponse:
    return SuiteListResponse(suites=list_suites())


@router.post("/experiments/direction", response_model=DirectionResponse)
def steering_direction(payload: DirectionRequest, request: Request) -> DirectionResponse:
    store = get_store(request)
    try:
        positive = _get_experiment(store, payload.positive_experiment_id)
        negative = _get_experiment(store, payload.negative_experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return compute_direction(positive, negative, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/experiments/batch", response_model=BatchJobResponse, status_code=202)
def start_batch(payload: BatchRequest, request: Request) -> BatchJobResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    payload.capture_module_ids = list(dict.fromkeys(payload.capture_module_ids))
    try:
        return get_registry(request).start(payload)
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        raise _fail(exc) from exc


@router.get("/experiments/batch/{job_id}", response_model=BatchJobResponse)
def batch_status(
    job_id: str,
    request: Request,
    include_runs: bool = Query(default=False),
) -> BatchJobResponse:
    try:
        return get_registry(request).get(job_id, include_runs=include_runs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/experiments/batch/{job_id}/cancel", response_model=BatchJobResponse)
def cancel_batch(job_id: str, request: Request) -> BatchJobResponse:
    try:
        return get_registry(request).cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/experiments/lens", response_model=LensResponse)
def logit_lens(payload: LensRequest, request: Request) -> LensResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    try:
        return compute_lens(manager, payload)
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        raise _fail(exc) from exc


@router.post("/experiments/patch", response_model=PatchResponse)
def activation_patch(payload: PatchRequest, request: Request) -> PatchResponse:
    manager = get_manager(request)
    _require_loaded(manager)
    try:
        return run_patch(manager, payload)
    except (ModelNotLoadedError, torch.cuda.OutOfMemoryError, ValueError) as exc:
        raise _fail(exc) from exc


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def experiment_detail(experiment_id: str, request: Request) -> ExperimentDetail:
    store = get_store(request)
    try:
        return store.build_detail(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/experiments/{experiment_id}/activations", response_model=list[ActivationSummary])
def experiment_activations(experiment_id: str, request: Request) -> list[ActivationSummary]:
    store = get_store(request)
    try:
        record = store.get_run(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not record.captured_modules and not record.activation_snapshot_path:
        return []
    try:
        return store.activation_store.list(experiment_id)
    except KeyError:
        try:
            store.restore_activations(experiment_id)
        except KeyError:
            return []
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return store.activation_store.list(experiment_id)


@router.get(
    "/experiments/{experiment_id}/activations/{module_id:path}",
    response_model=ActivationResponse,
)
def experiment_activation(
    experiment_id: str,
    module_id: str,
    request: Request,
    feature: int | None = None,
    position: int = -1,
    max_positions: int = Query(default=128, ge=1, le=512),
    max_features: int = Query(default=128, ge=1, le=512),
    compare_with: str | None = None,
) -> ActivationResponse:
    store = get_store(request)
    try:
        store.get_run(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        experiment = _get_experiment(store, experiment_id)
        reference = _get_experiment(store, compare_with) if compare_with else None
        return build_activation_response(
            experiment,
            module_id,
            feature=feature,
            position=position,
            max_positions=max_positions,
            max_features=max_features,
            compare_with=reference,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
