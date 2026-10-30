"""HTTP routes for the LFM3 Neuron Explorer."""

from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas.api import (
    ArchitectureResponse,
    BaselineRequest,
    BaselineResponse,
    DiagnosticsResponse,
    HealthResponse,
    LoadModelRequest,
    ModelStatus,
    ModuleDetail,
)
from backend.app.services.architecture_inspector import build_architecture, describe_module
from backend.app.services.generation import generate_baseline
from backend.app.services.model_loader import (
    ModelLoadError,
    ModelManager,
    ModelNotLoadedError,
)

router = APIRouter(prefix="/api")


def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


def _require_loaded(manager: ModelManager) -> None:
    if not manager.loaded:
        raise HTTPException(status_code=409, detail="No model is loaded. POST /api/model/load first.")


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
        return generate_baseline(manager, payload)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise HTTPException(
            status_code=507,
            detail=f"GPU out of memory: {exc}. Reduce max_new_tokens or prompt length.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
