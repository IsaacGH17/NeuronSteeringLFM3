"""Pydantic request/response contracts for the HTTP API.

The backend is the authority on module names, shapes, and results; these models
describe exactly what the frontend may display.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelState = Literal["unloaded", "loading", "loaded", "error"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "lfm3-neuron-explorer"


class ModelStatus(BaseModel):
    state: ModelState
    model_id: str
    revision: str
    device: str | None = None
    dtype: str | None = None
    loaded: bool = False
    error: str | None = None


class LoadModelRequest(BaseModel):
    model_id: str | None = None
    revision: str | None = None
    device: str | None = None


class MemoryStats(BaseModel):
    device: str
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None


class DiagnosticsResponse(BaseModel):
    python_version: str
    torch_version: str
    transformers_version: str
    fastapi_version: str
    cuda_available: bool
    hip_version: str | None = None
    backend: Literal["rocm", "cuda", "cpu"]
    device_count: int
    device_name: str | None = None
    inference_device: str | None = None
    model_state: ModelState
    memory: MemoryStats | None = None
    notes: list[str] = Field(default_factory=list)


class ModuleNode(BaseModel):
    id: str
    name: str
    class_name: str
    parameter_count: int = 0
    child_count: int = 0
    layer_type: str | None = None
    supports_steering: bool = False
    steering_reason: str | None = None
    children: list[ModuleNode] = Field(default_factory=list)


class LayerBlock(BaseModel):
    layer_index: int
    module_id: str
    layer_type: str
    operator_type: Literal["attention", "conv"]
    operator_module_id: str
    operator_class: str
    ffn_module_id: str
    parameter_count: int


class ArchitectureResponse(BaseModel):
    model_id: str
    revision: str
    model_class: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    layer_types: list[str]
    tree: ModuleNode
    layer_overview: list[LayerBlock]


class ParameterInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    num_parameters: int


class ModuleDetail(BaseModel):
    id: str
    name: str
    class_name: str
    parent_id: str | None = None
    parameter_count: int
    direct_parameter_count: int
    layer_type: str | None = None
    supports_steering: bool
    steering_reason: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    parameters: list[ParameterInfo] = Field(default_factory=list)
    captured: bool = False


class GenerationSettings(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    enable_thinking: bool = False


class BaselineRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class BaselineResponse(BaseModel):
    id: str
    timestamp: datetime
    condition: Literal["baseline"] = "baseline"
    model_id: str
    revision: str
    device: str
    prompt: str
    system: str | None = None
    output: str
    prompt_tokens: int
    generated_tokens: int
    duration_ms: float
    tokens_per_second: float | None = None
    settings: GenerationSettings
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
