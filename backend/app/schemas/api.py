"""Pydantic request/response contracts for the HTTP API.

The backend is the authority on module names, shapes, and results; these models
describe exactly what the frontend may display.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class SteeringSpec(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    module_id: str = Field(min_length=1, max_length=256)
    feature_index: int = Field(ge=0)
    strength: float = Field(allow_inf_nan=False)


class BaselineRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    capture_module_ids: list[str] = Field(default_factory=list, max_length=8)
    persist_activations: bool | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class SteeringRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    steering: SteeringSpec
    capture_module_ids: list[str] = Field(default_factory=list, max_length=8)
    persist_activations: bool | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class BaselineResponse(BaseModel):
    id: str
    timestamp: datetime
    condition: Literal["baseline", "steered", "control"] = "baseline"
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
    steering: SteeringSpec | None = None
    captured_modules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SteeringResponse(BaseModel):
    baseline: BaselineResponse
    steered: BaselineResponse


class EnvironmentInfo(BaseModel):
    python_version: str
    torch_version: str
    transformers_version: str
    fastapi_version: str
    app_version: str
    platform: str
    git_commit: str


class ExperimentSummary(BaseModel):
    id: str
    timestamp: datetime
    model_id: str
    revision: str
    resolved_revision: str
    device: str
    condition: Literal["baseline", "steered", "control"]
    condition_detail: Literal["none", "positive", "negative", "zero", "control"]
    sweep_id: str | None = None
    sweep_index: int | None = None
    repeat_index: int | None = None
    module_id: str | None = None
    feature_index: int | None = None
    strength: float | None = None
    prompt_preview: str
    output_preview: str
    has_activations: bool
    activation_source: Literal["live", "snapshot"] | None = None
    duration_ms: float
    tokens_per_second: float | None = None


class CaptureInfo(BaseModel):
    module_id: str
    shape: list[int]
    dtype: str


class ExperimentDetail(ExperimentSummary):
    system: str | None = None
    prompt: str
    output: str
    prompt_tokens: int
    generated_tokens: int
    settings: GenerationSettings
    steering: SteeringSpec | None = None
    captured_modules: list[str] = Field(default_factory=list)
    captures: list[CaptureInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    environment: EnvironmentInfo
    tokens: list[TokenInfo]


class ExperimentListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ExperimentSummary]


class SweepRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    module_id: str = Field(min_length=1, max_length=256)
    feature_index: int = Field(ge=0)
    strengths: list[float] = Field(min_length=1, max_length=9)
    include_control: bool = False
    control_seed: int | None = None
    repeats: int = Field(default=1, ge=1, le=3)
    capture_module_ids: list[str] = Field(default_factory=list, max_length=8)
    persist_activations: bool | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("strengths")
    @classmethod
    def _strengths_finite(cls, value: list[float]) -> list[float]:
        for strength in value:
            if not math.isfinite(strength):
                raise ValueError("strengths must be finite")
        return value


class SweepRun(BaseModel):
    kind: Literal["baseline", "strength", "control"]
    sweep_index: int
    repeat_index: int
    run: BaselineResponse


class SweepResponse(BaseModel):
    sweep_id: str
    runs: list[SweepRun]
    warnings: list[str] = Field(default_factory=list)


class TokenInfo(BaseModel):
    index: int
    token_id: int
    text: str
    phase: Literal["prompt", "generated"]


class ActivationSummary(BaseModel):
    module_id: str
    shape: list[int]
    dtype: str


class HeatmapData(BaseModel):
    positions: list[int]
    features: list[int]
    values: list[list[float]]
    sampling: str


class HistogramData(BaseModel):
    bin_edges: list[float]
    counts: list[int]


class ActivationStats(BaseModel):
    minimum: float
    maximum: float
    mean: float
    std: float
    abs_mean: float


class TopDimension(BaseModel):
    index: int
    value: float
    abs_value: float


class FeatureSeries(BaseModel):
    feature: int
    positions: list[int]
    values: list[float]


class ActivationResponse(BaseModel):
    experiment_id: str
    module_id: str
    shape: list[int]
    dtype: str
    aggregation: str
    prompt_token_count: int
    tokens: list[TokenInfo]
    heatmap: HeatmapData
    histogram: HistogramData
    stats: ActivationStats
    position: int
    top_dimensions: list[TopDimension]
    feature_series: FeatureSeries
    comparison: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
