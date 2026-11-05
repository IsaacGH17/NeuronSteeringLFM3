"""Pydantic request/response contracts for the HTTP API.

The backend is the authority on module names, shapes, and results; these models
describe exactly what the frontend may display.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class SteeringComponent(BaseModel):
    """One hidden dimension and its relative weight in a steering direction."""

    feature_index: int = Field(ge=0)
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _weight_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight must be finite")
        return value


class SteeringSpec(BaseModel):
    """A steering direction applied to one module's hidden-state output.

    ``components`` holds one or more dimensions with relative weights; the
    request-level ``strength`` scales all of them at once. The legacy
    single-dimension form (``feature_index`` + ``strength``) is still accepted
    and normalized to one component with weight 1.

    ``normalization`` decides how ``strength`` maps to activation units:

    * ``raw`` adds ``strength * weight`` to each dimension.
    * ``sigma`` adds ``strength * weight * sigma`` where sigma is that
      dimension's standard deviation over batch and tokens in the current
      forward pass, which makes the same strength comparable across layers.
    """

    model_config = ConfigDict(from_attributes=True)

    module_id: str = Field(min_length=1, max_length=256)
    strength: float = Field(allow_inf_nan=False)
    components: list[SteeringComponent] = Field(default_factory=list, max_length=512)
    normalization: Literal["raw", "sigma"] = "raw"
    feature_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _normalize_components(self) -> SteeringSpec:
        merged: dict[int, float] = {}
        for component in self.components:
            merged[component.feature_index] = merged.get(component.feature_index, 0.0) + component.weight
        if not merged:
            if self.feature_index is None:
                raise ValueError("steering must define feature_index or components")
            merged[self.feature_index] = 1.0
        self.components = [
            SteeringComponent(feature_index=index, weight=weight) for index, weight in merged.items()
        ]
        if len(self.components) == 1:
            if self.feature_index is None:
                self.feature_index = self.components[0].feature_index
        else:
            self.feature_index = None
        return self

    @property
    def dimension_count(self) -> int:
        return len(self.components)

    def describe(self) -> str:
        dims = len(self.components)
        unit = "" if self.normalization == "raw" else " sigma"
        sign = "+" if self.strength >= 0 else "-"
        label = f"dim {self.feature_index}" if dims == 1 else f"{dims} dims"
        return f"{self.module_id} {label} {sign}{abs(self.strength):.6g}{unit}"


CheckKind = Literal[
    "max_words",
    "min_words",
    "contains",
    "not_contains",
    "one_word",
    "is_number",
    "list_items",
    "regex",
]


class MetricCheck(BaseModel):
    """One deterministic expectation about a generated text."""

    name: str
    kind: CheckKind
    expected: str | int | float | None = None
    passed: bool
    detail: str | None = None


class TextMetrics(BaseModel):
    """Deterministic text measurements attached to every run.

    These are lexical and structural heuristics, not semantic judgments: they
    make a trait usable as a number (tone markers, verbosity, repetition) and
    keep capability visible through the ``checks`` of a prompt suite.
    """

    words: int
    characters: int
    sentences: int
    avg_words_per_sentence: float
    distinct_3: float
    repeated_4gram_ratio: float
    refusal: bool
    lexicon: dict[str, float] = Field(default_factory=dict)
    checks: list[MetricCheck] = Field(default_factory=list)
    checks_passed: int | None = None
    checks_total: int | None = None


class BaselineRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    capture_module_ids: list[str] = Field(default_factory=list, max_length=128)
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
    capture_module_ids: list[str] = Field(default_factory=list, max_length=128)
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
    metrics: TextMetrics | None = None
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
    steering_summary: str | None = None
    metrics: TextMetrics | None = None
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
    feature_index: int | None = Field(default=None, ge=0)
    components: list[SteeringComponent] = Field(default_factory=list, max_length=512)
    normalization: Literal["raw", "sigma"] = "raw"
    strengths: list[float] = Field(min_length=1, max_length=32)
    include_control: bool = False
    control_seed: int | None = None
    repeats: int = Field(default=1, ge=1, le=10)
    capture_module_ids: list[str] = Field(default_factory=list, max_length=128)
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

    @model_validator(mode="after")
    def _needs_a_direction(self) -> SweepRequest:
        if not self.components and self.feature_index is None:
            raise ValueError("provide feature_index or components")
        return self


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


class FeatureStats(BaseModel):
    """Per hidden-dimension statistics over the captured positions."""

    mean: list[float]
    std: list[float]
    minimum: list[float]
    maximum: list[float]


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
    feature_stats: FeatureStats | None = None
    comparison: str | None = None
    warnings: list[str] = Field(default_factory=list)


class LensToken(BaseModel):
    token_id: int
    text: str
    probability: float


class LensLayer(BaseModel):
    layer_index: int
    module_id: str
    top1_ids: list[int]
    top1_probs: list[float]
    top_tokens: list[LensToken]
    ablation_kl: float | None = None
    ablation_top1: LensToken | None = None


class LensRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    position: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class LensResponse(BaseModel):
    prompt: str
    system: str | None = None
    output: str
    prompt_tokens: int
    position: int
    tokens: list[TokenInfo]
    layers: list[LensLayer]
    vocab: dict[int, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PatchRequest(BaseModel):
    recipient_prompt: str = Field(min_length=1, max_length=32_000)
    recipient_system: str | None = Field(default=None, max_length=8_000)
    donor_prompt: str = Field(min_length=1, max_length=32_000)
    donor_system: str | None = Field(default=None, max_length=8_000)
    module_id: str = Field(min_length=1, max_length=256)
    position: int = -1
    settings: GenerationSettings = Field(default_factory=GenerationSettings)


class PatchResponse(BaseModel):
    module_id: str
    position: int
    donor_output: str
    baseline_output: str
    patched_output: str
    changed: bool
    prompt_tokens: int
    warnings: list[str] = Field(default_factory=list)


class PatchSweepRequest(BaseModel):
    recipient_prompt: str = Field(min_length=1, max_length=32_000)
    recipient_system: str | None = Field(default=None, max_length=8_000)
    donor_prompt: str = Field(min_length=1, max_length=32_000)
    donor_system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    layer_indices: list[int] = Field(default_factory=list, max_length=24)
    positions: list[int] = Field(default_factory=lambda: [-1], min_length=1, max_length=8)

    @field_validator("recipient_prompt", "donor_prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class PatchSweepCell(BaseModel):
    layer_index: int
    module_id: str
    position: int
    donor_position: int
    patched_output: str
    changed: bool
    first_token_changed: bool
    common_prefix_tokens: int
    generated_tokens: int
    warnings: list[str] = Field(default_factory=list)


class PatchSweepResponse(BaseModel):
    recipient_prompt: str
    donor_prompt: str
    baseline_output: str
    donor_output: str
    prompt_tokens: int
    donor_prompt_tokens: int
    layers: list[int]
    positions: list[int]
    cells: list[PatchSweepCell]
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)


class NeuronTop(BaseModel):
    neuron: int
    weight: float
    mean_abs: float
    sigma: float


class NeuronCluster(BaseModel):
    id: int
    size: int
    mass: float
    cohesion: float
    peak_position: int
    top_neurons: list[NeuronTop]
    profile: list[float]
    profile_positions: list[int]


class NeuronTraceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    layer_index: int = Field(default=16, ge=0)
    cluster_k: int = Field(default=6, ge=2, le=12)
    cluster_max_positions: int = Field(default=128, ge=16, le=512)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class NeuronTraceResponse(BaseModel):
    experiment_id: str
    prompt: str
    system: str | None = None
    output: str
    tokens: list[TokenInfo]
    prompt_tokens: int
    generated_tokens: int
    layer_index: int
    layer_type: str
    intermediate_size: int
    hidden_size: int
    residual_rms: float
    clusters: list[NeuronCluster]
    top_neurons: list[NeuronTop]
    assignments: list[int]
    cluster_positions: int
    warnings: list[str] = Field(default_factory=list)


class NeuronComponent(BaseModel):
    neuron: int = Field(ge=0)
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _weight_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight must be finite")
        return value


class NeuronDelta(BaseModel):
    neuron: int
    delta: float


class NeuronSteeringRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    layer_index: int = Field(default=16, ge=0)
    neurons: list[NeuronComponent] = Field(min_length=1, max_length=64)
    alpha: float = Field(default=0.7, allow_inf_nan=False)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class NeuronSteeringResponse(BaseModel):
    layer_index: int
    alpha: float
    neurons: list[NeuronComponent]
    direction_norm: float
    residual_rms: float
    deltas: list[NeuronDelta]
    baseline: BaselineResponse
    steered: BaselineResponse
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)


class SuiteCheck(BaseModel):
    """A verifiable expectation attached to a suite prompt."""

    kind: CheckKind
    value: str | int | float | None = None


class SuitePrompt(BaseModel):
    id: str
    text: str
    tags: list[str] = Field(default_factory=list)
    checks: list[SuiteCheck] = Field(default_factory=list)


class PromptSuite(BaseModel):
    id: str
    name: str
    description: str
    kind: Literal["trait", "capability"]
    prompts: list[SuitePrompt]


class SuiteListResponse(BaseModel):
    suites: list[PromptSuite]


class DirectionRequest(BaseModel):
    positive_experiment_id: str
    negative_experiment_id: str
    module_id: str | None = None
    position_mode: Literal["mean_prompt", "mean_all", "last_prompt"] = "mean_prompt"
    top_k: int = Field(default=8, ge=1, le=512)
    normalize: bool = True


class DirectionResponse(BaseModel):
    module_id: str
    components: list[SteeringComponent]
    sigma: list[float]
    norm: float
    separation: float
    positions_compared: int
    warnings: list[str] = Field(default_factory=list)


class FlowRequest(BaseModel):
    """Trace one generation across every decoder layer.

    The flow endpoint always captures all decoder-layer outputs; it does not
    accept an arbitrary capture list. ``cluster_k`` and
    ``cluster_max_positions`` control the descriptive clustering of hidden
    dimensions (see ``services/flow.py``).
    """

    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    cluster_k: int = Field(default=6, ge=2, le=12)
    cluster_max_positions: int = Field(default=128, ge=16, le=512)
    include_ablation: bool = False
    persist_activations: bool | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class FlowDimension(BaseModel):
    """One hidden dimension inside a cluster, with the numbers used to steer.

    ``sigma`` is measured over the sampled trace positions. The steering engine
    recalibrates its own σ on the prompt prefill for ``normalization="sigma"``.
    """

    index: int
    weight: float
    mean_abs: float
    sigma: float


class FlowCluster(BaseModel):
    """A group of hidden dimensions with similar activation profiles.

    Clustering is descriptive: it groups dimensions by how they move across the
    captured positions. It is not a causal claim about behaviour.
    """

    id: int
    size: int
    mass: float
    cohesion: float
    peak_position: int
    direction_norm: float
    top_dimensions: list[FlowDimension]
    profile: list[float]
    profile_positions: list[int]
    components: list[SteeringComponent]


class FlowLayer(BaseModel):
    layer_index: int
    module_id: str
    layer_type: str
    operator_type: Literal["attention", "conv"]
    mean_norm: float
    delta_norm: float
    write: float
    similarity: float
    gain: float
    energy: list[float]
    delta: list[float]
    clusters: list[FlowCluster]
    ablation_kl: float | None = None


class FlowEdge(BaseModel):
    source_layer: int
    target_layer: int
    write: float
    similarity: float
    gain: float


class FlowResponse(BaseModel):
    experiment_id: str
    prompt: str
    system: str | None = None
    output: str
    prompt_tokens: int
    generated_tokens: int
    tokens: list[TokenInfo]
    layers: list[FlowLayer]
    edges: list[FlowEdge]
    global_max_norm: float
    cluster_k: int
    cluster_positions: int
    settings: GenerationSettings
    warnings: list[str] = Field(default_factory=list)


class NextTokenRequest(BaseModel):
    """One forward at the last prompt token; no text is generated."""

    prompt: str = Field(min_length=1, max_length=32_000)
    system: str | None = Field(default=None, max_length=8_000)
    enable_thinking: bool = False
    top_k: int = Field(default=10, ge=1, le=50)
    steering: SteeringSpec | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class TokenProbability(BaseModel):
    token_id: int
    text: str
    probability: float


class NextTokenDistribution(BaseModel):
    tokens: list[TokenProbability]
    entropy: float


class NextTokenCandidate(BaseModel):
    """A token in the union of the baseline and steered top-k lists."""

    token_id: int
    text: str
    baseline_probability: float
    steered_probability: float | None = None
    delta: float | None = None


class NextTokenResponse(BaseModel):
    prompt: str
    system: str | None = None
    model_id: str
    device: str
    prompt_tokens: int
    position: int
    top_k: int
    baseline: NextTokenDistribution
    steered: NextTokenDistribution | None = None
    candidates: list[NextTokenCandidate]
    kl_divergence: float | None = None
    steering: SteeringSpec | None = None
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)


class BatchPlan(BaseModel):
    """The run matrix: modules x strengths x repeats, plus optional control."""

    module_ids: list[str] = Field(min_length=1, max_length=64)
    components: list[SteeringComponent] = Field(min_length=1, max_length=512)
    normalization: Literal["raw", "sigma"] = "raw"
    strengths: list[float] = Field(min_length=1, max_length=32)
    include_control: bool = False
    control_seed: int | None = None
    repeats: int = Field(default=1, ge=1, le=10)
    baseline_per_prompt: bool = True

    @field_validator("strengths")
    @classmethod
    def _strengths_finite(cls, value: list[float]) -> list[float]:
        for strength in value:
            if not math.isfinite(strength):
                raise ValueError("strengths must be finite")
        return value


class BatchRequest(BaseModel):
    suite_id: str | None = None
    prompts: list[str] = Field(default_factory=list, max_length=64)
    system: str | None = Field(default=None, max_length=8_000)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    plan: BatchPlan
    capture_module_ids: list[str] = Field(default_factory=list, max_length=128)
    persist_activations: bool | None = None

    @model_validator(mode="after")
    def _one_prompt_source(self) -> BatchRequest:
        if bool(self.suite_id) == bool(self.prompts):
            raise ValueError("provide exactly one of suite_id or prompts")
        return self


BatchState = Literal["queued", "running", "completed", "cancelled", "failed"]


class BatchCellSummary(BaseModel):
    """Aggregate of every run that shares module and strength."""

    kind: Literal["baseline", "steered", "control"]
    module_id: str | None = None
    strength: float | None = None
    runs: int
    words_mean: float
    distinct_3_mean: float
    refusal_rate: float
    checks_passed_rate: float | None = None
    lexicon_mean: dict[str, float] = Field(default_factory=dict)


class BatchRunRecord(BaseModel):
    index: int
    kind: Literal["baseline", "steered", "control"]
    sweep_index: int
    repeat_index: int
    prompt_id: str
    prompt: str
    run: BaselineResponse


class BatchJobResponse(BaseModel):
    job_id: str
    state: BatchState
    total: int
    completed: int
    current: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    plan: str
    warnings: list[str] = Field(default_factory=list)
    runs: list[BatchRunRecord] = Field(default_factory=list)
    cells: list[BatchCellSummary] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
