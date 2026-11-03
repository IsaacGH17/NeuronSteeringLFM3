export type ModelState = "unloaded" | "loading" | "loaded" | "error";

export interface ModelStatus {
  state: ModelState;
  model_id: string;
  revision: string;
  device: string | null;
  dtype: string | null;
  loaded: boolean;
  error: string | null;
}

export interface MemoryStats {
  device: string;
  allocated_bytes: number | null;
  reserved_bytes: number | null;
  total_bytes: number | null;
  free_bytes: number | null;
}

export interface Diagnostics {
  python_version: string;
  torch_version: string;
  transformers_version: string;
  fastapi_version: string;
  cuda_available: boolean;
  hip_version: string | null;
  backend: "rocm" | "cuda" | "cpu";
  device_count: number;
  device_name: string | null;
  inference_device: string | null;
  model_state: ModelState;
  memory: MemoryStats | null;
  notes: string[];
}

export interface ModuleNode {
  id: string;
  name: string;
  class_name: string;
  parameter_count: number;
  child_count: number;
  layer_type: string | null;
  supports_steering: boolean;
  steering_reason: string | null;
  children: ModuleNode[];
}

export interface LayerBlock {
  layer_index: number;
  module_id: string;
  layer_type: string;
  operator_type: "attention" | "conv";
  operator_module_id: string;
  operator_class: string;
  ffn_module_id: string;
  parameter_count: number;
}

export interface Architecture {
  model_id: string;
  revision: string;
  model_class: string;
  num_hidden_layers: number;
  hidden_size: number;
  vocab_size: number;
  layer_types: string[];
  tree: ModuleNode;
  layer_overview: LayerBlock[];
}

export interface ParameterInfo {
  name: string;
  shape: number[];
  dtype: string;
  num_parameters: number;
}

export interface ModuleDetail {
  id: string;
  name: string;
  class_name: string;
  parent_id: string | null;
  parameter_count: number;
  direct_parameter_count: number;
  layer_type: string | null;
  supports_steering: boolean;
  steering_reason: string | null;
  child_ids: string[];
  parameters: ParameterInfo[];
  captured: boolean;
}

export interface GenerationSettings {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  seed: number | null;
  enable_thinking: boolean;
}

export interface SteeringSpec {
  module_id: string;
  feature_index: number;
  strength: number;
}

export type ExperimentCondition = "baseline" | "steered" | "control";
export type ExperimentConditionDetail = "none" | "positive" | "negative" | "zero" | "control";

export interface BaselineResponse {
  id: string;
  timestamp: string;
  condition: ExperimentCondition;
  model_id: string;
  revision: string;
  device: string;
  prompt: string;
  system: string | null;
  output: string;
  prompt_tokens: number;
  generated_tokens: number;
  duration_ms: number;
  tokens_per_second: number | null;
  settings: GenerationSettings;
  steering: SteeringSpec | null;
  captured_modules: string[];
  warnings: string[];
}

export interface SteeringResponse {
  baseline: BaselineResponse;
  steered: BaselineResponse;
}

export interface EnvironmentInfo {
  python_version: string;
  torch_version: string;
  transformers_version: string;
  fastapi_version: string;
  app_version: string;
  platform: string;
  git_commit: string;
}

export interface ExperimentSummary {
  id: string;
  timestamp: string;
  model_id: string;
  revision: string;
  resolved_revision: string;
  device: string;
  condition: ExperimentCondition;
  condition_detail: ExperimentConditionDetail;
  sweep_id: string | null;
  sweep_index: number | null;
  repeat_index: number | null;
  module_id: string | null;
  feature_index: number | null;
  strength: number | null;
  prompt_preview: string;
  output_preview: string;
  has_activations: boolean;
  activation_source: "live" | "snapshot" | null;
  duration_ms: number;
  tokens_per_second: number | null;
}

export interface CaptureInfo {
  module_id: string;
  shape: number[];
  dtype: string;
}

export interface ExperimentDetail extends ExperimentSummary {
  system: string | null;
  prompt: string;
  output: string;
  prompt_tokens: number;
  generated_tokens: number;
  settings: GenerationSettings;
  steering: SteeringSpec | null;
  captured_modules: string[];
  captures: CaptureInfo[];
  warnings: string[];
  environment: EnvironmentInfo;
  tokens: TokenInfo[];
}

export interface ExperimentListResponse {
  total: number;
  offset: number;
  limit: number;
  items: ExperimentSummary[];
}

export interface SweepRun {
  kind: "baseline" | "strength" | "control";
  sweep_index: number;
  repeat_index: number;
  run: BaselineResponse;
}

export interface SweepResponse {
  sweep_id: string;
  runs: SweepRun[];
  warnings: string[];
}

export interface TokenInfo {
  index: number;
  token_id: number;
  text: string;
  phase: "prompt" | "generated";
}

export interface ActivationSummary {
  module_id: string;
  shape: number[];
  dtype: string;
}

export interface HeatmapData {
  positions: number[];
  features: number[];
  values: number[][];
  sampling: string;
}

export interface HistogramData {
  bin_edges: number[];
  counts: number[];
}

export interface ActivationStats {
  minimum: number;
  maximum: number;
  mean: number;
  std: number;
  abs_mean: number;
}

export interface TopDimension {
  index: number;
  value: number;
  abs_value: number;
}

export interface FeatureSeries {
  feature: number;
  positions: number[];
  values: number[];
}

export interface ActivationResponse {
  experiment_id: string;
  module_id: string;
  shape: number[];
  dtype: string;
  aggregation: string;
  prompt_token_count: number;
  tokens: TokenInfo[];
  heatmap: HeatmapData;
  histogram: HistogramData;
  stats: ActivationStats;
  position: number;
  top_dimensions: TopDimension[];
  feature_series: FeatureSeries;
  comparison: string | null;
  warnings: string[];
}
