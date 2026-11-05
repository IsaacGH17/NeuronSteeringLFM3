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

export interface SteeringComponent {
  feature_index: number;
  weight: number;
}

export type SteeringNormalization = "raw" | "sigma";

export interface SteeringSpec {
  module_id: string;
  strength: number;
  feature_index: number | null;
  components?: SteeringComponent[];
  normalization?: SteeringNormalization;
}

export interface FeatureStats {
  mean: number[];
  std: number[];
  minimum: number[];
  maximum: number[];
}

export interface MetricCheck {
  name: string;
  kind: string;
  expected: string | number | null;
  passed: boolean;
  detail: string | null;
}

export interface TextMetrics {
  words: number;
  characters: number;
  sentences: number;
  avg_words_per_sentence: number;
  distinct_3: number;
  repeated_4gram_ratio: number;
  refusal: boolean;
  lexicon: Record<string, number>;
  checks: MetricCheck[];
  checks_passed: number | null;
  checks_total: number | null;
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
  metrics: TextMetrics | null;
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
  steering_summary: string | null;
  metrics: TextMetrics | null;
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
  feature_stats: FeatureStats | null;
  comparison: string | null;
  warnings: string[];
}

export interface SuiteCheck {
  kind: string;
  value: string | number | null;
}

export interface SuitePrompt {
  id: string;
  text: string;
  tags: string[];
  checks: SuiteCheck[];
}

export interface PromptSuite {
  id: string;
  name: string;
  description: string;
  kind: "trait" | "capability";
  prompts: SuitePrompt[];
}

export interface ModulePreset {
  id: string;
  label: string;
  modules: { module_id: string; layer_index: number }[];
}

export interface DirectionRequest {
  positive_experiment_id: string;
  negative_experiment_id: string;
  module_id?: string | null;
  position_mode?: "mean_prompt" | "mean_all" | "last_prompt";
  top_k?: number;
  normalize?: boolean;
}

export interface DirectionResponse {
  module_id: string;
  components: SteeringComponent[];
  sigma: number[];
  norm: number;
  separation: number;
  positions_compared: number;
  warnings: string[];
}

export interface BatchPlan {
  module_ids: string[];
  components: SteeringComponent[];
  normalization: SteeringNormalization;
  strengths: number[];
  include_control: boolean;
  control_seed: number | null;
  repeats: number;
  baseline_per_prompt: boolean;
}

export interface BatchRequest {
  suite_id: string | null;
  prompts: string[];
  system: string | null;
  settings: GenerationSettings;
  plan: BatchPlan;
  capture_module_ids: string[];
  persist_activations: boolean | null;
}

export type BatchState = "queued" | "running" | "completed" | "cancelled" | "failed";

export interface BatchCellSummary {
  kind: "baseline" | "steered" | "control";
  module_id: string | null;
  strength: number | null;
  runs: number;
  words_mean: number;
  distinct_3_mean: number;
  refusal_rate: number;
  checks_passed_rate: number | null;
  lexicon_mean: Record<string, number>;
}

export interface BatchRunRecord {
  index: number;
  kind: "baseline" | "steered" | "control";
  sweep_index: number;
  repeat_index: number;
  prompt_id: string;
  prompt: string;
  run: BaselineResponse;
}

export interface BatchJob {
  job_id: string;
  state: BatchState;
  total: number;
  completed: number;
  current: string | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  plan: string;
  warnings: string[];
  runs: BatchRunRecord[];
  cells: BatchCellSummary[];
}

export interface LensToken {
  token_id: number;
  text: string;
  probability: number;
}

export interface LensLayer {
  layer_index: number;
  module_id: string;
  top1_ids: number[];
  top1_probs: number[];
  top_tokens: LensToken[];
  ablation_kl: number | null;
  ablation_top1: LensToken | null;
}

export interface LensRequest {
  prompt: string;
  system?: string | null;
  settings: GenerationSettings;
  position?: number | null;
  top_k?: number;
}

export interface LensResponse {
  prompt: string;
  system: string | null;
  output: string;
  prompt_tokens: number;
  position: number;
  tokens: TokenInfo[];
  layers: LensLayer[];
  vocab: Record<string, string>;
  warnings: string[];
}

export interface PatchRequest {
  recipient_prompt: string;
  recipient_system?: string | null;
  donor_prompt: string;
  donor_system?: string | null;
  module_id: string;
  position?: number;
  settings: GenerationSettings;
}

export interface PatchResponse {
  module_id: string;
  position: number;
  donor_output: string;
  baseline_output: string;
  patched_output: string;
  changed: boolean;
  prompt_tokens: number;
  warnings: string[];
}

export interface FlowDimension {
  index: number;
  weight: number;
  mean_abs: number;
  sigma: number;
}

export interface FlowCluster {
  id: number;
  size: number;
  mass: number;
  cohesion: number;
  peak_position: number;
  direction_norm: number;
  top_dimensions: FlowDimension[];
  profile: number[];
  profile_positions: number[];
  components: SteeringComponent[];
}

export interface FlowLayer {
  layer_index: number;
  module_id: string;
  layer_type: string;
  operator_type: "attention" | "conv";
  mean_norm: number;
  delta_norm: number;
  write: number;
  similarity: number;
  gain: number;
  energy: number[];
  delta: number[];
  clusters: FlowCluster[];
  ablation_kl: number | null;
}

export interface FlowEdge {
  source_layer: number;
  target_layer: number;
  write: number;
  similarity: number;
  gain: number;
}

export interface FlowRequest {
  prompt: string;
  system?: string | null;
  settings: GenerationSettings;
  cluster_k?: number;
  cluster_max_positions?: number;
  include_ablation?: boolean;
}

export interface FlowResponse {
  experiment_id: string;
  prompt: string;
  system: string | null;
  output: string;
  prompt_tokens: number;
  generated_tokens: number;
  tokens: TokenInfo[];
  layers: FlowLayer[];
  edges: FlowEdge[];
  global_max_norm: number;
  cluster_k: number;
  cluster_positions: number;
  settings: GenerationSettings;
  warnings: string[];
}

export interface TokenProbability {
  token_id: number;
  text: string;
  probability: number;
}

export interface NextTokenDistribution {
  tokens: TokenProbability[];
  entropy: number;
}

export interface NextTokenCandidate {
  token_id: number;
  text: string;
  baseline_probability: number;
  steered_probability: number | null;
  delta: number | null;
}

export interface NextTokenRequest {
  prompt: string;
  system?: string | null;
  enable_thinking?: boolean;
  top_k?: number;
  steering?: SteeringSpec | null;
}

export interface NextTokenResponse {
  prompt: string;
  system: string | null;
  model_id: string;
  device: string;
  prompt_tokens: number;
  position: number;
  top_k: number;
  baseline: NextTokenDistribution;
  steered: NextTokenDistribution | null;
  candidates: NextTokenCandidate[];
  kl_divergence: number | null;
  steering: SteeringSpec | null;
  duration_ms: number;
  warnings: string[];
}

export interface ProbeSeed {
  prompt: string;
  system: string | null;
  steering: SteeringSpec | null;
}

export interface PatchSweepRequest {
  recipient_prompt: string;
  recipient_system?: string | null;
  donor_prompt: string;
  donor_system?: string | null;
  settings: GenerationSettings;
  layer_indices?: number[];
  positions?: number[];
}

export interface PatchSweepCell {
  layer_index: number;
  module_id: string;
  position: number;
  donor_position: number;
  patched_output: string;
  changed: boolean;
  first_token_changed: boolean;
  common_prefix_tokens: number;
  generated_tokens: number;
  warnings: string[];
}

export interface PatchSweepResponse {
  recipient_prompt: string;
  donor_prompt: string;
  baseline_output: string;
  donor_output: string;
  prompt_tokens: number;
  donor_prompt_tokens: number;
  layers: number[];
  positions: number[];
  cells: PatchSweepCell[];
  duration_ms: number;
  warnings: string[];
}

export interface NeuronTop {
  neuron: number;
  weight: number;
  mean_abs: number;
  sigma: number;
}

export interface NeuronCluster {
  id: number;
  size: number;
  mass: number;
  cohesion: number;
  peak_position: number;
  top_neurons: NeuronTop[];
  profile: number[];
  profile_positions: number[];
}

export interface NeuronTraceRequest {
  prompt: string;
  system?: string | null;
  settings: GenerationSettings;
  layer_index?: number;
  cluster_k?: number;
  cluster_max_positions?: number;
}

export interface NeuronTraceResponse {
  experiment_id: string;
  prompt: string;
  system: string | null;
  output: string;
  tokens: TokenInfo[];
  prompt_tokens: number;
  generated_tokens: number;
  layer_index: number;
  layer_type: string;
  intermediate_size: number;
  hidden_size: number;
  residual_rms: number;
  clusters: NeuronCluster[];
  top_neurons: NeuronTop[];
  assignments: number[];
  cluster_positions: number;
  warnings: string[];
}

export interface NeuronComponent {
  neuron: number;
  weight: number;
}

export interface NeuronDelta {
  neuron: number;
  delta: number;
}

export interface NeuronSteeringRequest {
  prompt: string;
  system?: string | null;
  settings: GenerationSettings;
  layer_index: number;
  neurons: NeuronComponent[];
  alpha: number;
}

export interface NeuronSteeringResponse {
  layer_index: number;
  alpha: number;
  neurons: NeuronComponent[];
  direction_norm: number;
  residual_rms: number;
  deltas: NeuronDelta[];
  baseline: BaselineResponse;
  steered: BaselineResponse;
  duration_ms: number;
  warnings: string[];
}
