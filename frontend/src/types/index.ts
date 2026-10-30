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

export interface BaselineResponse {
  id: string;
  timestamp: string;
  condition: "baseline";
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
  warnings: string[];
}
