import type {
  ActivationResponse,
  Architecture,
  BaselineResponse,
  Diagnostics,
  ExperimentDetail,
  ExperimentSummary,
  GenerationSettings,
  ModuleDetail,
  ModuleNode,
  ModelStatus,
} from "../types";

export const DEFAULT_SETTINGS: GenerationSettings = {
  max_new_tokens: 128,
  temperature: 0,
  top_p: 1,
  seed: null,
  enable_thinking: false,
};

export function moduleNode(overrides: Partial<ModuleNode> & Pick<ModuleNode, "id" | "name">): ModuleNode {
  return {
    class_name: "Linear",
    parameter_count: 1000,
    child_count: 0,
    layer_type: null,
    supports_steering: false,
    steering_reason: null,
    children: [],
    ...overrides,
  };
}

export function architectureFixture(): Architecture {
  const ffn = moduleNode({
    id: "model.layers.0.feed_forward",
    name: "feed_forward",
    class_name: "Lfm3MLP",
    parameter_count: 31_457_280,
    supports_steering: true,
    steering_reason: "Output is [batch, tokens, hidden]; a single hidden dimension can be shifted.",
  });
  const conv = moduleNode({
    id: "model.layers.0.conv",
    name: "conv",
    class_name: "Lfm3ShortConv",
    parameter_count: 43_008,
    supports_steering: true,
    steering_reason: "Output is [batch, tokens, hidden]; a single hidden dimension can be shifted.",
  });
  const layer0 = moduleNode({
    id: "model.layers.0",
    name: "0",
    class_name: "Lfm3DecoderLayer",
    layer_type: "conv",
    parameter_count: 31_500_288,
    supports_steering: true,
    steering_reason: "Output is [batch, tokens, hidden]; a single hidden dimension can be shifted.",
    children: [conv, ffn],
  });
  const layers = moduleNode({
    id: "model.layers",
    name: "layers",
    class_name: "ModuleList",
    parameter_count: 31_500_288,
    children: [layer0],
  });
  const model = moduleNode({
    id: "model",
    name: "model",
    class_name: "Lfm3Model",
    parameter_count: 1_700_000_000,
    children: [layers],
  });
  const root = moduleNode({
    id: "root",
    name: "Lfm3ForCausalLM",
    class_name: "Lfm3ForCausalLM",
    parameter_count: 1_700_000_000,
    children: [model],
  });

  return {
    model_id: "LiquidAI/LFM3-1.7B",
    revision: "main",
    model_class: "Lfm3ForCausalLM",
    num_hidden_layers: 24,
    hidden_size: 2048,
    vocab_size: 128_000,
    layer_types: ["conv"],
    tree: root,
    layer_overview: [
      {
        layer_index: 0,
        module_id: "model.layers.0",
        layer_type: "conv",
        operator_type: "conv",
        operator_module_id: "model.layers.0.conv",
        operator_class: "Lfm3ShortConv",
        ffn_module_id: "model.layers.0.feed_forward",
        parameter_count: 31_500_288,
      },
      {
        layer_index: 2,
        module_id: "model.layers.2",
        layer_type: "full_attention",
        operator_type: "attention",
        operator_module_id: "model.layers.2.self_attn",
        operator_class: "Lfm3Attention",
        ffn_module_id: "model.layers.2.feed_forward",
        parameter_count: 36_000_000,
      },
    ],
  };
}

export function moduleDetailFixture(overrides: Partial<ModuleDetail> = {}): ModuleDetail {
  return {
    id: "model.layers.0.feed_forward",
    name: "feed_forward",
    class_name: "Lfm3MLP",
    parent_id: "model.layers.0",
    parameter_count: 31_457_280,
    direct_parameter_count: 31_457_280,
    layer_type: "conv",
    supports_steering: true,
    steering_reason: "Output is [batch, tokens, hidden]; a single hidden dimension can be shifted.",
    child_ids: ["model.layers.0.feed_forward.w1", "model.layers.0.feed_forward.w2"],
    parameters: [
      {
        name: "w2.weight",
        shape: [2048, 7680],
        dtype: "torch.bfloat16",
        num_parameters: 15_728_640,
      },
    ],
    captured: false,
    ...overrides,
  };
}

export function baselineFixture(overrides: Partial<BaselineResponse> = {}): BaselineResponse {
  return {
    id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    timestamp: "2026-11-03T10:00:00Z",
    condition: "baseline",
    model_id: "LiquidAI/LFM3-1.7B",
    revision: "main",
    device: "cuda",
    prompt: "Hello",
    system: null,
    output: "A transformer activation is a hidden-state vector.",
    prompt_tokens: 3,
    generated_tokens: 12,
    duration_ms: 1234.5,
    tokens_per_second: 9.7,
    settings: DEFAULT_SETTINGS,
    steering: null,
    captured_modules: [],
    warnings: [],
    ...overrides,
  };
}

export function activationFixture(overrides: Partial<ActivationResponse> = {}): ActivationResponse {
  return {
    experiment_id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    module_id: "model.layers.0.feed_forward",
    shape: [4, 2048],
    dtype: "float32",
    aggregation: "All token positions of the prompt and generated tokens.",
    prompt_token_count: 2,
    tokens: [
      { index: 0, token_id: 1, text: "Hello", phase: "prompt" },
      { index: 1, token_id: 2, text: " world", phase: "prompt" },
      { index: 2, token_id: 3, text: " activated", phase: "generated" },
    ],
    heatmap: {
      positions: [0, 2],
      features: [0, 1],
      values: [
        [0.5, -0.25],
        [1.5, -1.75],
      ],
      sampling: "2 of 3 positions × 2 of 2048 features (evenly sampled)",
    },
    histogram: {
      bin_edges: [-2, -1, 0, 1, 2],
      counts: [1, 2, 3, 1],
    },
    stats: { minimum: -1.75, maximum: 1.5, mean: 0.0, std: 1.0, abs_mean: 1.1 },
    position: 2,
    top_dimensions: [
      { index: 7, value: 1.5, abs_value: 1.5 },
      { index: 42, value: -1.75, abs_value: 1.75 },
    ],
    feature_series: {
      feature: 7,
      positions: [0, 1, 2],
      values: [0.1, 0.4, 1.5],
    },
    comparison: null,
    warnings: [],
    ...overrides,
  };
}

export function summaryFixture(overrides: Partial<ExperimentSummary> = {}): ExperimentSummary {
  return {
    id: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    timestamp: "2026-11-03T10:00:00Z",
    model_id: "LiquidAI/LFM3-1.7B",
    revision: "main",
    resolved_revision: "abc123",
    device: "cuda",
    condition: "baseline",
    condition_detail: "none",
    sweep_id: null,
    sweep_index: null,
    repeat_index: null,
    module_id: null,
    feature_index: null,
    strength: null,
    prompt_preview: "Hello",
    output_preview: "A transformer activation is a hidden-state vector.",
    has_activations: false,
    activation_source: null,
    duration_ms: 1234.5,
    tokens_per_second: 9.7,
    ...overrides,
  };
}

export function detailFixture(overrides: Partial<ExperimentDetail> = {}): ExperimentDetail {
  return {
    ...summaryFixture(),
    system: null,
    prompt: "Hello",
    output: "A transformer activation is a hidden-state vector.",
    prompt_tokens: 3,
    generated_tokens: 12,
    settings: DEFAULT_SETTINGS,
    steering: null,
    captured_modules: [],
    captures: [],
    warnings: [],
    environment: {
      python_version: "3.13.15",
      torch_version: "2.14.1+rocm7.2",
      transformers_version: "5.21.0",
      fastapi_version: "0.115.0",
      app_version: "0.1.0",
      platform: "Linux",
      git_commit: "abcdef1",
    },
    tokens: [],
    ...overrides,
  };
}

export const loadedStatus: ModelStatus = {
  state: "loaded",
  model_id: "LiquidAI/LFM3-1.7B",
  revision: "main",
  device: "cuda",
  dtype: "bfloat16",
  loaded: true,
  error: null,
};

export const unloadedStatus: ModelStatus = {
  state: "unloaded",
  model_id: "LiquidAI/LFM3-1.7B",
  revision: "main",
  device: null,
  dtype: null,
  loaded: false,
  error: null,
};

export const diagnosticsFixture: Diagnostics = {
  python_version: "3.13.15",
  torch_version: "2.14.1+rocm7.2",
  transformers_version: "5.21.0",
  fastapi_version: "0.115.0",
  cuda_available: true,
  hip_version: "7.2",
  backend: "rocm",
  device_count: 1,
  device_name: "AMD Radeon RX 7600",
  inference_device: "cuda",
  model_state: "loaded",
  memory: null,
  notes: [],
};
