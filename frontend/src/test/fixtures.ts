import type {
  ActivationResponse,
  Architecture,
  BaselineResponse,
  Diagnostics,
  ExperimentDetail,
  ExperimentSummary,
  FlowResponse,
  GenerationSettings,
  NextTokenResponse,
  NeuronSteeringResponse,
  NeuronTraceResponse,
  PatchSweepResponse,
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
    metrics: null,
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
    feature_stats: null,
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
    steering_summary: null,
    metrics: null,
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

function flowClusters(layerIndex: number) {
  const base = layerIndex * 10;
  return [
    {
      id: 0,
      size: 3,
      mass: 0.6,
      cohesion: 0.8,
      peak_position: 2,
      direction_norm: 1.4,
      top_dimensions: [
        { index: base, weight: 1, mean_abs: 0.5, sigma: 0.12 },
        { index: base + 5, weight: 0.5, mean_abs: 0.4, sigma: 0.09 },
      ],
      profile: [0.1, 0.2, 0.3, 0.2],
      profile_positions: [0, 1, 2, 3],
      components: [
        { feature_index: base, weight: 1 },
        { feature_index: base + 5, weight: 0.5 },
      ],
    },
    {
      id: 1,
      size: 2,
      mass: 0.4,
      cohesion: 0.7,
      peak_position: 3,
      direction_norm: 1.1,
      top_dimensions: [
        { index: base + 9, weight: -0.8, mean_abs: 0.3, sigma: 0.05 },
      ],
      profile: [-0.1, -0.2, -0.1, -0.3],
      profile_positions: [0, 1, 2, 3],
      components: [{ feature_index: base + 9, weight: -0.8 }],
    },
  ];
}

export function flowFixture(overrides: Partial<FlowResponse> = {}): FlowResponse {
  return {
    experiment_id: "d".repeat(32),
    prompt: "Explain activations.",
    system: null,
    output: "An activation is a hidden-state vector.",
    prompt_tokens: 3,
    generated_tokens: 1,
    tokens: [
      { index: 0, token_id: 10, text: "The", phase: "prompt" },
      { index: 1, token_id: 11, text: " sky", phase: "prompt" },
      { index: 2, token_id: 12, text: " is", phase: "prompt" },
      { index: 3, token_id: 13, text: " blue", phase: "generated" },
    ],
    layers: [0, 1, 2].map((layerIndex) => ({
      layer_index: layerIndex,
      module_id: `model.layers.${layerIndex}`,
      layer_type: layerIndex === 0 ? "conv" : "full_attention",
      operator_type: layerIndex === 0 ? ("conv" as const) : ("attention" as const),
      mean_norm: 2.5 + layerIndex,
      delta_norm: 0.8 + layerIndex * 0.2,
      write: layerIndex === 0 ? 0 : 0.42 + layerIndex * 0.1,
      similarity: layerIndex === 0 ? 1 : 0.9 - layerIndex * 0.03,
      gain: layerIndex === 0 ? 1 : 1.1 + layerIndex * 0.05,
      energy: [1, 2, 3, 4].map((value) => value + layerIndex),
      delta: [0.5, 1, 1.5, 2].map((value) => value + layerIndex * 0.2),
      clusters: flowClusters(layerIndex),
      ablation_kl: 0.2 + layerIndex * 0.1,
    })),
    edges: [
      { source_layer: 0, target_layer: 1, write: 0.42, similarity: 0.9, gain: 1.1 },
      { source_layer: 1, target_layer: 2, write: 0.52, similarity: 0.87, gain: 1.15 },
    ],
    global_max_norm: 6,
    cluster_k: 2,
    cluster_positions: 4,
    settings: DEFAULT_SETTINGS,
    warnings: [],
    ...overrides,
  };
}


export function nextTokenFixture(overrides: Partial<NextTokenResponse> = {}): NextTokenResponse {
  const candidates = [
    {
      token_id: 100,
      text: "A",
      baseline_probability: 0.42,
      steered_probability: 0.31,
      delta: -0.11,
    },
    {
      token_id: 101,
      text: "The",
      baseline_probability: 0.21,
      steered_probability: 0.38,
      delta: 0.17,
    },
    {
      token_id: 102,
      text: "This",
      baseline_probability: 0.12,
      steered_probability: 0.08,
      delta: -0.04,
    },
  ];
  return {
    prompt: "Hello",
    system: null,
    model_id: "LiquidAI/LFM3-1.7B",
    device: "cuda",
    prompt_tokens: 12,
    position: 11,
    top_k: 3,
    baseline: {
      tokens: candidates.map((candidate) => ({
        token_id: candidate.token_id,
        text: candidate.text,
        probability: candidate.baseline_probability,
      })),
      entropy: 1.2,
    },
    steered: {
      tokens: [
        { token_id: 101, text: "The", probability: 0.38 },
        { token_id: 100, text: "A", probability: 0.31 },
        { token_id: 102, text: "This", probability: 0.08 },
      ],
      entropy: 1.1,
    },
    candidates,
    kl_divergence: 0.23,
    steering: {
      module_id: "model.layers.12",
      strength: 1,
      feature_index: null,
      components: [{ feature_index: 1, weight: 1 }],
      normalization: "sigma",
    },
    duration_ms: 12.3,
    warnings: [],
    ...overrides,
  };
}


export function patchSweepFixture(overrides: Partial<PatchSweepResponse> = {}): PatchSweepResponse {
  const cells = [
    {
      layer_index: 0,
      module_id: "model.layers.0",
      position: 10,
      donor_position: 10,
      patched_output: "The baseline answer.",
      changed: false,
      first_token_changed: false,
      common_prefix_tokens: 4,
      generated_tokens: 4,
      warnings: [],
    },
    {
      layer_index: 0,
      module_id: "model.layers.0",
      position: 11,
      donor_position: 11,
      patched_output: "A different answer.",
      changed: true,
      first_token_changed: false,
      common_prefix_tokens: 1,
      generated_tokens: 3,
      warnings: [],
    },
    {
      layer_index: 1,
      module_id: "model.layers.1",
      position: 10,
      donor_position: 10,
      patched_output: "Another answer.",
      changed: true,
      first_token_changed: true,
      common_prefix_tokens: 0,
      generated_tokens: 2,
      warnings: [],
    },
    {
      layer_index: 1,
      module_id: "model.layers.1",
      position: 11,
      donor_position: 11,
      patched_output: "The baseline answer.",
      changed: false,
      first_token_changed: false,
      common_prefix_tokens: 4,
      generated_tokens: 4,
      warnings: [],
    },
  ];
  return {
    recipient_prompt: "Hello",
    donor_prompt: "World",
    baseline_output: "The baseline answer.",
    donor_output: "A different answer.",
    prompt_tokens: 12,
    donor_prompt_tokens: 12,
    layers: [0, 1],
    positions: [10, 11],
    cells,
    duration_ms: 1234,
    warnings: [],
    ...overrides,
  };
}


export function neuronTraceFixture(
  overrides: Partial<NeuronTraceResponse> = {},
): NeuronTraceResponse {
  const assignments = Array.from({ length: 256 }, (_, index) => (index < 160 ? 0 : 1));
  const clusters = [
    {
      id: 0,
      size: 160,
      mass: 0.62,
      cohesion: 0.71,
      peak_position: 2,
      top_neurons: [
        { neuron: 7, weight: 1, mean_abs: 0.9, sigma: 0.2 },
        { neuron: 31, weight: 0.6, mean_abs: 0.7, sigma: 0.15 },
      ],
      profile: [0.1, 0.2, 0.3, 0.2],
      profile_positions: [0, 1, 2, 3],
    },
    {
      id: 1,
      size: 96,
      mass: 0.38,
      cohesion: 0.55,
      peak_position: 1,
      top_neurons: [
        { neuron: 200, weight: -0.8, mean_abs: 0.5, sigma: 0.1 },
      ],
      profile: [-0.1, -0.2, -0.1, -0.15],
      profile_positions: [0, 1, 2, 3],
    },
  ];
  return {
    experiment_id: "e".repeat(32),
    prompt: "Hello",
    system: null,
    output: "The baseline answer.",
    tokens: [
      { index: 0, token_id: 1, text: "Hello", phase: "prompt" },
      { index: 1, token_id: 2, text: " world", phase: "prompt" },
      { index: 2, token_id: 3, text: " answer", phase: "generated" },
    ],
    prompt_tokens: 2,
    generated_tokens: 1,
    layer_index: 1,
    layer_type: "full_attention",
    intermediate_size: 256,
    hidden_size: 128,
    residual_rms: 2.3,
    clusters,
    top_neurons: clusters[0].top_neurons,
    assignments,
    cluster_positions: 4,
    warnings: [],
    ...overrides,
  };
}

export function neuronSteeringFixture(
  overrides: Partial<NeuronSteeringResponse> = {},
): NeuronSteeringResponse {
  return {
    layer_index: 1,
    alpha: 0.7,
    neurons: [
      { neuron: 7, weight: 1 },
      { neuron: 31, weight: 0.6 },
    ],
    direction_norm: 1.2,
    residual_rms: 2.3,
    deltas: [
      { neuron: 7, delta: 1.34 },
      { neuron: 31, delta: 0.8 },
    ],
    baseline: baselineFixture({
      id: "a".repeat(32),
      output: "The baseline answer.",
    }),
    steered: baselineFixture({
      id: "b".repeat(32),
      condition: "steered",
      output: "A steered answer.",
    }),
    duration_ms: 120,
    warnings: [],
    ...overrides,
  };
}
