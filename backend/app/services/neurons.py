"""MLP-neuron inspection and group steering through the w2 intermediate.

LFM3's MLP intermediate is the input of ``feed_forward.w2`` (7680 dimensions for
LFM3-1.7B). A neuron is perturbed by adding a scalar to that intermediate; its
downstream contribution is the neuron's column of ``w2``. Because a sandwich
RMSNorm follows the MLP, the real per-hidden-dimension gain is
``w2[:, i] * post_ffn_norm.weight``.

The trace captures the intermediate for one layer during a real generation and
clusters neurons by their z-scored activation profiles. Group steering computes

    direction = sum_i w_i * (w2[:, i] * post_weight)
    delta_i   = alpha * residual_rms / ||direction|| * w_i

so ``alpha`` is a target displacement in residual-RMS units. Hooks are
out-of-place and removed in ``finally``; weights never change.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from backend.app.schemas.api import (
    BaselineRequest,
    NeuronCluster,
    NeuronComponent,
    NeuronDelta,
    NeuronSteeringRequest,
    NeuronSteeringResponse,
    NeuronTop,
    NeuronTraceRequest,
    NeuronTraceResponse,
)
from backend.app.services.activation_capture import capture_modules, finalize_captures
from backend.app.services.architecture_inspector import build_layer_overview, get_module
from backend.app.services.generation import generate_baseline, generate_chat
from backend.app.services.model_loader import ModelManager

MAX_NEURON_COMPONENTS = 64
MAX_CLUSTER_NEURONS = 12
PROFILE_POINTS = 48
CLUSTER_ITERATIONS = 8
_EPS = 1e-12


def _even_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.linspace(0, size - 1, num=limit).round().astype(np.int64))


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    norms[norms < _EPS] = 1.0
    return matrix / norms


def _rounded(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in np.nan_to_num(values, nan=0.0)]


@contextmanager
def capture_intermediate(model: Any, layer_index: int) -> Iterator[list[torch.Tensor]]:
    """Capture the input of ``model.layers.<i>.feed_forward.w2`` per forward."""
    chunks: list[torch.Tensor] = []
    module = get_module(model, f"model.layers.{layer_index}.feed_forward.w2")

    def hook(module: Any, args: tuple[Any, ...]) -> None:
        if not args:
            return
        tensor = args[0]
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            chunks.append(tensor.detach().to(device="cpu", dtype=torch.float32, copy=True))

    handle = module.register_forward_pre_hook(hook)
    try:
        yield chunks
    finally:
        handle.remove()


def _finalize_intermediate(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise ValueError("The w2 intermediate was never captured.")
    tensor = torch.cat(chunks, dim=1)
    if tensor.shape[0] != 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    return tensor.squeeze(0)


def _cluster_neurons(
    array: np.ndarray,
    *,
    k: int,
    max_positions: int,
) -> tuple[list[NeuronCluster], list[NeuronTop], int, list[int]]:
    positions = _even_indices(int(array.shape[0]), max_positions)
    sampled = np.asarray(array[positions], dtype=np.float32)
    mean_abs = np.mean(np.abs(sampled), axis=0)
    mean_signed = np.mean(sampled, axis=0)
    std = np.std(sampled, axis=0)
    centered = sampled - sampled.mean(axis=0)
    spread = sampled.std(axis=0)
    spread[spread < 1e-8] = 1.0
    profiles = _unit_columns(centered / spread)
    cluster_count = max(1, min(int(k), profiles.shape[1]))
    seeds = np.argsort(-mean_abs, kind="stable")[:cluster_count]
    centroids = profiles[:, seeds].T.copy()
    assignment = np.zeros(profiles.shape[1], dtype=np.int64)
    for _ in range(CLUSTER_ITERATIONS):
        scores = profiles.T @ centroids.T
        assignment = np.argmax(scores, axis=1)
        for cluster_id in range(cluster_count):
            members = assignment == cluster_id
            if not np.any(members):
                continue
            updated = profiles[:, members].mean(axis=1)
            norm = float(np.linalg.norm(updated))
            if norm > _EPS:
                centroids[cluster_id] = updated / norm
    scores = profiles.T @ centroids.T
    assignment = np.argmax(scores, axis=1)

    total_mass = float(np.sum(mean_abs)) or 1.0
    clusters: list[NeuronCluster] = []
    for cluster_id in range(cluster_count):
        members = np.flatnonzero(assignment == cluster_id)
        if members.size == 0:
            continue
        by_mass = members[np.argsort(-mean_abs[members], kind="stable")]
        top = by_mass[:MAX_CLUSTER_NEURONS]
        signed = mean_signed[top]
        scale = float(np.max(np.abs(signed))) if top.size else 0.0
        if scale <= 1e-8:
            magnitude = np.abs(mean_abs[top])
            largest = float(np.max(magnitude)) if top.size else 0.0
            weights = magnitude / largest if largest > _EPS else np.ones_like(magnitude)
        else:
            weights = signed / scale
        profile = np.mean(sampled[:, members], axis=1)
        profile_indices = _even_indices(profile.size, PROFILE_POINTS)
        peak_local = int(np.argmax(np.abs(profile)))
        clusters.append(
            NeuronCluster(
                id=0,
                size=int(members.size),
                mass=round(float(np.sum(mean_abs[members]) / total_mass), 6),
                cohesion=round(float(np.mean(scores[members, cluster_id])), 6),
                peak_position=int(positions[peak_local]),
                top_neurons=[
                    NeuronTop(
                        neuron=int(index),
                        weight=round(float(weight), 6),
                        mean_abs=round(float(mean_abs[index]), 6),
                        sigma=round(float(std[index]), 6),
                    )
                    for index, weight in zip(top, weights, strict=True)
                ],
                profile=_rounded(profile[profile_indices]),
                profile_positions=[int(positions[index]) for index in profile_indices],
            )
        )
    clusters.sort(key=lambda cluster: (-cluster.mass, cluster.id))
    for index, cluster in enumerate(clusters):
        cluster.id = index

    global_top = np.argsort(-mean_abs, kind="stable")[:20]
    global_signed = mean_signed[global_top]
    global_scale = float(np.max(np.abs(global_signed))) if global_top.size else 0.0
    if global_scale <= 1e-8:
        global_weights = np.abs(mean_abs[global_top])
        largest = float(np.max(global_weights)) if global_top.size else 0.0
        global_weights = global_weights / largest if largest > _EPS else np.ones_like(global_weights)
    else:
        global_weights = global_signed / global_scale
    top_neurons = [
        NeuronTop(
            neuron=int(index),
            weight=round(float(weight), 6),
            mean_abs=round(float(mean_abs[index]), 6),
            sigma=round(float(std[index]), 6),
        )
        for index, weight in zip(global_top, global_weights, strict=True)
    ]
    return clusters, top_neurons, int(positions.size), assignment.tolist()


def _layer_meta(model: Any, layer_index: int) -> tuple[str, str]:
    for block in build_layer_overview(model):
        if block.layer_index == layer_index:
            return block.layer_type, block.module_id
    raise ValueError(f"Layer {layer_index} is outside the available decoder layers.")


def _residual_rms(layer_tensor: torch.Tensor, prompt_tokens: int, hidden_size: int) -> float:
    index = min(max(0, prompt_tokens - 1), layer_tensor.shape[0] - 1)
    return float(layer_tensor[index].float().norm() / math.sqrt(hidden_size))


def trace_neurons(manager: ModelManager, request: NeuronTraceRequest) -> tuple[Any, NeuronTraceResponse]:
    model = manager.model
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    intermediate_size = int(getattr(model.config, "intermediate_size", 0))
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("The loaded model does not expose hidden/intermediate sizes.")
    layer_type, layer_module_id = _layer_meta(model, request.layer_index)

    baseline_request = BaselineRequest(
        prompt=request.prompt,
        system=request.system,
        settings=request.settings,
    )
    warnings: list[str] = []
    with capture_modules(model, [layer_module_id]) as (layer_chunks, layer_warnings):
        with capture_intermediate(model, request.layer_index) as intermediate_chunks:
            result = generate_baseline(manager, baseline_request)
    layer_captures, layer_finalize_warnings = finalize_captures(layer_chunks)
    intermediate = _finalize_intermediate(intermediate_chunks).numpy()
    warnings.extend(layer_warnings)
    warnings.extend(layer_finalize_warnings)
    if layer_module_id not in layer_captures:
        raise ValueError(f"Layer {layer_module_id!r} did not produce an activation.")
    layer_tensor = layer_captures[layer_module_id]
    position_count = min(layer_tensor.shape[0], intermediate.shape[0])
    if layer_tensor.shape[0] != intermediate.shape[0]:
        warnings.append(
            f"Layer output covers {layer_tensor.shape[0]} positions and the w2 intermediate "
            f"{intermediate.shape[0]}; the trace uses the first {position_count}."
        )
    intermediate = intermediate[:position_count]
    layer_tensor = layer_tensor[:position_count]
    prompt_tokens = result.response.prompt_tokens
    residual_rms = _residual_rms(layer_tensor, prompt_tokens, hidden_size)
    clusters, top_neurons, sampled, assignments = _cluster_neurons(
        intermediate,
        k=request.cluster_k,
        max_positions=request.cluster_max_positions,
    )
    if sampled and sampled < position_count:
        warnings.append(
            f"Neuron clusters use {sampled} evenly sampled positions of {position_count}; "
            "the residual RMS uses the last prompt token."
        )
    response = NeuronTraceResponse(
        experiment_id=result.response.id,
        prompt=result.response.prompt,
        system=result.response.system,
        output=result.response.output,
        tokens=result.tokens,
        prompt_tokens=prompt_tokens,
        generated_tokens=result.response.generated_tokens,
        layer_index=request.layer_index,
        layer_type=layer_type,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        residual_rms=round(residual_rms, 6),
        clusters=clusters,
        top_neurons=top_neurons,
        assignments=assignments,
        cluster_positions=sampled,
        warnings=warnings,
    )
    return result, response


def _w2_and_post(model: Any, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    w2 = get_module(model, f"model.layers.{layer_index}.feed_forward.w2").weight.detach().float()
    try:
        post = get_module(model, f"model.layers.{layer_index}.post_ffn_norm").weight.detach().float()
    except KeyError:
        post = torch.ones(w2.shape[0], device=w2.device, dtype=torch.float32)
    return w2, post


def _direction(w2: torch.Tensor, post: torch.Tensor, components: list[NeuronComponent]) -> torch.Tensor:
    direction = torch.zeros(w2.shape[0], device=w2.device, dtype=torch.float32)
    for component in components:
        direction = direction + component.weight * (w2[:, component.neuron] * post)
    return direction


@contextmanager
def steer_neurons(model: Any, layer_index: int, deltas: dict[int, float]) -> Iterator[None]:
    module = get_module(model, f"model.layers.{layer_index}.feed_forward.w2")

    def hook(module: Any, args: tuple[Any, ...]) -> Any:
        if not args:
            return None
        tensor = args[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return None
        steered = tensor.to(torch.float32).clone()
        for neuron, delta in deltas.items():
            steered[..., neuron] = steered[..., neuron] + delta
        return (steered.to(tensor.dtype),)

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def steer_neuron_group(manager: ModelManager, request: NeuronSteeringRequest) -> NeuronSteeringResponse:
    model = manager.model
    if not request.neurons:
        raise ValueError("Neuron steering needs at least one neuron.")
    if len(request.neurons) > MAX_NEURON_COMPONENTS:
        raise ValueError(f"At most {MAX_NEURON_COMPONENTS} neurons can be steered at once.")
    intermediate_size = int(getattr(model.config, "intermediate_size", 0))
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    for component in request.neurons:
        if not 0 <= component.neuron < intermediate_size:
            raise ValueError(f"Neuron {component.neuron} is outside [0, {intermediate_size - 1}].")
    _layer_meta(model, request.layer_index)
    w2, post = _w2_and_post(model, request.layer_index)
    direction = _direction(w2, post, request.neurons)
    direction_norm = float(direction.norm())
    if not math.isfinite(direction_norm) or direction_norm <= _EPS:
        raise ValueError("The selected neuron group has a zero-length w2 direction.")

    baseline_request = BaselineRequest(
        prompt=request.prompt,
        system=request.system,
        settings=request.settings,
    )
    warnings: list[str] = []
    with capture_modules(model, [f"model.layers.{request.layer_index}"]) as (
        layer_chunks,
        layer_warnings,
    ):
        baseline = generate_baseline(manager, baseline_request)
    layer_captures, layer_finalize_warnings = finalize_captures(layer_chunks)
    warnings.extend(layer_warnings)
    warnings.extend(layer_finalize_warnings)
    layer_id = f"model.layers.{request.layer_index}"
    if layer_id not in layer_captures:
        raise ValueError(f"Layer {layer_id!r} did not produce an activation.")
    residual_rms = _residual_rms(layer_captures[layer_id], baseline.response.prompt_tokens, hidden_size)
    scale = request.alpha * residual_rms / direction_norm
    deltas = {component.neuron: scale * component.weight for component in request.neurons}
    start = time.perf_counter()
    with steer_neurons(model, request.layer_index, deltas):
        steered = generate_chat(manager, baseline_request, condition="steered")
    duration_ms = (time.perf_counter() - start) * 1000.0
    if not any(abs(delta) > _EPS for delta in deltas.values()):
        warnings.append("All neuron deltas are zero; the steered output should equal the baseline.")
    return NeuronSteeringResponse(
        layer_index=request.layer_index,
        alpha=request.alpha,
        neurons=request.neurons,
        direction_norm=round(direction_norm, 6),
        residual_rms=round(residual_rms, 6),
        deltas=[NeuronDelta(neuron=neuron, delta=round(delta, 6)) for neuron, delta in deltas.items()],
        baseline=baseline.response,
        steered=steered.response,
        duration_ms=round(duration_ms, 2),
        warnings=warnings,
    )
