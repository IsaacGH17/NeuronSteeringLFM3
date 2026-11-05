"""Trace a real generation across every decoder layer.

The flow view needs three things that the activation store already holds:
one captured tensor per decoder layer. From those tensors this service derives

* per-layer residual write, direction retention, and gain between consecutive
  layers (the widths of the graph edges),
* per-token activation energy and change (the playback signal), and
* clusters of hidden dimensions whose activation profiles move together.

The clustering is descriptive. A cluster says "these dimensions behaved
similarly on this trace"; it does not claim that the group controls a
behaviour. Every number returned here is computed from the captured tensors.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from backend.app.schemas.api import (
    FlowCluster,
    FlowDimension,
    FlowEdge,
    FlowLayer,
    FlowRequest,
    FlowResponse,
    SteeringComponent,
)
from backend.app.services.ablation import compute_ablation_kl
from backend.app.services.activation_store import StoredExperiment
from backend.app.services.architecture_inspector import build_layer_overview
from backend.app.services.generation import (
    GenerationResult,
    count_prompt_tokens,
    generate_baseline,
)
from backend.app.services.model_loader import ModelManager

# Capturing every decoder layer is much heavier than capturing one module. At
# 24 layers x 2048 hidden x 4 bytes, 512 positions is about 100 MB; 2048 would
# be 400 MB and would also duplicate the decode-step chunks while concatenating.
MAX_FLOW_POSITIONS = 512
MAX_CLUSTER_DIMENSIONS = 12
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


def cluster_profiles(
    array: np.ndarray,
    *,
    k: int,
    max_positions: int,
) -> tuple[list[FlowCluster], int]:
    """Group hidden dimensions by their z-scored profile across positions.

    ``array`` is ``[positions, hidden]``. Each hidden dimension becomes a vector
    over the sampled positions; z-scoring removes its scale and L2-normalizing
    makes the comparison about shape, not magnitude. Spherical k-means then
    groups dimensions that rise and fall together.

    Returns ``(clusters, sampled_positions)``. Clusters are sorted by mass, so
    cluster 0 is the group carrying the most activation.
    """
    position_count = int(array.shape[0])
    if position_count == 0 or array.shape[1] == 0:
        return [], 0
    positions = _even_indices(position_count, max_positions)
    sampled = np.asarray(array[positions], dtype=np.float32)
    if sampled.shape[0] == 0:
        return [], 0

    mean_abs = np.mean(np.abs(sampled), axis=0)
    mean_signed = np.mean(sampled, axis=0)
    std = np.std(sampled, axis=0)

    centered = sampled - sampled.mean(axis=0)
    spread = sampled.std(axis=0)
    spread[spread < 1e-8] = 1.0
    profiles = _unit_columns(centered / spread)

    cluster_count = max(1, min(int(k), profiles.shape[1]))
    # Deterministic seeds: the highest-mass dimensions start one cluster each.
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
    # One final assignment keeps membership consistent with the last centroids.
    assignment = np.argmax(scores, axis=1)
    total_mass = float(np.sum(mean_abs))
    if total_mass <= _EPS:
        total_mass = 1.0

    clusters: list[FlowCluster] = []
    for cluster_id in range(cluster_count):
        members = np.flatnonzero(assignment == cluster_id)
        if members.size == 0:
            continue
        by_mass = members[np.argsort(-mean_abs[members], kind="stable")]
        top = by_mass[:MAX_CLUSTER_DIMENSIONS]
        signed = mean_signed[top]
        scale = float(np.max(np.abs(signed))) if top.size else 0.0
        if scale <= 1e-8:
            # Dimensions cancel out on average; fall back to magnitudes so the
            # cluster still has a usable (positive) direction.
            magnitude = np.abs(mean_abs[top])
            largest = float(np.max(magnitude)) if top.size else 0.0
            weights = magnitude / largest if largest > _EPS else np.ones_like(magnitude)
        else:
            weights = signed / scale

        profile = np.mean(sampled[:, members], axis=1)
        profile_indices = _even_indices(profile.size, PROFILE_POINTS)
        peak_local = int(np.argmax(np.abs(profile)))

        clusters.append(
            FlowCluster(
                id=0,
                size=int(members.size),
                mass=round(float(np.sum(mean_abs[members]) / total_mass), 6),
                cohesion=round(float(np.mean(scores[members, cluster_id])), 6),
                peak_position=int(positions[peak_local]),
                direction_norm=round(float(np.linalg.norm(weights)), 6),
                top_dimensions=[
                    FlowDimension(
                        index=int(index),
                        weight=round(float(weight), 6),
                        mean_abs=round(float(mean_abs[index]), 6),
                        sigma=round(float(std[index]), 6),
                    )
                    for index, weight in zip(top, weights, strict=True)
                ],
                profile=_rounded(profile[profile_indices]),
                profile_positions=[int(positions[index]) for index in profile_indices],
                components=[
                    SteeringComponent(
                        feature_index=int(index),
                        weight=round(float(weight), 6),
                    )
                    for index, weight in zip(top, weights, strict=True)
                ],
            )
        )

    clusters.sort(key=lambda cluster: (-cluster.mass, cluster.id))
    for index, cluster in enumerate(clusters):
        cluster.id = index
    return clusters, int(positions.size)


def compute_flow(
    experiment: StoredExperiment,
    request: FlowRequest,
    *,
    model: object,
) -> FlowResponse:
    """Build the flow payload from one experiment's captured layer tensors."""
    overview = build_layer_overview(model)
    if not overview:
        raise ValueError("The loaded model exposes no decoder layers to trace.")

    missing = [block.module_id for block in overview if block.module_id not in experiment.captures]
    if missing:
        raise ValueError(
            "This experiment did not capture every decoder layer; "
            f"missing {len(missing)} layer(s) starting with {missing[0]!r}."
        )

    arrays = [
        np.asarray(experiment.captures[block.module_id].numpy(), dtype=np.float32) for block in overview
    ]
    position_counts = {array.shape[0] for array in arrays}
    position_count = min(position_counts)
    warnings = list(experiment.warnings)
    if len(position_counts) > 1:
        warnings.append(
            f"Captured layers have different lengths {sorted(position_counts)}; "
            f"the trace covers the first {position_count} positions."
        )
    arrays = [array[:position_count] for array in arrays]
    if position_count == 0:
        raise ValueError("The captured layers are empty; there is nothing to trace.")

    energies = [np.linalg.norm(array, axis=1) for array in arrays]
    global_max_norm = max(float(np.max(energy)) for energy in energies)

    layers: list[FlowLayer] = []
    edges: list[FlowEdge] = []
    sampled_positions = 0
    previous: np.ndarray | None = None
    previous_energy: np.ndarray | None = None

    for index, (block, array, energy) in enumerate(zip(overview, arrays, energies, strict=True)):
        mean_norm = float(np.mean(energy))
        if previous is None or previous_energy is None:
            delta = energy
            write = 0.0
            similarity = 1.0
            gain = 1.0
        else:
            delta = np.linalg.norm(array - previous, axis=1)
            previous_mean = float(np.mean(previous_energy))
            write = float(np.mean(delta)) / (previous_mean + _EPS)
            cosine = np.sum(array * previous, axis=1) / (energy * previous_energy + _EPS)
            similarity = float(np.mean(cosine))
            gain = mean_norm / (previous_mean + _EPS)
            edges.append(
                FlowEdge(
                    source_layer=index - 1,
                    target_layer=index,
                    write=round(write, 6),
                    similarity=round(similarity, 6),
                    gain=round(gain, 6),
                )
            )

        clusters, sampled = cluster_profiles(
            array,
            k=request.cluster_k,
            max_positions=request.cluster_max_positions,
        )
        sampled_positions = sampled
        layers.append(
            FlowLayer(
                layer_index=index,
                module_id=block.module_id,
                layer_type=block.layer_type,
                operator_type=block.operator_type,
                mean_norm=round(mean_norm, 6),
                delta_norm=round(float(np.mean(delta)), 6),
                write=round(write, 6),
                similarity=round(similarity, 6),
                gain=round(gain, 6),
                energy=_rounded(energy),
                delta=_rounded(delta),
                clusters=clusters,
            )
        )
        previous = array
        previous_energy = energy

    tokens = experiment.tokens[:position_count]
    if len(experiment.tokens) > position_count and not any(
        "final generated token" in warning for warning in warnings
    ):
        warnings.append(
            "The final generated token is never forwarded, so it has no activation; "
            f"the trace covers the first {position_count} tokens."
        )
    if sampled_positions and sampled_positions < position_count:
        warnings.append(
            f"Clusters were computed on {sampled_positions} evenly sampled positions "
            f"of {position_count}; edges and energy use all {position_count}."
        )

    return FlowResponse(
        experiment_id=experiment.id,
        prompt=experiment.prompt,
        system=experiment.system,
        output=experiment.output,
        prompt_tokens=experiment.prompt_token_count,
        generated_tokens=experiment.generated_tokens,
        tokens=tokens,
        layers=layers,
        edges=edges,
        global_max_norm=round(global_max_norm, 6),
        cluster_k=request.cluster_k,
        cluster_positions=sampled_positions,
        settings=request.settings,
        warnings=warnings,
    )


def trace_flow(
    manager: ModelManager,
    request: FlowRequest,
    *,
    on_generation: Callable[[GenerationResult], None] | None = None,
) -> tuple[GenerationResult, FlowResponse]:
    """Run one real generation with every decoder layer captured, then trace it.

    ``on_generation`` runs immediately after generation and before analysis, so
    a caller can persist the completed run even if clustering fails.
    """
    model = manager.model
    overview = build_layer_overview(model)
    if not overview:
        raise ValueError("The loaded model exposes no decoder layers to trace.")

    max_new_tokens = min(request.settings.max_new_tokens, manager.settings.max_new_tokens)
    prompt_tokens = count_prompt_tokens(
        manager.tokenizer,
        request.prompt,
        request.system,
        request.settings.enable_thinking,
    )
    needed = prompt_tokens + max_new_tokens
    if needed > MAX_FLOW_POSITIONS:
        raise ValueError(
            f"Flow tracing captures every decoder layer and is limited to "
            f"{MAX_FLOW_POSITIONS} positions; this run could need {needed}. "
            "Lower max_new_tokens or shorten the prompt."
        )

    result = generate_baseline(
        manager,
        request,
        capture_ids=[block.module_id for block in overview],
    )
    if on_generation is not None:
        on_generation(result)
    stored = StoredExperiment(
        id=result.response.id,
        created_at=result.response.timestamp,
        model_id=result.response.model_id,
        revision=result.response.revision,
        prompt=result.response.prompt,
        system=result.response.system,
        output=result.response.output,
        prompt_token_count=result.response.prompt_tokens,
        generated_tokens=result.response.generated_tokens,
        tokens=result.tokens,
        captures=result.captures,
        warnings=result.response.warnings,
    )
    flow = compute_flow(stored, request, model=model)
    if request.include_ablation:
        values, ablation_warnings = compute_ablation_kl(
            manager,
            prompt=request.prompt,
            system=request.system,
            enable_thinking=request.settings.enable_thinking,
            layer_ids=[block.module_id for block in overview],
        )
        for layer, value in zip(flow.layers, values, strict=True):
            layer.ablation_kl = value
        flow.warnings.extend(ablation_warnings)
    return result, flow
