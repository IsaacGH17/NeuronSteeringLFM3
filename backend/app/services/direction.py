"""Build a steering direction by subtracting two captured states.

This is the contrastive-activation-addition recipe reduced to what the tool can
measure: take two runs whose only intended difference is style, average each
module output over the same token positions, and subtract. The result is one
vector of ``hidden_size`` numbers; the largest components are the dimensions
that separate the two states, and the per-dimension standard deviation says how
large a displacement is relative to that dimension's normal spread.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from backend.app.schemas.api import DirectionRequest, DirectionResponse, SteeringComponent
from backend.app.services.activation_store import StoredExperiment

POSITION_MODES = ("mean_prompt", "mean_all", "last_prompt")


def _select_positions(experiment: StoredExperiment, mode: str, shared: int) -> np.ndarray:
    prompt_tokens = min(max(experiment.prompt_token_count, 1), shared)
    if mode == "last_prompt":
        return np.array([prompt_tokens - 1])
    if mode == "mean_prompt":
        return np.arange(prompt_tokens)
    return np.arange(shared)


def compute_direction(
    positive: StoredExperiment,
    negative: StoredExperiment,
    request: DirectionRequest,
) -> DirectionResponse:
    shared_modules = [module for module in positive.captures if module in negative.captures]
    if not shared_modules:
        raise ValueError("The two experiments share no captured modules.")
    module_id = request.module_id or shared_modules[0]
    if module_id not in positive.captures or module_id not in negative.captures:
        raise ValueError(f"Module {module_id!r} was not captured in both experiments.")

    pos_tensor: torch.Tensor = positive.captures[module_id]
    neg_tensor: torch.Tensor = negative.captures[module_id]
    if pos_tensor.shape[1] != neg_tensor.shape[1]:
        raise ValueError(
            f"Cannot compare: hidden sizes differ ({pos_tensor.shape[1]} vs {neg_tensor.shape[1]})."
        )
    shared_positions = min(pos_tensor.shape[0], neg_tensor.shape[0])
    if shared_positions == 0:
        raise ValueError("The captures have no positions to compare.")

    pos_array = pos_tensor[:shared_positions].numpy().astype(np.float64)
    neg_array = neg_tensor[:shared_positions].numpy().astype(np.float64)
    warnings: list[str] = []
    if pos_tensor.shape[0] != neg_tensor.shape[0]:
        warnings.append(
            f"Runs have different lengths ({pos_tensor.shape[0]} vs {neg_tensor.shape[0]} positions); "
            f"the direction uses the first {shared_positions}."
        )

    positions = _select_positions(positive, request.position_mode, shared_positions)
    pos_sample = pos_array[positions]
    neg_sample = neg_array[positions]
    difference = pos_sample.mean(axis=0) - neg_sample.mean(axis=0)
    norm = float(np.linalg.norm(difference))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("The two captured states are identical at the compared positions.")

    spread = float(np.sqrt(np.mean(np.var(pos_sample, axis=0))))
    separation = float(norm / spread) if spread > 0 else float("inf")

    order = np.argsort(-np.abs(difference))[: request.top_k]
    max_abs = float(np.max(np.abs(difference[order]))) if order.size else 0.0
    scale = max_abs if request.normalize and max_abs > 0 else 1.0
    components = [
        SteeringComponent(
            feature_index=int(index),
            weight=round(float(difference[index] / scale), 6),
        )
        for index in order
    ]
    sigma = [round(float(np.std(pos_array[positions, int(index)])) + 1e-12, 6) for index in order]

    if len(positive.tokens) and positive.prompt_token_count:
        if positions.max() >= len(positive.tokens):
            warnings.append("Some compared positions fall outside the decoded prompt tokens.")
    if norm < 1e-6:
        warnings.append("The difference is very small; the direction may be noise.")
    return DirectionResponse(
        module_id=module_id,
        components=components,
        sigma=sigma,
        norm=round(norm, 6),
        separation=round(separation, 4),
        positions_compared=int(len(positions)),
        warnings=warnings,
    )
