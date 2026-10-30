"""Ephemeral in-memory storage and analysis of captured activations.

Complete activation histories are never persisted by default. Entries are
bounded by count and total bytes; the oldest experiment is evicted first.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

from backend.app.schemas.api import (
    ActivationResponse,
    ActivationStats,
    ActivationSummary,
    FeatureSeries,
    HeatmapData,
    HistogramData,
    TokenInfo,
    TopDimension,
)

MAX_STORED_EXPERIMENTS = 8
MAX_STORED_BYTES = 256 * 1024 * 1024
TOP_DIMENSIONS = 20


@dataclass
class StoredExperiment:
    id: str
    created_at: datetime
    model_id: str
    revision: str
    prompt: str
    system: str | None
    output: str
    prompt_token_count: int
    generated_tokens: int
    tokens: list[TokenInfo]
    captures: dict[str, torch.Tensor] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.captures.values())


class ActivationStore:
    def __init__(
        self,
        max_experiments: int = MAX_STORED_EXPERIMENTS,
        max_bytes: int = MAX_STORED_BYTES,
    ) -> None:
        self.max_experiments = max_experiments
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, StoredExperiment] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def add(self, experiment: StoredExperiment) -> None:
        with self._lock:
            if experiment.id in self._entries:
                self._bytes -= self._entries.pop(experiment.id).nbytes()
            self._entries[experiment.id] = experiment
            self._bytes += experiment.nbytes()
            while self._entries and (
                len(self._entries) > self.max_experiments or self._bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.nbytes()

    def get(self, experiment_id: str) -> StoredExperiment:
        with self._lock:
            try:
                return self._entries[experiment_id]
            except KeyError as exc:
                raise KeyError(f"Unknown experiment {experiment_id!r}.") from exc

    def list(self, experiment_id: str) -> list[ActivationSummary]:
        experiment = self.get(experiment_id)
        return [
            ActivationSummary(
                module_id=module_id,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype).replace("torch.", ""),
            )
            for module_id, tensor in experiment.captures.items()
        ]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


def _sample_indices(size: int, limit: int) -> list[int]:
    if size <= limit:
        return list(range(size))
    return sorted({int(index) for index in np.linspace(0, size - 1, num=limit).round()})


def _resolve_position(position: int, size: int) -> int:
    resolved = size + position if position < 0 else position
    if resolved < 0 or resolved >= size:
        raise ValueError(f"position {position} is outside the captured range [0, {size - 1}].")
    return resolved


def build_activation_response(
    experiment: StoredExperiment,
    module_id: str,
    *,
    feature: int | None = None,
    position: int = -1,
    max_positions: int = 128,
    max_features: int = 128,
) -> ActivationResponse:
    try:
        tensor = experiment.captures[module_id]
    except KeyError as exc:
        raise KeyError(f"Module {module_id!r} was not captured for experiment {experiment.id!r}.") from exc

    array = tensor.numpy()
    token_count, hidden_size = array.shape
    warnings = list(experiment.warnings)
    if len(experiment.tokens) != token_count:
        warnings.append(
            f"Captured {token_count} positions but decoded {len(experiment.tokens)} tokens; "
            "token alignment is approximate."
        )

    resolved_position = _resolve_position(position, token_count)
    row = array[resolved_position]
    order = np.argsort(-np.abs(row))[:TOP_DIMENSIONS]
    top_dimensions = [
        TopDimension(
            index=int(index),
            value=float(row[index]),
            abs_value=float(abs(row[index])),
        )
        for index in order
    ]

    resolved_feature = feature if feature is not None else int(order[0])
    if resolved_feature < 0 or resolved_feature >= hidden_size:
        raise ValueError(f"feature {feature} is outside the captured range [0, {hidden_size - 1}].")

    position_indices = _sample_indices(token_count, max_positions)
    feature_indices = _sample_indices(hidden_size, max_features)
    values = array[np.ix_(position_indices, feature_indices)]

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        finite = np.zeros(1, dtype=np.float32)
    counts, bin_edges = np.histogram(finite, bins=40)

    return ActivationResponse(
        experiment_id=experiment.id,
        module_id=module_id,
        shape=[token_count, hidden_size],
        dtype=str(tensor.dtype).replace("torch.", ""),
        aggregation="raw values, single sequence; hooks detach and copy to CPU float32",
        prompt_token_count=experiment.prompt_token_count,
        tokens=experiment.tokens,
        heatmap=HeatmapData(
            positions=position_indices,
            features=feature_indices,
            values=values.tolist(),
            sampling=(
                f"evenly sampled {len(position_indices)}/{token_count} positions × "
                f"{len(feature_indices)}/{hidden_size} features; no pooling"
            ),
        ),
        histogram=HistogramData(
            bin_edges=[float(value) for value in bin_edges],
            counts=[int(value) for value in counts],
        ),
        stats=ActivationStats(
            minimum=float(np.min(finite)),
            maximum=float(np.max(finite)),
            mean=float(np.mean(finite)),
            std=float(np.std(finite)),
            abs_mean=float(np.mean(np.abs(finite))),
        ),
        position=resolved_position,
        top_dimensions=top_dimensions,
        feature_series=FeatureSeries(
            feature=resolved_feature,
            positions=list(range(token_count)),
            values=[float(value) for value in array[:, resolved_feature]],
        ),
        warnings=warnings,
    )


activation_store = ActivationStore()


def clear_store() -> None:
    activation_store.clear()


def store_experiment(
    *,
    experiment_id: str,
    created_at: datetime,
    model_id: str,
    revision: str,
    prompt: str,
    system: str | None,
    output: str,
    prompt_token_count: int,
    generated_tokens: int,
    tokens: list[TokenInfo],
    captures: dict[str, torch.Tensor],
    warnings: list[str] | None = None,
) -> StoredExperiment:
    experiment = StoredExperiment(
        id=experiment_id,
        created_at=created_at,
        model_id=model_id,
        revision=revision,
        prompt=prompt,
        system=system,
        output=output,
        prompt_token_count=prompt_token_count,
        generated_tokens=generated_tokens,
        tokens=tokens,
        captures=captures,
        warnings=list(warnings or []),
    )
    activation_store.add(experiment)
    return experiment
