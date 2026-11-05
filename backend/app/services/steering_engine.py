"""Inference-time activation steering via forward hooks.

A steering direction adds a weighted offset to one or more feature dimensions
of a module's hidden-state output:

``h[..., i] += strength * weight_i`` (``normalization="raw"``)

``h[..., i] += strength * weight_i * sigma_i`` (``normalization="sigma"``)

``sigma_i`` is the standard deviation of dimension ``i`` measured on the first
forward with more than one token (the prompt prefill) and reused for the rest
of the generation, so the same ``strength`` means roughly the same displacement
regardless of the layer or dimension scale. Decode steps carry a single token,
where a per-step spread would always be zero; the cached calibration avoids
that degenerate case. ``strength == 0`` leaves the tensor numerically identical
to the baseline in both modes.

The intervention is strictly out-of-place: the hook clones the module output on
its own device (converting to float32 for the arithmetic, then back to the
original dtype) before modifying it, because the conv path mutates cache state
in place and the same cached tensor may flow elsewhere. Hooks are installed
only for the requested experiment and always removed in a ``finally`` block.
Model weights are never touched.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from backend.app.services.architecture_inspector import STEERABLE_CLASSES, get_module

MAX_STEERING_COMPONENTS = 512
Component = tuple[int, float]


@dataclass(frozen=True)
class SteeringSpec:
    """One module plus a weighted set of hidden dimensions."""

    module_id: str
    feature_index: int | None
    strength: float
    components: tuple[Component, ...] = field(default_factory=tuple)
    normalization: str = "raw"

    def __post_init__(self) -> None:
        merged: dict[int, float] = {}
        for index, weight in self.components:
            merged[index] = merged.get(index, 0.0) + float(weight)
        if not merged:
            if self.feature_index is None:
                raise ValueError("a steering spec needs feature_index or components")
            merged[self.feature_index] = 1.0
        object.__setattr__(self, "components", tuple(sorted(merged.items())))
        if len(self.components) == 1 and self.feature_index is None:
            object.__setattr__(self, "feature_index", self.components[0][0])
        if len(self.components) > 1:
            object.__setattr__(self, "feature_index", None)

    @property
    def indices(self) -> list[int]:
        return [index for index, _ in self.components]

    def summary(self) -> str:
        label = f"dim {self.feature_index}" if len(self.components) == 1 else f"{len(self.components)} dims"
        unit = "" if self.normalization == "raw" else " sigma"
        return f"{self.module_id} {label} {'+' if self.strength >= 0 else '-'}{abs(self.strength):.6g}{unit}"


def to_engine_spec(spec: Any) -> SteeringSpec:
    """Accept either the API model or the engine dataclass."""
    if isinstance(spec, SteeringSpec):
        return spec
    return SteeringSpec(
        module_id=spec.module_id,
        feature_index=spec.feature_index,
        strength=float(spec.strength),
        components=tuple((component.feature_index, float(component.weight)) for component in spec.components),
        normalization=spec.normalization,
    )


def _hidden_size(model: Any) -> int:
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int):
        raise ValueError("The loaded model does not expose a numeric hidden_size in its config.")
    return hidden_size


def validate_steering_spec(model: Any, spec: Any, *, strength_limit: float) -> SteeringSpec:
    """Verify module, dimensions, weights, and strength before any hook is installed."""
    spec = to_engine_spec(spec)
    spec = SteeringSpec(
        module_id=spec.module_id,
        feature_index=spec.feature_index,
        strength=float(spec.strength),
        components=spec.components,
        normalization=spec.normalization,
    )
    if spec.normalization not in {"raw", "sigma"}:
        raise ValueError(f"Unknown steering normalization {spec.normalization!r}.")
    if not math.isfinite(spec.strength):
        raise ValueError("Steering strength must be finite.")
    if abs(spec.strength) > strength_limit:
        raise ValueError(f"Steering strength {spec.strength} exceeds the safety limit ±{strength_limit}.")
    if not spec.components:
        raise ValueError("Steering needs at least one dimension.")
    if len(spec.components) > MAX_STEERING_COMPONENTS:
        raise ValueError(
            f"Steering accepts at most {MAX_STEERING_COMPONENTS} dimensions; got {len(spec.components)}."
        )
    try:
        module = get_module(model, spec.module_id)
    except KeyError as exc:
        raise ValueError(f"Unknown steering module {spec.module_id!r}.") from exc
    class_name = type(module).__name__
    if class_name not in STEERABLE_CLASSES:
        raise ValueError(
            f"Module {spec.module_id!r} ({class_name}) cannot be steered: "
            "its forward output is not a hidden-state tensor."
        )
    hidden_size = _hidden_size(model)
    for index, weight in spec.components:
        if not 0 <= index < hidden_size:
            raise ValueError(
                f"feature_index {index} is outside the hidden-size range [0, {hidden_size - 1}]."
            )
        if not math.isfinite(weight):
            raise ValueError(f"Steering weight for dimension {index} must be finite.")
    return spec


def _steered_output(
    output: Any,
    *,
    feature_index: int | None = None,
    strength: float = 0.0,
    components: tuple[Component, ...] | None = None,
    normalization: str = "raw",
    skipped: set[int] | None = None,
    scales: dict[int, float] | None = None,
) -> tuple[Any, bool]:
    """Return a steered copy of a module output plus whether it was modified.

    Handles both a bare tensor and a tuple whose first element is the
    hidden-state tensor (Lfm3Attention returns ``(attn_output, attn_weights)``).
    """
    if isinstance(output, torch.Tensor):
        target, container = output, None
    elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        target, container = output[0], output
    else:
        return output, False
    if target.ndim != 3:
        return output, False

    resolved = components
    if not resolved:
        if feature_index is None:
            return output, False
        resolved = ((feature_index, 1.0),)

    # Float32 keeps small displacements visible in bfloat16 activations, and
    # converting back is exact, so strength == 0 reproduces the baseline.
    # ``.to`` returns the same tensor when the dtype already matches, so clone
    # unconditionally to keep the intervention out-of-place.
    steered = target.to(torch.float32).clone()
    if normalization == "sigma":
        cache = scales if scales is not None else {}
        if not cache and target.shape[1] > 1:
            # Calibrate once on the prefill; decode steps carry one token and
            # would otherwise report a spread of zero for every dimension.
            for index, _ in resolved:
                cache[index] = float(steered[..., index].detach().std(unbiased=False))
        for index, weight in resolved:
            sigma = cache.get(index, 0.0)
            if not math.isfinite(sigma) or sigma == 0.0:
                if skipped is not None:
                    skipped.add(index)
                continue
            steered[..., index] = steered[..., index] + strength * weight * sigma
    else:
        for index, weight in resolved:
            steered[..., index] = steered[..., index] + strength * weight
    result = steered.to(target.dtype)
    if container is None:
        return result, True
    return (result, *container[1:]), True


@contextmanager
def steer_modules(model: Any, specs: list[SteeringSpec]) -> Iterator[tuple[dict[str, int], list[str]]]:
    """Install steering hooks around a generation; remove them in ``finally``.

    Yields ``(applied_calls, warnings)`` where ``applied_calls`` counts how
    many forwards each module had its dimensions shifted.
    """
    applied_calls: dict[str, int] = {spec.module_id: 0 for spec in specs}
    warnings: list[str] = []
    warned: set[str] = set()
    handles: list[Any] = []

    def make_hook(spec: SteeringSpec):
        skipped_dims: set[int] = set()
        calibrated: dict[int, float] = {}

        def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            steered, modified = _steered_output(
                output,
                feature_index=spec.feature_index,
                strength=spec.strength,
                components=spec.components,
                normalization=spec.normalization,
                skipped=skipped_dims,
                scales=calibrated if spec.normalization == "sigma" else None,
            )
            if modified:
                applied_calls[spec.module_id] += 1
            elif spec.module_id not in warned:
                warnings.append(
                    f"{spec.module_id} did not return a [batch, seq, hidden] tensor; "
                    "no dimension was shifted."
                )
                warned.add(spec.module_id)
            if skipped_dims and spec.normalization == "sigma":
                message = (
                    f"{spec.module_id}: dimensions {sorted(skipped_dims)} had zero variance "
                    "and were left unchanged in sigma mode."
                )
                if message not in warnings:
                    warnings.append(message)
            return steered

        return hook

    try:
        for spec in specs:
            handles.append(get_module(model, spec.module_id).register_forward_hook(make_hook(spec)))
        yield applied_calls, warnings
    finally:
        for handle in handles:
            handle.remove()
