"""Inference-time activation steering via forward hooks.

A steering intervention adds a strength ``alpha`` to one feature dimension of a
module's hidden-state output:

``h[..., i] = h[..., i] + alpha``

The intervention is strictly out-of-place: the hook clones the module output on
its own device and dtype before modifying it, because the conv path mutates
cache state in place and the same cached tensor may flow elsewhere. Hooks are
installed only for the requested experiment and always removed in a ``finally``
block. Model weights are never touched.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from backend.app.services.architecture_inspector import STEERABLE_CLASSES, get_module


@dataclass(frozen=True)
class SteeringSpec:
    module_id: str
    feature_index: int
    strength: float

    def summary(self) -> str:
        return (
            f"{self.module_id} dim {self.feature_index} "
            f"{'+' if self.strength >= 0 else '-'}{abs(self.strength):.6g}"
        )


def _hidden_size(model: Any) -> int:
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int):
        raise ValueError("The loaded model does not expose a numeric hidden_size in its config.")
    return hidden_size


def validate_steering_spec(model: Any, spec: SteeringSpec, *, strength_limit: float) -> SteeringSpec:
    """Verify module, dimension, and strength before any hook is installed."""
    spec = SteeringSpec(
        module_id=spec.module_id,
        feature_index=spec.feature_index,
        strength=float(spec.strength),
    )
    if not math.isfinite(spec.strength):
        raise ValueError("Steering strength must be finite.")
    if abs(spec.strength) > strength_limit:
        raise ValueError(f"Steering strength {spec.strength} exceeds the safety limit ±{strength_limit}.")
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
    if not 0 <= spec.feature_index < hidden_size:
        raise ValueError(
            f"feature_index {spec.feature_index} is outside the hidden-size range [0, {hidden_size - 1}]."
        )
    return spec


def _steered_output(output: Any, *, feature_index: int, strength: float) -> tuple[Any, bool]:
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
    steered = target.clone()
    steered[..., feature_index] = steered[..., feature_index] + strength
    if container is None:
        return steered, True
    return (steered, *container[1:]), True


@contextmanager
def steer_modules(model: Any, specs: list[SteeringSpec]) -> Iterator[tuple[dict[str, int], list[str]]]:
    """Install steering hooks around a generation; remove them in ``finally``.

    Yields ``(applied_calls, warnings)`` where ``applied_calls`` counts how
    many forwards each module had its dimension shifted.
    """
    applied_calls: dict[str, int] = {spec.module_id: 0 for spec in specs}
    warnings: list[str] = []
    warned: set[str] = set()
    handles: list[Any] = []

    def make_hook(spec: SteeringSpec):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            steered, modified = _steered_output(
                output, feature_index=spec.feature_index, strength=spec.strength
            )
            if modified:
                applied_calls[spec.module_id] += 1
            elif spec.module_id not in warned:
                warnings.append(
                    f"{spec.module_id} did not return a [batch, seq, hidden] tensor; "
                    f"dim {spec.feature_index} was not shifted."
                )
                warned.add(spec.module_id)
            return steered

        return hook

    try:
        for spec in specs:
            handles.append(get_module(model, spec.module_id).register_forward_hook(make_hook(spec)))
        yield applied_calls, warnings
    finally:
        for handle in handles:
            handle.remove()
