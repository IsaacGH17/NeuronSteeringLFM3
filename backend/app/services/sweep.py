"""Multi-strength sweep orchestration with optional randomized controls."""

from __future__ import annotations

import math
import random
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from backend.app.schemas.api import (
    BaselineRequest,
    GenerationSettings,
    SweepRequest,
)
from backend.app.schemas.api import (
    SteeringSpec as ApiSteeringSpec,
)
from backend.app.services.generation import (
    GenerationResult,
    generate_baseline,
    generate_chat,
    generate_steered,
)
from backend.app.services.model_loader import ModelManager
from backend.app.services.steering_engine import (
    SteeringSpec as EngineSteeringSpec,
)
from backend.app.services.steering_engine import (
    validate_steering_spec,
)

SweepKind = Literal["baseline", "strength", "control"]


@dataclass
class SweepRunResult:
    kind: SweepKind
    sweep_index: int
    repeat_index: int
    result: GenerationResult


@dataclass
class SweepResult:
    sweep_id: str
    runs: list[SweepRunResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _hidden_size(model) -> int:
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    hidden_size = getattr(text_config, "hidden_size", None)
    if not isinstance(hidden_size, int):
        raise ValueError("The loaded model does not expose a numeric hidden_size in its config.")
    return hidden_size


def _effective_seed(settings: GenerationSettings, repeat_index: int, repeats: int) -> int | None:
    """Determine the seed for a repeated run under the sweep seed rules."""
    base_seed = settings.seed
    if base_seed is None or repeats <= 1:
        return base_seed
    if settings.temperature == 0:
        # Greedy repeats are identical by design; keep the same seed.
        return base_seed
    if repeat_index == 0:
        return base_seed
    return base_seed + repeat_index


def run_sweep(
    manager: ModelManager,
    payload: SweepRequest,
    *,
    capture_ids: list[str] | None = None,
    on_run: Callable[[SweepRunResult], None] | None = None,
    sweep_id: str | None = None,
) -> SweepResult:
    """Run baseline + strengths (+ optional control) across repeats."""
    if not payload.strengths:
        raise ValueError("sweep strengths must contain at least one value.")
    if not (1 <= len(payload.strengths) <= 9):
        raise ValueError("sweep strengths must contain between 1 and 9 values.")
    if not (1 <= payload.repeats <= 3):
        raise ValueError("repeats must be between 1 and 3.")
    for strength in payload.strengths:
        if not math.isfinite(strength):
            raise ValueError("strengths must be finite.")

    # Validate the target module/feature and every strength up front.
    for strength in payload.strengths:
        validate_steering_spec(
            manager.model,
            EngineSteeringSpec(payload.module_id, payload.feature_index, strength),
            strength_limit=manager.settings.max_steering_strength,
        )

    total = 1 + payload.repeats * (len(payload.strengths) + (1 if payload.include_control else 0))
    if total > 31:
        raise ValueError(
            f"Sweep would create {total} runs; the limit is 31 "
            f"(1 + repeats * (len(strengths) + (1 if include_control else 0)))."
        )

    warnings: list[str] = []
    if payload.settings.temperature == 0 and payload.repeats > 1:
        warnings.append("Greedy decoding with repeats>1 keeps the same seed, so repeated runs are identical.")
    if payload.settings.temperature > 0 and payload.settings.seed is None and payload.repeats > 1:
        warnings.append(
            "Sampling runs have no seed, so repeated runs are not reproducible; set a seed to compare repeats."
        )

    sweep_id = sweep_id or uuid.uuid4().hex
    runs: list[SweepRunResult] = []

    # Baseline run.
    baseline_request = BaselineRequest(
        prompt=payload.prompt,
        system=payload.system,
        settings=payload.settings,
        capture_module_ids=payload.capture_module_ids,
    )
    baseline_result = generate_baseline(manager, baseline_request, capture_ids=capture_ids)
    baseline_run = SweepRunResult(kind="baseline", sweep_index=-1, repeat_index=0, result=baseline_result)
    runs.append(baseline_run)
    if on_run is not None:
        on_run(baseline_run)

    # Control configuration: one random feature matching target magnitude.
    hidden_size = _hidden_size(manager.model)
    control_feature: int | None = None
    control_strength: float | None = None
    if payload.include_control:
        if hidden_size < 2:
            raise ValueError("A random-feature control needs a hidden size of at least 2.")
        if payload.control_seed is not None:
            control_seed = payload.control_seed
        elif payload.settings.seed is not None:
            control_seed = payload.settings.seed
        else:
            control_seed = secrets.randbelow(2**31)
        rng = random.Random(control_seed)
        control_feature = rng.randrange(0, hidden_size)
        if control_feature == payload.feature_index:
            control_feature = (control_feature + 1) % hidden_size
        max_magnitude = max(abs(s) for s in payload.strengths)
        sign = math.copysign(1.0, payload.strengths[-1])
        control_strength = sign * max_magnitude
        warnings.append(
            f"Control steered random feature {control_feature} (seed {control_seed}) with the same "
            "max magnitude as the sweep; it is not a claim about the target feature's behavior."
        )

    for repeat_index in range(payload.repeats):
        seed = _effective_seed(payload.settings, repeat_index, payload.repeats)
        run_settings = payload.settings.model_copy(update={"seed": seed})
        run_request = BaselineRequest(
            prompt=payload.prompt,
            system=payload.system,
            settings=run_settings,
            capture_module_ids=payload.capture_module_ids,
        )

        for sweep_index, strength in enumerate(payload.strengths):
            steering = ApiSteeringSpec(
                module_id=payload.module_id,
                feature_index=payload.feature_index,
                strength=strength,
            )
            result = generate_steered(
                manager,
                run_request,
                steering,
                capture_ids=capture_ids,
            )
            strength_run = SweepRunResult(
                kind="strength",
                sweep_index=sweep_index,
                repeat_index=repeat_index,
                result=result,
            )
            runs.append(strength_run)
            if on_run is not None:
                on_run(strength_run)

        if payload.include_control and control_feature is not None and control_strength is not None:
            control_steering = ApiSteeringSpec(
                module_id=payload.module_id,
                feature_index=control_feature,
                strength=control_strength,
            )
            control_result = generate_chat(
                manager,
                run_request,
                capture_ids=capture_ids,
                steering=control_steering,
                condition="control",
            )
            control_run = SweepRunResult(
                kind="control",
                sweep_index=len(payload.strengths),
                repeat_index=repeat_index,
                result=control_result,
            )
            runs.append(control_run)
            if on_run is not None:
                on_run(control_run)

    return SweepResult(sweep_id=sweep_id, runs=runs, warnings=warnings)
