"""One-forward next-token distributions, with optional inference-time steering.

The probe answers a narrower question than generation: *what does the model
want to say next, right now?* It renders the chat prompt, runs a single forward
over the prompt (no decoding), and reads the softmax at the last position. When
a steering spec is supplied, the same prompt is forwarded a second time with
the hooks installed, so the response can show the baseline and steered
distributions side by side.

Nothing is generated and nothing is written to experiment history.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from backend.app.schemas.api import (
    NextTokenCandidate,
    NextTokenDistribution,
    NextTokenRequest,
    NextTokenResponse,
    TokenProbability,
)
from backend.app.services.generation import render_chat_prompt
from backend.app.services.model_loader import ModelManager
from backend.app.services.steering_engine import steer_modules, validate_steering_spec

_EPS = 1e-12


def _decode(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])


def last_token_logits(manager: ModelManager, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Forward the prompt once and return the last-position logits as float32."""
    output = manager.model(**inputs, use_cache=False, logits_to_keep=1)
    logits = getattr(output, "logits", None)
    if logits is None:
        logits = output[0]
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    elif logits.ndim == 2:
        logits = logits[-1:, :]
    return logits[0].float()


def _distribution(
    logits: torch.Tensor,
    tokenizer: Any,
    top_k: int,
) -> tuple[NextTokenDistribution, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=-1)
    count = min(top_k, int(probabilities.numel()))
    top = torch.topk(probabilities, k=count)
    tokens = [
        TokenProbability(
            token_id=int(token_id),
            text=_decode(tokenizer, int(token_id)),
            probability=round(float(probability), 8),
        )
        for token_id, probability in zip(top.indices.tolist(), top.values.tolist(), strict=True)
    ]
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(_EPS))).sum())
    return NextTokenDistribution(tokens=tokens, entropy=round(entropy, 6)), probabilities


def _candidates(
    baseline: NextTokenDistribution,
    steered: NextTokenDistribution | None,
    baseline_probabilities: torch.Tensor,
    steered_probabilities: torch.Tensor | None,
    tokenizer: Any,
) -> list[NextTokenCandidate]:
    ids: list[int] = []
    for token in baseline.tokens:
        if token.token_id not in ids:
            ids.append(token.token_id)
    if steered is not None:
        for token in steered.tokens:
            if token.token_id not in ids:
                ids.append(token.token_id)
    candidates: list[NextTokenCandidate] = []
    for token_id in ids:
        baseline_probability = float(baseline_probabilities[token_id])
        steered_probability = (
            float(steered_probabilities[token_id]) if steered_probabilities is not None else None
        )
        candidates.append(
            NextTokenCandidate(
                token_id=token_id,
                text=_decode(tokenizer, token_id),
                baseline_probability=round(baseline_probability, 8),
                steered_probability=(
                    round(steered_probability, 8) if steered_probability is not None else None
                ),
                delta=(
                    round(steered_probability - baseline_probability, 8)
                    if steered_probability is not None
                    else None
                ),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            -max(candidate.baseline_probability, candidate.steered_probability or 0.0),
            candidate.token_id,
        )
    )
    return candidates


def probe_next_token(manager: ModelManager, request: NextTokenRequest) -> NextTokenResponse:
    """Run the baseline forward and, when asked, the steered forward."""
    model = manager.model
    tokenizer = manager.tokenizer
    engine_spec = None
    if request.steering is not None:
        engine_spec = validate_steering_spec(
            model, request.steering, strength_limit=manager.settings.max_steering_strength
        )

    templated = render_chat_prompt(
        tokenizer,
        request.prompt,
        request.system,
        request.enable_thinking,
    )
    encoded = tokenizer(templated, return_tensors="pt")
    inputs = {
        key: value.to(manager.device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    prompt_tokens = int(inputs["input_ids"].shape[1])
    warnings: list[str] = []
    start = time.perf_counter()
    with manager.inference_lock, torch.inference_mode():
        baseline_logits = last_token_logits(manager, inputs)
        steered_logits = None
        if engine_spec is not None:
            with steer_modules(model, [engine_spec]) as (applied_calls, steer_warnings):
                steered_logits = last_token_logits(manager, inputs)
            warnings.extend(steer_warnings)
            if applied_calls.get(engine_spec.module_id, 0) == 0:
                warnings.append(
                    f"Steering hook on {engine_spec.module_id} never fired; "
                    "the steered distribution is the baseline."
                )
    duration_ms = (time.perf_counter() - start) * 1000.0

    baseline, baseline_probabilities = _distribution(baseline_logits, tokenizer, request.top_k)
    steered: NextTokenDistribution | None = None
    steered_probabilities: torch.Tensor | None = None
    kl_divergence: float | None = None
    if steered_logits is not None:
        steered, steered_probabilities = _distribution(steered_logits, tokenizer, request.top_k)
        kl_divergence = max(
            0.0,
            float(
                (
                    steered_probabilities
                    * (
                        torch.log(steered_probabilities.clamp_min(_EPS))
                        - torch.log(baseline_probabilities.clamp_min(_EPS))
                    )
                ).sum()
            ),
        )

    status = manager.status()
    return NextTokenResponse(
        prompt=request.prompt,
        system=request.system,
        model_id=status.model_id,
        device=manager.device,
        prompt_tokens=prompt_tokens,
        position=prompt_tokens - 1,
        top_k=request.top_k,
        baseline=baseline,
        steered=steered,
        candidates=_candidates(
            baseline,
            steered,
            baseline_probabilities,
            steered_probabilities,
            tokenizer,
        ),
        kl_divergence=round(kl_divergence, 6) if kl_divergence is not None else None,
        steering=request.steering,
        duration_ms=round(duration_ms, 2),
        warnings=warnings,
    )
