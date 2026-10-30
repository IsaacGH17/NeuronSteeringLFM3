"""Baseline chat generation.

The model is the only source of truth: prompt templating, tokenization,
generation, and decoding all go through the loaded tokenizer/model.
"""

from __future__ import annotations

import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import torch

from backend.app.schemas.api import BaselineRequest, BaselineResponse, GenerationSettings, TokenInfo
from backend.app.services.activation_capture import (
    MAX_CAPTURE_POSITIONS,
    capture_modules,
    finalize_captures,
    validate_capture_module_ids,
)
from backend.app.services.model_loader import ModelManager


@dataclass
class GenerationResult:
    response: BaselineResponse
    tokens: list[TokenInfo] = field(default_factory=list)
    captures: dict[str, torch.Tensor] = field(default_factory=dict)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])


def _build_tokens(tokenizer: Any, prompt_ids: list[int], generated_ids: list[int]) -> list[TokenInfo]:
    tokens: list[TokenInfo] = []
    for index, token_id in enumerate(prompt_ids):
        tokens.append(
            TokenInfo(
                index=index, token_id=int(token_id), text=_decode_token(tokenizer, token_id), phase="prompt"
            )
        )
    offset = len(prompt_ids)
    for index, token_id in enumerate(generated_ids):
        tokens.append(
            TokenInfo(
                index=offset + index,
                token_id=int(token_id),
                text=_decode_token(tokenizer, token_id),
                phase="generated",
            )
        )
    return tokens


def _apply_chat_template(
    tokenizer: Any,
    prompt: str,
    system: str | None,
    enable_thinking: bool,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _tokenize(tokenizer: Any, text: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _generation_kwargs(
    model: Any,
    tokenizer: Any,
    settings: GenerationSettings,
    max_new_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": settings.temperature > 0,
        "use_cache": True,
    }
    if settings.temperature > 0:
        kwargs["temperature"] = settings.temperature
        kwargs["top_p"] = settings.top_p
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def generate_baseline(
    manager: ModelManager,
    request: BaselineRequest,
    *,
    capture_ids: list[str] | None = None,
) -> GenerationResult:
    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device
    settings = request.settings
    capture_ids = validate_capture_module_ids(model, capture_ids or [])

    warnings: list[str] = []
    max_new_tokens = min(settings.max_new_tokens, manager.settings.max_new_tokens)
    if max_new_tokens != settings.max_new_tokens:
        warnings.append(
            f"max_new_tokens clamped from {settings.max_new_tokens} to {max_new_tokens} "
            "by the server-side safety limit."
        )
    if settings.temperature == 0 and settings.top_p != 1.0:
        warnings.append("top_p is ignored because temperature is 0 (greedy decoding).")

    templated = _apply_chat_template(tokenizer, request.prompt, request.system, settings.enable_thinking)
    inputs = _tokenize(tokenizer, templated, device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    if capture_ids and prompt_tokens + max_new_tokens > MAX_CAPTURE_POSITIONS:
        raise ValueError(
            f"Activation capture is limited to {MAX_CAPTURE_POSITIONS} positions; this run needs up to "
            f"{prompt_tokens + max_new_tokens}. Shorten the prompt or max_new_tokens."
        )
    kwargs = _generation_kwargs(model, tokenizer, settings, max_new_tokens)

    if settings.seed is not None:
        torch.manual_seed(settings.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(settings.seed)

    capture_context = capture_modules(model, capture_ids) if capture_ids else nullcontext(({}, []))
    start = time.perf_counter()
    with (
        manager.inference_lock,
        torch.inference_mode(),
        capture_context as (capture_chunks, capture_warnings),
    ):
        output = model.generate(**inputs, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000.0

    generated_ids = output[0, prompt_tokens:]
    generated_tokens = int(generated_ids.shape[0])
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_per_second = generated_tokens / (duration_ms / 1000.0) if duration_ms > 0 else None

    captures, finalize_warnings = finalize_captures(capture_chunks)
    captured_token_count = prompt_tokens + generated_tokens
    if captures:
        captured_token_count = min(tensor.shape[0] for tensor in captures.values())
        if captured_token_count < prompt_tokens + generated_tokens:
            warnings.append(
                f"Activations cover the {captured_token_count} positions that were forwarded; "
                "the final generated token has no activation."
            )
    token_infos = _build_tokens(
        tokenizer,
        inputs["input_ids"][0].tolist(),
        generated_ids.tolist()[: max(0, captured_token_count - prompt_tokens)],
    )
    status = manager.status()
    response = BaselineResponse(
        id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        model_id=status.model_id,
        revision=status.revision,
        device=device,
        prompt=request.prompt,
        system=request.system,
        output=text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        duration_ms=round(duration_ms, 2),
        tokens_per_second=round(tokens_per_second, 2) if tokens_per_second is not None else None,
        settings=settings,
        captured_modules=list(captures),
        warnings=warnings + capture_warnings + finalize_warnings,
    )
    return GenerationResult(response=response, tokens=token_infos, captures=captures)
