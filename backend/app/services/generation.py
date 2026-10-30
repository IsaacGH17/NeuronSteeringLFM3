"""Baseline chat generation.

The model is the only source of truth: prompt templating, tokenization,
generation, and decoding all go through the loaded tokenizer/model.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import torch

from backend.app.schemas.api import BaselineRequest, BaselineResponse, GenerationSettings
from backend.app.services.model_loader import ModelManager


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


def generate_baseline(manager: ModelManager, request: BaselineRequest) -> BaselineResponse:
    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device
    settings = request.settings

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
    kwargs = _generation_kwargs(model, tokenizer, settings, max_new_tokens)

    if settings.seed is not None:
        torch.manual_seed(settings.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(settings.seed)

    start = time.perf_counter()
    with manager.inference_lock, torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000.0

    generated_ids = output[0, prompt_tokens:]
    generated_tokens = int(generated_ids.shape[0])
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_per_second = generated_tokens / (duration_ms / 1000.0) if duration_ms > 0 else None

    return BaselineResponse(
        id=uuid.uuid4().hex,
        timestamp=datetime.now(UTC),
        model_id=manager.status().model_id,
        revision=manager.status().revision,
        device=device,
        prompt=request.prompt,
        system=request.system,
        output=text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        duration_ms=round(duration_ms, 2),
        tokens_per_second=round(tokens_per_second, 2) if tokens_per_second is not None else None,
        settings=settings,
        warnings=warnings,
    )
