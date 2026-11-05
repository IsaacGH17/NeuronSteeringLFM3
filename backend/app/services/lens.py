"""Logit lens and per-layer ablation over a real generation.

The lens projects every decoder-layer activation through the model's final norm
and language head, which shows how the next-token prediction sharpens with depth.
The ablation zeroes one decoder layer at a time on the prompt prefill and measures
the KL divergence of the next-token distribution, which gives a cheap causal
importance map per layer.

Nothing is fabricated: every token and probability comes from the loaded model's
own forward pass.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from backend.app.schemas.api import (
    LensLayer,
    LensRequest,
    LensResponse,
    LensToken,
    TokenInfo,
)
from backend.app.services.architecture_inspector import STEERABLE_CLASSES, get_module
from backend.app.services.generation import (
    _apply_chat_template,
    _build_tokens,
    _generation_kwargs,
    _tokenize,
)
from backend.app.services.model_loader import ModelManager

MAX_POSITIONS = 512


def _decode(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:  # pragma: no cover - defensive for exotic tokenizers
        return str(token_id)


def _resolve(position: int, size: int) -> int:
    resolved = size + position if position < 0 else position
    if resolved < 0 or resolved >= size:
        raise ValueError(f"position {position} is outside the range [0, {size - 1}].")
    return resolved


def _top_token(tokenizer: Any, probabilities: torch.Tensor, index: int, prob: float) -> LensToken:
    return LensToken(
        token_id=int(index),
        text=_decode(tokenizer, int(index)),
        probability=round(float(prob), 6),
    )


def compute_lens(manager: ModelManager, request: LensRequest) -> LensResponse:
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

    templated = _apply_chat_template(tokenizer, request.prompt, request.system, settings.enable_thinking)
    inputs = _tokenize(tokenizer, templated, device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    kwargs = _generation_kwargs(model, tokenizer, settings, max_new_tokens)
    if settings.seed is not None:
        torch.manual_seed(settings.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(settings.seed)

    with manager.inference_lock, torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    generated_ids = generated[0, prompt_tokens:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if generated_ids.numel() == 0:
        warnings.append("The model generated no tokens; the lens covers the prompt only.")

    full_ids = generated[0:1]
    full_mask = torch.ones_like(full_ids)
    if full_ids.shape[1] > MAX_POSITIONS:
        raise ValueError(
            f"The lens covers at most {MAX_POSITIONS} positions; this run has {full_ids.shape[1]}."
        )

    with manager.inference_lock, torch.inference_mode():
        output = model(input_ids=full_ids, attention_mask=full_mask, output_hidden_states=True)
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise ValueError("The loaded model did not return hidden states for the lens.")
    total_positions = int(full_ids.shape[1])
    position = _resolve(
        request.position if request.position is not None else prompt_tokens - 1,
        total_positions,
    )

    vocab: dict[int, str] = {}
    layers: list[LensLayer] = []
    for layer_index, hidden in enumerate(hidden_states):
        with torch.inference_mode():
            logits = model.lm_head(model.model.embedding_norm(hidden))
            probs = torch.softmax(logits[0].float(), dim=-1)
        top1_probs, top1_ids = probs.max(dim=-1)
        selected = probs[position]
        values, indices = torch.topk(selected, k=min(request.top_k, selected.shape[0]))
        top_tokens = []
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            token = _top_token(tokenizer, selected, index, value)
            vocab[token.token_id] = token.text
            top_tokens.append(token)
        module_id = "embeddings" if layer_index == 0 else f"model.layers.{layer_index - 1}"
        ids = top1_ids.tolist()[:MAX_POSITIONS]
        probs_list = [round(float(value), 6) for value in top1_probs.tolist()[:MAX_POSITIONS]]
        for token_id in ids:
            vocab.setdefault(int(token_id), _decode(tokenizer, int(token_id)))
        layers.append(
            LensLayer(
                layer_index=layer_index,
                module_id=module_id,
                top1_ids=[int(value) for value in ids],
                top1_probs=probs_list,
                top_tokens=top_tokens,
            )
        )
        del logits

    # ------------------------------------------------------------- ablation
    prompt_ids = inputs["input_ids"]
    prompt_mask = inputs.get("attention_mask", torch.ones_like(prompt_ids))
    with manager.inference_lock, torch.inference_mode():
        base_logits = model(input_ids=prompt_ids, attention_mask=prompt_mask).logits[0, -1].float()
    base_probs = torch.softmax(base_logits, dim=-1)
    last = prompt_tokens - 1
    for layer_index in range(1, len(hidden_states)):
        module_id = f"model.layers.{layer_index - 1}"
        module = get_module(model, module_id)
        if type(module).__name__ not in STEERABLE_CLASSES:
            continue

        def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                return output
            zeroed = torch.zeros_like(tensor)
            return (zeroed, *output[1:]) if isinstance(output, tuple) else zeroed

        handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                logits = model(input_ids=prompt_ids, attention_mask=prompt_mask).logits[0, last].float()
        finally:
            handle.remove()
        probs = torch.softmax(logits, dim=-1)
        kl = float(torch.sum(base_probs * (torch.log(base_probs + 1e-12) - torch.log(probs + 1e-12))))
        value, index = probs.max(dim=-1)
        token = _top_token(tokenizer, probs, int(index), float(value))
        vocab.setdefault(token.token_id, token.text)
        target = layers[layer_index]
        target.ablation_kl = round(kl, 6) if math.isfinite(kl) else None
        target.ablation_top1 = token

    tokens: list[TokenInfo] = _build_tokens(
        tokenizer,
        full_ids[0, :prompt_tokens].tolist(),
        generated_ids.tolist(),
    )
    return LensResponse(
        prompt=request.prompt,
        system=request.system,
        output=output_text,
        prompt_tokens=prompt_tokens,
        position=position,
        tokens=tokens,
        layers=layers,
        vocab=vocab,
        warnings=warnings,
    )
