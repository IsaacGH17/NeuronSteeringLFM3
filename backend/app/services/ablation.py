"""Per-layer ablation importance for the next-token distribution.

Skipping one decoder layer's contribution on the prompt prefill and measuring the
KL divergence of the next-token distribution gives a cheap causal-importance map.
In a residual stream, zeroing the layer output would also cut the stream for
every later layer; the ablation therefore returns the layer's *input* instead,
which preserves the residual and removes only that layer's computation. This is
the same intervention the Interpret drawer's ablation uses, exposed as a small
reusable service so the flow trace can attach it to its layer nodes.

The ablation never changes weights; hooks are installed per layer and removed in
``finally``. The baseline and ablated distributions both come from the model's
own forward pass.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from backend.app.services.architecture_inspector import get_module
from backend.app.services.generation import render_chat_prompt
from backend.app.services.model_loader import ModelManager
from backend.app.services.next_token import last_token_logits

_EPS = 1e-12


def _skip_layer_hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
    """Return the layer input, skipping the layer while preserving the residual."""
    if not args or not isinstance(args[0], torch.Tensor):
        return output
    replacement = args[0].clone()
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (replacement, *output[1:])
    return output


def compute_ablation_kl(
    manager: ModelManager,
    *,
    prompt: str,
    system: str | None,
    enable_thinking: bool,
    layer_ids: list[str],
) -> tuple[list[float | None], list[str]]:
    """Return per-layer KL and warnings, in the same order as ``layer_ids``."""
    model = manager.model
    tokenizer = manager.tokenizer
    templated = render_chat_prompt(tokenizer, prompt, system, enable_thinking)
    encoded = tokenizer(templated, return_tensors="pt")
    inputs = {
        key: value.to(manager.device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }

    results: list[float | None] = []
    warnings: list[str] = []
    with manager.inference_lock, torch.inference_mode():
        baseline = torch.softmax(last_token_logits(manager, inputs), dim=-1)
        for module_id in layer_ids:
            try:
                module = get_module(model, module_id)
            except KeyError as exc:
                results.append(None)
                warnings.append(f"Ablation of {module_id} failed: {exc}")
                continue
            handle = module.register_forward_hook(_skip_layer_hook)
            try:
                ablated = torch.softmax(last_token_logits(manager, inputs), dim=-1)
                kl = float(torch.sum(baseline * (torch.log(baseline + _EPS) - torch.log(ablated + _EPS))))
                results.append(round(max(0.0, kl), 6) if math.isfinite(kl) else None)
            except Exception as exc:  # noqa: BLE001 - report the layer, keep the rest
                results.append(None)
                warnings.append(f"Ablation of {module_id} failed: {exc}")
            finally:
                handle.remove()
    return results, warnings
