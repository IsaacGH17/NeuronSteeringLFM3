"""Activation patching: run a recipient with one module state taken from a donor.

The donor's activation at a chosen module and prompt position replaces the
recipient's state at the same module and position during the prefill; generation
continues normally from there. The intervention is out-of-place, lasts for the
first forward only, and the hook is removed in ``finally``. Weights never change.

This is the standard causal-tracing move: if the donor's answer travels with the
activation, the module is part of the circuit that computes it.
"""

from __future__ import annotations

from typing import Any

import torch

from backend.app.schemas.api import (
    BaselineRequest,
    PatchRequest,
    PatchResponse,
)
from backend.app.services.activation_capture import (
    capture_modules,
    finalize_captures,
    validate_capture_module_ids,
)
from backend.app.services.architecture_inspector import get_module
from backend.app.services.generation import generate_chat
from backend.app.services.model_loader import ModelManager


def _resolve(position: int, size: int, label: str) -> int:
    resolved = size + position if position < 0 else position
    if resolved < 0 or resolved >= size:
        raise ValueError(f"{label} position {position} is outside the range [0, {size - 1}].")
    return resolved


def _donor_vector(manager: ModelManager, request: PatchRequest) -> tuple[torch.Tensor, int, list[str]]:
    tokenizer = manager.tokenizer
    templated = _apply(tokenizer, request.donor_prompt, request.donor_system)
    encoded = tokenizer(templated, return_tensors="pt")
    inputs = {key: value.to(manager.device) for key, value in encoded.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    with manager.inference_lock, torch.inference_mode():
        with capture_modules(manager.model, [request.module_id]) as (chunks, warnings):
            manager.model(**inputs)
    captures, finalize_warnings = finalize_captures(chunks)
    if request.module_id not in captures:
        raise ValueError(f"Module {request.module_id!r} did not produce an activation.")
    tensor = captures[request.module_id]
    position = _resolve(request.position, prompt_tokens, "donor")
    return tensor[position].to(manager.device), prompt_tokens, warnings + finalize_warnings


def _apply(tokenizer: Any, prompt: str, system: str | None) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:  # pragma: no cover - older templates
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def run_patch(manager: ModelManager, request: PatchRequest) -> PatchResponse:
    validate_capture_module_ids(manager.model, [request.module_id])
    donor_vector, donor_tokens, donor_warnings = _donor_vector(manager, request)

    recipient_request = BaselineRequest(
        prompt=request.recipient_prompt,
        system=request.recipient_system,
        settings=request.settings,
    )
    baseline = generate_chat(manager, recipient_request, condition="baseline")
    prompt_tokens = baseline.response.prompt_tokens
    position = _resolve(request.position, prompt_tokens, "recipient")

    module = get_module(manager.model, request.module_id)
    state = {"calls": 0}

    def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
        if state["calls"] > 0:
            return output
        state["calls"] += 1
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return output
        if tensor.shape[1] <= position:
            return output
        patched = tensor.clone()
        patched[:, position, :] = donor_vector.to(patched.dtype)
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

    handle = module.register_forward_hook(hook)
    try:
        patched = generate_chat(manager, recipient_request, condition="steered")
    finally:
        handle.remove()

    donor_request = BaselineRequest(
        prompt=request.donor_prompt,
        system=request.donor_system,
        settings=request.settings,
    )
    donor = generate_chat(manager, donor_request, condition="baseline")
    warnings = list(donor_warnings)
    if state["calls"] == 0:
        warnings.append(
            f"The patch hook on {request.module_id} never fired; the patched output equals the baseline."
        )
    return PatchResponse(
        module_id=request.module_id,
        position=position,
        donor_output=donor.response.output,
        baseline_output=baseline.response.output,
        patched_output=patched.response.output,
        changed=patched.response.output != baseline.response.output,
        prompt_tokens=prompt_tokens,
        warnings=warnings,
    )
