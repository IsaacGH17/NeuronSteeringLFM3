"""Patch sweep: causal tracing over layers and prompt positions.

One baseline generation and one donor prefill are shared by the whole grid. Each
cell then runs one recipient generation with the donor's activation patched into
one decoder layer at one prompt position for the first forward only. The grid
reports whether the text changed, whether the first generated token changed, and
how many generated tokens still share a prefix.

The intervention is out-of-place, lasts for the first forward only, and every
hook is removed in ``finally``. Weights never change.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from backend.app.schemas.api import (
    BaselineRequest,
    PatchSweepCell,
    PatchSweepRequest,
    PatchSweepResponse,
)
from backend.app.services.activation_capture import (
    capture_modules,
    finalize_captures,
    validate_capture_module_ids,
)
from backend.app.services.architecture_inspector import build_layer_overview, get_module
from backend.app.services.generation import generate_chat, render_chat_prompt
from backend.app.services.model_loader import ModelManager

MAX_PATCH_SWEEP_CELLS = 48


def _resolve(position: int, size: int) -> int | None:
    resolved = size + position if position < 0 else position
    if resolved < 0 or resolved >= size:
        return None
    return resolved


def _generated_ids(result) -> list[int]:
    return [token.token_id for token in result.tokens if token.phase == "generated"]


def run_patch_sweep(manager: ModelManager, request: PatchSweepRequest) -> PatchSweepResponse:
    model = manager.model
    tokenizer = manager.tokenizer
    overview = build_layer_overview(model)
    available = [block.layer_index for block in overview]
    if not available:
        raise ValueError("The loaded model exposes no decoder layers to patch.")

    if request.layer_indices:
        layers = sorted(set(request.layer_indices))
        invalid = [index for index in layers if index not in available]
        if invalid:
            raise ValueError(f"Layer indices {invalid} are outside the available layers {available}.")
    else:
        layers = available

    baseline_request = BaselineRequest(
        prompt=request.recipient_prompt,
        system=request.recipient_system,
        settings=request.settings,
    )
    baseline = generate_chat(manager, baseline_request, condition="baseline")
    prompt_tokens = baseline.response.prompt_tokens

    warnings: list[str] = []
    positions: list[int] = []
    for requested in request.positions:
        resolved = _resolve(requested, prompt_tokens)
        if resolved is None:
            warnings.append(
                f"Position {requested} is outside the recipient prompt "
                f"[0, {prompt_tokens - 1}] and was skipped."
            )
            continue
        if resolved not in positions:
            positions.append(resolved)
    if not positions:
        raise ValueError("No valid recipient positions were left after resolving the request.")
    if len(layers) * len(positions) > MAX_PATCH_SWEEP_CELLS:
        raise ValueError(
            f"A patch sweep is limited to {MAX_PATCH_SWEEP_CELLS} cells; "
            f"this grid would run {len(layers) * len(positions)}. "
            "Narrow the layer range or use fewer positions."
        )

    donor_request = BaselineRequest(
        prompt=request.donor_prompt,
        system=request.donor_system,
        settings=request.settings,
    )
    donor_templated = render_chat_prompt(
        tokenizer,
        request.donor_prompt,
        request.donor_system,
        request.settings.enable_thinking,
    )
    encoded = tokenizer(donor_templated, return_tensors="pt")
    donor_inputs = {
        key: value.to(manager.device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    donor_prompt_tokens = int(donor_inputs["input_ids"].shape[1])
    layer_ids = [f"model.layers.{index}" for index in layers]
    validate_capture_module_ids(model, layer_ids)
    with manager.inference_lock, torch.inference_mode():
        with capture_modules(model, layer_ids) as (chunks, capture_warnings):
            model(**donor_inputs, use_cache=False)
    captures, finalize_warnings = finalize_captures(chunks)
    warnings.extend(capture_warnings)
    warnings.extend(finalize_warnings)
    donor = generate_chat(manager, donor_request, condition="baseline")
    baseline_generated = _generated_ids(baseline)

    start = time.perf_counter()
    cells: list[PatchSweepCell] = []
    for layer_index in layers:
        module_id = f"model.layers.{layer_index}"
        if module_id not in captures:
            raise ValueError(f"Layer {module_id!r} did not produce a donor activation.")
        donor_tensor = captures[module_id]
        module = get_module(model, module_id)
        for position in positions:
            donor_position = min(position, donor_prompt_tokens - 1)
            donor_vector = donor_tensor[donor_position].to(manager.device)
            state = {"calls": 0}

            def hook(
                module: Any,
                args: tuple[Any, ...],
                output: Any,
                *,
                donor_vector: torch.Tensor = donor_vector,
                position: int = position,
                state: dict[str, int] = state,
            ) -> Any:
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
                patched = generate_chat(manager, baseline_request, condition="baseline")
            finally:
                handle.remove()

            patched_generated = _generated_ids(patched)
            first_token_changed = bool(
                baseline_generated and patched_generated and baseline_generated[0] != patched_generated[0]
            )
            common_prefix = 0
            for baseline_token, patched_token in zip(baseline_generated, patched_generated, strict=False):
                if baseline_token != patched_token:
                    break
                common_prefix += 1
            cell_warnings: list[str] = []
            if state["calls"] == 0:
                cell_warnings.append(
                    f"The patch hook on {module_id} never fired; this cell equals the baseline."
                )
            if donor_position != position:
                cell_warnings.append(
                    f"The donor has {donor_prompt_tokens} prompt tokens; its last token was used "
                    f"instead of position {position}."
                )
            cells.append(
                PatchSweepCell(
                    layer_index=layer_index,
                    module_id=module_id,
                    position=position,
                    donor_position=donor_position,
                    patched_output=patched.response.output,
                    changed=patched.response.output != baseline.response.output,
                    first_token_changed=first_token_changed,
                    common_prefix_tokens=common_prefix,
                    generated_tokens=len(patched_generated),
                    warnings=cell_warnings,
                )
            )
    duration_ms = (time.perf_counter() - start) * 1000.0

    return PatchSweepResponse(
        recipient_prompt=request.recipient_prompt,
        donor_prompt=request.donor_prompt,
        baseline_output=baseline.response.output,
        donor_output=donor.response.output,
        prompt_tokens=prompt_tokens,
        donor_prompt_tokens=donor_prompt_tokens,
        layers=layers,
        positions=positions,
        cells=cells,
        duration_ms=round(duration_ms, 2),
        warnings=warnings,
    )
