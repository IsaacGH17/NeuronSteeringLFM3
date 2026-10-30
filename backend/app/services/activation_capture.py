"""Forward-hook capture of selected module activations.

Hooks are installed only for the requested generation and always removed in a
``finally`` block. Captured tensors are detached, copied to CPU float32, and
never kept on the accelerator. Only explicitly selected modules are captured.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from backend.app.services.architecture_inspector import STEERABLE_CLASSES, get_module

MAX_CAPTURE_MODULES = 8
MAX_CAPTURE_POSITIONS = 2048


def validate_capture_module_ids(model: Any, module_ids: list[str]) -> list[str]:
    """Deduplicate and verify capture targets before generation starts."""
    deduplicated = list(dict.fromkeys(module_ids))
    if len(deduplicated) > MAX_CAPTURE_MODULES:
        raise ValueError(f"At most {MAX_CAPTURE_MODULES} modules can be captured in one run.")
    for module_id in deduplicated:
        try:
            module = get_module(model, module_id)
        except KeyError as exc:
            raise ValueError(f"Unknown capture module {module_id!r}.") from exc
        class_name = type(module).__name__
        if class_name not in STEERABLE_CLASSES:
            raise ValueError(
                f"Module {module_id!r} ({class_name}) cannot be captured: "
                "its forward output is not a hidden-state tensor."
            )
    return deduplicated


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    return None


@contextmanager
def capture_modules(
    model: Any, module_ids: list[str]
) -> Iterator[tuple[dict[str, list[torch.Tensor]], list[str]]]:
    """Capture the output tensor of each module for every forward call."""
    captures: dict[str, list[torch.Tensor]] = {module_id: [] for module_id in module_ids}
    warnings: list[str] = []
    warned: set[str] = set()
    handles = []

    def make_hook(module_id: str):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            tensor = _extract_tensor(output)
            if tensor is None:
                if module_id not in warned:
                    warnings.append(f"{module_id} did not return a tensor; no activation captured.")
                    warned.add(module_id)
                return
            if tensor.ndim != 3:
                if module_id not in warned:
                    warnings.append(
                        f"{module_id} returned shape {list(tensor.shape)}; expected [batch, seq, hidden]."
                    )
                    warned.add(module_id)
                return
            captures[module_id].append(tensor.detach().to(device="cpu", dtype=torch.float32, copy=True))

        return hook

    try:
        for module_id in module_ids:
            handles.append(get_module(model, module_id).register_forward_hook(make_hook(module_id)))
        yield captures, warnings
    finally:
        for handle in handles:
            handle.remove()


def finalize_captures(
    captures: dict[str, list[torch.Tensor]],
    *,
    token_count: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Concatenate per-call chunks along the sequence dimension."""
    finalized: dict[str, torch.Tensor] = {}
    warnings: list[str] = []
    for module_id, chunks in captures.items():
        if not chunks:
            warnings.append(f"{module_id} was never called during generation.")
            continue
        tensor = torch.cat(chunks, dim=1)
        if tensor.shape[0] != 1:
            warnings.append(f"{module_id} batch size {tensor.shape[0]} was averaged into one row.")
            tensor = tensor.mean(dim=0, keepdim=True)
        tensor = tensor.squeeze(0)
        if token_count is not None and tensor.shape[0] != token_count:
            warnings.append(
                f"{module_id} captured {tensor.shape[0]} positions for {token_count} tokens; "
                "token alignment may be approximate."
            )
        finalized[module_id] = tensor
    return finalized, warnings
