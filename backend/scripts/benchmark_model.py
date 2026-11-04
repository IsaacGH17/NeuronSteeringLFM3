"""Measure real LFM3 latency and VRAM use on the local device.

Run from the repository root:

    uv run python -m backend.scripts.benchmark_model
    uv run python -m backend.scripts.benchmark_model --device cpu --max-new-tokens 8

The first generation includes the ROCm/SDPA warmup; the script warms up twice
before timing so the reported numbers reflect steady-state decoding.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from backend.app.core.settings import Settings
from backend.app.schemas.api import BaselineRequest, GenerationSettings, SteeringSpec
from backend.app.services.generation import GenerationResult, generate_baseline, generate_steered
from backend.app.services.model_loader import ModelManager

PROMPT = "Reply with one short sentence about the ocean."
DEFAULT_CAPTURE = ["model.layers.22.feed_forward"]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed(label: str, device: str, call: Callable[[], GenerationResult]) -> None:
    _sync(device)
    started = time.perf_counter()
    result = call()
    _sync(device)
    elapsed = time.perf_counter() - started
    response = result.response
    rate = response.tokens_per_second or 0.0
    print(
        f"{label:<34} {elapsed:6.2f}s  {response.generated_tokens:3d} tok  "
        f"{rate:6.1f} tok/s  {response.duration_ms:7.0f} ms (engine)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--capture-module", default=DEFAULT_CAPTURE[0])
    args = parser.parse_args()

    settings = Settings(device=args.device, max_new_tokens=max(args.max_new_tokens, 64))
    manager = ModelManager(settings)
    started = time.perf_counter()
    manager.load()
    load_seconds = time.perf_counter() - started
    status = manager.status()
    print(f"load {load_seconds:.2f}s · device={manager.device} · dtype={status.dtype}")

    def request(tokens: int) -> BaselineRequest:
        return BaselineRequest(
            prompt=PROMPT,
            settings=GenerationSettings(
                max_new_tokens=tokens,
                temperature=0,
                top_p=1.0,
                enable_thinking=False,
            ),
        )

    if manager.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for _ in range(2):
        generate_baseline(manager, request(8))

    for tokens in (8, args.max_new_tokens):
        _timed(
            f"baseline n={tokens}", manager.device, lambda t=tokens: generate_baseline(manager, request(t))
        )

    capture = [args.capture_module]
    _timed(
        f"baseline + capture n={args.max_new_tokens}",
        manager.device,
        lambda: generate_baseline(manager, request(args.max_new_tokens), capture_ids=capture),
    )
    steering = SteeringSpec(module_id=args.capture_module, feature_index=500, strength=-12)
    _timed(
        f"steered α=-12 + capture n={args.max_new_tokens}",
        manager.device,
        lambda: generate_steered(manager, request(args.max_new_tokens), steering, capture_ids=capture),
    )

    if manager.device.startswith("cuda"):
        print(
            f"peak allocated {torch.cuda.max_memory_allocated() / 1e9:.2f} GB · "
            f"reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )


if __name__ == "__main__":
    main()
