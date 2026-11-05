"""Format pilot: is bullet-vs-prose steerable with a multi-layer direction?

Extraction uses the format contrast in the only place where format exists: the
generation itself. Each prompt is generated under a "three bullet points" system
and under a "one flowing paragraph" system, and the per-layer difference of the
mean generated-token state is the direction. The direction is then applied at
several residual layers (full vectors, alpha / sqrt(L)) and generation is scored
with structural features: bullet lines, headings, words, sentences.

Pre-registered criteria:
  * install: at the chosen alpha, bullet lines increase on >= 3 of 4 held-out
    prompts.
  * control: random directions of the same magnitude do so on <= 1 of 4.
  * coherence: outputs are non-empty and distinct-3 >= 0.65.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from backend.app.core.settings import Settings
from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.generation import generate_baseline
from backend.app.services.metrics import compute_metrics
from backend.app.services.model_loader import ModelManager
from backend.scripts.multilayer_probe import MultiLayerSteering

BULLET_SYSTEM = (
    "Answer with exactly three bullet points. Each bullet is one line. No introduction, no conclusion."
)
PROSE_SYSTEM = "Answer in one flowing paragraph of three sentences. No lists, no bullet points, no headings."
PROMPTS = (
    "How do I learn to play guitar?",
    "What are the tradeoffs of working from home?",
    "Describe a rainy morning.",
    "Describe how a bicycle works.",
)
LADDER = PROMPTS[:2]
HELD_OUT = PROMPTS[2:]
LAYERS = (8, 12, 16, 20, 22)
ALPHAS = (0.5, 1.0, 2.0, 3.0)


@dataclass
class Run:
    phase: str
    label: str
    prompt: str
    alpha: float | None
    bullets: int
    headings: int
    words: int
    sentences: int
    distinct_3: float
    output: str


@dataclass
class Report:
    created_at: str
    device: str
    dtype: str
    config: dict[str, object]
    directions: list[dict[str, object]] = field(default_factory=list)
    ladder: list[dict[str, object]] = field(default_factory=list)
    held_out: list[dict[str, object]] = field(default_factory=list)
    capability: list[dict[str, object]] = field(default_factory=list)
    chosen: dict[str, object] = field(default_factory=dict)
    runs: list[Run] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def format_features(text: str) -> dict[str, int]:
    lines = text.splitlines()
    return {
        "bullets": sum(1 for line in lines if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)),
        "headings": sum(1 for line in lines if line.strip().startswith("#")),
        "words": len(re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’\-]*", text)),
        "sentences": max(1, len([part for part in re.split(r"[.!?]+", text) if part.strip()])),
    }


def record(phase: str, label: str, prompt: str, alpha: float | None, output: str) -> Run:
    features = format_features(output)
    return Run(
        phase=phase,
        label=label,
        prompt=prompt,
        alpha=alpha,
        bullets=features["bullets"],
        headings=features["headings"],
        words=features["words"],
        sentences=features["sentences"],
        distinct_3=compute_metrics(output).distinct_3,
        output=output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"format-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=128))
    started = time.perf_counter()
    status = manager.load()
    model = manager.model
    settings = GenerationSettings(max_new_tokens=args.tokens)
    modules = [f"model.layers.{layer}" for layer in LAYERS]
    report = Report(
        created_at=datetime.now(UTC).isoformat(),
        device=status.device or args.device,
        dtype=status.dtype or "?",
        config={
            "layers": list(LAYERS),
            "alphas": list(ALPHAS),
            "prompts": list(PROMPTS),
            "bullet_system": BULLET_SYSTEM,
            "prose_system": PROSE_SYSTEM,
            "tokens": args.tokens,
        },
    )
    print(f"[load] {status.state} {status.device} {status.dtype} in {time.perf_counter() - started:.1f}s")

    # ------------------------------------------------------------- extraction
    bullet_means: list[dict[str, np.ndarray]] = []
    prose_means: list[dict[str, np.ndarray]] = []
    for prompt in PROMPTS:
        for label, system, bucket in (
            ("bullets", BULLET_SYSTEM, bullet_means),
            ("prose", PROSE_SYSTEM, prose_means),
        ):
            result = generate_baseline(
                manager, BaselineRequest(prompt=prompt, system=system, settings=settings), capture_ids=modules
            )
            prompt_tokens = result.response.prompt_tokens
            means = {}
            for module in modules:
                tensor = result.captures[module]
                generated = tensor[prompt_tokens:] if tensor.shape[0] > prompt_tokens else tensor[-1:]
                means[module] = generated.numpy().astype(np.float64).mean(axis=0)
            bucket.append(means)
            report.runs.append(record("extraction", label, prompt, None, result.response.output))
            print(
                f"[extract] {label:7s} «{prompt[:28]}» bullets={format_features(result.response.output)['bullets']} "
                f"words={format_features(result.response.output)['words']}"
            )

    vectors: dict[str, np.ndarray] = {}
    for module in modules:
        differences = [
            bullet[module] - prose[module] for bullet, prose in zip(bullet_means, prose_means, strict=True)
        ]
        vector = np.mean(differences, axis=0)
        vectors[module] = vector
        report.directions.append(
            {
                "module_id": module,
                "norm": round(float(np.linalg.norm(vector)), 4),
                "top_dimensions": [int(index) for index in np.argsort(-np.abs(vector))[:8]],
            }
        )
        print(f"[direction] {module}: norm {np.linalg.norm(vector):.4f}")

    unit_vectors = {
        module: torch.tensor(
            vectors[module] / max(np.linalg.norm(vectors[module]), 1e-9), device=status.device
        )
        for module in modules
    }

    def steer(alpha: float, directions=None) -> MultiLayerSteering:
        chosen = directions or unit_vectors
        scale = {module: alpha / math.sqrt(len(chosen)) for module in chosen}
        return MultiLayerSteering(model, chosen, scale)

    # ----------------------------------------------------------------- ladder
    ladder_rows: list[dict[str, object]] = []
    for prompt in LADDER:
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        base_record = record("ladder", "baseline", prompt, None, base.output)
        report.runs.append(base_record)
        for alpha in ALPHAS:
            with steer(alpha):
                result = generate_baseline(
                    manager, BaselineRequest(prompt=prompt, settings=settings)
                ).response
            row = record("ladder", f"alpha {alpha}", prompt, alpha, result.output)
            report.runs.append(row)
            ladder_rows.append(
                {
                    "prompt": prompt,
                    "alpha": alpha,
                    "bullets": row.bullets,
                    "bullets_delta": row.bullets - base_record.bullets,
                    "words": row.words,
                    "distinct_3": row.distinct_3,
                    "output": row.output,
                }
            )
            print(
                f"[ladder] «{prompt[:22]}» α={alpha}: bullets={row.bullets} "
                f"(base {base_record.bullets}) words={row.words}"
            )
    report.ladder = ladder_rows

    def bullets_delta(alpha: float) -> float:
        rows = [row for row in ladder_rows if row["alpha"] == alpha]
        return statistics.fmean([int(row["bullets_delta"]) for row in rows]) if rows else -99.0

    scores = {alpha: bullets_delta(alpha) for alpha in ALPHAS}
    chosen_alpha = max(ALPHAS, key=bullets_delta)
    report.chosen = {
        "alpha": chosen_alpha,
        "ladder_mean_bullet_delta": {str(alpha): value for alpha, value in scores.items()},
        "rule": "max(mean bullet-line delta) on the two ladder prompts",
    }
    print(f"[phase2] chosen α={chosen_alpha} scores={scores}")

    # -------------------------------------------------------------- held-out
    for prompt in HELD_OUT:
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        base_record = record("held_out", "baseline", prompt, None, base.output)
        with steer(chosen_alpha):
            steered = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        steered_record = record("held_out", "steered", prompt, chosen_alpha, steered.output)
        rng = random.Random(101)
        random_directions = {}
        for module in modules:
            raw = np.array([rng.gauss(0.0, 1.0) for _ in range(model.config.hidden_size)])
            random_directions[module] = torch.tensor(raw / np.linalg.norm(raw), device=status.device)
        with steer(chosen_alpha, random_directions):
            control = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        control_record = record("held_out", "random control", prompt, chosen_alpha, control.output)
        report.runs += [base_record, steered_record, control_record]
        report.held_out.append(
            {
                "prompt": prompt,
                "baseline_bullets": base_record.bullets,
                "steered_bullets": steered_record.bullets,
                "control_bullets": control_record.bullets,
                "steered_words": steered_record.words,
                "distinct_3": steered_record.distinct_3,
                "output": steered_record.output,
            }
        )
        print(
            f"[held-out] «{prompt[:26]}» bullets {base_record.bullets} -> {steered_record.bullets} "
            f"(random {control_record.bullets})"
        )

    # ------------------------------------------------------------ capability
    for prompt, needle in (
        ("Reply with exactly one word: the opposite of 'hot'.", "cold"),
        ("What is 17 + 26? Reply with only the number.", "43"),
    ):
        small = GenerationSettings(max_new_tokens=16)
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=small)).response
        with steer(chosen_alpha):
            steered = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=small)).response
        report.capability.append(
            {
                "prompt": prompt,
                "baseline_ok": needle in base.output.lower(),
                "steered_ok": needle in steered.output.lower(),
            }
        )

    moved = sum(1 for row in report.held_out if row["steered_bullets"] > row["baseline_bullets"])
    control_moved = sum(1 for row in report.held_out if row["control_bullets"] > row["baseline_bullets"])
    report.notes = [
        f"Bullet lines increased on {moved}/{len(report.held_out)} held-out prompts at α={chosen_alpha}; "
        f"random-direction control on {control_moved}/{len(report.held_out)}.",
        f"Capability needles: baseline {sum(1 for row in report.capability if row['baseline_ok'])}/2, "
        f"steered {sum(1 for row in report.capability if row['steered_ok'])}/2.",
        "Direction extracted from generated-token states under bullet vs prose instructions.",
        "Structural metrics only: bullet lines, headings, words, sentences, distinct-3.",
    ]
    print("\n".join(f"- {note}" for note in report.notes))
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
