"""Pilot: find a tone direction that moves style without breaking utility.

Runs on the real loaded model, in three phases:

1. **Extract.** The same neutral prompts are generated under a warm/casual and
   a clinical/terse system prompt; for each candidate module the difference of
   the last prompt-token states is a direction in activation space (contrastive
   activation addition, reduced to its top-k dimensions).
2. **Ladder.** Steer the most separable direction in sigma units across a range
   and measure tone markers, length, and repetition.
3. **Validate.** Apply the chosen alpha to held-out prompts and measure how much
   of the baseline-to-warm-target style distance it recovers, plus capability
   checks and a random-direction control at the same magnitude.

Every number in the report comes from the runs stored in the JSON file next to
it. This is a pilot, not proof: the sample is small and the metrics are lexical
heuristics.
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
from backend.app.schemas.api import BaselineRequest, GenerationSettings, SteeringSpec, SuiteCheck
from backend.app.services.generation import generate_baseline, generate_steered
from backend.app.services.metrics import compute_metrics
from backend.app.services.model_loader import ModelManager

WARM_SYSTEM = (
    "You are a bubbly friend texting. Use exclamation marks, contractions, and at least one emoji. "
    "Keep it short, personal, and enthusiastic."
)
CLINICAL_SYSTEM = (
    "You are a technical report writer. Use only declarative sentences with no adjectives, no emojis, "
    "no exclamation marks, and no contractions. Be terse and factual."
)

EXTRACTION_PROMPTS = (
    "Describe a rainy morning.",
    "What should I cook tonight? Give me one idea.",
    "Describe your ideal weekend in three sentences.",
    "Give me a two-sentence review of a coffee shop.",
)
LADDER_PROMPTS = (
    "Explain why the sky is blue.",
    "Describe how a bicycle works.",
)
HELD_OUT_PROMPTS = (
    "How do I learn to play guitar?",
    "Summarize the plot of a heist film in three sentences.",
    "Write a short note to a coworker about a delayed report.",
    "What are the tradeoffs of working from home?",
    "A friend asks for advice about a difficult decision. Reply briefly.",
    "Explain what a transformer activation is in two sentences.",
)
CAPABILITY_PROMPTS: tuple[tuple[str, tuple[SuiteCheck, ...]], ...] = (
    (
        "Reply with exactly one word: the opposite of 'hot'.",
        (SuiteCheck(kind="max_words", value=3), SuiteCheck(kind="contains", value="cold")),
    ),
    (
        "What is 17 + 26? Reply with only the number.",
        (SuiteCheck(kind="is_number"), SuiteCheck(kind="contains", value="43")),
    ),
    (
        "List exactly three colors, one per line. No other text.",
        (SuiteCheck(kind="list_items", value=3),),
    ),
    (
        'Reply with JSON only: {"ok": true}',
        (SuiteCheck(kind="regex", value=r"\{.*\}"),),
    ),
    (
        "Name the largest planet in the solar system. One word.",
        (SuiteCheck(kind="contains", value="jupiter"),),
    ),
    (
        "Translate to English: 'la casa es grande'.",
        (SuiteCheck(kind="regex", value=r"house"), SuiteCheck(kind="regex", value=r"big|large")),
    ),
)


# ----------------------------------------------------------------- data model
@dataclass
class Capture:
    prompt: str
    system: str
    label: str
    captures: dict[str, torch.Tensor]
    prompt_tokens: int
    record: RunRecord


@dataclass
class RunRecord:
    phase: str
    label: str
    prompt: str
    system: str | None
    module_id: str | None
    alpha: float | None
    normalization: str | None
    output: str
    words: int
    sentences: int
    distinct_3: float
    repeated_4gram_ratio: float
    refusal: bool
    lexicon: dict[str, float]
    style: dict[str, int]
    checks: list[dict[str, object]]
    duration_ms: float


@dataclass
class DirectionSummary:
    module_id: str
    components: list[dict[str, float]]
    sigma: list[float]
    norm: float
    separation: float
    top_dimensions: list[int]


@dataclass
class PilotReport:
    created_at: str
    device: str
    dtype: str
    config: dict[str, object]
    directions: list[DirectionSummary] = field(default_factory=list)
    chosen: dict[str, object] = field(default_factory=dict)
    ladder: list[dict[str, object]] = field(default_factory=list)
    held_out: list[dict[str, object]] = field(default_factory=list)
    capability: list[dict[str, object]] = field(default_factory=list)
    control: list[dict[str, object]] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------- helpers
def record_run(
    phase: str,
    label: str,
    request: BaselineRequest,
    response,
    *,
    module_id: str | None,
    alpha: float | None = None,
    normalization: str | None = None,
    checks: tuple[SuiteCheck, ...] = (),
) -> RunRecord:
    metrics = compute_metrics(response.output, list(checks))
    return RunRecord(
        phase=phase,
        label=label,
        prompt=request.prompt,
        system=request.system,
        module_id=module_id,
        alpha=alpha,
        normalization=normalization,
        output=response.output,
        words=metrics.words,
        sentences=metrics.sentences,
        distinct_3=metrics.distinct_3,
        repeated_4gram_ratio=metrics.repeated_4gram_ratio,
        refusal=metrics.refusal,
        lexicon=metrics.lexicon,
        style=style_features(response.output),
        checks=[check.model_dump() for check in metrics.checks],
        duration_ms=response.duration_ms,
    )


_EMOJI_RE = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")
_CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|we|our|us)\b")
STYLE_KEYS = ("emoji", "exclamations", "contractions", "first_person")


def style_features(text: str) -> dict[str, int]:
    """Cheap, content-light markers of the warm/casual vs clinical/terse axis."""
    return {
        "emoji": len(_EMOJI_RE.findall(text)),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "contractions": len(_CONTRACTION_RE.findall(text)),
        "first_person": len(_FIRST_PERSON_RE.findall(text)),
    }


def style_distance(left: dict[str, int], right: dict[str, int]) -> float:
    return float(sum(abs(int(left.get(key, 0)) - int(right.get(key, 0))) for key in STYLE_KEYS))


def prompt_slice(tensor: torch.Tensor, prompt_tokens: int) -> np.ndarray:
    rows = min(int(tensor.shape[0]), max(1, prompt_tokens))
    return tensor[:rows].numpy().astype(np.float64)


def compute_direction(
    module_id: str,
    positives: list[Capture],
    negatives: list[Capture],
    top_k: int,
    position_mode: str = "last_prompt",
) -> DirectionSummary:
    differences = []
    spreads = []
    for positive, negative in zip(positives, negatives, strict=True):
        pos = prompt_slice(positive.captures[module_id], positive.prompt_tokens)
        neg = prompt_slice(negative.captures[module_id], negative.prompt_tokens)
        shared = min(len(pos), len(neg))
        if position_mode == "last_prompt":
            differences.append(pos[shared - 1] - neg[shared - 1])
        else:
            differences.append(pos[:shared].mean(axis=0) - neg[:shared].mean(axis=0))
        # Scale always comes from the prompt spread, not from a single token.
        spreads.append(pos[:shared].std(axis=0))
    difference = np.mean(differences, axis=0)
    sigma = np.mean(spreads, axis=0)
    norm = float(np.linalg.norm(difference))
    mean_variance = float(np.mean(sigma**2))
    separation = norm / math.sqrt(mean_variance) if mean_variance > 0 else float("inf")
    order = np.argsort(-np.abs(difference))[:top_k]
    max_abs = float(np.max(np.abs(difference[order]))) or 1.0
    return DirectionSummary(
        module_id=module_id,
        components=[
            {"feature_index": int(index), "weight": round(float(difference[index] / max_abs), 5)}
            for index in order
        ],
        sigma=[round(float(sigma[int(index)]), 6) for index in order],
        norm=round(norm, 5),
        separation=round(separation, 4),
        top_dimensions=[int(index) for index in order],
    )


def tone_delta(steered: RunRecord, baseline: RunRecord) -> dict[str, float]:
    return {
        name: round(steered.lexicon.get(name, 0.0) - baseline.lexicon.get(name, 0.0), 3)
        for name in sorted(set(steered.lexicon) | set(baseline.lexicon))
    }


def markdown_report(report: PilotReport) -> str:
    lines = [
        "# Personality pilot",
        "",
        f"- created: `{report.created_at}`",
        f"- device: `{report.device}` · dtype `{report.dtype}`",
        f"- modules probed: {', '.join(f'`{item}`' for item in report.config['modules'])}",
        f"- top-k dimensions: {report.config['top_k']} · tokens per run: {report.config['tokens']}",
        "",
        "## Directions",
        "",
        "| module | separation | norm | top dimensions |",
        "|---|---:|---:|---|",
    ]
    for direction in report.directions:
        dims = ", ".join(str(index) for index in direction.top_dimensions)
        lines.append(
            f"| `{direction.module_id}` | {direction.separation:.3f} | {direction.norm:.3f} | {dims} |"
        )
    chosen = report.chosen
    lines += [
        "",
        "## Chosen setting",
        "",
        f"- module: `{chosen.get('module_id')}`",
        f"- alpha: **{chosen.get('alpha')} σ**",
        f"- rule: {chosen.get('rule')}",
        f"- zero strength reproduced the baseline: **{chosen.get('zero_matches_baseline')}**",
        "",
        "## Alpha ladder",
        "",
        "| prompt | α (σ) | emoji | ! | contractions | 1st person | words | distinct-3 | refusal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.ladder:
        style = row["style"]
        lines.append(
            f"| {row['prompt'][:26]} | {row['alpha']} | {style['emoji']} | {style['exclamations']} | "
            f"{style['contractions']} | {style['first_person']} | {row['words']} | "
            f"{row['distinct_3']:.2f} | {row['refusal']} |"
        )
    lines += [
        "",
        "## Held-out prompts",
        "",
        "| prompt | Δemoji | Δ! | Δcontractions | Δ1st | Δwords | toward warm target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.held_out:
        delta = row["delta_style"]
        fraction = row.get("fraction_warm")
        fraction_text = "—" if fraction is None else f"{fraction * 100:+.0f}%"
        lines.append(
            f"| {row['prompt'][:40]} | {delta['emoji']:+d} | {delta['exclamations']:+d} | "
            f"{delta['contractions']:+d} | {delta['first_person']:+d} | {row['delta_words']:+d} | "
            f"{fraction_text} |"
        )
    lines += [
        "",
        "## Capability checks",
        "",
        "| prompt | baseline | steered |",
        "|---|---|---|",
    ]
    for row in report.capability:
        lines.append(f"| {row['prompt'][:44]} | {row['baseline']} | {row['steered']} |")
    lines += [
        "",
        "## Random-direction control",
        "",
        "| run | style | toward warm target | checks |",
        "|---|---:|---:|---|",
    ]
    for row in report.control:
        checks = f"{row['checks_passed']}/{row['checks_total']}" if row.get("checks_total") else "—"
        style = row["style"]
        style_text = (
            f"emoji {style['emoji']}, ! {style['exclamations']}, "
            f"contr {style['contractions']}, 1st {style['first_person']}"
        )
        fraction = row.get("fraction_warm")
        fraction_text = "—" if fraction is None else f"{fraction * 100:+.0f}%"
        lines.append(f"| {row['label']} | {style_text} | {fraction_text} | {checks} |")
    lines += ["", "## Notes", ""] + [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--modules",
        default="model.layers.6.feed_forward,model.layers.12.feed_forward,"
        "model.layers.18.feed_forward,model.layers.22.feed_forward",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--capability-tokens", type=int, default=16)
    parser.add_argument("--alphas", default="-120,-80,-40,0,40,80,120")
    parser.add_argument("--position-mode", choices=("last_prompt", "mean_prompt"), default="last_prompt")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    modules = [item.strip() for item in args.modules.split(",") if item.strip()]
    alphas = [float(value) for value in args.alphas.split(",")]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"personality-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=256))
    load_started = time.perf_counter()
    status = manager.load()
    print(
        f"[load] {status.state} {status.device} {status.dtype} in {time.perf_counter() - load_started:.1f}s"
    )

    settings = GenerationSettings(max_new_tokens=args.tokens)
    capability_settings = GenerationSettings(max_new_tokens=args.capability_tokens)
    report = PilotReport(
        created_at=datetime.now(UTC).isoformat(),
        device=status.device or args.device,
        dtype=status.dtype or "?",
        config={
            "modules": modules,
            "top_k": args.top_k,
            "tokens": args.tokens,
            "capability_tokens": args.capability_tokens,
            "alphas": alphas,
            "warm_system": WARM_SYSTEM,
            "clinical_system": CLINICAL_SYSTEM,
            "extraction_prompts": list(EXTRACTION_PROMPTS),
            "ladder_prompts": list(LADDER_PROMPTS),
            "held_out_prompts": list(HELD_OUT_PROMPTS),
        },
    )

    # ------------------------------------------------------------- phase 1
    phase_started = time.perf_counter()
    positives: list[Capture] = []
    negatives: list[Capture] = []
    for prompt in EXTRACTION_PROMPTS:
        for system, label, bucket in (
            (WARM_SYSTEM, "warm", positives),
            (CLINICAL_SYSTEM, "clinical", negatives),
        ):
            request = BaselineRequest(prompt=prompt, system=system, settings=settings)
            result = generate_baseline(manager, request, capture_ids=modules)
            record = record_run("extraction", label, request, result.response, module_id=None)
            report.runs.append(record)
            bucket.append(
                Capture(
                    prompt=prompt,
                    system=system,
                    label=label,
                    captures=result.captures,
                    prompt_tokens=result.response.prompt_tokens,
                    record=record,
                )
            )
            print(f"[extract] {label:8s} «{prompt[:30]}» {result.response.duration_ms:6.0f} ms")
    report.directions = [
        compute_direction(module_id, positives, negatives, args.top_k, args.position_mode)
        for module_id in modules
    ]
    best = max(report.directions, key=lambda item: item.separation)
    print(
        "[phase1] "
        + ", ".join(f"{item.module_id.split('.')[2]}:{item.separation:.2f}" for item in report.directions)
        + f" -> chosen layer {best.module_id}"
        + f" ({time.perf_counter() - phase_started:.1f}s)"
    )

    # ------------------------------------------------------------- phase 2
    phase_started = time.perf_counter()
    ladder_rows: list[dict[str, object]] = []
    ladder_baselines: dict[str, RunRecord] = {}
    ladder_targets: dict[str, dict[str, int]] = {}
    zero_matches: dict[str, bool] = {}
    for ladder_prompt in LADDER_PROMPTS:
        request = BaselineRequest(prompt=ladder_prompt, settings=settings)
        baseline = generate_baseline(manager, request)
        baseline_record = record_run("ladder", "baseline", request, baseline.response, module_id=None)
        report.runs.append(baseline_record)
        ladder_baselines[ladder_prompt] = baseline_record
        zero_matches[ladder_prompt] = False
        for system, reference_label in ((WARM_SYSTEM, "warm target"), (CLINICAL_SYSTEM, "clinical target")):
            reference_request = BaselineRequest(prompt=ladder_prompt, system=system, settings=settings)
            reference = generate_baseline(manager, reference_request)
            reference_record = record_run(
                "ladder", reference_label, reference_request, reference.response, module_id=None
            )
            report.runs.append(reference_record)
            if reference_label == "warm target":
                ladder_targets[ladder_prompt] = reference_record.style
        for alpha in alphas:
            steering = SteeringSpec(
                module_id=best.module_id,
                strength=alpha,
                components=best.components,
                normalization="sigma",
            )
            result = generate_steered(manager, request, steering)
            record = record_run(
                "ladder",
                f"alpha {alpha:+.1f}",
                request,
                result.response,
                module_id=best.module_id,
                alpha=alpha,
                normalization="sigma",
            )
            report.runs.append(record)
            if alpha == 0.0:
                zero_matches[ladder_prompt] = record.output == baseline_record.output
            ladder_rows.append(
                {
                    "prompt": ladder_prompt,
                    "alpha": alpha,
                    "words": record.words,
                    "sentences": record.sentences,
                    "distinct_3": record.distinct_3,
                    "lexicon": record.lexicon,
                    "style": record.style,
                    "refusal": record.refusal,
                    "output": record.output,
                }
            )
            print(
                f"[ladder] «{ladder_prompt[:24]}» α={alpha:+.1f}σ words={record.words:3d} "
                f"warm={record.lexicon.get('warmth', 0):5.1f} tech={record.lexicon.get('technical', 0):5.1f}"
            )
    report.ladder = ladder_rows

    def rows_for(alpha: float) -> list[dict[str, object]]:
        return [row for row in ladder_rows if float(row["alpha"]) == alpha]

    def admissible(alpha: float) -> bool:
        return all(
            not bool(row["refusal"])
            and float(row["distinct_3"]) >= 0.65 * ladder_baselines[str(row["prompt"])].distinct_3
            and int(row["words"]) <= max(6, 2.5 * ladder_baselines[str(row["prompt"])].words)
            for row in rows_for(alpha)
        )

    def score(alpha: float) -> float:
        values = [
            -style_distance(row["style"], ladder_targets[str(row["prompt"])]) for row in rows_for(alpha)
        ]
        return statistics.fmean(values) if values else -999.0

    ladder_scores = {alpha: round(score(alpha), 3) for alpha in alphas}
    candidates = [alpha for alpha in alphas if alpha != 0.0 and admissible(alpha)]
    chosen_alpha = max(candidates, key=score) if candidates else alphas[-1]
    report.chosen = {
        "module_id": best.module_id,
        "components": best.components,
        "sigma": best.sigma,
        "alpha": chosen_alpha,
        "rule": (
            "min(style distance to the warm-target run) averaged over the two ladder prompts, "
            "admissible if no refusal, distinct-3 >= 65% of baseline and words <= 2.5x baseline on both"
        ),
        "zero_matches_baseline": all(zero_matches.values()),
        "ladder_scores": {str(key): value for key, value in ladder_scores.items()},
    }
    print(
        f"[phase2] chosen α={chosen_alpha:+.1f}σ scores={ladder_scores} ({time.perf_counter() - phase_started:.1f}s)"
    )

    # ------------------------------------------------------------- phase 3
    phase_started = time.perf_counter()
    steering = SteeringSpec(
        module_id=best.module_id,
        strength=chosen_alpha,
        components=best.components,
        normalization="sigma",
    )
    held_out_warm: dict[str, dict[str, int]] = {}
    for prompt in HELD_OUT_PROMPTS:
        request = BaselineRequest(prompt=prompt, settings=settings)
        baseline = generate_baseline(manager, request)
        baseline_record = record_run("held_out", "baseline", request, baseline.response, module_id=None)
        steered = generate_steered(manager, request, steering)
        steered_record = record_run(
            "held_out",
            "steered",
            request,
            steered.response,
            module_id=best.module_id,
            alpha=chosen_alpha,
            normalization="sigma",
        )
        warm_target = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        )
        warm_record = record_run("held_out", "warm target", request, warm_target.response, module_id=None)
        clinical_target = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=CLINICAL_SYSTEM, settings=settings)
        )
        clinical_record = record_run(
            "held_out", "clinical target", request, clinical_target.response, module_id=None
        )
        report.runs += [baseline_record, steered_record, warm_record, clinical_record]
        held_out_warm[prompt] = warm_record.style
        distance_baseline = style_distance(baseline_record.style, warm_record.style)
        distance_steered = style_distance(steered_record.style, warm_record.style)
        fraction = 1.0 - distance_steered / distance_baseline if distance_baseline else None
        delta_style = {
            key: int(steered_record.style.get(key, 0)) - int(baseline_record.style.get(key, 0))
            for key in STYLE_KEYS
        }
        report.held_out.append(
            {
                "prompt": prompt,
                "delta_style": delta_style,
                "delta_words": steered_record.words - baseline_record.words,
                "fraction_warm": round(fraction, 4) if fraction is not None else None,
                "baseline_style": baseline_record.style,
                "steered_style": steered_record.style,
                "warm_target_style": warm_record.style,
                "clinical_target_style": clinical_record.style,
                "baseline_output": baseline_record.output,
                "steered_output": steered_record.output,
            }
        )
        print(
            f"[held-out] «{prompt[:30]}» Δemoji={delta_style['emoji']:+d} "
            f"Δ!={delta_style['exclamations']:+d} Δcontr={delta_style['contractions']:+d} "
            f"toward warm={fraction * 100 if fraction is not None else float('nan'):+.0f}%"
        )

    for prompt, checks in CAPABILITY_PROMPTS:
        request = BaselineRequest(prompt=prompt, settings=capability_settings)
        baseline = generate_baseline(manager, request, checks=list(checks))
        baseline_record = record_run(
            "capability", "baseline", request, baseline.response, module_id=None, checks=checks
        )
        steered = generate_steered(manager, request, steering, checks=list(checks))
        steered_record = record_run(
            "capability",
            "steered",
            request,
            steered.response,
            module_id=best.module_id,
            alpha=chosen_alpha,
            normalization="sigma",
            checks=checks,
        )
        report.runs += [baseline_record, steered_record]

        def passed(record: RunRecord) -> str:
            total = len(record.checks)
            hit = sum(1 for check in record.checks if check["passed"])
            return f"{hit}/{total}"

        report.capability.append(
            {
                "prompt": prompt,
                "baseline": passed(baseline_record),
                "steered": passed(steered_record),
                "baseline_output": baseline_record.output,
                "steered_output": steered_record.output,
            }
        )
        print(f"[capability] «{prompt[:30]}» {passed(baseline_record)} -> {passed(steered_record)}")

    # random-direction control: same weights, random dimensions
    rng = random.Random(7)
    hidden_size = int(manager.model.config.get_text_config().hidden_size)
    target_dims = {component["feature_index"] for component in best.components}
    choices = [index for index in range(hidden_size) if index not in target_dims]
    shuffled = rng.sample(choices, len(best.components))
    control_components = [
        {"feature_index": index, "weight": component["weight"]}
        for index, component in zip(shuffled, best.components, strict=True)
    ]
    control_steering = SteeringSpec(
        module_id=best.module_id,
        strength=chosen_alpha,
        components=control_components,
        normalization="sigma",
    )
    for label, control_spec in (
        ("random direction", control_steering),
        (
            "reverse α",
            SteeringSpec(
                module_id=best.module_id,
                strength=-chosen_alpha,
                components=best.components,
                normalization="sigma",
            ),
        ),
    ):
        request = BaselineRequest(prompt=HELD_OUT_PROMPTS[0], settings=settings)
        result = generate_steered(manager, request, control_spec)
        record = record_run(
            "control",
            label,
            request,
            result.response,
            module_id=best.module_id,
            alpha=chosen_alpha,
            normalization="sigma",
        )
        report.runs.append(record)
        warm_style = held_out_warm[HELD_OUT_PROMPTS[0]]
        baseline_style = next(
            row["baseline_style"] for row in report.held_out if row["prompt"] == HELD_OUT_PROMPTS[0]
        )
        distance_baseline = style_distance(baseline_style, warm_style)
        distance_control = style_distance(record.style, warm_style)
        fraction = 1.0 - distance_control / distance_baseline if distance_baseline else None
        report.control.append(
            {
                "label": label,
                "words": record.words,
                "lexicon": record.lexicon,
                "style": record.style,
                "fraction_warm": round(fraction, 4) if fraction is not None else None,
                "checks_passed": None,
                "checks_total": None,
                "output": record.output,
            }
        )
        print(
            f"[control] {label}: emoji={record.style['emoji']} != {record.style['exclamations']} "
            f"contr={record.style['contractions']} toward warm={fraction * 100 if fraction is not None else float('nan'):+.0f}%"
        )

    # -------------------------------------------------------------- summary
    fractions = [
        float(row["fraction_warm"]) for row in report.held_out if row.get("fraction_warm") is not None
    ]
    toward = sum(1 for value in fractions if value > 0)
    emoji_delta = sum(int(row["delta_style"]["emoji"]) for row in report.held_out)
    baseline_passed = sum(int(row["baseline"].split("/")[0]) for row in report.capability)
    steered_passed = sum(int(row["steered"].split("/")[0]) for row in report.capability)
    possible = sum(int(row["steered"].split("/")[1]) for row in report.capability)
    control_row = next((row for row in report.control if row["label"] == "random direction"), None)
    report.notes = [
        f"Held-out prompts moved toward the warm target in {toward}/{len(fractions)} cases; "
        f"mean recovery {statistics.fmean(fractions) * 100:+.0f}% of the baseline-to-target style distance."
        if fractions
        else "No held-out style movement could be measured.",
        f"Total Δemoji across held-out prompts: {emoji_delta:+d}.",
        f"Capability checks: baseline {baseline_passed}/{possible}, steered {steered_passed}/{possible}.",
        (
            f"Random-direction control recovered "
            f"{control_row['fraction_warm'] * 100:+.0f}% of the same distance."
            if control_row and control_row.get("fraction_warm") is not None
            else "Random-direction control unavailable."
        ),
        f"Direction = mean difference of {len(EXTRACTION_PROMPTS)} warm/clinical prompt pairs at the last prompt "
        f"token ({args.position_mode}), top {args.top_k} dimensions, weights normalized to max |w| = 1.",
        "Selection used the ladder prompts; held-out prompts were never used to pick alpha.",
        f"Zero-strength equality with the baseline: {all(zero_matches.values())}.",
        "Style markers are lexical/structural heuristics over ~40-token generations; treat them as effect sizes, not verdicts.",
    ]
    print("", markdown_report(report))
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - load_started:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
