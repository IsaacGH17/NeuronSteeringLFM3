"""Phase 3: does an individual MLP-neuron push reach the generated text?

The screen found two neurons whose intermediate perturbation reliably shifts the
next-token probability of style tokens (L16 n=3177, L12 n=4961). This script
turns the same intervention into generation: for every prompt it compares the
baseline, a warm-system target, the two neurons across a small alpha grid (+/-),
and a random-neuron control at the strongest alpha.

Pre-registered criteria (written before the run):
  * primary (L16 n=3177, alpha=0.7, +): the style score (emoji + "!" +
    contractions + first person) increases on >= 3 of 4 prompts.
  * control: a random neuron at the same alpha increases the score on <= 1 of 4.
  * coherence: distinct-3 >= 0.65, no refusals, non-empty text at that setting.
  * capability: at least as many needles pass as in the baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

from backend.app.core.settings import Settings
from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.architecture_inspector import get_module
from backend.app.services.generation import generate_baseline
from backend.app.services.model_loader import ModelManager
from backend.scripts.multilayer_probe import WARM_SYSTEM
from backend.scripts.neuron_probe import chat_inputs
from backend.scripts.personality_pilot import style_distance, style_features

PROMPTS = (
    "Explain why the sky is blue.",
    "How do I learn to play guitar?",
    "Write a short note to a coworker about a delayed report.",
    "A friend asks for advice about a difficult decision. Reply briefly.",
)
PRIMARY = (16, 3177)
SECONDARY = (12, 4961)
CONFIGS = (
    (PRIMARY, 0.3, 1),
    (PRIMARY, 0.5, 1),
    (PRIMARY, 0.7, 1),
    (PRIMARY, 0.5, -1),
    (PRIMARY, 0.7, -1),
    (SECONDARY, 0.5, 1),
    (SECONDARY, 0.7, 1),
)
CAPABILITY = (
    ("Reply with exactly one word: the opposite of 'hot'.", "cold"),
    ("What is 17 + 26? Reply with only the number.", "43"),
    ("Name the largest planet in the solar system. One word.", "jupiter"),
    ("List exactly three colors, one per line. No other text.", "red"),
)


@dataclass
class RunRecord:
    prompt: str
    label: str
    layer: int | None
    neuron: int | None
    sign: int
    alpha: float
    output: str
    words: int
    distinct_3: float
    refusal: bool
    style: dict[str, int]
    style_score: int


@dataclass
class Report:
    created_at: str
    device: str
    dtype: str
    config: dict[str, object]
    runs: list[RunRecord] = field(default_factory=list)
    per_prompt: list[dict[str, object]] = field(default_factory=list)
    capability: list[dict[str, object]] = field(default_factory=list)
    verdict: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def w2_gain(model, layer: int, neuron: int) -> float:
    w2 = get_module(model, f"model.layers.{layer}.feed_forward.w2").weight.detach().float()
    post = get_module(model, f"model.layers.{layer}.post_ffn_norm").weight.detach().float()
    return float((w2[:, neuron] * post).norm())


def residual_rms(model, inputs, layer: int) -> float:
    hidden = int(model.config.hidden_size)
    store: dict[str, float] = {}

    def hook(mod, args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store["rms"] = float(tensor[0, -1].norm() / math.sqrt(hidden))

    handle = get_module(model, f"model.layers.{layer}").register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        handle.remove()
    return store["rms"]


@contextmanager
def steer_neuron(model, layer: int, neuron: int, delta: float):
    module = get_module(model, f"model.layers.{layer}.feed_forward.w2")

    def hook(mod, args):
        tensor = args[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return None
        steered = tensor.to(torch.float32).clone()
        steered[..., neuron] = steered[..., neuron] + delta
        return (steered.to(tensor.dtype),)

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def describe(record: RunRecord) -> str:
    return (
        f"{record.label:22s} style={record.style} words={record.words:3d} "
        f"d3={record.distinct_3:.2f} | {record.output[:64].replace(chr(10), ' ')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"neuron-gen-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=128))
    started = time.perf_counter()
    status = manager.load()
    model, tokenizer = manager.model, manager.tokenizer
    device = status.device or args.device
    settings = GenerationSettings(max_new_tokens=args.tokens)
    gains = {(layer, neuron): w2_gain(model, layer, neuron) for layer, neuron in (PRIMARY, SECONDARY)}
    rng = random.Random(31)
    random_neuron = (PRIMARY[0], rng.randrange(0, int(model.config.intermediate_size)))
    while (random_neuron[0], random_neuron[1]) in (PRIMARY, SECONDARY):
        random_neuron = (PRIMARY[0], rng.randrange(0, int(model.config.intermediate_size)))
    random_gain = w2_gain(model, *random_neuron)

    report = Report(
        created_at=datetime.now(UTC).isoformat(),
        device=device,
        dtype=status.dtype or "?",
        config={
            "prompts": list(PROMPTS),
            "primary": PRIMARY,
            "secondary": SECONDARY,
            "configs": [
                {"layer": layer, "neuron": neuron, "alpha": alpha, "sign": sign}
                for (layer, neuron), alpha, sign in CONFIGS
            ],
            "random_control": {"layer": random_neuron[0], "neuron": random_neuron[1], "alpha": 0.7},
            "tokens": args.tokens,
            "style_score": "emoji + exclamations + contractions + first_person",
        },
    )
    print(f"[load] {status.state} {device} {status.dtype} in {time.perf_counter() - started:.1f}s")
    print(f"[gains] primary {gains[PRIMARY]:.4f} secondary {gains[SECONDARY]:.4f} random {random_gain:.4f}")

    for prompt in PROMPTS:
        request = BaselineRequest(prompt=prompt, settings=settings)
        baseline = generate_baseline(manager, request).response
        base_record = RunRecord(
            prompt,
            "baseline",
            None,
            None,
            0,
            0.0,
            baseline.output,
            baseline.metrics.words,
            baseline.metrics.distinct_3,
            baseline.metrics.refusal,
            style_features(baseline.output),
            sum(style_features(baseline.output).values()),
        )
        warm = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        ).response
        warm_record = RunRecord(
            prompt,
            "warm target",
            None,
            None,
            0,
            0.0,
            warm.output,
            warm.metrics.words,
            warm.metrics.distinct_3,
            warm.metrics.refusal,
            style_features(warm.output),
            sum(style_features(warm.output).values()),
        )
        report.runs += [base_record, warm_record]
        inputs = chat_inputs(tokenizer, device, prompt)
        rows: list[dict[str, object]] = []

        def run_config(
            label: str,
            layer: int,
            neuron: int,
            sign: int,
            alpha: float,
            gain: float,
            *,
            prompt: str = prompt,
            request: BaselineRequest = request,
            inputs: dict = inputs,
            base_record: RunRecord = base_record,
            warm_record: RunRecord = warm_record,
            rows: list[dict[str, object]] = rows,
        ) -> RunRecord:
            delta = sign * alpha * residual_rms(model, inputs, layer) / max(gain, 1e-9)
            with steer_neuron(model, layer, neuron, float(delta)):
                response = generate_baseline(manager, request).response
            style = style_features(response.output)
            record = RunRecord(
                prompt,
                f"{label} α={alpha:+.1f}",
                layer,
                neuron,
                sign,
                alpha,
                response.output,
                response.metrics.words,
                response.metrics.distinct_3,
                response.metrics.refusal,
                style,
                sum(style.values()),
            )
            report.runs.append(record)
            row = {
                "label": label,
                "layer": layer,
                "neuron": neuron,
                "sign": sign,
                "alpha": alpha,
                "style": style,
                "score_delta": record.style_score - base_record.style_score,
                "fraction_warm": round(
                    1.0
                    - style_distance(style, warm_record.style)
                    / max(1.0, style_distance(base_record.style, warm_record.style)),
                    4,
                ),
                "words": record.words,
                "distinct_3": record.distinct_3,
                "output": record.output,
            }
            rows.append(row)
            print(f"  {describe(record)}")
            return record

        for (layer, neuron), alpha, sign in CONFIGS:
            label = "primary" if (layer, neuron) == PRIMARY else "secondary"
            run_config(label, layer, neuron, sign, alpha, gains[(layer, neuron)])
        run_config("random control", random_neuron[0], random_neuron[1], 1, 0.7, random_gain)

        report.per_prompt.append(
            {
                "prompt": prompt,
                "baseline": asdict(base_record),
                "warm_target": asdict(warm_record),
                "runs": rows,
            }
        )
        print(f"[prompt] «{prompt[:34]}» done ({time.perf_counter() - started:.1f}s)")

    # ---------------------------------------------------------- capability
    primary_delta = (
        0.7
        * residual_rms(model, chat_inputs(tokenizer, device, CAPABILITY[0][0]), PRIMARY[0])
        / gains[PRIMARY]
    )
    for prompt, needle in CAPABILITY:
        base = generate_baseline(
            manager, BaselineRequest(prompt=prompt, settings=GenerationSettings(max_new_tokens=16))
        ).response
        with steer_neuron(model, PRIMARY[0], PRIMARY[1], float(primary_delta)):
            steered = generate_baseline(
                manager, BaselineRequest(prompt=prompt, settings=GenerationSettings(max_new_tokens=16))
            ).response
        report.capability.append(
            {
                "prompt": prompt,
                "baseline_ok": needle.lower() in base.output.lower(),
                "steered_ok": needle.lower() in steered.output.lower(),
                "baseline_output": base.output,
                "steered_output": steered.output,
            }
        )

    # ------------------------------------------------------------- verdict
    primary_rows = [
        row
        for entry in report.per_prompt
        for row in entry["runs"]
        if row["layer"] == PRIMARY[0]
        and row["neuron"] == PRIMARY[1]
        and row["alpha"] == 0.7
        and row["sign"] == 1
    ]
    control_rows = [
        row for entry in report.per_prompt for row in entry["runs"] if row["label"] == "random control"
    ]
    primary_moved = sum(1 for row in primary_rows if row["score_delta"] > 0)
    control_moved = sum(1 for row in control_rows if row["score_delta"] > 0)
    coherent = all(row["distinct_3"] >= 0.65 and not row["style"] == {} for row in primary_rows)
    capability_base = sum(1 for row in report.capability if row["baseline_ok"])
    capability_steered = sum(1 for row in report.capability if row["steered_ok"])
    report.verdict = {
        "primary_moved_prompts": primary_moved,
        "primary_total": len(primary_rows),
        "control_moved_prompts": control_moved,
        "control_total": len(control_rows),
        "mean_primary_score_delta": round(statistics.fmean([row["score_delta"] for row in primary_rows]), 3)
        if primary_rows
        else 0.0,
        "coherent": coherent,
        "capability_baseline": capability_base,
        "capability_steered": capability_steered,
    }
    report.notes = [
        f"Primary L{PRIMARY[0]} n={PRIMARY[1]} at α=0.7 (+): style score increased on "
        f"{primary_moved}/{len(primary_rows)} prompts (mean Δ {report.verdict['mean_primary_score_delta']:+.2f}).",
        f"Random control at the same α increased the score on {control_moved}/{len(control_rows)} prompts.",
        f"Capability needles: baseline {capability_base}/{len(CAPABILITY)}, steered {capability_steered}/{len(CAPABILITY)}.",
        "Style score = emoji + exclamations + contractions + first person, counted lexically.",
        "Neuron deltas are α · residual_rms / ||w2[:, i] * post_ffn_norm.weight||, applied at every token.",
    ]
    print("\n".join(f"- {note}" for note in report.notes))
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
