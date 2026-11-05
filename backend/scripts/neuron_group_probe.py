"""Group-of-neurons steering: top-k contrast neurons as one summed direction.

For each layer the group is the top-k intermediate neurons by the teacher-forced
warm/clinical contrast, with signed weights normalized to max |w| = 1. The
downstream direction is the weighted sum of their w2 columns (times the post-FFN
norm weight), and the perturbation is scaled so that this summed direction has
length ``alpha * residual_rms``:

    G = || sum_i w_i * (w2[:, i] * post_weight) ||
    delta_i = alpha * residual_rms / G * w_i

The screen compares the group against random groups of the same size and weight
multiset. If a group beats the controls, the script generates text at the chosen
setting and checks whether the style reaches the output.
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

import numpy as np
import torch

from backend.app.core.settings import Settings
from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.architecture_inspector import get_module
from backend.app.services.generation import generate_baseline
from backend.app.services.model_loader import ModelManager
from backend.scripts.multilayer_probe import (
    CLINICAL_SYSTEM,
    PASSAGES,
    TEACHER_USER,
    WARM_SYSTEM,
    render_conversation,
)
from backend.scripts.neuron_probe import STYLE_STRINGS, chat_inputs, last_position_probs
from backend.scripts.personality_pilot import style_distance, style_features

SCREEN_PROMPTS = (
    "Explain why the sky is blue.",
    "How do I learn to play guitar?",
    "Describe a rainy morning.",
)
GEN_PROMPTS = (
    "Explain why the sky is blue.",
    "How do I learn to play guitar?",
    "Write a short note to a coworker about a delayed report.",
    "A friend asks for advice about a difficult decision. Reply briefly.",
)
CAPABILITY = (
    ("Reply with exactly one word: the opposite of 'hot'.", "cold"),
    ("What is 17 + 26? Reply with only the number.", "43"),
    ("Name the largest planet in the solar system. One word.", "jupiter"),
    ("List exactly three colors, one per line. No other text.", "red"),
)


@dataclass
class Group:
    layer: int
    neurons: list[int]
    weights: list[float]
    direction_norm: float
    contrast_norm: float


@dataclass
class ScreenRow:
    layer: int
    alpha: float
    sign: int
    kl: float
    style_delta: float
    prompt: str


@dataclass
class GenRow:
    prompt: str
    label: str
    alpha: float
    sign: int
    style: dict[str, int]
    score_delta: int
    fraction_warm: float
    words: int
    distinct_3: float
    output: str


@dataclass
class Report:
    created_at: str
    device: str
    dtype: str
    config: dict[str, object]
    groups: list[Group] = field(default_factory=list)
    screens: list[ScreenRow] = field(default_factory=list)
    controls: list[ScreenRow] = field(default_factory=list)
    gen: list[GenRow] = field(default_factory=list)
    capability: list[dict[str, object]] = field(default_factory=list)
    verdict: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def layer_gain_matrix(model, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    w2 = get_module(model, f"model.layers.{layer}.feed_forward.w2").weight.detach().float()
    post = get_module(model, f"model.layers.{layer}.post_ffn_norm").weight.detach().float()
    return w2, post


def group_direction(
    w2: torch.Tensor, post: torch.Tensor, neurons: list[int], weights: list[float]
) -> torch.Tensor:
    direction = torch.zeros(w2.shape[0], device=w2.device, dtype=torch.float32)
    for neuron, weight in zip(neurons, weights, strict=True):
        direction = direction + weight * (w2[:, neuron] * post)
    return direction


@contextmanager
def steer_group(model, layer: int, deltas: dict[int, float]):
    module = get_module(model, f"model.layers.{layer}.feed_forward.w2")

    def hook(mod, args):
        tensor = args[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return None
        steered = tensor.to(torch.float32).clone()
        for neuron, delta in deltas.items():
            steered[..., neuron] = steered[..., neuron] + delta
        return (steered.to(tensor.dtype),)

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def residual_rms_all(model, inputs, layers: list[int]) -> dict[int, float]:
    hidden = int(model.config.hidden_size)
    store: dict[int, float] = {}
    handles = []
    for layer in layers:

        def hook(mod, args, output, *, layer=layer):
            tensor = output[0] if isinstance(output, tuple) else output
            store[layer] = float(tensor[0, -1].norm() / math.sqrt(hidden))

        handles.append(get_module(model, f"model.layers.{layer}").register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="12,16,20,22")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--alphas", default="0.3,0.5,0.7")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",")]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"neuron-groups-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=128))
    started = time.perf_counter()
    status = manager.load()
    model, tokenizer = manager.model, manager.tokenizer
    device = status.device or args.device
    intermediate = int(model.config.intermediate_size)

    report = Report(
        created_at=datetime.now(UTC).isoformat(),
        device=device,
        dtype=status.dtype or "?",
        config={
            "layers": layers,
            "top_k": args.top_k,
            "alphas": alphas,
            "screen_prompts": list(SCREEN_PROMPTS),
            "gen_prompts": list(GEN_PROMPTS),
            "tokens": args.tokens,
        },
    )
    print(f"[load] {status.state} {device} {status.dtype} in {time.perf_counter() - started:.1f}s")

    # ------------------------------------------------------------- extraction
    warm: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    clinical: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for passage in PASSAGES:
        for label, system in (("warm", WARM_SYSTEM), ("clinical", CLINICAL_SYSTEM)):
            rendered = render_conversation(tokenizer, system, TEACHER_USER, passage)
            start = rendered.rfind(passage)
            encoding = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True)
            offsets = encoding.pop("offset_mapping")[0].tolist()
            positions = [
                index
                for index, (begin, end) in enumerate(offsets)
                if end > start and begin < start + len(passage)
            ]
            inputs = {key: value.to(device) for key, value in encoding.items()}
            store: dict[int, torch.Tensor] = {}
            handles = []
            for layer in layers:

                def hook(mod, hook_args, *, layer=layer, store=store):
                    tensor = hook_args[0]
                    if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                        store[layer] = tensor.detach().to("cpu", dtype=torch.float32)

                handles.append(
                    get_module(model, f"model.layers.{layer}.feed_forward.w2").register_forward_pre_hook(hook)
                )
            with torch.inference_mode():
                model(**inputs)
            for handle in handles:
                handle.remove()
            target = warm if label == "warm" else clinical
            for layer in layers:
                target[layer].append(store[layer][0, positions].numpy().astype(np.float64))
            print(f"[extract] {label:8s} «{passage[:22]}…» {len(positions)} tokens")

    directions = {}
    for layer in layers:
        diffs = [w.mean(0) - c.mean(0) for w, c in zip(warm[layer], clinical[layer], strict=True)]
        directions[layer] = np.mean(diffs, axis=0)

    for layer in layers:
        contrast = directions[layer]
        order = np.argsort(-np.abs(contrast))[: args.top_k]
        weights = (contrast[order] / np.max(np.abs(contrast[order]))).tolist()
        w2, post = layer_gain_matrix(model, layer)
        direction = group_direction(w2, post, [int(i) for i in order], weights)
        report.groups.append(
            Group(
                layer=layer,
                neurons=[int(i) for i in order],
                weights=[round(float(w), 4) for w in weights],
                direction_norm=round(float(direction.norm()), 5),
                contrast_norm=round(float(np.linalg.norm(contrast[order])), 4),
            )
        )
        print(
            f"[group] L{layer}: top {args.top_k} |w|<=1, direction norm {float(direction.norm()):.4f}, "
            f"contrast norm {np.linalg.norm(contrast[order]):.3f}"
        )

    # ----------------------------------------------------------------- screen
    style_ids = {
        int(tokenizer(text, add_special_tokens=False)["input_ids"][0])
        for text in STYLE_STRINGS
        if tokenizer(text, add_special_tokens=False)["input_ids"]
    }
    style_t = torch.tensor(sorted(style_ids), device=device)
    screen_inputs = {prompt: chat_inputs(tokenizer, device, prompt) for prompt in SCREEN_PROMPTS}
    bases = {}
    for prompt, inputs in screen_inputs.items():
        _, probs = last_position_probs(model, inputs)
        bases[prompt] = {"probs": probs, "mass": float(probs[style_t].sum())}
    rms = {prompt: residual_rms_all(model, inputs, layers) for prompt, inputs in screen_inputs.items()}
    print(f"[screen] style ids {sorted(style_ids)}")

    def measure(
        layer: int, neurons: list[int], weights: list[float], alpha: float, sign: int
    ) -> tuple[float, float]:
        w2, post = layer_gain_matrix(model, layer)
        direction = group_direction(w2, post, neurons, weights)
        norm = float(direction.norm())
        deltas = {}
        for prompt in SCREEN_PROMPTS:
            scale = sign * alpha * rms[prompt][layer] / max(norm, 1e-9)
            deltas[prompt] = {n: scale * w for n, w in zip(neurons, weights, strict=True)}
        kls = []
        styles = []
        for prompt in SCREEN_PROMPTS:
            with steer_group(model, layer, deltas[prompt]):
                _, probs = last_position_probs(model, screen_inputs[prompt])
            kls.append(
                float(
                    torch.sum(
                        bases[prompt]["probs"] * torch.log((bases[prompt]["probs"] + 1e-12) / (probs + 1e-12))
                    ).item()
                )
            )
            styles.append(float(probs[style_t].sum()) - bases[prompt]["mass"])
        return max(kls), float(statistics.fmean(styles))

    best = None
    for group in report.groups:
        for sign in (1, -1):
            for alpha in alphas:
                kl, style = measure(group.layer, group.neurons, group.weights, alpha, sign)
                row = ScreenRow(group.layer, alpha, sign, round(kl, 6), round(style, 6), "mean")
                report.screens.append(row)
                if best is None or abs(row.style_delta) > abs(best.style_delta):
                    best = row
                print(
                    f"[screen] L{group.layer} α={alpha:+.1f} sign={sign:+d}: Δstyle={style:+.5f} KL={kl:.5f}"
                )

    control_max = 0.0
    for group in report.groups:
        pool = [i for i in range(intermediate) if i not in set(group.neurons)]
        for seed in (1, 2, 3):
            local = random.Random(seed * 100 + group.layer)
            neurons = local.sample(pool, args.top_k)
            shuffled = group.weights[:]
            local.shuffle(shuffled)
            for sign in (1, -1):
                kl, style = measure(group.layer, neurons, shuffled, 0.7, sign)
                report.controls.append(
                    ScreenRow(group.layer, 0.7, sign, round(kl, 6), round(style, 6), "control")
                )
                control_max = max(control_max, abs(style))
    print(f"[screen] control max |Δstyle| = {control_max:.5f}")

    # ------------------------------------------------------------- generation
    primary = best
    group = next(item for item in report.groups if item.layer == primary.layer)
    w2, post = layer_gain_matrix(model, primary.layer)
    pool = [i for i in range(intermediate) if i not in set(group.neurons)]
    control_neurons = random.Random(7).sample(pool, args.top_k)
    control_weights = group.weights[:]
    random.Random(8).shuffle(control_weights)

    for prompt in GEN_PROMPTS:
        inputs = chat_inputs(tokenizer, device, prompt)
        settings = GenerationSettings(max_new_tokens=args.tokens)
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        warm_reference = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        ).response
        base_style = style_features(base.output)
        warm_style = style_features(warm_reference.output)
        base_score = sum(base_style.values())
        report.gen.append(
            GenRow(
                prompt,
                "baseline",
                0.0,
                0,
                base_style,
                0,
                0.0,
                base.metrics.words,
                base.metrics.distinct_3,
                base.output,
            )
        )
        report.gen.append(
            GenRow(
                prompt,
                "warm target",
                0.0,
                0,
                warm_style,
                sum(warm_style.values()) - base_score,
                1.0,
                warm_reference.metrics.words,
                warm_reference.metrics.distinct_3,
                warm_reference.output,
            )
        )

        def generate_arm(
            label: str,
            neurons: list[int],
            weights: list[float],
            alpha: float,
            sign: int,
            *,
            prompt: str = prompt,
            inputs: dict = inputs,
            settings: GenerationSettings = settings,
            warm_style: dict[str, int] = warm_style,
            base_style: dict[str, int] = base_style,
            base_score: int = base_score,
        ) -> None:
            direction = group_direction(w2, post, neurons, weights)
            scale = (
                sign
                * alpha
                * residual_rms_all(model, inputs, [primary.layer])[primary.layer]
                / max(float(direction.norm()), 1e-9)
            )
            deltas = {n: scale * w for n, w in zip(neurons, weights, strict=True)}
            with steer_group(model, primary.layer, deltas):
                response = generate_baseline(
                    manager, BaselineRequest(prompt=prompt, settings=settings)
                ).response
            style = style_features(response.output)
            fraction = 1.0 - style_distance(style, warm_style) / max(
                1.0, style_distance(base_style, warm_style)
            )
            row = GenRow(
                prompt,
                label,
                alpha,
                sign,
                style,
                sum(style.values()) - base_score,
                round(fraction, 3),
                response.metrics.words,
                response.metrics.distinct_3,
                response.output,
            )
            report.gen.append(row)
            print(f"  {prompt[:22]:22s} {label:16s} style={style} Δ={row.score_delta:+d}")

        generate_arm("group best", group.neurons, group.weights, primary.alpha, primary.sign)
        generate_arm("group opposite", group.neurons, group.weights, primary.alpha, -primary.sign)
        generate_arm("random group", control_neurons, control_weights, primary.alpha, primary.sign)

    # ------------------------------------------------------------ capability
    inputs = chat_inputs(tokenizer, device, CAPABILITY[0][0])
    scale = primary.alpha * residual_rms_all(model, inputs, [primary.layer])[primary.layer]
    direction = group_direction(w2, post, group.neurons, group.weights)
    deltas = {
        n: scale / max(float(direction.norm()), 1e-9) * w
        for n, w in zip(group.neurons, group.weights, strict=True)
    }
    for prompt, needle in CAPABILITY:
        base = generate_baseline(
            manager, BaselineRequest(prompt=prompt, settings=GenerationSettings(max_new_tokens=16))
        ).response
        with steer_group(model, primary.layer, deltas):
            steered = generate_baseline(
                manager, BaselineRequest(prompt=prompt, settings=GenerationSettings(max_new_tokens=16))
            ).response
        report.capability.append(
            {
                "prompt": prompt,
                "baseline_ok": needle.lower() in base.output.lower(),
                "steered_ok": needle.lower() in steered.output.lower(),
            }
        )

    # --------------------------------------------------------------- verdict
    def moved(label_prefix: str) -> tuple[int, int]:
        rows = [row for row in report.gen if row.label.startswith(label_prefix)]
        return sum(1 for row in rows if row.score_delta > 0), len(rows)

    primary_moved, primary_total = moved("group best")
    control_moved, control_total = moved("random group")
    report.verdict = {
        "best_screen": {
            "layer": primary.layer,
            "alpha": primary.alpha,
            "sign": primary.sign,
            "style_delta": primary.style_delta,
            "kl": primary.kl,
        },
        "control_max_style_delta": round(control_max, 6),
        "screen_above_control": abs(primary.style_delta) > control_max,
        "primary_moved": primary_moved,
        "primary_total": primary_total,
        "control_moved": control_moved,
        "control_total": control_total,
        "capability_baseline": sum(1 for row in report.capability if row["baseline_ok"]),
        "capability_steered": sum(1 for row in report.capability if row["steered_ok"]),
    }
    report.notes = [
        f"Best screen: L{primary.layer} α={primary.alpha} sign={primary.sign:+d} "
        f"(Δstyle {primary.style_delta:+.5f}, KL {primary.kl:.5f}, control max {control_max:.5f}).",
        f"Generation: group moved the style score on {primary_moved}/{primary_total} prompts; "
        f"random group on {control_moved}/{control_total}.",
        f"Capability needles: baseline {report.verdict['capability_baseline']}/4, "
        f"steered {report.verdict['capability_steered']}/4.",
        "Group direction = sum of signed w2 columns weighted by the contrast, normalised by its own norm.",
        "Style score = emoji + exclamations + contractions + first person.",
    ]
    print("\n".join(f"- {note}" for note in report.notes))
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
