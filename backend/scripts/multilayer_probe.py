"""Multi-layer steering probe: many layers, full vectors, one budget.

Directions come from the same contrastive design used by the personality pilot
(same prompts under a warm/casual and a clinical/terse system prompt, last
prompt token), but they are extracted independently for several decoder-layer
outputs and applied *together* during generation:

    h_l <- h_l + alpha_l * rms(h_l) * d_hat_l        alpha_l = alpha / sqrt(L)

so the total displacement budget is constant regardless of how many layers are
steered. Vectors are full 2048-d and unit-normalised (not top-k, not per-
dimension sigma). The probe measures three things the single-layer pilot could
not: whether accumulation installs the tone, whether the same vector with the
opposite sign cancels a system-prompt tone, and whether shuffled or random
layer directions do the same.

Pre-registered success criteria (written before the run):
  * install:  multi-layer +alpha recovers >= 50% of the baseline-to-warm style
    distance on >= 4 of 6 held-out prompts.
  * cancel:   multi-layer -alpha on the warm prompt removes >= 50% of the same
    distance on >= 4 of 6 prompts.
  * controls: shuffled and random layer directions stay below 20% recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from backend.app.core.settings import Settings
from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.activation_capture import capture_modules, finalize_captures
from backend.app.services.architecture_inspector import get_module
from backend.app.services.generation import generate_baseline
from backend.app.services.metrics import compute_metrics
from backend.app.services.model_loader import ModelManager
from backend.scripts.personality_pilot import (
    CLINICAL_SYSTEM,
    HELD_OUT_PROMPTS,
    WARM_SYSTEM,
    style_distance,
    style_features,
)

EXTRACTION_PROMPTS = (
    "Describe a rainy morning.",
    "What should I cook tonight? Give me one idea.",
    "Describe your ideal weekend in three sentences.",
    "Give me a two-sentence review of a coffee shop.",
)
LADDER_PROMPTS = ("Explain why the sky is blue.", "Describe how a bicycle works.")
CONTROL_PROMPTS = HELD_OUT_PROMPTS[:2]
ALPHAS = (0.25, 0.5, 1.0, 2.0)


@dataclass
class LayerDirection:
    module_id: str
    layer: int
    vector: list[float]
    norm: float
    separation: float


@dataclass
class RunRecord:
    phase: str
    label: str
    prompt: str
    system: str | None
    alpha: float | None
    output: str
    words: int
    distinct_3: float
    refusal: bool
    style: dict[str, int]
    checks_passed: int | None
    checks_total: int | None
    duration_ms: float


@dataclass
class ProbeReport:
    created_at: str
    device: str
    dtype: str
    config: dict[str, object]
    directions: list[LayerDirection] = field(default_factory=list)
    cosines: list[list[float]] = field(default_factory=list)
    ladder: list[dict[str, object]] = field(default_factory=list)
    held_out: list[dict[str, object]] = field(default_factory=list)
    single_layer: list[dict[str, object]] = field(default_factory=list)
    controls: list[dict[str, object]] = field(default_factory=list)
    capability: list[dict[str, object]] = field(default_factory=list)
    chosen: dict[str, object] = field(default_factory=dict)
    runs: list[RunRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def record(phase: str, label: str, prompt: str, system: str | None, response, *, alpha=None) -> RunRecord:
    metrics = compute_metrics(response.output)
    return RunRecord(
        phase=phase,
        label=label,
        prompt=prompt,
        system=system,
        alpha=alpha,
        output=response.output,
        words=metrics.words,
        distinct_3=metrics.distinct_3,
        refusal=metrics.refusal,
        style=style_features(response.output),
        checks_passed=metrics.checks_passed,
        checks_total=metrics.checks_total,
        duration_ms=response.duration_ms,
    )


class MultiLayerSteering:
    """Out-of-place residual steering on several decoder layers at once."""

    def __init__(self, model, directions: dict[str, torch.Tensor], alphas: dict[str, float]) -> None:
        self.model = model
        self.directions = directions
        self.alphas = alphas
        self.handles: list[object] = []
        self.calls = {module_id: 0 for module_id in directions}

    def __enter__(self) -> MultiLayerSteering:
        for module_id, direction in self.directions.items():
            alpha = self.alphas[module_id]
            vector = direction

            def hook(module, args, output, *, module_id=module_id, alpha=alpha, vector=vector):
                tensor = output[0] if isinstance(output, tuple) else output
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                    return output
                base = tensor.to(torch.float32)
                rms = base.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                steered = (base + alpha * rms * vector).to(tensor.dtype)
                self.calls[module_id] += 1
                return (steered, *output[1:]) if isinstance(output, tuple) else steered

            self.handles.append(get_module(self.model, module_id).register_forward_hook(hook))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def unit(vector: np.ndarray) -> torch.Tensor:
    tensor = torch.tensor(vector, dtype=torch.float32)
    return tensor / tensor.norm().clamp_min(1e-9)


def fraction_toward(
    steered: dict[str, int], baseline: dict[str, int], target: dict[str, int]
) -> float | None:
    base_distance = style_distance(baseline, target)
    if base_distance == 0:
        return None
    return 1.0 - style_distance(steered, target) / base_distance


PASSAGES = (
    (
        "Rain begins as water vapour condenses in cool air. The droplets grow until they are "
        "heavy enough to fall. A rainy morning changes the light, the sound, and the smell of "
        "the streets."
    ),
    (
        "A bicycle transfers power from the pedals through a chain to the rear wheel. Gears "
        "change how far the wheel turns for each pedal stroke, which is why climbing is easier "
        "in a low gear."
    ),
    (
        "Coffee is brewed by passing hot water through ground beans. The grind size, the water "
        "temperature, and the contact time decide how much of the bean ends up in the cup."
    ),
)
TEACHER_USER = "Write a short explanation."


def render_conversation(tokenizer, system: str | None, user: str, assistant: str) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def extract_generation(manager, modules, settings, report) -> tuple[dict, dict]:
    """Current design: last prompt token difference between two free generations."""
    captures: dict[str, list[np.ndarray]] = {"warm": [], "clinical": []}
    prompt_tokens: dict[str, list[int]] = {"warm": [], "clinical": []}
    for prompt in EXTRACTION_PROMPTS:
        for label, system in (("warm", WARM_SYSTEM), ("clinical", CLINICAL_SYSTEM)):
            request = BaselineRequest(prompt=prompt, system=system, settings=settings)
            result = generate_baseline(manager, request, capture_ids=modules)
            captures[label].append(
                np.stack([result.captures[module].numpy().astype(np.float64) for module in modules])
            )
            prompt_tokens[label].append(result.response.prompt_tokens)
            report.runs.append(record("extraction", label, prompt, system, result.response))
            print(f"[extract] {label:8s} «{prompt[:30]}» {result.response.duration_ms:6.0f} ms")

    vectors: dict[str, np.ndarray] = {}
    spreads: dict[str, np.ndarray] = {}
    for layer_index, module_id in enumerate(modules):
        differences = []
        layer_spreads = []
        for pair in range(len(EXTRACTION_PROMPTS)):
            warm_tensor = captures["warm"][pair][layer_index]
            clinical_tensor = captures["clinical"][pair][layer_index]
            position = min(prompt_tokens["warm"][pair], prompt_tokens["clinical"][pair]) - 1
            differences.append(warm_tensor[position] - clinical_tensor[position])
            layer_spreads.append(warm_tensor[: position + 1].std(axis=0))
        vectors[module_id] = np.mean(differences, axis=0)
        spreads[module_id] = np.mean(layer_spreads, axis=0)
    return vectors, spreads


def extract_teacher_forced(manager, modules, report) -> tuple[dict, dict]:
    """Same assistant text under both systems; average over its token positions."""
    tokenizer = manager.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Teacher-forced extraction needs a fast tokenizer with offset mapping.")
    per_condition: dict[str, list[dict[str, np.ndarray]]] = {"warm": [], "clinical": []}
    for passage in PASSAGES:
        for label, system in (("warm", WARM_SYSTEM), ("clinical", CLINICAL_SYSTEM)):
            rendered = render_conversation(tokenizer, system, TEACHER_USER, passage)
            start = rendered.rfind(passage)
            if start < 0:
                raise RuntimeError("Assistant text was not found verbatim in the rendered prompt.")
            encoding = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True)
            offsets = encoding.pop("offset_mapping")[0].tolist()
            positions = [
                index
                for index, (begin, end) in enumerate(offsets)
                if end > start and begin < start + len(passage)
            ]
            if not positions:
                raise RuntimeError("Could not map the assistant text to token positions.")
            inputs = {key: value.to(manager.device) for key, value in encoding.items()}
            with capture_modules(manager.model, modules) as (chunks, _warnings):
                with torch.inference_mode():
                    manager.model(**inputs)
            tensors, _finalize_warnings = finalize_captures(chunks)
            per_condition[label].append(
                {module: tensors[module].numpy().astype(np.float64)[positions] for module in modules}
            )
            print(f"[extract] {label:8s} teacher-forced «{passage[:26]}…» {len(positions)} assistant tokens")

    vectors: dict[str, np.ndarray] = {}
    spreads: dict[str, np.ndarray] = {}
    for module_id in modules:
        differences = []
        warm_positions = []
        for warm, clinical in zip(per_condition["warm"], per_condition["clinical"], strict=True):
            differences.append(warm[module_id].mean(axis=0) - clinical[module_id].mean(axis=0))
            warm_positions.append(warm[module_id])
        vectors[module_id] = np.mean(differences, axis=0)
        spreads[module_id] = np.concatenate(warm_positions, axis=0).std(axis=0)
    return vectors, spreads


def summarise(vectors: dict, spreads: dict, modules, layers, report) -> None:
    for layer_index, module_id in enumerate(modules):
        difference = vectors[module_id]
        sigma = spreads[module_id]
        separation = float(np.linalg.norm(difference) / math.sqrt(float(np.mean(sigma**2)) or 1.0))
        report.directions.append(
            LayerDirection(
                module_id=module_id,
                layer=layers[layer_index],
                vector=[round(float(value), 6) for value in difference],
                norm=round(float(np.linalg.norm(difference)), 4),
                separation=round(separation, 3),
            )
        )
    names = list(vectors)
    report.cosines = [
        [
            round(
                float(
                    np.dot(vectors[a], vectors[b])
                    / max(1e-9, np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b]))
                ),
                3,
            )
            for b in names
        ]
        for a in names
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="8,12,16,20,22")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--alphas", default="1.0,2.0,4.0,8.0")
    parser.add_argument("--extraction", choices=("generation", "teacher_forced"), default="generation")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    modules = [f"model.layers.{layer}" for layer in layers]
    alphas = [float(item) for item in args.alphas.split(",")]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"multilayer-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=256))
    started = time.perf_counter()
    status = manager.load()
    model = manager.model
    device = status.device or args.device
    settings = GenerationSettings(max_new_tokens=args.tokens)
    report = ProbeReport(
        created_at=datetime.now(UTC).isoformat(),
        device=device,
        dtype=status.dtype or "?",
        config={
            "layers": layers,
            "alphas": alphas,
            "tokens": args.tokens,
            "extraction": args.extraction,
            "extraction_prompts": list(EXTRACTION_PROMPTS),
            "teacher_forced_passages": list(PASSAGES) if args.extraction == "teacher_forced" else [],
            "ladder_prompts": list(LADDER_PROMPTS),
            "held_out_prompts": list(HELD_OUT_PROMPTS),
            "control_prompts": list(CONTROL_PROMPTS),
            "warm_system": WARM_SYSTEM,
            "clinical_system": CLINICAL_SYSTEM,
        },
    )
    print(f"[load] {status.state} {device} {status.dtype} in {time.perf_counter() - started:.1f}s")

    # ---------------------------------------------------------------- extract
    phase = time.perf_counter()
    if args.extraction == "teacher_forced":
        vectors, spreads = extract_teacher_forced(manager, modules, report)
    else:
        vectors, spreads = extract_generation(manager, modules, settings, report)
    summarise(vectors, spreads, modules, layers, report)
    best_layer = max(report.directions, key=lambda item: item.separation)
    print(
        f"[phase1] {args.extraction} separations "
        + ", ".join(f"L{item.layer}:{item.separation:.1f}" for item in report.directions)
        + f" -> best L{best_layer.layer} ({time.perf_counter() - phase:.1f}s)"
    )

    unit_vectors = {module_id: unit(vectors[module_id]).to(device) for module_id in modules}
    layer_count = len(modules)

    def steer(
        alpha_total: float,
        *,
        directions: dict[str, torch.Tensor] | None = None,
        per_layer: bool = False,
    ) -> MultiLayerSteering:
        chosen = directions or unit_vectors
        if per_layer:
            scale = {module_id: alpha_total for module_id in chosen}
        else:
            scale = {module_id: alpha_total / math.sqrt(layer_count) for module_id in chosen}
        return MultiLayerSteering(model, chosen, scale)

    # ----------------------------------------------------------------- ladder
    phase = time.perf_counter()
    ladder_rows: list[dict[str, object]] = []
    lenient_targets: dict[str, dict[str, int]] = {}
    for prompt in LADDER_PROMPTS:
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        warm = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        ).response
        base_record = record("ladder", "baseline", prompt, None, base)
        warm_record = record("ladder", "warm target", prompt, WARM_SYSTEM, warm)
        report.runs += [base_record, warm_record]
        lenient_targets[prompt] = warm_record.style
        for alpha in alphas:
            with steer(alpha):
                steered = generate_baseline(
                    manager, BaselineRequest(prompt=prompt, settings=settings)
                ).response
            steered_record = record("ladder", f"alpha {alpha:+.2f}", prompt, None, steered, alpha=alpha)
            report.runs.append(steered_record)
            fraction = fraction_toward(steered_record.style, base_record.style, warm_record.style)
            ladder_rows.append(
                {
                    "prompt": prompt,
                    "alpha": alpha,
                    "style": steered_record.style,
                    "words": steered_record.words,
                    "distinct_3": steered_record.distinct_3,
                    "refusal": steered_record.refusal,
                    "fraction_warm": round(fraction, 4) if fraction is not None else None,
                    "output": steered_record.output,
                }
            )
            print(
                f"[ladder] «{prompt[:22]}» α={alpha:+.2f} style={steered_record.style} "
                f"toward warm={fraction * 100 if fraction is not None else float('nan'):+.0f}%"
            )
    report.ladder = ladder_rows

    def rows_for(alpha: float) -> list[dict[str, object]]:
        return [row for row in ladder_rows if float(row["alpha"]) == alpha]

    def admissible(alpha: float) -> bool:
        return all(not row["refusal"] and float(row["distinct_3"]) >= 0.65 for row in rows_for(alpha))

    def score(alpha: float) -> float:
        values = [float(row["fraction_warm"] or 0.0) for row in rows_for(alpha)]
        return statistics.fmean(values) if values else 0.0

    candidates = [alpha for alpha in alphas if admissible(alpha)]
    # Ties are broken upward: when every alpha scores the same (usually zero),
    # validate the strongest admissible intervention instead of the weakest.
    chosen_alpha = max(candidates, key=lambda value: (score(value), value)) if candidates else alphas[0]
    report.chosen = {
        "alpha_total": chosen_alpha,
        "per_layer": round(chosen_alpha / math.sqrt(layer_count), 4),
        "layers": modules,
        "best_single_layer": best_layer.module_id,
        "ladder_scores": {str(alpha): round(score(alpha), 4) for alpha in alphas},
        "rule": "max(mean fraction toward warm target) on the ladder prompts, admissible if no refusal and distinct-3 >= 0.65",
    }
    print(f"[phase2] chosen α_total={chosen_alpha} ({time.perf_counter() - phase:.1f}s)")

    # -------------------------------------------------------------- held-out
    phase = time.perf_counter()
    for prompt in HELD_OUT_PROMPTS:
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        warm = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        ).response
        base_record = record("held_out", "baseline", prompt, None, base)
        warm_record = record("held_out", "warm target", prompt, WARM_SYSTEM, warm)
        with steer(chosen_alpha):
            steered = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        steered_record = record("held_out", "multi-layer +α", prompt, None, steered, alpha=chosen_alpha)
        with steer(-chosen_alpha):
            cancelled = generate_baseline(
                manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
            ).response
        cancelled_record = record(
            "held_out", "multi-layer -α on warm", prompt, WARM_SYSTEM, cancelled, alpha=-chosen_alpha
        )
        report.runs += [base_record, warm_record, steered_record, cancelled_record]
        install = fraction_toward(steered_record.style, base_record.style, warm_record.style)
        cancel = fraction_toward(cancelled_record.style, warm_record.style, base_record.style)
        report.held_out.append(
            {
                "prompt": prompt,
                "install_fraction": round(install, 4) if install is not None else None,
                "cancel_fraction": round(cancel, 4) if cancel is not None else None,
                "styles": {
                    "baseline": base_record.style,
                    "warm": warm_record.style,
                    "steered": steered_record.style,
                    "cancelled": cancelled_record.style,
                },
                "outputs": {
                    "baseline": base_record.output,
                    "warm": warm_record.output,
                    "steered": steered_record.output,
                    "cancelled": cancelled_record.output,
                },
            }
        )
        print(
            f"[held-out] «{prompt[:26]}» install={install * 100 if install is not None else float('nan'):+.0f}% "
            f"cancel={cancel * 100 if cancel is not None else float('nan'):+.0f}%"
        )

    # --------------------------------------------------------- single layer
    for prompt in HELD_OUT_PROMPTS[:3]:
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        warm = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        ).response
        base_record = record("single_layer", "baseline", prompt, None, base)
        warm_record = record("single_layer", "warm target", prompt, WARM_SYSTEM, warm)
        with steer(
            chosen_alpha,
            directions={best_layer.module_id: unit_vectors[best_layer.module_id]},
            per_layer=True,
        ):
            steered = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        steered_record = record("single_layer", "single +α", prompt, None, steered, alpha=chosen_alpha)
        report.runs += [base_record, warm_record, steered_record]
        fraction = fraction_toward(steered_record.style, base_record.style, warm_record.style)
        report.single_layer.append(
            {
                "prompt": prompt,
                "layer": best_layer.layer,
                "fraction_warm": round(fraction, 4) if fraction is not None else None,
                "style": steered_record.style,
                "output": steered_record.output,
            }
        )

    # ------------------------------------------------------------- controls
    rng = random.Random(11)
    rotated = {
        module_id: unit_vectors[modules[(index + 1) % layer_count]] for index, module_id in enumerate(modules)
    }
    random_directions = {}
    for module_id in modules:
        raw = np.array([rng.gauss(0.0, 1.0) for _ in range(2048)])
        random_directions[module_id] = unit(raw).to(device)
    for prompt in CONTROL_PROMPTS:
        base = generate_baseline(manager, BaselineRequest(prompt=prompt, settings=settings)).response
        warm = generate_baseline(
            manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
        ).response
        base_record = record("control", "baseline", prompt, None, base)
        warm_record = record("control", "warm target", prompt, WARM_SYSTEM, warm)
        report.runs += [base_record, warm_record]
        for label, directions in (("shuffled layers", rotated), ("random directions", random_directions)):
            with steer(chosen_alpha, directions=directions):
                steered = generate_baseline(
                    manager, BaselineRequest(prompt=prompt, settings=settings)
                ).response
            steered_record = record("control", label, prompt, None, steered, alpha=chosen_alpha)
            report.runs.append(steered_record)
            fraction = fraction_toward(steered_record.style, base_record.style, warm_record.style)
            report.controls.append(
                {
                    "prompt": prompt,
                    "label": label,
                    "arm": "install",
                    "fraction_warm": round(fraction, 4) if fraction is not None else None,
                    "style": steered_record.style,
                }
            )
            print(
                f"[control] «{prompt[:22]}» {label} install: toward warm="
                f"{fraction * 100 if fraction is not None else float('nan'):+.0f}%"
            )

            # The same control directions on an *active* persona: does any
            # generic perturbation cancel the warm register as well as ours?
            with steer(-chosen_alpha, directions=directions):
                cancelled = generate_baseline(
                    manager, BaselineRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings)
                ).response
            cancelled_record = record(
                "control", f"{label} (cancel)", prompt, WARM_SYSTEM, cancelled, alpha=-chosen_alpha
            )
            report.runs.append(cancelled_record)
            cancel_fraction = fraction_toward(cancelled_record.style, warm_record.style, base_record.style)
            report.controls.append(
                {
                    "prompt": prompt,
                    "label": label,
                    "arm": "cancel",
                    "fraction_cancel": round(cancel_fraction, 4) if cancel_fraction is not None else None,
                    "style": cancelled_record.style,
                }
            )
            print(
                f"[control] «{prompt[:22]}» {label} cancel: toward baseline="
                f"{cancel_fraction * 100 if cancel_fraction is not None else float('nan'):+.0f}%"
            )

    # ------------------------------------------------------------ capability
    capability_prompts = (
        ("Reply with exactly one word: the opposite of 'hot'.", "cold"),
        ("What is 17 + 26? Reply with only the number.", "43"),
        ("Name the largest planet in the solar system. One word.", "jupiter"),
        ("Translate to English: 'la casa es grande'.", "house"),
    )
    for prompt, needle in capability_prompts:
        base = generate_baseline(
            manager, BaselineRequest(prompt=prompt, settings=GenerationSettings(max_new_tokens=16))
        ).response
        with steer(chosen_alpha):
            steered = generate_baseline(
                manager, BaselineRequest(prompt=prompt, settings=GenerationSettings(max_new_tokens=16))
            ).response
        base_ok = needle.lower() in base.output.lower()
        steered_ok = needle.lower() in steered.output.lower()
        report.capability.append(
            {
                "prompt": prompt,
                "baseline_ok": base_ok,
                "steered_ok": steered_ok,
                "baseline_output": base.output,
                "steered_output": steered.output,
            }
        )

    # --------------------------------------------------------------- summary
    installs = [row["install_fraction"] for row in report.held_out if row["install_fraction"] is not None]
    cancels = [row["cancel_fraction"] for row in report.held_out if row["cancel_fraction"] is not None]
    install_hits = sum(1 for value in installs if value >= 0.5)
    cancel_hits = sum(1 for value in cancels if value >= 0.5)
    control_values = [row["fraction_warm"] for row in report.controls if row.get("fraction_warm") is not None]
    control_cancel_values = [
        row["fraction_cancel"] for row in report.controls if row.get("fraction_cancel") is not None
    ]
    capability_base = sum(1 for row in report.capability if row["baseline_ok"])
    capability_steered = sum(1 for row in report.capability if row["steered_ok"])
    report.notes = [
        f"Install criterion: {install_hits}/{len(installs)} held-out prompts recovered >= 50% of the baseline-to-warm distance "
        f"(mean {statistics.fmean(installs) * 100:+.0f}%)."
        if installs
        else "Install criterion: no measurable distance.",
        f"Cancel criterion: {cancel_hits}/{len(cancels)} warm prompts lost >= 50% of the distance "
        f"(mean {statistics.fmean(cancels) * 100:+.0f}%)."
        if cancels
        else "Cancel criterion: no measurable distance.",
        (
            f"Install controls recovery: {statistics.fmean(control_values) * 100:+.0f}% (target: below 20%)."
            if control_values
            else "Install controls unavailable."
        ),
        (
            f"Cancel controls recovery: {statistics.fmean(control_cancel_values) * 100:+.0f}% "
            f"vs the real direction's {statistics.fmean(cancels) * 100:+.0f}%."
            if control_cancel_values and cancels
            else "Cancel controls unavailable."
        ),
        f"Capability needle present: baseline {capability_base}/{len(report.capability)}, "
        f"steered {capability_steered}/{len(report.capability)}.",
        f"Total displacement budget α={chosen_alpha}, per layer α={chosen_alpha / math.sqrt(layer_count):.3f}, "
        f"layers {layers}, full 2048-d unit vectors scaled by per-token residual RMS.",
        "Style metrics are lexical/structural; small samples and one model. Treat this as a pilot.",
    ]
    report.chosen["criteria"] = {
        "install_hits": install_hits,
        "held_out": len(installs),
        "cancel_hits": cancel_hits,
    }
    print("\n".join(f"- {note}" for note in report.notes))
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
