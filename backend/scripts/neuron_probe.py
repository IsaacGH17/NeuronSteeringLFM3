"""Individual MLP-neuron steering: candidate screen without generating text.

The MLP intermediate is the input of ``feed_forward.w2`` (7680 dimensions for
LFM3-1.7B). A neuron is steered by adding a scalar to that intermediate; its
downstream contribution is the neuron's column of ``w2``. Because LFM3 applies a
sandwich RMSNorm *after* the MLP, the real gain of neuron ``i`` is

    gain_i = || w2.weight[:, i] * post_ffn_norm.weight ||

so a perturbation of ``alpha / gain_i`` aims at a pre-norm residual displacement
of ``alpha``. The probe measures the actual post-norm attenuation and screens
candidate neurons with a single forward pass each:

  * KL(base || steered) at the next-token distribution,
  * change in probability mass of style tokens (emoji, "!", "Oh", "Hey", ...),
  * max |delta logits|,

against random-neuron controls of the same magnitude. Candidates are chosen by
the teacher-forced warm/clinical contrast *before* any effect is measured.
Only neurons that beat every random control on both screen prompts are kept.
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
from backend.app.services.architecture_inspector import get_module
from backend.app.services.model_loader import ModelManager
from backend.scripts.multilayer_probe import (
    CLINICAL_SYSTEM,
    PASSAGES,
    TEACHER_USER,
    WARM_SYSTEM,
    render_conversation,
)

SCREEN_PROMPTS = ("Explain why the sky is blue.", "How do I learn to play guitar?")
STYLE_STRINGS = ("!", "!!", "😊", "🌧", "☔", "✨", "Oh", "Hey", "Sure", "Wow")


@dataclass
class Candidate:
    layer: int
    neuron: int
    contrast: float
    gain: float


@dataclass
class ScreenResult:
    layer: int
    neuron: int
    sign: int
    alpha: float
    kl: float
    style_delta: float
    max_logit_delta: float
    prompt: str


@dataclass
class NeuronReport:
    created_at: str
    device: str
    dtype: str
    config: dict[str, object]
    engineering: dict[str, object] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    controls: list[Candidate] = field(default_factory=list)
    screens: list[ScreenResult] = field(default_factory=list)
    verdict: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def intermediate_capture(model, layers: list[int]):
    """Forward pre-hooks on every w2 that store the intermediate tensor."""
    store: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        module = get_module(model, f"model.layers.{layer}.feed_forward.w2")

        def hook(mod, args, *, layer=layer):
            tensor = args[0]
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                store[layer] = tensor.detach().to("cpu", dtype=torch.float32)
            return None

        handles.append(module.register_forward_pre_hook(hook))
    return store, handles


def forward_hook_capture(model, module_ids: list[str]):
    store: dict[str, torch.Tensor] = {}
    handles = []
    for module_id in module_ids:
        module = get_module(model, module_id)

        def hook(mod, args, output, *, module_id=module_id):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                store[module_id] = tensor.detach().to("cpu", dtype=torch.float32)
            return None

        handles.append(module.register_forward_hook(hook))
    return store, handles


def neuron_hook(model, layer: int, neuron: int, delta: float):
    module = get_module(model, f"model.layers.{layer}.feed_forward.w2")

    def hook(mod, args):
        tensor = args[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            return None
        steered = tensor.to(torch.float32).clone()
        steered[..., neuron] = steered[..., neuron] + delta
        return (steered.to(tensor.dtype),)

    return module.register_forward_pre_hook(hook)


def chat_inputs(tokenizer, device, prompt: str, system: str | None = None):
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoding = tokenizer(rendered, return_tensors="pt")
    return {key: value.to(device) for key, value in encoding.items()}


def last_position_probs(model, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    return logits, torch.softmax(logits, dim=-1)


def kl_divergence(base: torch.Tensor, other: torch.Tensor) -> float:
    eps = 1e-12
    return float(torch.sum(base * (torch.log(base + eps) - torch.log(other + eps))).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="12,16,20,22")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--alphas", default="0.1,0.3")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    layers = [int(item) for item in args.layers.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",")]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"neurons-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=64))
    started = time.perf_counter()
    status = manager.load()
    model, tokenizer = manager.model, manager.tokenizer
    device = status.device or args.device
    config = model.config
    intermediate = int(config.intermediate_size)
    hidden = int(config.hidden_size)
    sandwich = bool(getattr(config, "use_sandwich_norm", False))
    post_weight = {
        layer: get_module(model, f"model.layers.{layer}.post_ffn_norm").weight.detach()
        if sandwich
        else torch.ones(hidden)
        for layer in layers
    }
    gains: dict[int, np.ndarray] = {}
    for layer in layers:
        w2 = get_module(model, f"model.layers.{layer}.feed_forward.w2").weight.detach().float()
        gains[layer] = (w2 * post_weight[layer][:, None]).norm(dim=0).cpu().numpy()

    report = NeuronReport(
        created_at=datetime.now(UTC).isoformat(),
        device=device,
        dtype=status.dtype or "?",
        config={
            "layers": layers,
            "intermediate_size": intermediate,
            "hidden_size": hidden,
            "use_sandwich_norm": sandwich,
            "top_k": args.top_k,
            "random_controls": args.random_controls,
            "alphas": alphas,
            "screen_prompts": list(SCREEN_PROMPTS),
            "style_strings": list(STYLE_STRINGS),
            "warm_system": WARM_SYSTEM,
            "clinical_system": CLINICAL_SYSTEM,
        },
    )
    print(f"[load] {status.state} {device} {status.dtype} in {time.perf_counter() - started:.1f}s")

    # ---------------------------------------------------------------- phase 0
    store, handles = intermediate_capture(model, layers)
    probe_inputs = chat_inputs(tokenizer, device, "Describe a rainy morning.")
    residual_store, residual_handles = forward_hook_capture(
        model, [f"model.layers.{layer}" for layer in layers]
    )
    ffn_store, ffn_handles = forward_hook_capture(
        model, [f"model.layers.{layer}.feed_forward" for layer in layers]
    )
    with torch.inference_mode():
        model(**probe_inputs)
    for group in (handles, residual_handles, ffn_handles):
        for handle in group:
            handle.remove()
    shapes = {layer: list(store[layer].shape) for layer in layers}
    assert all(shape[-1] == intermediate for shape in shapes.values()), shapes
    residual_rms = {
        layer: float(residual_store[f"model.layers.{layer}"][0, -1].norm() / math.sqrt(hidden))
        for layer in layers
    }
    ffn_rms = {
        layer: float(ffn_store[f"model.layers.{layer}.feed_forward"][0, -1].norm() / math.sqrt(hidden))
        for layer in layers
    }
    report.engineering = {
        "intermediate_shapes": shapes,
        "gain_min": {layer: float(gains[layer].min()) for layer in layers},
        "gain_median": {layer: float(np.median(gains[layer])) for layer in layers},
        "gain_max": {layer: float(gains[layer].max()) for layer in layers},
        "residual_rms_last_token": residual_rms,
        "ffn_out_rms_last_token": ffn_rms,
        "attenuation_ffn_over_residual": {
            layer: round(ffn_rms[layer] / residual_rms[layer], 4) for layer in layers
        },
    }
    print(
        "[phase0] intermediate",
        {layer: shapes[layer] for layer in layers},
        f"| gain median {np.median([np.median(gains[layer]) for layer in layers]):.3f}",
        f"| ffn/residual rms {report.engineering['attenuation_ffn_over_residual']}",
    )

    # ---------------------------------------------------------------- phase 1
    phase = time.perf_counter()
    warm_positions: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    clinical_positions: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
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
            store.clear()
            handles = []
            for layer in layers:
                module = get_module(model, f"model.layers.{layer}.feed_forward.w2")

                def hook(mod, hook_args, *, layer=layer):
                    tensor = hook_args[0]
                    if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                        store[layer] = tensor.detach().to("cpu", dtype=torch.float32)
                    return None

                handles.append(module.register_forward_pre_hook(hook))
            with torch.inference_mode():
                model(**inputs)
            for handle in handles:
                handle.remove()
            target = warm_positions if label == "warm" else clinical_positions
            for layer in layers:
                target[layer].append(store[layer][0, positions].numpy().astype(np.float64))
            print(f"[phase1] {label:8s} «{passage[:24]}…» {len(positions)} tokens")
    vectors: dict[int, np.ndarray] = {}
    spreads: dict[int, np.ndarray] = {}
    for layer in layers:
        differences = [warm.mean(0) - clinical.mean(0) for warm, clinical in zip(warm_positions[layer], clinical_positions[layer], strict=True)]
        vectors[layer] = np.mean(differences, axis=0)
        spreads[layer] = np.concatenate(warm_positions[layer], axis=0).std(axis=0)

    rng = random.Random(23)
    for layer in layers:
        contrast = vectors[layer]
        order = np.argsort(-np.abs(contrast))[: args.top_k]
        for neuron in order:
            report.candidates.append(
                Candidate(
                    layer=layer,
                    neuron=int(neuron),
                    contrast=round(float(contrast[neuron]), 5),
                    gain=round(float(gains[layer][neuron]), 5),
                )
            )
        candidates = [item.neuron for item in report.candidates if item.layer == layer]
        pool = [index for index in range(intermediate) if index not in set(candidates)]
        for neuron in rng.sample(pool, min(args.random_controls, len(pool))):
            report.controls.append(
                Candidate(
                    layer=layer,
                    neuron=int(neuron),
                    contrast=round(float(contrast[neuron]), 5),
                    gain=round(float(gains[layer][neuron]), 5),
                )
            )
    print(f"[phase1] {len(report.candidates)} candidates, {len(report.controls)} controls ({time.perf_counter() - phase:.1f}s)")

    # ---------------------------------------------------------------- phase 2
    phase = time.perf_counter()
    style_ids = set()
    for text in STYLE_STRINGS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            style_ids.add(int(ids[0]))
    style_ids_tensor = torch.tensor(sorted(style_ids), device=device)
    print(f"[phase2] style token ids: {sorted(style_ids)}")

    prompts = []
    for prompt in SCREEN_PROMPTS:
        inputs = chat_inputs(tokenizer, device, prompt)
        logits, probs = last_position_probs(model, inputs)
        prompts.append(
            {
                "prompt": prompt,
                "inputs": inputs,
                "logits": logits,
                "probs": probs,
                "style_mass": float(probs[style_ids_tensor].sum().item()),
            }
        )

    def screen(layer: int, neuron: int, sign: int, alpha_fraction: float) -> dict[str, float]:
        results = {"kl": 0.0, "style_delta": 0.0, "max_logit_delta": 0.0}
        delta = sign * alpha_fraction * residual_rms[layer] / max(gains[layer][neuron], 1e-9)
        handle = neuron_hook(model, layer, neuron, float(delta))
        try:
            for entry in prompts:
                logits, probs = last_position_probs(model, entry["inputs"])
                results["kl"] = max(results["kl"], kl_divergence(entry["probs"], probs))
                results["style_delta"] = max(
                    results["style_delta"],
                    abs(float(probs[style_ids_tensor].sum().item()) - entry["style_mass"]),
                )
                results["max_logit_delta"] = max(
                    results["max_logit_delta"], float((logits - entry["logits"]).abs().max().item())
                )
        finally:
            handle.remove()
        return results

    def signed_effect(layer: int, neuron: int, sign: int, alpha_fraction: float) -> float:
        """Signed style-mass change, averaged over the two screen prompts."""
        delta = sign * alpha_fraction * residual_rms[layer] / max(gains[layer][neuron], 1e-9)
        handle = neuron_hook(model, layer, neuron, float(delta))
        changes = []
        try:
            for entry in prompts:
                _, probs = last_position_probs(model, entry["inputs"])
                changes.append(float(probs[style_ids_tensor].sum().item()) - entry["style_mass"])
        finally:
            handle.remove()
        return float(statistics.fmean(changes))

    control_effects = []
    for candidate in report.controls:
        for sign in (1, -1):
            for alpha_fraction in alphas:
                control_effects.append(abs(signed_effect(candidate.layer, candidate.neuron, sign, alpha_fraction)))
    control_cutoff = max(control_effects) if control_effects else 0.0
    print(f"[phase2] control max |Δstyle| = {control_cutoff:.5f}")

    for candidate in report.candidates:
        best = None
        for sign in (1, -1):
            for alpha_fraction in alphas:
                effect = signed_effect(candidate.layer, candidate.neuron, sign, alpha_fraction)
                metrics = screen(candidate.layer, candidate.neuron, sign, alpha_fraction)
                record = ScreenResult(
                    layer=candidate.layer,
                    neuron=candidate.neuron,
                    sign=sign,
                    alpha=alpha_fraction,
                    kl=round(metrics["kl"], 6),
                    style_delta=round(effect, 6),
                    max_logit_delta=round(metrics["max_logit_delta"], 4),
                    prompt="mean",
                )
                report.screens.append(record)
                if best is None or abs(effect) > abs(best.style_delta):
                    best = record
        print(
            f"[screen] L{candidate.layer} n={candidate.neuron} contrast={candidate.contrast:+.4f} "
            f"best Δstyle={best.style_delta:+.5f} KL={best.kl:.5f} maxΔlogit={best.max_logit_delta:.3f}"
        )

    passing = [
        record
        for record in report.screens
        if abs(record.style_delta) > control_cutoff
    ]
    report.verdict = {
        "control_cutoff": round(control_cutoff, 6),
        "control_effects_max": round(max(control_effects, default=0.0), 6),
        "passing_screens": len(passing),
        "passing_neurons": sorted({(record.layer, record.neuron) for record in passing}),
        "max_kl": round(max((record.kl for record in report.screens), default=0.0), 6),
        "max_style_delta": round(
            max((abs(record.style_delta) for record in report.screens), default=0.0), 6
        ),
    }
    best_screens = sorted(report.screens, key=lambda record: -abs(record.style_delta))[:10]
    report.notes = [
        f"{len(report.candidates)} candidates (top {args.top_k} per layer by contrast) and "
        f"{len(report.controls)} random controls, {len(report.screens)} screen forwards.",
        f"Neurons whose signed style-token shift beats every random control: "
        f"{len(report.verdict['passing_neurons'])}.",
        f"Post-norm attenuation (ffn_out rms / residual rms): "
        f"{report.engineering['attenuation_ffn_over_residual']}.",
        "Screen prompts were not used to pick candidates; candidates come from teacher-forced passages.",
        "A pass here is only evidence of a measurable next-token effect, not of a visible style change.",
        "Style mass sums the probability of a small set of emoji/punctuation/opening-token ids.",
    ]
    print("\n".join(f"- {note}" for note in report.notes))
    print(
        "[top] "
        + ", ".join(
            f"L{record.layer}:{record.neuron} Δ{record.style_delta:+.4f}"
            for record in best_screens[:6]
        )
    )
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
