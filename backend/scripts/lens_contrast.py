"""Compare the logit lens of two system prompts on the same user prompt.

Runs the Interpret lens for a warm/casual and a clinical/terse system prompt and
measures, layer by layer:

  * top-1 agreement at the last prompt token (the state that predicts the first
    generated token),
  * when the two conditions first split and stay split,
  * how confident that prediction becomes with depth.

Generated positions are *not* compared: the two conditions produce different
continuations, so index k is a different context in each run.

The lens is a linear readout applied to each layer; it shows when the model's
next-token preference becomes condition-specific, not where the computation
"happens". Outputs a JSON report plus a compact table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from backend.app.core.settings import Settings
from backend.app.schemas.api import GenerationSettings, LensRequest, LensResponse
from backend.app.services.lens import compute_lens
from backend.app.services.model_loader import ModelManager
from backend.scripts.multilayer_probe import CLINICAL_SYSTEM, WARM_SYSTEM
from backend.scripts.personality_pilot import style_features

PROMPTS = ("Describe a rainy morning.", "How do I learn to play guitar?")
TOKENS = 40


@dataclass
class LayerRow:
    layer: int
    warm_token: str
    warm_prob: float
    clinical_token: str
    clinical_prob: float
    same: bool
    confidence: float


@dataclass
class PromptReport:
    prompt: str
    warm_prompt_tokens: int
    clinical_prompt_tokens: int
    generated_positions: int
    warm_output: str
    clinical_output: str
    warm_style: dict[str, int]
    clinical_style: dict[str, int]
    rows: list[LayerRow] = field(default_factory=list)


@dataclass
class Report:
    created_at: str
    device: str
    dtype: str
    tokens: int
    prompts: list[PromptReport] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def compare(warm: LensResponse, clinical: LensResponse) -> list[LayerRow]:
    rows: list[LayerRow] = []
    for warm_layer, clinical_layer in zip(warm.layers, clinical.layers, strict=True):
        warm_top = warm_layer.top_tokens[0]
        clinical_top = clinical_layer.top_tokens[0]
        rows.append(
            LayerRow(
                layer=warm_layer.layer_index,
                warm_token=warm_top.text,
                warm_prob=round(warm_top.probability, 4),
                clinical_token=clinical_top.text,
                clinical_prob=round(clinical_top.probability, 4),
                same=warm_top.token_id == clinical_top.token_id,
                confidence=round(min(warm_top.probability, clinical_top.probability), 4),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("data/pilot") / f"lens-contrast-{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager(Settings(device=args.device, max_new_tokens=128))
    started = time.perf_counter()
    status = manager.load()
    settings = GenerationSettings(max_new_tokens=args.tokens)
    report = Report(
        created_at=datetime.now(UTC).isoformat(),
        device=status.device or args.device,
        dtype=status.dtype or "?",
        tokens=args.tokens,
    )
    print(f"[load] {status.state} {status.device} {status.dtype} in {time.perf_counter() - started:.1f}s")

    for prompt in PROMPTS:
        warm = compute_lens(manager, LensRequest(prompt=prompt, system=WARM_SYSTEM, settings=settings))
        clinical = compute_lens(
            manager, LensRequest(prompt=prompt, system=CLINICAL_SYSTEM, settings=settings)
        )
        entry = PromptReport(
            prompt=prompt,
            warm_prompt_tokens=warm.prompt_tokens,
            clinical_prompt_tokens=clinical.prompt_tokens,
            generated_positions=min(
                len(warm.tokens) - warm.prompt_tokens, len(clinical.tokens) - clinical.prompt_tokens
            ),
            warm_output=warm.output,
            clinical_output=clinical.output,
            warm_style=style_features(warm.output),
            clinical_style=style_features(clinical.output),
            rows=compare(warm, clinical),
        )
        report.prompts.append(entry)
        print(f"\n== «{prompt}»")
        print(f"   warm     ({warm.prompt_tokens} prompt tok): {warm.output[:70].replace(chr(10), ' ')}")
        print(
            f"   clinical ({clinical.prompt_tokens} prompt tok): {clinical.output[:70].replace(chr(10), ' ')}"
        )
        for row in entry.rows:
            marker = "=" if row.same else "!"
            print(
                f"   L{row.layer:02d} {marker} warm {row.warm_token!r:14s} {row.warm_prob:.3f} | "
                f"clinical {row.clinical_token!r:14s} {row.clinical_prob:.3f}"
            )

    splits: dict[str, int | None] = {}
    last_agreements: dict[str, int] = {}
    for prompt in report.prompts:
        rows = prompt.rows
        split = None
        for index, row in enumerate(rows):
            if not row.same and all(not later.same for later in rows[index:]):
                split = row.layer
                break
        splits[prompt.prompt] = split
        agreements = [row.layer for row in rows if row.same]
        last_agreements[prompt.prompt] = max(agreements) if agreements else -1
    confident = [row for prompt in report.prompts for row in prompt.rows if row.layer >= 15 and not row.same]
    mean_confidence_late = (
        round(statistics.fmean(row.confidence for row in confident), 4) if confident else 0.0
    )
    report.summary = {
        "stable_split_layer": dict(splits),
        "last_agreeing_layer": dict(last_agreements),
        "mean_min_confidence_from_layer_15": mean_confidence_late,
        "final_tokens": {
            prompt.prompt: {
                "warm": prompt.rows[-1].warm_token,
                "warm_prob": prompt.rows[-1].warm_prob,
                "clinical": prompt.rows[-1].clinical_token,
                "clinical_prob": prompt.rows[-1].clinical_prob,
            }
            for prompt in report.prompts
        },
    }
    report.notes = [
        "Generated positions are not compared: the two conditions produce different continuations.",
        f"Stable split at layer(s) {splits}; last agreement before that at {last_agreements}.",
        f"Mean min-probability of the split prediction from layer 15: {mean_confidence_late:.3f}.",
        "The lens is a linear readout; it shows when the preference becomes condition-specific, not where it is computed.",
    ]
    print("\n".join(f"- {note}" for note in report.notes))
    output_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {output_path} ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
