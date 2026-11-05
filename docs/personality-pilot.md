# Personality pilot: what moved, what did not

Date: 2026-11-04 · model `LiquidAI/LFM3-1.7B` · device `cuda` (RX 7600, bf16) · greedy decoding.

Script: `backend/scripts/personality_pilot.py` (reproducible; writes JSON next to a Markdown report).
Raw run: `data/pilot/personality-*.json` (gitignored).

## The question

Can one activation direction — a weighted set of hidden dimensions — reproduce a
system-prompt "personality" (warm/casual vs clinical/terse) while the model keeps
passing its capability checks?

## Design

- **Contrast.** Same four neutral prompts, two system prompts: a bubbly friend
  (exclamations, contractions, emoji) vs a technical report writer (declarative,
  no adjectives, no emoji, no contractions).
- **Direction.** For each candidate module, average the last-prompt-token state
  difference across the four prompt pairs; keep the top 8 dimensions, weights
  normalized to max |w| = 1.
- **Ladder.** Steer at α ∈ {−120, −80, −40, 0, 40, 80, 120} σ on two prompts that
  were not used to look for an effect, and choose α by style distance to the
  warm-target run.
- **Validation.** Six unseen prompts, each with baseline, steered, warm target and
  clinical target; capability checks on six tasks; random-direction and reverse-α
  controls.
- **Style metrics.** Emoji, exclamation marks, contractions, first-person
  pronouns, words, distinct-3, refusal. Lexical heuristics over ~40-token
  generations.

## Result 1 — the system prompt does move the personality

Same prompt, 60 tokens:

| condition | emoji | `!` | contractions | first person | words |
|---|---:|---:|---:|---:|---:|
| plain | 0 | 0 | 0 | 0 | 45 |
| warm system | 3 | 1 | 1 | 1 | 38 |
| clinical system | 0 | 0 | 0 | 0 | 47 |

The effect is large, immediate and free: the warm run opens with "Oh wow, that's
such a cozy vibe! ☔️" and the clinical one with "The sky is gray. Rain falls
steadily on the pavement." (short declaratives). This is the reference behaviour
an activation direction would have to reproduce.

## Result 2 — a single-layer mean-difference direction does not carry it

Layer separations were large and rose with depth:

| module | separation (‖d‖ / RMS σ) |
|---|---:|
| `layers.4.feed_forward` | 58.2 |
| `layers.8.feed_forward` | 56.4 |
| `layers.12.feed_forward` | 62.9 |
| `layers.16.feed_forward` | 58.3 |
| `layers.20.feed_forward` | **84.6** |
| `layers.22.feed_forward` | 64.1 |

Large separation did **not** mean causal grip. In the ladder, no α produced a
single emoji or exclamation mark on either prompt; every steered row had the same
style vector as the baseline. On the six held-out prompts the steered outputs
moved toward the warm target in **0/6** cases (mean −21 % of the baseline-to-target
distance), while capability checks did not degrade (baseline 5/9, steered 6/9).
The random-direction control (−50 %) and reverse-α (+50 %) were within the same
noise band.

Follow-up probes (same model, same prompt):

- Steering the **residual stream** (`model.layers.16` / `22`) at 40–100 σ did not
  restyle the text: it collapsed into repetition ("ours ours ours", "evident
  evident 7433…"). So the residual has a much smaller safe budget than the FFN
  output, and large α there is destructive rather than expressive.
- **Subtracting** the direction from the warm prompt did not cancel the persona:
  the output stayed fully warm (2 emoji, 1 `!`, 3 contractions). If the vector
  encoded the trait, the opposite sign should attenuate it.
- The same null result held with the direction built from **generated-token**
  states, with the **full 2048-d vector**, and when scaled by the per-token
  residual RMS instead of per-dimension σ.

Conclusion: within this pilot, the mean-difference vector at one layer encodes
"the prompt contained different instruction text", not a transferable style
policy. It perturbs wording but neither installs nor removes the persona.

## Result 3 — a real bug: σ mode was a no-op during decoding

While diagnosing the silence, the app's σ normalization turned out to be broken:
it used the standard deviation of the current forward pass, and every decode step
carries a single token, where the spread is exactly zero. Every dimension was
skipped after the prompt, and the run warned about it only once.

Fixed in `backend/app/services/steering_engine.py`: σ is now calibrated on the
first forward with more than one token (the prompt prefill) and reused for the
rest of the generation. Regression tests cover both the unit (`_steered_output`)
and the hook path. Any earlier σ-mode experiment with this build should be
re-read: after the first token it was running unmodified.


## Multi-layer follow-up (same day, second experiment)

Question: does applying the warm/clinical direction at **several residual layers
at once**, with **full 2048-d vectors** instead of top-8 dimensions, install or
cancel the persona?

Script: `backend/scripts/multilayer_probe.py` (raw run: `data/pilot/multilayer-*.json`).

Design: one direction per decoder layer for layers 8, 12, 16, 20 and 22, extracted
from the same four prompt pairs (last prompt token). During generation every
layer is shifted by `alpha_l * rms(h_l) * d_hat_l`, with `alpha_l = alpha / sqrt(5)`
so the total displacement budget is constant. The ladder used α_total ∈ {1, 2, 4, 8};
α_total = 8 (≈3.58 per layer) was selected by the pre-registered rule. Held-out:
six prompts with four arms each, three single-layer comparisons, four control runs,
and four capability needles.

**A first pass at α_total ∈ {0.25, 0.5, 1, 2} produced byte-identical outputs.**
A direct logit check showed the perturbation was real but tiny (max |Δlogits| = 0.5
at α = 1 per layer), well below the threshold that changes greedy tokens. That pass
is underpowered and is not counted as evidence; the numbers below come from the
second, calibrated pass.

| arm | result |
|---|---|
| install (plain prompt + direction) | **0/6** prompts recovered ≥50 % of the distance; mean −12.5 % |
| cancel (warm prompt − direction) | **2/6** ≥50 %, mean −4.2 %; two prompts became *warmer* (−100 %, −50 %) |
| single layer (L8, same per-layer α) | 0/3, mean −8.3 % |
| shuffled / random layer directions | **0 %** in all four runs |
| capability needles | baseline 3/4, steered 4/4 |

Layer geometry: adjacent directions are only partly aligned (cosine 0.55 for
L8–L12, 0.39 for L16–L22) and distant layers are nearly orthogonal (0.03 for
L8–L22), so the multi-layer intervention is not a redundant copy of one vector.

At α_total = 8 the text stays coherent (distinct-3 = 1.0, no refusals) while the
wording clearly changes, so this is not a magnitude artifact: the intervention
took effect and still did not move the persona. The cancellation arm is the
strongest evidence — subtracting the direction sometimes removed formatting
markers, sometimes left them, and sometimes produced *more* warm markers than the
warm prompt itself.

Interpretation: the axis being probed here is largely **formatting/instruction
behaviour** (emoji, exclamation marks, contractions, terse sentences). The model
appears to represent "use emoji and exclamations" as an instruction-following
policy rather than as a linear direction in the residual stream, at least for the
extraction used. A subtler affect axis is not ruled out — but these lexical
markers would not see it, and arbitrary perturbation also fails to produce them.

Next step agreed with the user: teacher-forced matched extraction (identical
assistant text under both system prompts), then, if that still fails, a
probe-learned direction instead of a mean difference.


## Teacher-forced follow-up (same day, third experiment)

Motivation: the generation-based direction mixes "the prompt contained different
text" with style. Teacher forcing removes that confound: the **same assistant
passage** is rendered under both system prompts, one forward each, and the states
are averaged over the passage's token positions. Three passages × two systems,
layers 8–22, then the same multi-layer application (full vectors, α/√L) and the
same six held-out prompts.

This run also added the control the previous experiment was missing: shuffled and
random directions applied to the **warm** prompt, to see whether any directed
perturbation of the same magnitude cancels the persona.

| arm | result |
|---|---|
| install (plain + direction) | **0/6** ≥50 %, mean −4 % |
| cancel (warm − direction) | **4/6** ≥50 %, mean **+29 %** |
| **cancel controls (shuffled / random)** | **+25 %** (shuffled 0 %, +50 %; random +50 %, 0 %) |
| capability needles | 3/4 both |

Separations were much smaller than with generation-based extraction (L8 5.8,
L12 12.9, L16 14.4, L20 13.1, L22 14.8), as expected when the content is held
constant. The late-layer directions are coherent with each other (cosine 0.82
between L16 and L20, 0.84 between L20 and L22), so the extraction is not noise.

The decisive number is the last control row: a shuffled or random direction of the
same magnitude removes the warm register almost as well as the extracted
direction (+25 % vs +29 %, four control runs). The apparent cancellation signal is
therefore **generic perturbation**, not a property of the direction. The install
arm remains at zero and the ladder shows no dose-response.

**Corrected conclusion.** Under content-controlled extraction, the direction is a
real, layer-coherent difference, but it is not a causal persona vector: it does
not install the persona, and it cancels no better than an arbitrary direction. In
this model the warm/clinical contrast behaves like an instruction-following policy
over formatting (emoji, exclamation marks, contractions, sentence length) rather
than a linear direction in the residual stream — at least for these layers, this
extraction, and this scale.

Caveats: six held-out prompts, two control prompts per arm, a coarse lexical style
vector, one model, greedy decoding. A nonlinear encoding, a different intervention
site, or a multi-direction representation is not ruled out.

Next: a **probe-learned** direction (fit a linear classifier that predicts style
from activations and use its weights). If a probe that predicts the behaviour
still fails to cause it, the linear hypothesis is dead for this axis and the
system prompt is the practical tool.


## Individual-neuron follow-up (same day, fourth experiment)

Question: is the warm/clinical contrast concentrated in single MLP neurons, and
does pushing one of them reach the text?

Scripts: `backend/scripts/neuron_probe.py` (screen without generating) and
`backend/scripts/neuron_generation.py` (phase 3). Raw runs:
`data/pilot/neurons-*.json` and `data/pilot/neuron-gen-*.json`.

**Access.** The intermediate is the input of `feed_forward.w2` (`[B, T, 7680]`,
captured with a forward pre-hook). A neuron is steered by adding a scalar there;
LFM3 applies `post_ffn_norm` (sandwich RMSNorm) to the MLP output, so the
downstream gain of neuron *i* is `||w2.weight[:, i] * post_ffn_norm.weight||`.
The added scalar is `alpha * residual_rms / gain_i`, which targets a pre-norm
residual displacement of `alpha`. Measured attenuation (ffn_out RMS / residual
RMS): L12 0.64, L16 0.39, L20 0.23, L22 1.22.

**Screen (no generation).** Candidates: top 20 neurons per layer (L12, L16, L20,
L22) by the teacher-forced warm/clinical contrast, fixed before any effect was
measured; 8 random neurons per layer as controls. For each neuron, sign and alpha
(0.1, 0.3 of residual RMS) one forward measured KL between next-token
distributions and the change in probability mass of style tokens (emoji, "!",
"Oh", "Hey", "Sure", "Wow"). 320 screen forwards.

Only **2 of 80** candidates beat the strongest random control (|Δstyle| 0.0249):
L16 n=3177 (+0.045, KL 0.052) and L12 n=4961 (+0.040, KL 0.085).

A consistency check on three prompts showed a clean **dose-response** for L16
n=3177 (mean signed Δstyle at α = 0.1/0.3/0.5/0.7):
**+0.010 / +0.033 / +0.072 / +0.114**, and the opposite sign reduced it
(−0.008 / −0.016 / −0.019 / −0.021). L12 n=4961 showed the same pattern but
concentrated in one prompt.

**Phase 3 (generation).** 4 prompts × 48 tokens: baseline, warm target, the two
neurons across α = 0.3/0.5/0.7 (+ and −), and a random-neuron control at α = 0.7.
Style score = emoji + exclamations + contractions + first person; capability
needles at the primary setting.

| arm | style score increased | mean Δscore |
|---|---:|---:|
| primary L16 n=3177, α=0.7 (+) | **1/4** | −0.5 |
| random neuron, α=0.7 (+) | 1/4 | −0.5 |
| secondary L12 n=4961, α=0.7 (+) | 1/4 | +0.25 |
| capability needles | baseline 4/4, steered 4/4 | — |

Text stayed coherent (distinct-3 = 1.00) and competent. The per-prompt deltas
were mixed: +1 on the coworker note, −3 on the advice prompt, 0 on the other two.

**Verdict.** Individual-neuron steering does produce a real, dose-responsive
shift in the next-token distribution — the first causal local effect found in
this project — but it does **not** survive into 48-token generated text at these
alphas, and it is not distinguishable from a random neuron at the text level. The
screen signal is local to the first-token distribution (or decays after a few
tokens), which is consistent with the post-FFN normalization absorbing a rank-1
push and with the persona being an instruction-following policy rather than a
single feature.

**What remains.** (a) Steering a *group* of contrast neurons together, normalized
by the norm of the summed columns; (b) a probe-learned direction in the
intermediate space; (c) accepting the system prompt as the practical persona tool
and using neuron steering for features that are genuinely local.


## Group-of-neurons follow-up (same day, fifth experiment)

Question: if one neuron is too weak, does a **group** of the top contrast neurons
work? Script: `backend/scripts/neuron_group_probe.py`. Group = top 8 intermediate
neurons per layer by the teacher-forced contrast, signed weights normalized to
max |w| = 1, and the perturbation scaled so that the weighted sum of their w2
columns (times the post-FFN norm weight) has length `alpha * residual_rms`.

**Screen.** 4 layers × 3 alphas × 2 signs × 3 prompts, plus 3 random groups per
layer as controls (same size and weight multiset). The strongest screen was
L16 at α = 0.7 with **negative sign**: Δstyle +0.139 and KL 0.52, against a
control maximum of 0.059.

**The sign is the problem.** With positive sign the same group *reduces* style
mass (−0.020 at α = 0.7). The controls show the same asymmetry: negative
perturbations on these prompts raise style-token mass (+0.059, +0.018). The
screen metric therefore rewards the wrong direction, probably because perturbing
the model away from a confident path raises the probability of short/emphatic
tokens ("!", "Sure") rather than making the text warmer.

**Generation** at the chosen setting (L16, α = 0.7, negative sign), 4 prompts ×
48 tokens:

| arm | style score increased | mean Δ |
|---|---:|---:|
| group best (negative sign) | **0/4** | lower on 2 prompts |
| group opposite (positive sign) | 1/4 | +0.5 |
| random group, same α | 0/4 | −0.25 |
| capability needles | 4/4 both | — |

Text stayed coherent (distinct-3 = 1.00). The group's large next-token effect did
not reach the generated text, and the arm that the screen rejected was the only
one with any style movement.

**Verdict.** Grouping the top contrast neurons does not fix the problem: the
intervention produces a large but *anti-aligned* next-token shift, and the
text-level result is indistinguishable from a random group of the same size.
This also invalidates the next-token style-mass screen as a proxy for text-level
persona — it has a sign bias and is too local. The remaining principled option is
a probe-learned direction (fit a classifier that predicts style and steer its
weights); the practical one is to use the system prompt for persona.


## Format-steering follow-up (same day, sixth experiment)

Question: since persona failed, is a simpler **format** axis (bullet list vs
prose) steerable? Script: `backend/scripts/format_pilot.py`.

The contrast is applied where format exists: in the generation. Each prompt was
generated under a "three bullet points" system and a "one flowing paragraph"
system; the per-layer difference of the mean generated-token state is the
direction. It was applied at layers 8–22 (full vectors, alpha / sqrt(L)) with
structural metrics: bullet lines, headings, words, sentences, distinct-3.

Extraction worked (2–3 bullet lines vs 0), and the direction norms rose with
depth (0.22 at L8 to 1.40 at L22).

| arm | bullet lines |
|---|---|
| ladder (2 prompts), α = 0.5/1/2/3 | +1 on one prompt, 0 on the other; identical outputs across α |
| held-out (2 prompts), α = 0.5 | **0/2** increased |
| random directions, same α | 0/2 |
| capability needles | 2/2 both |

Generation stayed coherent. The only movement was a single bullet line on one
ladder prompt, and it did not generalize to held-out prompts. This is the same
pattern as the persona experiments: an instruction-bound behavior that the
model does not expose as a linear activation direction, at least with this
extraction and these layers.

**Caveat:** only two held-out prompts, so this pilot is small even by the
standards of the others; it is consistent with the previous five, not decisive
on its own.


## Lens contrast: where the first-token persona is decided

Script: `backend/scripts/lens_contrast.py` (uses the Interpret lens service).

Same user prompt under the warm and the clinical system, lens read at the **last
prompt token** (the state that predicts the first generated token). Generated
positions are not compared: the two conditions produce different continuations,
so index *k* is a different context in each run.

For "Describe a rainy morning.":

| layer | warm top-1 | p | clinical top-1 | p |
|---:|---|---:|---|---:|
| 0 | `' response'` | 1.00 | `' response'` | 1.00 |
| 6 | `' response'` | 0.03 | `' response'` | 0.02 |
| 13 | `' sh'` | 0.03 | `'ase'` | 0.01 |
| 15 | `'Absolutely'` | 0.14 | `' describe'` | 0.03 |
| 19 | `'Absolutely'` | 0.48 | `' rain'` | 0.06 |
| 22 | `'Oh'` | 0.35 | `'The'` | 0.77 |
| 24 | `'Oh'` | **0.78** | `'The'` | **0.81** |

The same pattern holds for "How do I learn to play guitar?" (final layer: `'Oh'`
0.33 vs `'To'` 0.60).

**Finding.** Early layers do not distinguish the conditions at all: both predict
the same chat-template token (`' response'`), because the residual is still
dominated by the prompt template. Layers 7–14 are uncommitted and low-confidence
(≤0.03, unstable across prompts). The condition-specific preference consolidates
in the second half of the network: the stable split starts at layer 13 (rainy)
and 9 (guitar), and the mean minimum probability of the split prediction from
layer 15 is **0.219**. The final layer's top-1 is exactly the token both
continuations start with ("Oh hey!" vs "The sky is gray…").

**Why this matters for steering.** The persona decision for the first token is
late-forming: interventions in layers 8–14 act on an uncommitted representation
that later layers can overwrite, which is consistent with every null we found.
It also points at the experiment we have not run: extract the direction from the
**last prompt token at layers 19–22**, where the conditions already differ with
high confidence, and steer only those layers. That is a different object from the
teacher-forced direction (assistant-text positions) and from the earlier
multi-layer sum that diluted late layers by 1/sqrt(L).


## Late-layer follow-up (same day, seventh experiment)

Motivation: the lens showed the condition-specific preference consolidating from
layer ~15, so the direction should be extracted at the **last prompt token of
layers 19–22** and applied only there.

Script: `backend/scripts/multilayer_probe.py --extraction generation
--layers 19,20,21,22 --alphas 1,2,4,8`. Full vectors per layer, alpha_l =
alpha_total / 2 (so 0.5–4 rms per layer), same six held-out prompts, shuffled
and random controls, capability needles.

| arm | result |
|---|---|
| install (plain + direction) | **0/6** ≥50 %, mean −21 % |
| cancel (warm − direction) | 2/6, mean +8 % |
| **cancel controls (shuffled/random)** | **+31 %** (better than the real direction) |
| first-token flips vs baseline | 1/6 ("there" → "working"), and not toward the warm target |
| first tokens (baseline → warm target) | learning→oh, the→oh, hey→hey, there→oh, take→hey, a→sure |
| capability | 3/4 both |

The late-layer direction did not install the persona, did not flip the first
token toward the warm one, and its cancellation was *worse* than random
direction controls.

**Interpretation.** The lens shows *when* the condition-specific preference
becomes readable, not *which direction causes it*. The mean difference at the
last prompt token still mostly encodes "a different system prompt was present",
and that is not the knob that switches the behavior — even at the layers where
the preference is already visible. This closes the last mean-difference variant
we had: single layer, multi-layer spread, multi-layer late-only, teacher-forced,
prompt states, generated states, full vectors, and top-k.

What is left with real support is a **probe-learned** direction (fit a classifier
that predicts the style and use its weights) or head-level interventions; neither
is a small change, and both would test a different hypothesis (learned readout
vs. mean difference).

## What this does not prove

- Small sample: 4 extraction prompts, 2 ladder prompts, 6 held-out prompts, one
  seed, greedy decoding, one model. These are effect sizes, not verdicts.
- The metrics are lexical: they count emoji, punctuation and contractions. The
  model's *prose* was already lyrical in all conditions, so a subtler warmth axis
  would need a better measure (human ratings or a learned probe).
- Only feed-forward outputs and residuals of one layer at a time were tested; the
  web of layers was not.

## Next experiments (in order)

1. **Matched completions.** Force the same content in both poles (e.g. finish a
   fixed prefix with "warm" vs "clinical" continuations), then average states over
   generated tokens. The current pairs differ in prompt text, so the direction is
   dominated by that difference.
2. **Multi-layer application.** Extract one direction per layer from the same
   pairs and apply them together with a per-layer scale, instead of a single
   layer.
3. **Learn the direction, do not average it.** Fit a linear probe that predicts
   style from activations; use the probe weights as the direction. This selects
   the axis that actually predicts the behaviour.
4. **Keep the system prompt for persona** and use activation steering for axes
   the model represents locally (verbosity, list format, refusal), where the
   effect is easier to reproduce.

Re-run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m backend.scripts.personality_pilot \
  --device cuda --tokens 40 --alphas="-120,-80,-40,0,40,80,120" \
  --modules model.layers.4.feed_forward,model.layers.8.feed_forward,model.layers.12.feed_forward,model.layers.16.feed_forward,model.layers.20.feed_forward,model.layers.22.feed_forward
```
