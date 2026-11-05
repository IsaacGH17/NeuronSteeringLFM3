import { useEffect, useRef } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
}

const TOPICS = [
  ["help-loop", "The experiment loop"],
  ["help-activation", "Activations"],
  ["help-dimension", "Hidden dimensions"],
  ["help-steering", "Steering α"],
  ["help-modes", "Run modes"],
  ["help-controls", "Controls and sweeps"],
  ["help-repro", "Reproducibility"],
  ["help-hooks", "Hooks and safety"],
  ["help-reading", "Reading results"],
  ["help-flow", "Live flow"],
  ["help-data", "Capture and storage"],
  ["help-limits", "Limitations"],
] as const;

export function HelpPanel({ open, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();

    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="help-backdrop" onClick={onClose}>
      <aside
        className="help-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="help-header">
          <div>
            <p className="help-kicker">Field notes</p>
            <h2 id="help-title">Help &amp; key concepts</h2>
          </div>
          <button ref={closeRef} className="secondary" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="help-body">
          <p className="help-lede">
            This is a bench for looking at the model's insides, not a chatbot. Every module, tensor, index,
            and output you see comes from the loaded model itself. Steering only shifts activations while the
            model runs; the weights never change.
          </p>

          <nav className="help-toc" aria-label="Help topics">
            {TOPICS.map(([id, label]) => (
              <a key={id} href={`#${id}`}>
                {label}
              </a>
            ))}
          </nav>

          <section id="help-loop" className="help-section">
            <h3>The experiment loop</h3>
            <ol>
              <li>Load the model and walk through its real architecture.</li>
              <li>Select a steerable module and press <strong>Set as steering target</strong>.</li>
              <li>Run a baseline to capture the reference output and optional activations.</li>
              <li>Switch to <strong>Steered</strong>, choose a hidden dimension and α, then compare.</li>
              <li>Use <strong>Sweep</strong> to measure several strengths and a random control.</li>
              <li>Inspect the history, reopen captures, and export the metadata.</li>
            </ol>
          </section>

          <section id="help-activation" className="help-section">
            <h3>Activations</h3>
            <p>
              An activation is the hidden-state tensor a module produces for every token position. Captures
              come shaped <code>[tokens, hidden]</code> — <code>[17, 2048]</code>, say. The heatmap samples
              positions against dimensions; the distribution and top dimensions summarize the same measured
              tensor.
            </p>
            <p>
              The final generated token has no activation because it is never forwarded through the model.
              That is why a capture may cover fewer positions than the visible token strip.
            </p>
          </section>

          <section id="help-dimension" className="help-section">
            <h3>Hidden dimensions</h3>
            <p>
              A hidden dimension (often called a “neuron”) is one column of the activation tensor. Index{" "}
              <code>i</code> is a slot in the model’s learned representation, not a guaranteed concept. A
              dimension can correlate with a behavior, but that requires repeated experiments, multiple
              prompts, and controls before you claim it.
            </p>
          </section>

          <section id="help-steering" className="help-section">
            <h3>Steering α</h3>
            <p>
              Steering adds α to one hidden dimension at every token produced by the target module, during
              inference only. Positive and negative α push in opposite directions. <strong>α = 0 must
              reproduce the baseline exactly</strong>; use it as your correctness check.
            </p>
            <p>
              Small values usually change style or wording. Large values can leak reasoning tokens or
              collapse into repetition. Find the useful range per module and dimension instead of assuming
              a fixed value.
            </p>
          </section>

          <section id="help-modes" className="help-section">
            <h3>Run modes</h3>
            <dl className="help-list">
              <div>
                <dt>Baseline</dt>
                <dd>One unmodified generation. It is the reference for every comparison.</dd>
              </div>
              <div>
                <dt>Steered</dt>
                <dd>
                  A baseline and a steered copy with the same prompt, settings, and seed. The workspace shows
                  the two outputs and the labeled <code>steered − baseline</code> activation difference.
                </dd>
              </div>
              <div>
                <dt>Sweep</dt>
                <dd>
                  One shared baseline plus a run for each strength and an optional random-feature control.
                  Results are tabular and stored individually.
                </dd>
              </div>
            </dl>
          </section>

          <section id="help-controls" className="help-section">
            <h3>Controls and sweeps</h3>
            <p>
              A control steers a seeded random dimension at the same magnitude and sign as the last real
              strength, excluding the target dimension. It shows how much output change comes from an
              arbitrary perturbation rather than the dimension you chose. If the control changes just as
              much, you have not isolated a meaningful direction.
            </p>
            <p>
              Sweeps accept 1–9 strengths and up to 3 repeats, with at most 31 runs per sweep. They execute
              sequentially.
            </p>
          </section>

          <section id="help-repro" className="help-section">
            <h3>Reproducibility</h3>
            <p>
              Greedy decoding (<code>temperature = 0</code>) is deterministic and best for comparisons. With
              sampling, set a seed and keep the same seed across baseline and steered runs. Unseeded sampling
              cannot be reproduced exactly. Repeats of a greedy run produce identical outputs and the UI warns
              you about that.
            </p>
          </section>

          <section id="help-hooks" className="help-section">
            <h3>Hooks and safety</h3>
            <p>
              Steering and capture use forward hooks installed only for the duration of one experiment. They
              are removed in a <code>finally</code> block even if generation fails, and steered tensors are
              copied before modification. The original weights are never changed.
            </p>
          </section>

          <section id="help-reading" className="help-section">
            <h3>Reading results</h3>
            <p>
              A single changed response is not proof that a dimension controls a behavior. Look for a
              consistent direction across several prompts, a clean comparison against controls, and a useful
              range of strengths. Separate style effects (length, formatting, tone) from semantic claims, and
              report negative results too.
            </p>
          </section>

          <section id="help-flow" className="help-section">
            <h3>Live flow</h3>
            <p>
              <strong>Flow</strong> traces one real generation across every decoder layer. Ribbon width is the
              measured residual write <code>mean ‖h_l − h_(l−1)‖ / ‖h_(l−1)‖</code>; opacity is the mean cosine
              between consecutive layer states. The glow and the moving dashes replay one token at a time; the
              timing is illustrative, the values are measured.
            </p>
            <p>
              Colored strands group hidden dimensions whose z-scored profiles move together on that trace. Click one to
              inspect its top dimensions and steer the group in σ units. Clusters are descriptive, strand colors
              are per-layer ranks (not matched across layers), and an unchanged output after steering is a real
              result, not a failure of the tool.
            </p>
          </section>

          <section id="help-data" className="help-section">
            <h3>Capture and storage</h3>
            <p>
              Captured tensors are ephemeral by default: they live in a bounded in-memory store and disappear
              on eviction or restart. <strong>Persist activation snapshot</strong> writes an <code>.npz</code>{" "}
              file so a run can be reopened later. Experiment metadata is always saved as append-only JSONL
              under <code>data/</code>. The first captured module is shown in the center panel.
            </p>
          </section>

          <section id="help-limits" className="help-section">
            <h3>Limitations</h3>
            <ul>
              <li>Single-dimension steering is a crude control; real behaviors are distributed across layers.</li>
              <li>Large α values can break coherence, so usefulness must be checked per setting.</li>
              <li>Sweeps are sequential and cannot be cancelled from the current API.</li>
              <li>The reference 8 GB GPU requires bounded prompts and generation lengths.</li>
              <li>Canvas heatmaps are visual summaries, not the full tensor.</li>
            </ul>
          </section>
        </div>
      </aside>
    </div>
  );
}
