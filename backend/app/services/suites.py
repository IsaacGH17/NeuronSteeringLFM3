"""Built-in prompt suites for batch experiments.

A *trait* suite is a set of neutral requests: the content stays constant while
the steering direction changes, so the only thing that should move is tone and
style. A *capability* suite adds verifiable checks per prompt, so a steering
setting can be judged on whether the model still does the job.

Both suites are data, not code paths: batches can also send their own prompts,
in which case no checks are attached.
"""

from __future__ import annotations

from backend.app.schemas.api import PromptSuite, SuiteCheck, SuitePrompt

TRAIT_PROMPTS: tuple[str, ...] = (
    "Explain what a transformer activation is in two sentences.",
    "What should I cook tonight? Give me one idea.",
    "Describe a rainy morning.",
    "How do I learn to play guitar?",
    "Summarize the plot of a heist film in three sentences.",
    "Write a short note to a coworker about a delayed report.",
    "What are the tradeoffs of working from home?",
    "Explain why the sky is blue.",
    "Give me a two-sentence review of a coffee shop.",
    "A friend asks for advice about a difficult decision. Reply briefly.",
    "Describe your ideal weekend in three sentences.",
    'Reply to this message: "I finally finished my thesis."',
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
        "Answer in Spanish: ¿cuál es la capital de Francia?",
        (SuiteCheck(kind="regex", value=r"par[ií]s"),),
    ),
    (
        "Repeat exactly this token and nothing else: alpha-beta-gamma",
        (SuiteCheck(kind="contains", value="alpha-beta-gamma"),),
    ),
    (
        "Write a single sentence about rain that contains the word 'cloud'.",
        (SuiteCheck(kind="contains", value="cloud"), SuiteCheck(kind="max_words", value=40)),
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
        "Count the letters in 'strawberry'. Reply with only the number.",
        (SuiteCheck(kind="is_number"), SuiteCheck(kind="contains", value="10")),
    ),
    (
        "Translate to English: 'la casa es grande'.",
        (SuiteCheck(kind="regex", value=r"house"), SuiteCheck(kind="regex", value=r"big|large")),
    ),
)

TRAIT_SUITE = PromptSuite(
    id="trait-neutral",
    name="Neutral requests (tone probe)",
    description=(
        "Twelve ordinary requests with no style instruction. Keep the prompt fixed and vary "
        "the steering direction: warmth/technical/hedging markers show whether tone moved, "
        "while length and repetition show whether the text degraded."
    ),
    kind="trait",
    prompts=[
        SuitePrompt(id=f"t{index + 1}", text=text, tags=["trait"]) for index, text in enumerate(TRAIT_PROMPTS)
    ],
)

CAPABILITY_SUITE = PromptSuite(
    id="capability-core",
    name="Capability checks (10 tasks)",
    description=(
        "Ten tasks with deterministic checks: format, arithmetic, translation, JSON, and "
        "instruction length. Use it to see what a steering setting costs in usefulness."
    ),
    kind="capability",
    prompts=[
        SuitePrompt(
            id=f"c{index + 1}",
            text=text,
            tags=["capability"],
            checks=list(checks),
        )
        for index, (text, checks) in enumerate(CAPABILITY_PROMPTS)
    ],
)

SUITES: dict[str, PromptSuite] = {suite.id: suite for suite in (TRAIT_SUITE, CAPABILITY_SUITE)}


def list_suites() -> list[PromptSuite]:
    return list(SUITES.values())


def get_suite(suite_id: str) -> PromptSuite:
    try:
        return SUITES[suite_id]
    except KeyError as exc:
        known = ", ".join(sorted(SUITES))
        raise ValueError(f"Unknown prompt suite {suite_id!r}; known suites: {known}.") from exc
