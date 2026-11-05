"""Deterministic text metrics and the prompt suite catalogue."""

from __future__ import annotations

import pytest

from backend.app.schemas.api import SuiteCheck, TextMetrics
from backend.app.services.metrics import aggregate_metrics, compute_metrics
from backend.app.services.suites import get_suite, list_suites


def test_metrics_measure_structure_and_tone():
    text = "Sure! I would love to help.\n\n- The system uses a vector.\n- It might work, and it might not."
    metrics = compute_metrics(text)
    assert metrics.words > 10
    assert metrics.sentences >= 2
    assert 0.0 < metrics.distinct_3 <= 1.0
    assert metrics.refusal is False
    assert metrics.lexicon["warmth"] > 0
    assert metrics.lexicon["technical"] > 0
    assert metrics.lexicon["hedging"] > 0
    assert metrics.checks_passed is None


def test_metrics_detect_refusal_and_empty_output():
    assert compute_metrics("I cannot help with that request.").refusal is True
    empty = compute_metrics("")
    assert empty.words == 0 and empty.sentences == 0
    assert empty.distinct_3 == 0.0


def test_repetition_ratio_grows_with_loops():
    once = compute_metrics("the cat sat on the mat and looked around")
    loop = compute_metrics("the cat sat on " * 6)
    assert loop.repeated_4gram_ratio > once.repeated_4gram_ratio


@pytest.mark.parametrize(
    ("check", "text", "expected"),
    [
        (SuiteCheck(kind="max_words", value=3), "one two three", True),
        (SuiteCheck(kind="max_words", value=3), "one two three four", False),
        (SuiteCheck(kind="min_words", value=2), "one", False),
        (SuiteCheck(kind="contains", value="cold"), "Cold.", True),
        (SuiteCheck(kind="not_contains", value="cold"), "Warm.", True),
        (SuiteCheck(kind="one_word"), "Cold.", True),
        (SuiteCheck(kind="one_word"), "Very cold.", False),
        (SuiteCheck(kind="is_number"), "43.", True),
        (SuiteCheck(kind="is_number"), "forty-three", False),
        (SuiteCheck(kind="list_items", value=2), "- red\n- blue", True),
        (SuiteCheck(kind="list_items", value=3), "- red\n- blue", False),
        (SuiteCheck(kind="regex", value=r"par[ií]s"), "París", True),
        (SuiteCheck(kind="regex", value=r"{"), "not json", False),
    ],
)
def test_check_kinds(check: SuiteCheck, text: str, expected: bool):
    metrics = compute_metrics(text, [check])
    assert metrics.checks[0].passed is expected
    assert metrics.checks_total == 1
    assert metrics.checks_passed == (1 if expected else 0)


def test_unknown_regex_is_reported_as_failed():
    metrics = compute_metrics("text", [SuiteCheck(kind="regex", value="(")])
    assert metrics.checks[0].passed is False
    assert metrics.checks[0].detail == "invalid regex"


def test_aggregate_metrics_means_and_pass_rate():
    first = TextMetrics(
        words=10,
        characters=50,
        sentences=2,
        avg_words_per_sentence=5.0,
        distinct_3=1.0,
        repeated_4gram_ratio=0.0,
        refusal=False,
        lexicon={"warmth": 2.0},
        checks_passed=1,
        checks_total=2,
    )
    second = TextMetrics(
        words=30,
        characters=150,
        sentences=3,
        avg_words_per_sentence=10.0,
        distinct_3=0.5,
        repeated_4gram_ratio=0.2,
        refusal=True,
        lexicon={"warmth": 4.0},
        checks_passed=2,
        checks_total=2,
    )
    aggregated = aggregate_metrics([first, second])
    assert aggregated["words_mean"] == 20.0
    assert aggregated["distinct_3_mean"] == 0.75
    assert aggregated["refusal_rate"] == 0.5
    assert aggregated["checks_passed_rate"] == 0.75
    assert aggregated["lexicon_mean"]["warmth"] == 3.0

    empty = aggregate_metrics([])
    assert empty["words_mean"] == 0.0
    assert empty["checks_passed_rate"] is None


def test_suite_catalogue_is_consistent():
    suites = list_suites()
    assert {suite.id for suite in suites} == {"trait-neutral", "capability-core"}
    trait = get_suite("trait-neutral")
    capability = get_suite("capability-core")
    assert trait.kind == "trait" and len(trait.prompts) >= 12
    assert all(not prompt.checks for prompt in trait.prompts)
    assert capability.kind == "capability" and len(capability.prompts) == 10
    assert all(prompt.checks for prompt in capability.prompts)
    ids = [prompt.id for prompt in capability.prompts]
    assert len(ids) == len(set(ids))
    with pytest.raises(ValueError, match="Unknown prompt suite"):
        get_suite("nope")
