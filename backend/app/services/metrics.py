"""Deterministic text measurements for every run.

These are lexical and structural heuristics, not semantic judgments. They exist
so a style axis can be discussed as a number ("warmth markers per 100 words")
and so capability stays visible next to it (prompt-suite checks). Nothing here
calls another model or invents a verdict about quality.
"""

from __future__ import annotations

import math
import re

from backend.app.schemas.api import MetricCheck, SuiteCheck, TextMetrics

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’\-]*")
_SENTENCE_RE = re.compile(r"[.!?]+")
_LIST_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

# Marker lists are deliberately short and transparent: they are a lens, not a
# judge. Extend them per experiment rather than treating them as ground truth.
LEXICONS: dict[str, tuple[str, ...]] = {
    "warmth": (
        "glad",
        "happy",
        "love",
        "lovely",
        "wonderful",
        "appreciate",
        "kind",
        "warm",
        "care",
        "hope",
        "friend",
        "thanks",
        "thank",
        "great",
        "absolutely",
        "delightful",
        "cozy",
    ),
    "technical": (
        "process",
        "system",
        "mechanism",
        "function",
        "parameter",
        "vector",
        "data",
        "input",
        "output",
        "therefore",
        "thus",
        "component",
        "architecture",
        "implementation",
    ),
    "hedging": (
        "might",
        "may",
        "perhaps",
        "possibly",
        "likely",
        "seems",
        "appears",
        "somewhat",
        "generally",
        "often",
        "could",
        "roughly",
        "probably",
    ),
    "assertive": (
        "must",
        "always",
        "never",
        "definitely",
        "clearly",
        "certainly",
        "obviously",
        "undoubtedly",
        "will",
        "is",
    ),
}

REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "as an ai",
    "i'm sorry, but",
    "i am sorry, but",
)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _lower_words(text: str) -> list[str]:
    return [word.lower() for word in words(text)]


def _marker_count(text: str, lower_words: list[str], marker: str) -> int:
    if " " in marker:
        return text.lower().count(marker)
    return sum(1 for word in lower_words if word == marker)


def _lexicon_rates(text: str, lower_words: list[str]) -> dict[str, float]:
    total = len(lower_words)
    if total == 0:
        return {name: 0.0 for name in LEXICONS}
    return {
        name: round(100.0 * sum(_marker_count(text, lower_words, marker) for marker in markers) / total, 3)
        for name, markers in LEXICONS.items()
    }


def _ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    if len(tokens) < size:
        return []
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len([part for part in _SENTENCE_RE.split(stripped) if part.strip()]))


def _evaluate(check: SuiteCheck, text: str, lower_words: list[str]) -> MetricCheck:
    kind = check.kind
    expected = check.value
    stripped = text.strip()
    passed = False
    detail: str | None = None

    if kind in {"max_words", "min_words"}:
        limit = int(expected) if expected is not None else 0
        count = len(lower_words)
        passed = count <= limit if kind == "max_words" else count >= limit
        detail = f"{count} words"
    elif kind == "contains":
        needle = str(expected or "").lower()
        passed = bool(needle) and needle in text.lower()
        detail = f"looked for {needle!r}"
    elif kind == "not_contains":
        needle = str(expected or "").lower()
        passed = bool(needle) and needle not in text.lower()
        detail = f"avoided {needle!r}"
    elif kind == "one_word":
        passed = len(lower_words) == 1
        detail = f"{len(lower_words)} words"
    elif kind == "is_number":
        candidate = stripped.strip(" .,;:!?()[]")
        passed = bool(_NUMBER_RE.match(candidate))
        detail = candidate[:40] if candidate else "empty"
    elif kind == "list_items":
        minimum = int(expected) if expected is not None else 1
        count = len(_LIST_ITEM_RE.findall(text))
        passed = count >= minimum
        detail = f"{count} list items"
    elif kind == "regex":
        pattern = str(expected or "")
        try:
            passed = bool(re.search(pattern, text, flags=re.IGNORECASE))
        except re.error:
            passed = False
            detail = "invalid regex"
        detail = detail or f"pattern {pattern!r}"
    else:  # pragma: no cover - guarded by the Literal type
        detail = f"unknown check {kind!r}"

    return MetricCheck(
        name=f"{kind}:{expected}" if expected is not None else kind,
        kind=kind,
        expected=expected,
        passed=passed,
        detail=detail,
    )


def compute_metrics(text: str, checks: list[SuiteCheck] | None = None) -> TextMetrics:
    """Measure one generated text; compute suite checks when provided."""
    lower_words = _lower_words(text)
    total_words = len(lower_words)
    sentences = _sentence_count(text)

    trigrams = _ngrams(lower_words, 3)
    distinct_3 = round(len(set(trigrams)) / len(trigrams), 4) if trigrams else 0.0
    fourgrams = _ngrams(lower_words, 4)
    repeated_4gram_ratio = round(1.0 - (len(set(fourgrams)) / len(fourgrams)), 4) if fourgrams else 0.0
    lowered = text.lower()
    evaluated = [_evaluate(check, text, lower_words) for check in (checks or [])]
    checks_passed = sum(1 for check in evaluated if check.passed) if evaluated else None

    return TextMetrics(
        words=total_words,
        characters=len(text),
        sentences=sentences,
        avg_words_per_sentence=(round(total_words / sentences, 2) if sentences else float(total_words)),
        distinct_3=distinct_3,
        repeated_4gram_ratio=repeated_4gram_ratio,
        refusal=any(marker in lowered for marker in REFUSAL_MARKERS),
        lexicon=_lexicon_rates(text, lower_words),
        checks=evaluated,
        checks_passed=checks_passed,
        checks_total=len(evaluated) if evaluated else None,
    )


def aggregate_metrics(metrics_list: list[TextMetrics]) -> dict[str, float | None]:
    """Mean of the numeric fields across runs that share one sweep cell."""
    usable = [metrics for metrics in metrics_list if metrics is not None]
    if not usable:
        return {
            "words_mean": 0.0,
            "distinct_3_mean": 0.0,
            "refusal_rate": 0.0,
            "checks_passed_rate": None,
            "lexicon_mean": {},
        }
    count = len(usable)
    checks = [m.checks_passed / m.checks_total for m in usable if m.checks_total]
    lexicon_names = {name for metrics in usable for name in metrics.lexicon}
    return {
        "words_mean": round(sum(m.words for m in usable) / count, 2),
        "distinct_3_mean": round(sum(m.distinct_3 for m in usable) / count, 4),
        "refusal_rate": round(sum(1.0 for m in usable if m.refusal) / count, 4),
        "checks_passed_rate": round(sum(checks) / len(checks), 4) if checks else None,
        "lexicon_mean": {
            name: round(
                sum(m.lexicon.get(name, 0.0) for m in usable) / count,
                3,
            )
            for name in sorted(lexicon_names)
        },
    }


def safe_float(value: float) -> float:
    """Guard against non-finite values in payloads."""
    return value if math.isfinite(value) else 0.0
