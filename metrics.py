"""Evaluation helpers for verifier-based reasoning experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from verifier import reward_response, verify_response


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def _responses(row: Mapping[str, Any]) -> list[str]:
    values = row.get("responses")
    if values is None:
        value = row.get("response", "")
        return [str(value)]
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def evaluate_generations(
    examples: Iterable[Mapping[str, Any]],
    *,
    k: int | None = None,
    format_bonus: float = 0.1,
    invalid_penalty: float = 0.1,
) -> dict[str, float | int]:
    """Compute pass@k, format rate, reward and response-length metrics.

    The function intentionally uses the deterministic verifier for every
    response.  ``pass@k`` here is the empirical "any of the first k" rate,
    which is appropriate for this fixed-candidate smoke and is labelled as
    such in the README (it is not an unbiased stochastic pass@k estimator).
    """

    rows = list(examples)
    if not rows:
        raise ValueError("at least one example is required")
    all_checks: list[list] = []
    all_texts: list[list[str]] = []
    lengths: list[float] = []
    for row in rows:
        gold = str(row.get("gold_answer", row.get("answer", "")))
        texts = _responses(row)
        if not texts:
            raise ValueError("each example must provide at least one response")
        checks = [verify_response(text, gold) for text in texts]
        all_checks.append(checks)
        all_texts.append(texts)
        lengths.extend(float(len(text.split())) for text in texts)

    max_available = max(len(checks) for checks in all_checks)
    requested_k = max_available if k is None else int(k)
    if requested_k <= 0:
        raise ValueError("k must be positive")
    effective_k = min(requested_k, max_available)

    top1_correct = [checks[0].is_correct for checks in all_checks]
    top1_format = [checks[0].format_valid for checks in all_checks]
    top1_rewards = [
        reward_response(
            texts[0],
            str(row.get("gold_answer", row.get("answer", ""))),
            format_bonus=format_bonus,
            invalid_penalty=invalid_penalty,
        )
        for row, texts in zip(rows, all_texts)
    ]
    pass_k = [any(check.is_correct for check in checks[:effective_k]) for checks in all_checks]
    format_k = [any(check.format_valid for check in checks[:effective_k]) for checks in all_checks]

    count = len(rows)
    return {
        "examples": count,
        "k": effective_k,
        "pass@1": sum(top1_correct) / count,
        "pass@k": sum(pass_k) / count,
        "format_rate@1": sum(top1_format) / count,
        "format_rate@k": sum(format_k) / count,
        "mean_reward@1": sum(top1_rewards) / count,
        # The smoke has no model tokenizer; these are whitespace-delimited
        # word counts, deliberately named as such rather than token counts.
        "mean_response_words": sum(lengths) / len(lengths),
        "p95_response_words": _p95(lengths),
        "correct_examples@1": sum(top1_correct),
        "correct_examples@k": sum(pass_k),
    }


def compare_policies(results: Mapping[str, Mapping[str, float | int]]) -> dict[str, dict[str, float | int]]:
    """Return a JSON-friendly copy of a named policy metric table."""

    return {
        str(name): {str(metric): value for metric, value in metrics.items()}
        for name, metrics in results.items()
    }


__all__ = ["evaluate_generations", "compare_policies"]
