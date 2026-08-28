"""Answer extraction and deterministic rewards for short math responses.

The verifier is deliberately conservative.  It accepts common GSM-style
``####`` and LaTeX ``\\boxed{...}`` answers, normalises exact numeric forms,
and never asks an untrusted language model to judge correctness.  It is a
small reference implementation for the smoke experiment; production RLVR
systems should add a sandboxed symbolic checker and a stricter schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Optional


_MARKER_RE = re.compile(
    r"(?:####|final\s+answer|answer|答案)\s*(?:is|为|是|[:：=])?\s*(?P<tail>[^\n]+)",
    flags=re.IGNORECASE,
)
_NUMBER_ATOM = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    rf"[+-]?{_NUMBER_ATOM}(?:[eE][+-]?\d+)?"
    rf"(?:\s*/\s*[+-]?{_NUMBER_ATOM}(?:[eE][+-]?\d+)?)?"
    r"(?![A-Za-z0-9_])"
)
_BOXED_RE = re.compile(r"\\boxed\s*\{")


@dataclass(frozen=True)
class VerificationResult:
    """Auditable result for one generated response."""

    extracted_answer: Optional[str]
    normalized_answer: Optional[str]
    normalized_gold: str
    is_correct: bool
    format_valid: bool
    reason: str


def _replace_frac_commands(value: str) -> str:
    """Convert simple ``\\frac{a}{b}`` commands to ``a/b``."""

    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = pattern.sub(r"\1/\2", value)
    return value


def _extract_boxed(value: str) -> Optional[str]:
    """Return the first balanced LaTeX boxed payload, if present."""

    match = _BOXED_RE.search(value)
    if match is None:
        return None
    depth = 1
    start = match.end()
    for index in range(start, len(value)):
        character = value[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                payload = value[start:index].strip()
                return payload or None
    return None


def _last_numeric(value: str) -> Optional[str]:
    matches = list(_NUMBER_RE.finditer(value))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def extract_answer(response: str) -> Optional[str]:
    """Extract a likely final answer without evaluating arbitrary code.

    Explicit markers take precedence.  If no marker is present, the final
    numeric token is returned so that format compliance can be measured
    separately from mathematical correctness.
    """

    if not isinstance(response, str) or not response.strip():
        return None

    boxed = _extract_boxed(response)
    if boxed is not None:
        return boxed

    marker_matches = list(_MARKER_RE.finditer(response))
    if marker_matches:
        tail = marker_matches[-1].group("tail")
        boxed_tail = _extract_boxed(tail)
        if boxed_tail is not None:
            return boxed_tail
        numeric = _last_numeric(tail)
        if numeric is not None:
            return numeric
        return tail.strip().rstrip("。.!?;；") or None

    return _last_numeric(response)


def _as_fraction(value: str) -> Optional[Fraction]:
    value = value.strip().replace(" ", "")
    if not value:
        return None
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            result = Fraction(numerator) / Fraction(denominator)
        else:
            result = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent else result


def normalize_answer(answer: str) -> Optional[str]:
    """Canonicalise numeric answers (including equivalent fractions).

    Non-numeric answers are lower-cased and whitespace-normalised.  Returning
    a canonical string keeps JSON logs easy to inspect while exact numeric
    comparison uses :class:`fractions.Fraction` internally.
    """

    if not isinstance(answer, str):
        return None
    value = _replace_frac_commands(answer)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("−", "-").replace("–", "-").replace(",", "")
    value = value.strip().strip("`$ \\t\\r\\n")
    value = value.rstrip("。.!?;；,，")
    if "=" in value:
        value = value.rsplit("=", 1)[1].strip()
    value = re.sub(r"^答案\s*[:：]?\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip("$ ")

    # Handle a simple mixed number such as ``1 1/2``.
    mixed = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", value)
    if mixed:
        sign = -1 if mixed.group(1).startswith("-") else 1
        whole = abs(int(mixed.group(1)))
        value = str(sign * (whole * int(mixed.group(3)) + int(mixed.group(2)))) + "/" + mixed.group(3)

    fraction = _as_fraction(value)
    if fraction is not None:
        return str(fraction.numerator) if fraction.denominator == 1 else f"{fraction.numerator}/{fraction.denominator}"

    # Keep a useful fallback for categorical answers such as ``yes``/``no``.
    words = re.sub(r"\s+", " ", value.lower()).strip()
    words = words.strip(".。!！?？")
    return words or None


def verify_response(response: str, gold_answer: str) -> VerificationResult:
    """Verify one response against a gold answer."""

    extracted = extract_answer(response)
    normalized = normalize_answer(extracted) if extracted is not None else None
    normalized_gold = normalize_answer(str(gold_answer))
    format_valid = bool(extracted) and (
        _extract_boxed(response) is not None or bool(_MARKER_RE.search(response))
    )
    is_correct = normalized is not None and normalized == normalized_gold
    if extracted is None:
        reason = "no_parseable_answer"
    elif is_correct and format_valid:
        reason = "correct_and_well_formatted"
    elif is_correct:
        reason = "correct_but_missing_explicit_marker"
    elif format_valid:
        reason = "wrong_answer"
    else:
        reason = "wrong_or_unformatted_answer"
    return VerificationResult(
        extracted_answer=extracted,
        normalized_answer=normalized,
        normalized_gold=normalized_gold or "",
        is_correct=is_correct,
        format_valid=format_valid,
        reason=reason,
    )


def reward_response(
    response: str,
    gold_answer: str,
    *,
    correctness_reward: float = 1.0,
    format_bonus: float = 0.1,
    invalid_penalty: float = 0.1,
) -> float:
    """Return a transparent scalar reward for verifier-based RL smoke tests.

    Correctness is primary.  A marker bonus rewards machine-readable output,
    while an unparsable response receives a small penalty.  The components are
    intentionally exposed so reward-hacking ablations can change them.
    """

    result = verify_response(response, gold_answer)
    reward = correctness_reward if result.is_correct else 0.0
    if result.format_valid:
        reward += format_bonus
    if result.extracted_answer is None:
        reward -= invalid_penalty
    return float(reward)


__all__ = [
    "VerificationResult",
    "extract_answer",
    "normalize_answer",
    "verify_response",
    "reward_response",
]
