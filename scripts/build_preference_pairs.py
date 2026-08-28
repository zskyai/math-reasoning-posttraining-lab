#!/usr/bin/env python
"""Build verifier-checked DPO pairs from the local math fixture."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verifier import normalize_answer, verify_response  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            for field in ("id", "question", "answer"):
                if field not in row:
                    raise ValueError(f"row {line_number} is missing {field!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"fixture is empty: {path}")
    return rows


def _wrong_answer(gold: str) -> str:
    """Create a deterministic numeric hard negative one unit away from gold."""

    normalized = normalize_answer(gold)
    if normalized is not None:
        try:
            value = Fraction(normalized) + 1
            return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        except (ValueError, ZeroDivisionError):
            pass
    return "0" if normalized != "0" else "1"


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        gold = str(row["answer"])
        wrong = _wrong_answer(gold)
        solution = str(row.get("solution", ""))
        chosen = f"{solution} Final answer: \\boxed{{{gold}}}."
        rejected = f"{solution} Final answer: \\boxed{{{wrong}}}."
        chosen_check = verify_response(chosen, gold)
        rejected_check = verify_response(rejected, gold)
        if not chosen_check.is_correct or rejected_check.is_correct:
            raise AssertionError(f"failed to construct a valid pair for {row['id']}")
        pairs.append(
            {
                "id": str(row["id"]),
                "prompt": str(row["question"]),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_answer": chosen_check.normalized_answer,
                "rejected_answer": rejected_check.normalized_answer,
                "chosen_is_correct": True,
                "rejected_is_correct": False,
                "source": "fixture_math",
            }
        )
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "fixture_math.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "preference_pairs.jsonl",
    )
    args = parser.parse_args(argv)
    pairs = build_pairs(_load_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "pairs": len(pairs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
