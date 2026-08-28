#!/usr/bin/env python
"""Run a deterministic CPU smoke for Base -> SFT -> DPO -> GRPO.

The policy is a categorical surrogate over four fixed candidate completions;
the verifier, preference objective and group-relative objective are real, but
there is no claim that this surrogate is a language-model benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses import categorical_logps, dpo_loss, grpo_loss, sft_loss  # noqa: E402
from metrics import evaluate_generations  # noqa: E402
from verifier import reward_response  # noqa: E402


class CategoricalPolicy(nn.Module):
    """One independent categorical policy per fixture prompt."""

    def __init__(self, initial_logits: torch.Tensor) -> None:
        super().__init__()
        if initial_logits.ndim != 2:
            raise ValueError("initial_logits must have shape [examples, actions]")
        self.logits = nn.Parameter(initial_logits.clone().detach())

    def log_probs(self) -> torch.Tensor:
        return F.log_softmax(self.logits, dim=-1)

    def entropy(self) -> torch.Tensor:
        log_probs = self.log_probs()
        return -(log_probs.exp() * log_probs).sum(dim=-1).mean()


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("id", "question", "answer", "solution"):
                if field not in row:
                    raise ValueError(f"row {line_number} is missing {field!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"fixture is empty: {path}")
    return rows


def _wrong_answer(gold: str) -> str:
    # The fixture answers are numeric.  Keeping this local avoids coupling the
    # smoke to a generator or an external model.
    from fractions import Fraction

    try:
        value = Fraction(str(gold)) + 1
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    except (ValueError, ZeroDivisionError):
        return "0" if str(gold) != "0" else "1"


def _candidate_responses(row: dict[str, Any]) -> list[str]:
    gold = str(row["answer"])
    wrong = _wrong_answer(gold)
    solution = str(row["solution"])
    return [
        f"{solution} Final answer: \\boxed{{{gold}}}.",
        f"{solution} Final answer: \\boxed{{{wrong}}}.",
        f"{solution} Therefore, the result is {gold}.",
        "I need more information before I can solve this problem.",
    ]


def _initial_policy(num_examples: int, num_actions: int) -> torch.Tensor:
    if num_actions != 4:
        raise ValueError("the smoke fixture expects exactly four candidate actions")
    # The base policy prefers a formatted but wrong answer.  This gives each
    # objective a visible, interpretable failure mode on the tiny fixture.
    row = torch.tensor([0.0, 1.0, 0.2, -0.2], dtype=torch.float32)
    return row.repeat(num_examples, 1)


def _policy_records(
    policy: CategoricalPolicy,
    rows: list[dict[str, Any]],
    candidates: list[list[str]],
) -> list[dict[str, Any]]:
    ranking = torch.argsort(policy.logits.detach(), dim=-1, descending=True)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        ordered = [candidates[index][int(action)] for action in ranking[index]]
        records.append({"id": row["id"], "gold_answer": row["answer"], "responses": ordered})
    return records


def _evaluate_policy(
    name: str,
    policy: CategoricalPolicy,
    rows: list[dict[str, Any]],
    candidates: list[list[str]],
    *,
    format_bonus: float,
    invalid_penalty: float,
) -> dict[str, float | int]:
    metrics = evaluate_generations(
        _policy_records(policy, rows, candidates),
        k=len(candidates[0]),
        format_bonus=format_bonus,
        invalid_penalty=invalid_penalty,
    )
    with torch.no_grad():
        probabilities = policy.log_probs().exp()
    metrics.update(
        {
            "policy": name,
            "mean_entropy": float(policy.entropy().detach()),
            "action0_rate": float((probabilities.argmax(dim=-1) == 0).float().mean()),
            "action1_rate": float((probabilities.argmax(dim=-1) == 1).float().mean()),
            "action2_rate": float((probabilities.argmax(dim=-1) == 2).float().mean()),
            "action3_rate": float((probabilities.argmax(dim=-1) == 3).float().mean()),
        }
    )
    return metrics


def _train_sft(policy: CategoricalPolicy, target_action: int, steps: int, lr: float) -> float:
    if not 0 <= target_action < policy.logits.shape[1]:
        raise ValueError("target_action is outside the action vocabulary")
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    targets = torch.full((policy.logits.shape[0],), target_action, dtype=torch.long)
    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        final_loss_tensor = sft_loss(policy.logits, targets)
        final_loss_tensor.backward()
        optimizer.step()
        final_loss = float(final_loss_tensor.detach())
    return final_loss


def _train_dpo(policy: CategoricalPolicy, steps: int, lr: float, beta: float) -> float:
    # Prefer action 0 (correct + explicit marker) over action 2 (correct but
    # unmarked).  This isolates formatting preference from answer correctness.
    reference = policy.logits.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        log_probs = policy.log_probs()
        reference_log_probs = F.log_softmax(reference, dim=-1)
        output = dpo_loss(
            log_probs[:, 0],
            log_probs[:, 2],
            reference_log_probs[:, 0],
            reference_log_probs[:, 2],
            beta=beta,
        )
        output.loss.backward()
        optimizer.step()
        final_loss = float(output.loss.detach())
    return final_loss


def _train_grpo(
    policy: CategoricalPolicy,
    rewards: torch.Tensor,
    *,
    steps: int,
    lr: float,
    clip_eps: float,
    eps: float,
) -> tuple[float, int]:
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    num_examples, num_actions = policy.logits.shape
    group_ids = torch.arange(num_examples).repeat_interleave(num_actions)
    flat_rewards = rewards.repeat(num_examples)
    final_loss = 0.0
    zero_variance_groups = 0
    for _ in range(steps):
        old_logps = policy.log_probs().detach().reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        output = grpo_loss(
            policy.log_probs().reshape(-1),
            flat_rewards,
            group_ids,
            old_logps=old_logps,
            clip_eps=clip_eps,
            eps=eps,
        )
        output.loss.backward()
        optimizer.step()
        final_loss = float(output.loss.detach())
        zero_variance_groups = output.zero_variance_groups
    return final_loss, zero_variance_groups


def run(config: dict[str, Any], fixture_path: Path, output_dir: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 17))
    torch.manual_seed(seed)
    rows = _load_fixture(fixture_path)
    num_actions = int(config.get("num_actions", 4))
    candidates = [_candidate_responses(row) for row in rows]
    if any(len(items) != num_actions for items in candidates):
        raise ValueError("num_actions does not match generated candidates")

    format_bonus = float(config.get("format_bonus", 0.1))
    invalid_penalty = float(config.get("invalid_penalty", 0.1))
    base = CategoricalPolicy(_initial_policy(len(rows), num_actions))
    sft = CategoricalPolicy(base.logits.detach())
    sft_loss_value = _train_sft(
        sft,
        target_action=int(config.get("sft", {}).get("target_action", 2)),
        steps=int(config.get("sft", {}).get("steps", 60)),
        lr=float(config.get("sft", {}).get("learning_rate", 0.35)),
    )
    dpo = CategoricalPolicy(sft.logits.detach())
    dpo_loss_value = _train_dpo(
        dpo,
        steps=int(config.get("dpo", {}).get("steps", 60)),
        lr=float(config.get("dpo", {}).get("learning_rate", 0.25)),
        beta=float(config.get("dpo", {}).get("beta", 0.5)),
    )

    # Rewards are computed by the same verifier used for evaluation.  The
    # ordering corresponds to candidate actions 0..3 for every prompt.
    reward_values = torch.tensor(
        [
            reward_response(candidates[0][action], rows[0]["answer"], format_bonus=format_bonus, invalid_penalty=invalid_penalty)
            for action in range(num_actions)
        ],
        dtype=torch.float32,
    )
    grpo = CategoricalPolicy(sft.logits.detach())
    grpo_loss_value, zero_variance_groups = _train_grpo(
        grpo,
        reward_values,
        steps=int(config.get("grpo", {}).get("steps", 80)),
        lr=float(config.get("grpo", {}).get("learning_rate", 0.4)),
        clip_eps=float(config.get("grpo", {}).get("clip_eps", 0.2)),
        eps=float(config.get("grpo", {}).get("eps", 1e-6)),
    )

    policy_metrics = {
        "Base": _evaluate_policy("Base", base, rows, candidates, format_bonus=format_bonus, invalid_penalty=invalid_penalty),
        "SFT": _evaluate_policy("SFT", sft, rows, candidates, format_bonus=format_bonus, invalid_penalty=invalid_penalty),
        "SFT+DPO": _evaluate_policy("SFT+DPO", dpo, rows, candidates, format_bonus=format_bonus, invalid_penalty=invalid_penalty),
        "SFT+GRPO": _evaluate_policy("SFT+GRPO", grpo, rows, candidates, format_bonus=format_bonus, invalid_penalty=invalid_penalty),
    }
    summary: dict[str, Any] = {
        "status": "smoke_completed",
        "device": str(config.get("device", "cpu")),
        "seed": seed,
        "fixture": str(fixture_path),
        "examples": len(rows),
        "candidate_actions": [
            "correct_with_boxed_answer",
            "wrong_with_boxed_answer",
            "correct_without_explicit_marker",
            "unparseable_response",
        ],
        "reward_values": reward_values.tolist(),
        "training": {
            "sft_final_loss": sft_loss_value,
            "dpo_final_loss": dpo_loss_value,
            "grpo_final_loss": grpo_loss_value,
            "grpo_zero_variance_groups": zero_variance_groups,
        },
        "policies": policy_metrics,
        "limitations": [
            "This is a categorical surrogate over fixed candidates, not a language-model benchmark.",
            "The fixture is hand-authored and intentionally tiny; no GPU result is reported.",
            "Replace the fixture with a versioned public split and add multi-seed evaluation before making claims.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(policy_metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "smoke.json")
    parser.add_argument("--fixture", type=Path, default=PROJECT_ROOT / "data" / "fixture_math.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "smoke")
    args = parser.parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    summary = run(config, args.fixture, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
