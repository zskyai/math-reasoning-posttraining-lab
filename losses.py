"""Small, trainer-independent SFT, DPO and GRPO objective implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class PreferenceLossOutput:
    """DPO loss and detached diagnostics."""

    loss: torch.Tensor
    per_example_loss: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    preference_margin: torch.Tensor


@dataclass(frozen=True)
class GRPOLossOutput:
    """GRPO surrogate and group-normalisation diagnostics."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    advantages: torch.Tensor
    ratios: torch.Tensor
    zero_variance_groups: int


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def sft_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute ordinary cross-entropy for categorical or flattened LM logits."""

    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match logits prefix "
            f"{tuple(logits.shape[:-1])}"
        )
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def categorical_logps(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for one categorical action per row."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, actions]")
    if actions.ndim != 1 or actions.shape[0] != logits.shape[0]:
        raise ValueError("actions must have shape [batch]")
    return F.log_softmax(logits, dim=-1).gather(1, actions.long().unsqueeze(1)).squeeze(1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reduction: Reduction = "mean",
) -> PreferenceLossOutput:
    """Compute the reference-corrected DPO logistic objective."""

    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if len({tuple(tensor.shape) for tensor in tensors}) != 1:
        raise ValueError("all DPO log-probability tensors must have the same shape")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")

    reference_chosen = reference_chosen_logps.detach()
    reference_rejected = reference_rejected_logps.detach()
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen - reference_rejected
    relative_margin = policy_margin - reference_margin
    scaled = beta * relative_margin
    losses = (
        -(1.0 - label_smoothing) * F.logsigmoid(scaled)
        - label_smoothing * F.logsigmoid(-scaled)
    )
    return PreferenceLossOutput(
        loss=_reduce(losses, reduction),
        per_example_loss=losses,
        chosen_rewards=(beta * (policy_chosen_logps - reference_chosen)).detach(),
        rejected_rewards=(beta * (policy_rejected_logps - reference_rejected)).detach(),
        preference_margin=relative_margin.detach(),
    )


def _group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    normalize: bool,
    eps: float,
) -> tuple[torch.Tensor, int]:
    if rewards.ndim != 1 or group_ids.ndim != 1 or rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have the same shape [samples]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    advantages = torch.zeros_like(rewards)
    zero_variance = 0
    for group in torch.unique(group_ids, sorted=True):
        mask = group_ids == group
        group_rewards = rewards[mask]
        centered = group_rewards - group_rewards.mean()
        if not normalize:
            advantages[mask] = centered
            continue
        std = group_rewards.std(unbiased=False)
        if group_rewards.numel() < 2 or float(std.detach()) < eps:
            zero_variance += 1
            # There is no relative signal when every rollout in a group scores
            # identically; dropping it avoids injecting arbitrary gradients.
            advantages[mask] = torch.zeros_like(group_rewards)
        else:
            advantages[mask] = centered / (std + eps)
    return advantages.detach(), zero_variance


def grpo_loss(
    policy_logps: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    old_logps: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    normalize_rewards: bool = True,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> GRPOLossOutput:
    """Compute a clipped GRPO policy surrogate for sampled completions.

    ``group_ids`` identifies the rollouts sampled for each prompt.  Rewards are
    centred (and optionally standardised) within each group, so a prompt with
    uniformly wrong or uniformly correct rollouts contributes no spurious
    direction.  The reference/old policy is represented by detached
    ``old_logps``; omitting it gives a one-step ratio of one for smoke tests.
    """

    if policy_logps.ndim != 1 or rewards.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("policy_logps, rewards and group_ids must be rank-1")
    if not (policy_logps.shape == rewards.shape == group_ids.shape):
        raise ValueError("policy_logps, rewards and group_ids must have equal shapes")
    if old_logps is None:
        old_logps = policy_logps.detach()
    if old_logps.shape != policy_logps.shape:
        raise ValueError("old_logps must have the same shape as policy_logps")
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps must be in (0, 1)")

    advantages, zero_variance = _group_advantages(
        rewards,
        group_ids,
        normalize=normalize_rewards,
        eps=eps,
    )
    ratios = torch.exp(policy_logps - old_logps.detach())
    clipped_ratios = ratios.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratios * advantages, clipped_ratios * advantages)
    per_sample = -surrogate
    return GRPOLossOutput(
        loss=_reduce(per_sample, reduction),
        per_sample_loss=per_sample,
        advantages=advantages,
        ratios=ratios.detach(),
        zero_variance_groups=zero_variance,
    )


__all__ = [
    "PreferenceLossOutput",
    "GRPOLossOutput",
    "sft_loss",
    "categorical_logps",
    "dpo_loss",
    "grpo_loss",
]
