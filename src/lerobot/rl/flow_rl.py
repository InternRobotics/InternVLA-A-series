"""Reward-weighted and group-relative RL for implicit flow policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FlowRLConfig:
    """Configuration for flow-policy RL.

    Both supported algorithms use the policy's conditional flow-matching loss
    as a tractable surrogate. RWFM normalizes returns over the local batch;
    GRPO normalizes returns only among rollouts sampled for the same condition.
    """

    enable: bool = False
    algorithm: str = "rwfm"
    gamma: float = 0.99
    reward_horizon: int | None = None
    normalize_advantage: bool = True
    temperature: float = 1.0
    min_weight: float = 0.05
    max_weight: float = 20.0
    loss_weight: float = 1.0
    sft_loss_weight: float = 0.1
    require_on_policy: bool = True
    group_size: int = 4

    def validate(self) -> None:
        if not self.enable:
            return
        if self.algorithm not in {"rwfm", "grpo"}:
            raise ValueError(
                f"rl.algorithm must be 'rwfm' or 'grpo', got {self.algorithm!r}"
            )
        if not 0 <= self.gamma <= 1:
            raise ValueError("rl.gamma must be in [0, 1]")
        if self.reward_horizon is not None and self.reward_horizon < 1:
            raise ValueError("rl.reward_horizon must be >= 1 when set")
        if self.temperature <= 0:
            raise ValueError("rl.temperature must be > 0")
        if not 0 <= self.min_weight <= self.max_weight or self.max_weight == 0:
            raise ValueError("Require 0 <= rl.min_weight <= rl.max_weight and max_weight > 0")
        if self.loss_weight < 0 or self.sft_loss_weight < 0:
            raise ValueError("RL loss weights must be non-negative")
        if self.algorithm == "grpo" and self.group_size < 2:
            raise ValueError("rl.group_size must be >= 2")


@dataclass(frozen=True)
class FlowRLResult:
    loss: Tensor
    returns: Tensor
    advantages: Tensor
    weights: Tensor


def _as_batch_sequence(value: Tensor, *, name: str) -> Tensor:
    value = value.float()
    if value.ndim == 1:
        return value[:, None]
    if value.ndim >= 2:
        return value.reshape(value.shape[0], -1)
    raise ValueError(f"{name} must include a batch dimension")


def discounted_returns(rewards: Tensor, reward_mask: Tensor | None, gamma: float) -> Tensor:
    """Compute a truncated discounted return for each rollout state."""

    rewards = _as_batch_sequence(rewards, name="rewards")
    if reward_mask is None:
        reward_mask = torch.ones_like(rewards, dtype=torch.bool)
    else:
        reward_mask = _as_batch_sequence(reward_mask, name="reward_mask").bool()
        if reward_mask.shape != rewards.shape:
            raise ValueError(
                f"reward_mask shape {reward_mask.shape} does not match rewards {rewards.shape}"
            )
    discounts = torch.pow(
        torch.as_tensor(gamma, device=rewards.device, dtype=rewards.dtype),
        torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype),
    )
    return (rewards * reward_mask * discounts.unsqueeze(0)).sum(dim=1)


def reward_weighted_flow_matching_loss(
    per_sample_flow_loss: Tensor,
    rewards: Tensor,
    reward_mask: Tensor | None,
    *,
    gamma: float,
    temperature: float,
    min_weight: float,
    max_weight: float,
    normalize_advantage: bool,
    supplied_advantage: Tensor | None = None,
    eps: float = 1e-6,
) -> FlowRLResult:
    """Apply AWR-style exponential return weights to flow-matching losses."""

    if per_sample_flow_loss.ndim != 1:
        raise ValueError("per_sample_flow_loss must have shape [batch]")
    returns = discounted_returns(rewards, reward_mask, gamma).detach()
    if supplied_advantage is None:
        advantages = returns
    else:
        advantages = supplied_advantage.float().reshape(-1).to(returns).detach()
        if advantages.shape != returns.shape:
            raise ValueError(
                f"supplied_advantage shape {advantages.shape} does not match returns {returns.shape}"
            )

    if normalize_advantage and advantages.numel() > 1:
        std = advantages.std(unbiased=False)
        if std > eps:
            advantages = (advantages - advantages.mean()) / (std + eps)
        else:
            advantages = torch.zeros_like(advantages)

    weights = torch.exp(advantages / temperature).clamp(min=min_weight, max=max_weight)
    # Preserve a predictable loss scale while retaining relative sample weights.
    if weights.numel() > 1:
        weights = weights / weights.mean().clamp_min(eps)
    weights = weights.clamp(min=min_weight, max=max_weight).detach()
    loss = (per_sample_flow_loss * weights).mean()
    return FlowRLResult(loss=loss, returns=returns, advantages=advantages, weights=weights)


def group_relative_advantages(
    returns: Tensor,
    group_ids: Tensor,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> Tensor:
    """Normalize returns inside each complete rollout group.

    A group must contain different actions sampled for the same observation
    and task. Requiring complete groups prevents a shuffled or split batch from
    silently changing GRPO into ordinary batch-relative reward weighting.
    """

    returns = returns.float().reshape(-1)
    group_ids = group_ids.reshape(-1).to(device=returns.device)
    if group_ids.shape != returns.shape:
        raise ValueError(
            f"group_ids shape {group_ids.shape} does not match returns {returns.shape}"
        )
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    advantages = torch.empty_like(returns)
    unique_groups = torch.unique(group_ids)
    for group_id in unique_groups:
        mask = group_ids == group_id
        count = int(mask.sum().item())
        if count != group_size:
            raise ValueError(
                f"GRPO group {group_id.item()!r} contains {count} samples; "
                f"expected exactly {group_size}. Keep each rollout group in one batch."
            )
        group_returns = returns[mask]
        std = group_returns.std(unbiased=False)
        if std > eps:
            advantages[mask] = (group_returns - group_returns.mean()) / (std + eps)
        else:
            advantages[mask] = 0.0
    return advantages


def group_relative_flow_matching_loss(
    per_sample_flow_loss: Tensor,
    rewards: Tensor,
    reward_mask: Tensor | None,
    group_ids: Tensor,
    *,
    gamma: float,
    temperature: float,
    min_weight: float,
    max_weight: float,
    group_size: int,
) -> FlowRLResult:
    """Apply robotics Flow-GRPO weights to conditional flow-matching loss.

    The update follows the group-relative flow-matching variant: returns are
    standardized within condition-matched groups and converted to non-negative
    exponential weights. It does not claim an exact action likelihood ratio.
    """

    if per_sample_flow_loss.ndim != 1:
        raise ValueError("per_sample_flow_loss must have shape [batch]")
    returns = discounted_returns(rewards, reward_mask, gamma).detach()
    if returns.shape != per_sample_flow_loss.shape:
        raise ValueError(
            f"returns shape {returns.shape} does not match flow loss {per_sample_flow_loss.shape}"
        )
    advantages = group_relative_advantages(
        returns,
        group_ids,
        group_size=group_size,
    )
    weights = torch.exp(advantages / temperature).clamp(
        min=min_weight,
        max=max_weight,
    ).detach()
    loss = (per_sample_flow_loss * weights).mean()
    return FlowRLResult(loss=loss, returns=returns, advantages=advantages, weights=weights)
