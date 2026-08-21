"""Reward-weighted reinforcement learning for implicit flow policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FlowRLConfig:
    """Configuration for reward-weighted flow matching (RWFM).

    The policy's conditional flow-matching loss is used as a tractable
    negative-log-likelihood surrogate. Samples with larger on-policy return
    receive exponentially larger regression weights.
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

    def validate(self) -> None:
        if not self.enable:
            return
        if self.algorithm != "rwfm":
            raise ValueError(f"Only rl.algorithm='rwfm' is supported, got {self.algorithm!r}")
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
