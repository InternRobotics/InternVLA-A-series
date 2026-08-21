import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.rl.flow_rl import (
    FlowRLConfig,
    discounted_returns,
    reward_weighted_flow_matching_loss,
)


def test_discounted_returns_ignore_padded_rewards() -> None:
    rewards = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[True, True, False]])
    returns = discounted_returns(rewards, mask, gamma=0.5)
    torch.testing.assert_close(returns, torch.tensor([2.0]))


def test_rwfm_emphasizes_high_return_rollouts() -> None:
    result = reward_weighted_flow_matching_loss(
        torch.tensor([1.0, 3.0]),
        torch.tensor([[0.0], [1.0]]),
        torch.ones(2, 1, dtype=torch.bool),
        gamma=0.99,
        temperature=1.0,
        min_weight=0.05,
        max_weight=20.0,
        normalize_advantage=True,
    )
    assert result.weights[1] > result.weights[0]
    assert result.loss > 2.0


def test_equal_returns_reduce_to_standard_flow_matching() -> None:
    losses = torch.tensor([1.0, 3.0])
    result = reward_weighted_flow_matching_loss(
        losses,
        torch.ones(2, 2),
        None,
        gamma=1.0,
        temperature=1.0,
        min_weight=0.05,
        max_weight=20.0,
        normalize_advantage=True,
    )
    torch.testing.assert_close(result.weights, torch.ones(2))
    torch.testing.assert_close(result.loss, losses.mean())


def test_single_rollout_keeps_reward_dependent_update_scale() -> None:
    result = reward_weighted_flow_matching_loss(
        torch.tensor([2.0]),
        torch.tensor([[1.0]]),
        None,
        gamma=1.0,
        temperature=1.0,
        min_weight=0.05,
        max_weight=20.0,
        normalize_advantage=True,
    )
    torch.testing.assert_close(result.weights, torch.tensor([torch.e]))


def test_precomputed_advantage_overrides_return_baseline() -> None:
    result = reward_weighted_flow_matching_loss(
        torch.ones(2),
        torch.zeros(2, 1),
        None,
        gamma=1.0,
        temperature=1.0,
        min_weight=0.05,
        max_weight=20.0,
        normalize_advantage=False,
        supplied_advantage=torch.tensor([-1.0, 1.0]),
    )
    assert result.weights[1] > result.weights[0]


def test_flow_rl_config_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        FlowRLConfig(enable=True, temperature=0).validate()


def test_train_config_enables_flow_rl_before_dataset_creation(tmp_path) -> None:
    class RolloutDataset:
        include_rl_signals = True
        rl_rollout_dataset = True

    class InternVLAPolicy:
        type = "internvla_a1_5"
        push_to_hub = False
        repo_id = None
        video_loss_only = False
        inference_backend = "standard"
        rl_enabled = False
        opd_enabled = False

        def get_optimizer_preset(self):
            return object()

        def get_scheduler_preset(self):
            return object()

    cfg = TrainPipelineConfig(
        dataset=RolloutDataset(),
        policy=InternVLAPolicy(),
        output_dir=tmp_path / "run",
    )
    cfg.rl.enable = True
    cfg.rl.reward_horizon = 75
    cfg.validate()

    assert cfg.policy.rl_enabled
    assert cfg.policy.rl_reward_horizon == 75
