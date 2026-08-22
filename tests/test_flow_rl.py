from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataloader
from lerobot.datasets.sampler import GroupedBatchSampler
from lerobot.rl.flow_rl import (
    FlowRLConfig,
    discounted_returns,
    group_relative_advantages,
    group_relative_flow_matching_loss,
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

    with pytest.raises(ValueError, match="rwfm.*grpo"):
        FlowRLConfig(enable=True, algorithm="ppo").validate()


def test_grpo_normalizes_returns_within_each_group() -> None:
    returns = torch.tensor([0.0, 2.0, 100.0, 102.0])
    groups = torch.tensor([10, 10, 20, 20])
    advantages = group_relative_advantages(returns, groups, group_size=2)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_grpo_emphasizes_better_rollouts_in_every_group() -> None:
    result = group_relative_flow_matching_loss(
        torch.ones(4),
        torch.tensor([[0.0], [2.0], [100.0], [102.0]]),
        None,
        torch.tensor([10, 10, 20, 20]),
        gamma=1.0,
        temperature=1.0,
        min_weight=0.05,
        max_weight=20.0,
        group_size=2,
    )
    assert result.weights[1] > result.weights[0]
    assert result.weights[3] > result.weights[2]
    torch.testing.assert_close(result.weights[0], result.weights[2])
    torch.testing.assert_close(result.weights[1], result.weights[3])


def test_grpo_rejects_incomplete_groups() -> None:
    with pytest.raises(ValueError, match="expected exactly 2"):
        group_relative_advantages(
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([10, 10, 20]),
            group_size=2,
        )


def test_grouped_batch_sampler_keeps_complete_groups_together() -> None:
    sampler = GroupedBatchSampler(
        [10, 20, 10, 20, 30, 40, 30, 40],
        group_size=2,
        batch_size=4,
        shuffle=False,
    )
    assert list(sampler) == [[0, 2, 1, 3], [4, 6, 5, 7]]


def test_grpo_dataloader_builds_batches_from_dataset_group_ids() -> None:
    class RawDataset:
        features = {"grpo_group_id": object()}

        def __getitem__(self, key):
            assert key == "grpo_group_id"
            return [10, 20, 10, 20]

    class GroupedDataset:
        hf_dataset = RawDataset()
        repo_id = "owner/grouped-rollouts"

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"index": index}

    cfg = SimpleNamespace(
        num_workers=0,
        batch_size=4,
        seed=7,
        policy=SimpleNamespace(type="internvla_a1_5", enable_vqa_loss=False),
        dataset=SimpleNamespace(rl_group_id_key="grpo_group_id"),
        rl=SimpleNamespace(enable=True, algorithm="grpo", group_size=2),
    )
    dataloader, self_managed = make_dataloader(cfg, GroupedDataset())

    assert not self_managed
    assert isinstance(dataloader.batch_sampler, GroupedBatchSampler)
    assert sorted(next(iter(dataloader.batch_sampler))) == [0, 1, 2, 3]


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


def test_train_config_enables_grpo_and_checks_group_layout(tmp_path) -> None:
    class GroupedRolloutDataset:
        include_rl_signals = True
        rl_rollout_dataset = True
        rl_group_id_key = "grpo_group_id"
        rl_advantage_key = None
        streaming = False

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
        dataset=GroupedRolloutDataset(),
        policy=InternVLAPolicy(),
        output_dir=tmp_path / "grpo-run",
        batch_size=8,
    )
    cfg.rl.enable = True
    cfg.rl.algorithm = "grpo"
    cfg.rl.group_size = 4
    cfg.validate()

    assert cfg.policy.rl_enabled
    assert cfg.policy.rl_algorithm == "grpo"
    assert cfg.policy.rl_group_size == 4
