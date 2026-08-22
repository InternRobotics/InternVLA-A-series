"""Reinforcement-learning utilities shared by VLA policies."""

from lerobot.rl.flow_rl import FlowRLConfig

RL_ADVANTAGE = "rl.advantage"
RL_GROUP_ID = "rl.group_id"
RL_IS_ON_POLICY = "rl.is_on_policy"
RL_REWARD = "rl.reward"
RL_REWARD_MASK = "rl.reward_mask"

__all__ = [
    "FlowRLConfig",
    "RL_ADVANTAGE",
    "RL_GROUP_ID",
    "RL_IS_ON_POLICY",
    "RL_REWARD",
    "RL_REWARD_MASK",
]
