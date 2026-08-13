from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.policies.factory import AVAILABLE_POLICIES, make_policy_config
from lerobot.policies.lingbot_vla_2.configuration_lingbot_vla_2 import (
    LingBotVLA2Config,
    UnifyLingBotVLA2InputsTransformFn,
    preprocess_lingbot_vla_2_sample,
)
from lerobot.policies.lingbot_vla_2.modeling_lingbot_vla_2 import (
    LingBotVLA2Policy,
    _apply_lingbot_vla_2_release_architecture,
    _set_lingbot_attention_implementations,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class _TokenizerStub:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        assert not add_generation_prompt
        return f"user: {messages[0]['content']}"

    def __call__(self, text, **kwargs):
        assert text.startswith("user:")
        length = kwargs["max_length"]
        return {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length),
        }


class _ImageProcessorStub:
    def __call__(self, image):
        assert image.dtype == torch.uint8
        return {
            "pixel_values": torch.full((4, 6), float(image.float().mean())),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


class _ProcessorStub:
    tokenizer = _TokenizerStub()
    image_processor = _ImageProcessorStub()


def test_lingbot_vla_2_is_registered_with_canonical_defaults() -> None:
    config = make_policy_config("lingbot_vla_2", device="cpu")

    assert "lingbot_vla_2" in AVAILABLE_POLICIES
    assert isinstance(config, LingBotVLA2Config)
    assert config.type == "lingbot_vla_2"
    assert config.max_state_dim == config.max_action_dim == 55
    assert config.chunk_size == 50
    assert config.moe_implementation == "fused"
    assert config.action_delta_indices == list(range(50))


def test_lingbot_vla_2_restores_release_moe_architecture() -> None:
    official_config = SimpleNamespace(
        use_moe=False,
        token_num_experts=0,
        token_moe_intermediate_size=0,
        token_shared_intermediate_size=0,
    )

    _apply_lingbot_vla_2_release_architecture(official_config)

    assert official_config.use_moe
    assert official_config.token_moe_layers == list(range(36))
    assert official_config.token_num_experts == 32
    assert official_config.token_top_k == 4
    assert official_config.token_moe_intermediate_size == 512
    assert official_config.token_shared_intermediate_size == 704
    assert official_config.router_activation == "sigmoid"
    assert official_config.routed_scaling_factor == 4.0
    assert not official_config.use_shared_expert_gate


def test_lingbot_vla_2_can_restore_eager_attention_after_official_init() -> None:
    qwen_config = SimpleNamespace(
        _attn_implementation="flash_attention_2",
        text_config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        vision_config=SimpleNamespace(_attn_implementation="flash_attention_2"),
    )
    expert_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    official_model = SimpleNamespace(
        model=SimpleNamespace(
            qwenvl_with_expert=SimpleNamespace(
                qwenvl=SimpleNamespace(config=qwen_config),
                qwen_expert=SimpleNamespace(config=expert_config),
            )
        )
    )

    _set_lingbot_attention_implementations(official_model, "eager", "sdpa")

    assert qwen_config._attn_implementation == "eager"
    assert qwen_config.text_config._attn_implementation == "eager"
    assert qwen_config.vision_config._attn_implementation == "sdpa"
    assert expert_config._attn_implementation == "eager"


def test_lingbot_vla_2_qwen_preprocess_emits_grid_and_tokens() -> None:
    result = preprocess_lingbot_vla_2_sample(
        _ProcessorStub(),
        [torch.full((3, 8, 8), 0.5), torch.ones(3, 8, 8)],
        [True, False],
        "pick up the block",
        tokenizer_max_length=8,
    )

    assert result["images"].shape == (2, 4, 6)
    assert result["image_grid_thw"].shape == (2, 3)
    assert result["lang_tokens"].shape == (8,)
    torch.testing.assert_close(result["img_masks"], torch.tensor([True, False]))


def test_lingbot_vla_2_dataset_unifier_can_preserve_raw_inputs() -> None:
    transform = UnifyLingBotVLA2InputsTransformFn(num_cameras=2, preprocess=False)
    sample = {
        OBS_STATE: torch.zeros(14),
        ACTION: torch.zeros(50, 14),
        f"{ACTION}_is_pad": torch.tensor([False] * 49 + [True]),
        "task": "move the cup",
        f"{OBS_IMAGES}.image0": torch.zeros(3, 8, 8),
        f"{OBS_IMAGES}.image1": torch.ones(3, 8, 8),
        f"{OBS_IMAGES}.image1_mask": torch.tensor(False),
    }

    result = transform(sample)

    assert result["task"] == "move the cup"
    assert result["action_is_pad"][-1]
    assert result[f"{OBS_IMAGES}.image0"].shape == (3, 8, 8)
    assert not result[f"{OBS_IMAGES}.image1_mask"]


def test_lingbot_vla_2_dimension_mapping_round_trip() -> None:
    config = LingBotVLA2Config(
        device="cpu",
        environment_action_dim=4,
        action_dim_mapping=[[0, 2, 0, 2], [2, 4, 28, 30]],
    )
    policy = object.__new__(LingBotVLA2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config

    raw = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    canonical, is_pad = policy._map_features(raw, 55, config.action_dim_mapping)

    assert canonical.shape == (1, 2, 55)
    assert not is_pad[0, :2].any()
    assert not is_pad[0, 28:30].any()
    torch.testing.assert_close(policy._unmap_actions(canonical), raw)


def test_lingbot_vla_2_forward_combines_dimension_and_horizon_masks() -> None:
    class _OfficialStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(use_cache=False)
            self.received = None

        def forward(self, **kwargs):
            self.received = kwargs
            loss = kwargs["actions"].sum() * 0 + 1
            zero = loss * 0
            return loss, loss, zero, zero, zero, zero, {}, None, None, None, None

        def get_optim_params(self):
            return self.parameters()

    config = LingBotVLA2Config(
        device="cpu",
        environment_action_dim=4,
        action_dim_mapping=[[0, 4, 0, 4]],
        state_dim_mapping=[[0, 4, 0, 4]],
    )
    policy = object.__new__(LingBotVLA2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = _OfficialStub()

    batch = {
        OBS_STATE: torch.zeros(2, 4),
        ACTION: torch.zeros(2, 50, 4),
        "action_is_pad": torch.tensor([[False] * 49 + [True], [False] * 50]),
        "images": torch.zeros(2, 3, 4, 6),
        "img_masks": torch.ones(2, 3, dtype=torch.bool),
        "image_grid_thw": torch.ones(2, 3, 3, dtype=torch.long),
        "lang_tokens": torch.ones(2, 8, dtype=torch.long),
        "lang_masks": torch.ones(2, 8, dtype=torch.bool),
    }

    loss, metrics = policy(batch)

    assert loss.item() == 1
    assert metrics["loss_action"].item() == 1
    joint_mask = policy.model.received["joint_mask"]
    assert joint_mask.shape == (2, 50, 55)
    assert joint_mask[0, -1].sum() == 0
    assert joint_mask[1, 0, :4].all()
    assert not joint_mask[1, 0, 4:].any()


def test_lingbot_vla_2_reserved_dimensions_are_always_masked() -> None:
    class _OfficialStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(use_cache=False)
            self.joint_mask = None

        def forward(self, **kwargs):
            self.joint_mask = kwargs["joint_mask"]
            loss = kwargs["actions"].sum() * 0
            return loss, loss, loss, loss, loss, loss, {}, None, None, None, None

    policy = object.__new__(LingBotVLA2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = LingBotVLA2Config(device="cpu")
    policy.model = _OfficialStub()
    batch = {
        OBS_STATE: torch.zeros(1, 55),
        ACTION: torch.zeros(1, 50, 55),
        "action_is_pad": torch.zeros(1, 50, dtype=torch.bool),
        "images": torch.zeros(1, 3, 4, 6),
        "img_masks": torch.ones(1, 3, dtype=torch.bool),
        "image_grid_thw": torch.ones(1, 3, 3, dtype=torch.long),
        "lang_tokens": torch.ones(1, 8, dtype=torch.long),
        "lang_masks": torch.ones(1, 8, dtype=torch.bool),
    }

    policy(batch)

    assert policy.model.joint_mask[0, 0, :51].all()
    assert not policy.model.joint_mask[0, 0, 51:].any()


def test_lingbot_vla_2_update_runs_moe_load_balance_hook() -> None:
    policy = object.__new__(LingBotVLA2Policy)
    torch.nn.Module.__init__(policy)
    called = []
    policy._moe_load_balance_hook = lambda optimizer: called.append(optimizer)

    policy.update()

    assert called == [None]
