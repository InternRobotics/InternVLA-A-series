from __future__ import annotations

import torch

from lerobot.policies.g05.configuration_g05 import G05Config, UnifyG05InputsTransformFn
from lerobot.policies.g05.modeling_g05 import G05Policy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_g05_config_defaults_match_canonical_action_layout() -> None:
    config = G05Config(device="cpu")

    assert config.type == "g05"
    assert sum(config.action_parts_meta.values()) == 27
    assert config.action_delta_indices == list(range(32))
    assert config.use_group_markers
    assert config.dropout_noop_parts


def test_g05_dimension_mapping_round_trip() -> None:
    config = G05Config(
        device="cpu",
        environment_action_dim=4,
        action_dim_mapping=[[0, 2, 0, 2], [2, 4, 10, 12]],
    )
    policy = object.__new__(G05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config

    raw = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    canonical, is_pad = policy._map_features(raw, 27, config.action_dim_mapping)

    assert canonical.shape == (1, 2, 27)
    assert not is_pad[0, :2].any()
    assert not is_pad[0, 10:12].any()
    assert is_pad[0, 2:10].all()
    torch.testing.assert_close(policy._unmap_actions(canonical), raw)


def test_g05_dataset_unifier_preserves_required_fields() -> None:
    transform = UnifyG05InputsTransformFn(num_cameras=2)
    sample = {
        OBS_STATE: torch.zeros(5),
        ACTION: torch.zeros(3, 4),
        f"{ACTION}_is_pad": torch.tensor([False, False, True]),
        "task": "pick up the block",
        f"{OBS_IMAGES}.image0": torch.zeros(3, 16, 16),
        f"{OBS_IMAGES}.image0_mask": torch.tensor(True),
        f"{OBS_IMAGES}.image1": torch.ones(3, 16, 16),
        f"{OBS_IMAGES}.image1_mask": torch.tensor(False),
    }

    result = transform(sample)

    assert set(result) == {
        OBS_STATE,
        ACTION,
        "action_is_pad",
        "task",
        f"{OBS_IMAGES}.image0",
        f"{OBS_IMAGES}.image0_mask",
        f"{OBS_IMAGES}.image1",
        f"{OBS_IMAGES}.image1_mask",
    }
    assert result["task"] == "pick up the block"


def test_g05_dataset_unifier_merges_raw_action_padding() -> None:
    transform = UnifyG05InputsTransformFn(
        num_cameras=1,
        action_keys=["action.left", "action.right"],
    )
    sample = {
        OBS_STATE: torch.zeros(5),
        ACTION: torch.zeros(3, 4),
        "action.left_is_pad": torch.tensor([False, False, True]),
        "action.right_is_pad": torch.tensor([False, True, False]),
        f"{OBS_IMAGES}.image0": torch.zeros(3, 16, 16),
    }

    result = transform(sample)

    torch.testing.assert_close(
        result["action_is_pad"],
        torch.tensor([False, True, True]),
    )


def test_g05_state_dict_excludes_action_codec_sidecar() -> None:
    class _OfficialStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.action_tokenizer = torch.nn.Module()
            self.action_tokenizer.action_tokenizer = torch.nn.Linear(2, 2)

    config = G05Config(device="cpu")
    policy = object.__new__(G05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = _OfficialStub()

    state = policy.state_dict()

    assert "model.backbone.weight" in state
    assert not any(key.startswith("model.action_tokenizer.action_tokenizer.") for key in state)


def test_g05_to_moves_plain_action_tokenizer_wrapper() -> None:
    class _TokenizerStub:
        def __init__(self) -> None:
            self.device = None

        def to(self, device) -> None:
            self.device = torch.device(device)

    class _OfficialStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.action_tokenizer = _TokenizerStub()

    config = G05Config(device="cpu")
    policy = object.__new__(G05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = _OfficialStub()

    policy.to("cpu")

    assert policy.model.action_tokenizer.device == torch.device("cpu")


def test_g05_forward_converts_lerobot_batch_to_official_contract() -> None:
    class _OfficialStub(torch.nn.Module):
        def forward(self, batch, inference_mode=False):
            assert not inference_mode
            assert batch["pixel_values"].shape == (2, 3, 3, 8, 8)
            assert batch["action"].shape == (2, 32, 27)
            assert batch["action_is_pad"].shape == (2, 32)
            assert batch["action_dim_is_pad"].shape == (2, 27)
            assert len(batch["samples"]) == 2
            loss = batch["action"].sum() * 0 + 1
            return loss, {"ce_loss": loss}

    config = G05Config(device="cpu")
    policy = object.__new__(G05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = _OfficialStub()

    batch = {
        OBS_STATE: torch.zeros(2, 27),
        ACTION: torch.zeros(2, 32, 27),
        "action_is_pad": torch.zeros(2, 32, dtype=torch.bool),
        "task": ["pick up the block", "put down the block"],
    }
    for index in range(3):
        batch[f"{OBS_IMAGES}.image{index}"] = torch.full((2, 3, 8, 8), 0.5)
        batch[f"{OBS_IMAGES}.image{index}_mask"] = torch.ones(2, dtype=torch.bool)

    loss, metrics = policy(batch)

    assert loss.item() == 1
    assert metrics["loss_action"].item() == 1
