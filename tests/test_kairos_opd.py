import json

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.opd.config import OPDConfig
from lerobot.opd.kairos_teacher import (
    KairosDatasetStats,
    KairosOPDTeacher,
    KairosWAMClient,
    StudentActionProjector,
    compose_kairos_image,
)


def test_kairos_stats_round_trip(tmp_path) -> None:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "state": {
                    "default": {"global_mean": [1.0, 2.0], "global_std": [2.0, 4.0]}
                },
                "action": {
                    "default": {"global_mean": [3.0, 4.0], "global_std": [5.0, 6.0]}
                },
            }
        ),
        encoding="utf-8",
    )
    stats = KairosDatasetStats(stats_path)
    value = torch.tensor([[3.0, 6.0]])
    torch.testing.assert_close(stats.state.denormalize(stats.state.normalize(value)), value)


def test_projector_reorders_and_pads_piper_actions() -> None:
    action_stats = {
        "mean": np.zeros(14, dtype=np.float32),
        "std": np.ones(14, dtype=np.float32),
    }
    projector = StudentActionProjector(
        {"piper_robotwin": {"action": action_stats}},
        normalization_mode="mean_std",
        action_mode="abs",
        max_action_dim=32,
    )
    actions = torch.arange(14, dtype=torch.float32).view(1, 1, 14)
    projected, mask = projector.project(actions, torch.zeros(14), "piper_robotwin")

    assert projected.shape == (1, 1, 32)
    assert mask.shape == (32,)
    assert mask.sum().item() == 14
    assert not mask[6] and not mask[14]
    torch.testing.assert_close(projected[0, 0, 7], torch.tensor(6.0))
    torch.testing.assert_close(projected[0, 0, 15], torch.tensor(13.0))


def test_short_teacher_horizon_is_masked() -> None:
    teacher = KairosOPDTeacher.__new__(KairosOPDTeacher)
    teacher.chunk_size = 4
    actions, mask = teacher._match_horizon(torch.tensor([[[1.0], [2.0]]]))
    torch.testing.assert_close(actions, torch.tensor([[[1.0], [2.0], [2.0], [2.0]]]))
    torch.testing.assert_close(mask, torch.tensor([True, True, False, False]))


def test_robotwin_image_layout() -> None:
    image = Image.new("RGB", (10, 10))
    assert compose_kairos_image([image, image, image], "robotwin").size == (320, 384)


def test_pickle_client_rejects_untrusted_remote_endpoint() -> None:
    cfg = OPDConfig(teacher_endpoint="http://teacher.example.com:8006")
    with pytest.raises(ValueError, match="Refusing a non-local endpoint"):
        KairosWAMClient(cfg)
