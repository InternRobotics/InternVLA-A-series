import pytest
import torch

from lerobot.opd.config import OPDConfig
from lerobot.opd.losses import diagonal_gaussian_reverse_kl


def test_reverse_kl_is_zero_for_identical_unit_gaussians() -> None:
    mean = torch.zeros(2, 3, 4)
    loss = diagonal_gaussian_reverse_kl(
        mean,
        mean,
        torch.ones_like(mean),
        student_std=1.0,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_reverse_kl_uses_mask_and_caps_each_dimension() -> None:
    student = torch.tensor([[[10.0, 1.0]]])
    teacher = torch.zeros_like(student)
    mask = torch.tensor([[[False, True]]])
    loss = diagonal_gaussian_reverse_kl(
        student,
        teacher,
        torch.ones_like(student),
        student_std=1.0,
        mask=mask,
        max_kl_per_dim=2.0,
    )
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_opd_config_requires_kairos_stats() -> None:
    with pytest.raises(ValueError, match="kairos_dataset_stats_path"):
        OPDConfig(enable=True).validate()
