"""Losses used by continuous-action OPD."""

from __future__ import annotations

import torch
from torch import Tensor


def diagonal_gaussian_reverse_kl(
    student_mean: Tensor,
    teacher_mean: Tensor,
    teacher_std: Tensor,
    *,
    student_std: float | Tensor,
    mask: Tensor | None = None,
    max_kl_per_dim: float | None = None,
) -> Tensor:
    """Compute ``KL(student || teacher)`` for diagonal Gaussian proxies.

    InternVLA-A1.5 and Kairos are implicit flow policies, so their exact
    continuous action log-probabilities are unavailable. This loss applies the
    reverse-KL objective to local diagonal Gaussian proxies: ``student_mean``
    is the clean action reconstructed by InternVLA's flow field, while the
    teacher moments are estimated from seeded Kairos samples.
    """

    if student_mean.shape != teacher_mean.shape or student_mean.shape != teacher_std.shape:
        raise ValueError(
            "student_mean, teacher_mean, and teacher_std must have identical shapes; "
            f"got {student_mean.shape}, {teacher_mean.shape}, {teacher_std.shape}"
        )
    teacher_std = teacher_std.to(device=student_mean.device, dtype=torch.float32).clamp_min(1e-8)
    teacher_mean = teacher_mean.to(device=student_mean.device, dtype=torch.float32)
    student_mean = student_mean.float()
    student_std_tensor = torch.as_tensor(
        student_std, device=student_mean.device, dtype=torch.float32
    ).clamp_min(1e-8)

    variance_ratio = student_std_tensor.square() / teacher_std.square()
    mean_term = (student_mean - teacher_mean).square() / teacher_std.square()
    per_dim = torch.log(teacher_std / student_std_tensor) + 0.5 * (
        variance_ratio + mean_term - 1.0
    )
    if max_kl_per_dim is not None:
        per_dim = per_dim.clamp(max=float(max_kl_per_dim))

    if mask is None:
        return per_dim.mean()
    mask = mask.to(device=per_dim.device, dtype=per_dim.dtype)
    if mask.shape != per_dim.shape:
        mask = torch.broadcast_to(mask, per_dim.shape)
    return (per_dim * mask).sum() / mask.sum().clamp_min(1.0)
