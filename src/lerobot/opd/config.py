"""Configuration for continuous-action on-policy distillation."""

from dataclasses import dataclass


@dataclass
class OPDConfig:
    """Kairos teacher and reverse-KL surrogate settings.

    Kairos exposes stochastic continuous action samples rather than action-token
    logits. Multiple seeded samples are therefore used to estimate a diagonal
    teacher distribution in the student's normalized action space.
    """

    enable: bool = False
    teacher_type: str = "kairos"
    teacher_endpoint: str = "http://127.0.0.1:8006"
    kairos_dataset_stats_path: str | None = None
    kairos_stats_key: str = "default"
    teacher_samples: int = 4
    teacher_seed: int = 42
    action_horizon: int | None = None
    num_inference_steps: int = 10
    cfg_scale: float = 8.0
    negative_prompt: str = ""
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    image_layout: str = "auto"  # auto | first | robotwin | horizontal
    timeout_s: float = 3600.0
    load_teacher_on_init: bool = True
    allow_remote_pickle: bool = False
    batch_requests: bool = True

    # Teacher uncertainty and continuous reverse-KL surrogate.
    teacher_std_min: float = 0.05
    teacher_std_max: float = 1.0
    student_std: float = 0.10
    max_kl_per_dim: float | None = 20.0
    loss_weight: float = 1.0
    sft_loss_weight: float = 0.0
    student_normalization_mode: str = "mean_std"
    require_on_policy: bool = True

    def validate(self) -> None:
        if not self.enable:
            return
        if self.teacher_type != "kairos":
            raise ValueError(f"Only teacher_type='kairos' is supported, got {self.teacher_type!r}")
        if not self.teacher_endpoint:
            raise ValueError("opd.teacher_endpoint must be set when OPD is enabled")
        if not self.kairos_dataset_stats_path:
            raise ValueError("opd.kairos_dataset_stats_path is required for Kairos action/state scaling")
        if self.teacher_samples < 1:
            raise ValueError("opd.teacher_samples must be >= 1")
        if self.action_horizon is not None and self.action_horizon < 1:
            raise ValueError("opd.action_horizon must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError("opd.num_inference_steps must be >= 1")
        if self.timeout_s <= 0:
            raise ValueError("opd.timeout_s must be > 0")
        if self.image_layout not in {"auto", "first", "robotwin", "horizontal"}:
            raise ValueError(f"Unsupported opd.image_layout: {self.image_layout!r}")
        if not 0 < self.teacher_std_min <= self.teacher_std_max:
            raise ValueError("Require 0 < teacher_std_min <= teacher_std_max")
        if self.student_std <= 0:
            raise ValueError("opd.student_std must be > 0")
        if self.loss_weight < 0 or self.sft_loss_weight < 0:
            raise ValueError("OPD loss weights must be non-negative")
        if self.max_kl_per_dim is not None and self.max_kl_per_dim <= 0:
            raise ValueError("opd.max_kl_per_dim must be > 0 when set")
        if self.student_normalization_mode not in {"mean_std", "min_max", "q01_q99"}:
            raise ValueError(
                "opd.student_normalization_mode must be mean_std, min_max, or q01_q99"
            )
