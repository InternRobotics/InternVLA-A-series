"""Configuration for training Galaxea G0.5 inside the InternVLA/LeRobot stack."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.dataset_schemas import get_schema
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DataTransformFn,
    DeltaActionTransformFn,
    NormalizeTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
    TransformGroup,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@DataTransformFn.register_subclass("unify_g05_inputs")
@dataclass
class UnifyG05InputsTransformFn(DataTransformFn):
    """Keep the native LeRobot tensors required by the G0.5 policy adapter."""

    num_cameras: int = 3
    action_keys: list[str] = field(default_factory=list)

    def hydrate(self, dataset) -> UnifyG05InputsTransformFn:
        """Remember raw action keys so their horizon padding can be merged."""
        schema = get_schema(dataset.meta.robot_type)
        return replace(self, action_keys=schema.get_action_keys())

    def __call__(self, data: dict) -> dict:
        result = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            "task": str(data.get("task", "")),
        }
        import torch

        action_is_pad = data.get(f"{ACTION}_is_pad")
        if action_is_pad is None:
            raw_padding = [
                torch.as_tensor(data[f"{key}_is_pad"], dtype=torch.bool)
                for key in self.action_keys
                if f"{key}_is_pad" in data
            ]
            action_is_pad = (
                torch.stack(raw_padding).any(dim=0)
                if raw_padding
                else torch.zeros(data[ACTION].shape[0], dtype=torch.bool)
            )
        result["action_is_pad"] = action_is_pad

        for index in range(self.num_cameras):
            key = f"{OBS_IMAGES}.image{index}"
            result[key] = data[key]
            result[f"{key}_mask"] = data.get(f"{key}_mask", True)
        return result


@DatasetConfig.register_subclass("g05")
@dataclass
class G05DatasetConfig(DatasetConfig):
    """LeRobot dataset pipeline used by G0.5.

    Images are emitted in the range [-1, 1], while state/action normalization
    remains configurable through the standard LeRobot statistics transform.
    """

    height: int = 256
    width: int = 256
    num_cameras: int = 3
    normalization_mode: str = "q01_q99"

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=G05DatasetConfig.height,
                    width=G05DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(mode=G05DatasetConfig.normalization_mode),
                ComposeFieldsTransform(),
                ReorderStateActionTransform(),
                UnifyG05InputsTransformFn(num_cameras=G05DatasetConfig.num_cameras),
            ],
            outputs=[],
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (1 <= self.num_cameras <= 3):
            raise ValueError("G0.5 currently supports one to three input cameras")
        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(transform, DeltaActionTransformFn) for transform in inputs)
        if self.action_mode == "delta" and not has_delta:
            inputs.insert(0, DeltaActionTransformFn())
        elif self.action_mode == "abs" and has_delta:
            inputs = [transform for transform in inputs if not isinstance(transform, DeltaActionTransformFn)]

        for transform in inputs:
            if isinstance(transform, ResizeImagesWithPadFn):
                transform.height, transform.width = self.height, self.width
            elif isinstance(transform, NormalizeTransformFn):
                transform.mode = self.normalization_mode
            elif isinstance(transform, UnifyG05InputsTransformFn):
                transform.num_cameras = self.num_cameras
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("g05")
@dataclass
class G05Config(PreTrainedConfig):
    """G0.5 model and adapter settings.

    ``g05_source_path`` points to an installed/checked-out official GalaxeaVLA
    tree. That package supplies the model and ActionCodec; the training loop,
    datasets, optimizer, logging and checkpoints are owned by this repository.
    """

    g05_source_path: Path | None = None
    official_model_config: Path | None = None
    official_tokenizer_config: Path | None = None
    official_pretrained_checkpoint: Path | None = None
    hf_processor_path: str | None = None
    action_tokenizer_path: Path | None = None
    license: str | None = "other"

    dtype: str = "bfloat16"
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 16
    num_cameras: int = 3
    image_resolution: tuple[int, int] = (256, 256)
    max_state_dim: int = 27
    max_action_dim: int = 27
    embodiment: str = "unknown"

    discrete_action: bool = True
    continuous_action: bool = False
    predict_cot: bool = False
    proprio_encoder: str = "mlp"
    gradient_checkpointing: bool = True
    use_fused_ce: bool = True
    block_wise_autoregressive: bool = False
    action_codec_num_residuals: int = 2
    action_parts_meta: dict[str, int] = field(
        default_factory=lambda: {
            "left_control": 9,
            "left_gripper": 1,
            "right_control": 9,
            "right_gripper": 1,
            "lower_body": 7,
        }
    )
    gripper_parts: list[str] = field(default_factory=lambda: ["left_gripper", "right_gripper"])
    use_group_markers: bool = True
    dropout_noop_parts: bool = True
    absent_key_fill_value: float = -100.0
    environment_action_dim: int | None = None
    action_dim_mapping: list[list[int]] | None = None
    state_dim_mapping: list[list[int]] | None = None
    active_action_dims: list[int] | None = None
    active_state_dims: list[int] | None = None

    optimizer_lr: float = 4e-5
    backbone_lr_multiplier: float = 1.0
    vision_lr_multiplier: float = 0.1
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.03
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 4e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"dtype must be 'bfloat16' or 'float32', got {self.dtype!r}")
        if not (1 <= self.n_action_steps <= self.chunk_size):
            raise ValueError("n_action_steps must be between 1 and chunk_size")
        if self.chunk_size != 32:
            raise ValueError("The released G0.5 ActionCodec checkpoint requires chunk_size=32")
        if self.n_obs_steps != 1:
            raise ValueError(
                "The current LeRobot adapter supports n_obs_steps=1; G0.5 memory is configured separately."
            )
        if not (1 <= self.num_cameras <= 3):
            raise ValueError("G0.5 currently supports one to three input cameras")
        if not self.action_parts_meta:
            raise ValueError("action_parts_meta cannot be empty")
        action_layout_dim = sum(self.action_parts_meta.values())
        if action_layout_dim != self.max_action_dim:
            raise ValueError(
                f"action_parts_meta sums to {action_layout_dim}, but max_action_dim={self.max_action_dim}"
            )
        for dims_name, dims in (
            ("active_action_dims", self.active_action_dims),
            ("active_state_dims", self.active_state_dims),
        ):
            if dims is not None and (not dims or any(dim <= 0 for dim in dims)):
                raise ValueError(f"{dims_name} must contain positive part sizes")
        if self.active_action_dims is not None and sum(self.active_action_dims) > self.max_action_dim:
            raise ValueError("active_action_dims exceeds max_action_dim")
        if self.active_state_dims is not None and sum(self.active_state_dims) > self.max_state_dim:
            raise ValueError("active_state_dims exceeds max_state_dim")
        for dims_name, dims in (
            ("active_action_dims", self.active_action_dims),
            ("active_state_dims", self.active_state_dims),
        ):
            if dims is not None and len(dims) != len(self.action_parts_meta):
                raise ValueError(f"{dims_name} must have one size per action_parts_meta entry")
        for mapping_name, mapping, target_dim in (
            ("action_dim_mapping", self.action_dim_mapping, self.max_action_dim),
            ("state_dim_mapping", self.state_dim_mapping, self.max_state_dim),
        ):
            if mapping is None:
                continue
            for entry in mapping:
                if len(entry) != 4:
                    raise ValueError(
                        f"{mapping_name} entries must be [src_start, src_end, dst_start, dst_end]"
                    )
                src_start, src_end, dst_start, dst_end = entry
                if src_start < 0 or dst_start < 0 or src_end <= src_start or dst_end <= dst_start:
                    raise ValueError(f"Invalid {mapping_name} entry: {entry}")
                if src_end - src_start != dst_end - dst_start or dst_end > target_dim:
                    raise ValueError(f"Incompatible {mapping_name} entry: {entry}")
                if (
                    mapping_name == "action_dim_mapping"
                    and self.environment_action_dim is not None
                    and src_end > self.environment_action_dim
                ):
                    raise ValueError(
                        f"{mapping_name} source {entry} exceeds environment_action_dim="
                        f"{self.environment_action_dim}"
                    )

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.max_state_dim,)
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.max_action_dim,)
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> None:
        return None
