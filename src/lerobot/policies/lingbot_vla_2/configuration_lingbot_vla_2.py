"""Configuration and LeRobot input transforms for LingBot-VLA 2.0."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

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


def _as_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Match the official pipeline's uint8 input contract for Qwen3-VL."""
    image = torch.as_tensor(image).detach().cpu()
    if image.is_floating_point():
        max_value = float(image.max()) if image.numel() else 0.0
        if max_value <= 2.0:
            image = (image * 255.0).round()
        image = image.clamp(0, 255).to(torch.uint8)
    return image


def preprocess_lingbot_vla_2_sample(
    processor: Any,
    images: list[torch.Tensor],
    image_masks: list[bool | torch.Tensor],
    task: str,
    tokenizer_max_length: int,
) -> dict[str, torch.Tensor]:
    """Apply the official Qwen3-VL image and language preprocessing contract."""
    pixel_values = []
    image_grid_thw = []
    for image in images:
        processed = processor.image_processor(_as_uint8_image(image))
        pixels = torch.as_tensor(processed["pixel_values"])
        if pixels.ndim >= 3 and pixels.shape[0] == 1:
            pixels = pixels.squeeze(0)
        grid = torch.as_tensor(processed["image_grid_thw"], dtype=torch.long).reshape(-1, 3)[0]
        pixel_values.append(pixels)
        image_grid_thw.append(grid)

    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": str(task)}],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = processor.tokenizer(
        prompt,
        padding="max_length",
        padding_side="right",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "images": torch.stack(pixel_values),
        "img_masks": torch.as_tensor(image_masks, dtype=torch.bool).reshape(-1),
        "image_grid_thw": torch.stack(image_grid_thw),
        "lang_tokens": tokenized["input_ids"].squeeze(0).to(torch.long),
        "lang_masks": tokenized["attention_mask"].squeeze(0).to(torch.bool),
    }


@DataTransformFn.register_subclass("unify_lingbot_vla_2_inputs")
@dataclass
class UnifyLingBotVLA2InputsTransformFn(DataTransformFn):
    """Convert one native LeRobot sample to the official LingBot-VLA inputs."""

    qwen_processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer_max_length: int = 72
    num_cameras: int = 3
    preprocess: bool = True
    action_keys: list[str] = field(default_factory=list)
    _processor: Any = field(default=None, init=False, repr=False)

    def hydrate(self, dataset) -> UnifyLingBotVLA2InputsTransformFn:
        schema = get_schema(dataset.meta.robot_type)
        return replace(self, action_keys=schema.get_action_keys())

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.qwen_processor_path,
                padding_side="right",
                trust_remote_code=True,
            )
        return self._processor

    def __call__(self, data: dict) -> dict:
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

        result = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            "action_is_pad": torch.as_tensor(action_is_pad, dtype=torch.bool),
            "task": str(data.get("task", "")),
        }
        images = []
        image_masks = []
        for index in range(self.num_cameras):
            key = f"{OBS_IMAGES}.image{index}"
            images.append(data[key])
            image_masks.append(data.get(f"{key}_mask", True))

        if self.preprocess:
            result.update(
                preprocess_lingbot_vla_2_sample(
                    self._get_processor(),
                    images,
                    image_masks,
                    result["task"],
                    self.tokenizer_max_length,
                )
            )
        else:
            for index, (image, mask) in enumerate(zip(images, image_masks, strict=True)):
                key = f"{OBS_IMAGES}.image{index}"
                result[key] = image
                result[f"{key}_mask"] = torch.as_tensor(mask, dtype=torch.bool)
        return result


@DatasetConfig.register_subclass("lingbot_vla_2")
@dataclass
class LingBotVLA2DatasetConfig(DatasetConfig):
    """LeRobot data pipeline used for LingBot-VLA 2.0 post-training."""

    height: int = 256
    width: int = 256
    num_cameras: int = 3
    normalization_mode: str = "q01_q99"
    qwen_processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer_max_length: int = 72
    preprocess_in_dataset: bool = True

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=LingBotVLA2DatasetConfig.height,
                    width=LingBotVLA2DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(mode=LingBotVLA2DatasetConfig.normalization_mode),
                ComposeFieldsTransform(),
                ReorderStateActionTransform(),
                UnifyLingBotVLA2InputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (1 <= self.num_cameras <= 3):
            raise ValueError("LingBot-VLA 2.0 currently supports one to three input cameras")

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
            elif isinstance(transform, UnifyLingBotVLA2InputsTransformFn):
                transform.qwen_processor_path = self.qwen_processor_path
                transform.tokenizer_max_length = self.tokenizer_max_length
                transform.num_cameras = self.num_cameras
                transform.preprocess = self.preprocess_in_dataset
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("lingbot_vla_2")
@dataclass
class LingBotVLA2Config(PreTrainedConfig):
    """Adapter configuration for the official LingBot-VLA 2.0 model."""

    lingbot_vla_source_path: Path | None = None
    official_pretrained_model: str | None = None
    official_config_path: str | None = None
    qwen_processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    license: str | None = "apache-2.0"

    dtype: str = "bfloat16"
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    num_cameras: int = 3
    max_state_dim: int = 55
    max_action_dim: int = 55
    tokenizer_max_length: int = 72
    num_inference_steps: int = 10

    attention_implementation: str | None = None
    vit_attn_implementation: str | None = None
    # The release stores experts as fused [num_experts, out, in] tensors.
    moe_implementation: str | None = "fused"
    gradient_checkpointing: bool = False
    disable_auxiliary_distillation: bool = True

    environment_action_dim: int | None = None
    action_dim_mapping: list[list[int]] | None = None
    state_dim_mapping: list[list[int]] | None = None

    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        if self.n_obs_steps != 1:
            raise ValueError("The LingBot-VLA 2.0 adapter currently supports n_obs_steps=1")
        if self.chunk_size != 50:
            raise ValueError("The released LingBot-VLA 2.0 checkpoint uses chunk_size=50")
        if not (1 <= self.n_action_steps <= self.chunk_size):
            raise ValueError("n_action_steps must be between 1 and chunk_size")
        if self.max_state_dim != 55 or self.max_action_dim != 55:
            raise ValueError("LingBot-VLA 2.0 uses a fixed 55D canonical state/action space")
        if not (1 <= self.num_cameras <= 3):
            raise ValueError("LingBot-VLA 2.0 currently supports one to three input cameras")
        if self.moe_implementation not in {None, "eager", "fused"}:
            raise ValueError("moe_implementation must be None, 'eager', or 'fused'")
        for mapping_name, mapping, target_dim in (
            ("action_dim_mapping", self.action_dim_mapping, self.max_action_dim),
            ("state_dim_mapping", self.state_dim_mapping, self.max_state_dim),
        ):
            if mapping is None:
                continue
            for entry in mapping:
                if len(entry) != 4:
                    raise ValueError(f"{mapping_name} entries must be [src_start, src_end, dst_start, dst_end]")
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
