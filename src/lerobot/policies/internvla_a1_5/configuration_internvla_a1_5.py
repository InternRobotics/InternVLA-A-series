import logging
from dataclasses import dataclass, field, replace
from typing import Sequence

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    CaptureOPDInputsTransformFn,
    ExtractVideoFramesTransformFn,
    FASTInternVLAA15ActionTokenizerTransformFn,
    InternVLAA15ChatProcessorTransformFn,
    InternVLAA15VQAProcessorTransformFn,
)
from lerobot.opd import (
    OPD_IMAGE_MASK,
    OPD_IMAGE_PREFIX,
    OPD_IS_ON_POLICY,
    OPD_RAW_STATE,
    OPD_ROBOT_TYPE,
    OPD_TASK,
)
from lerobot.rl import RL_ADVANTAGE, RL_IS_ON_POLICY, RL_REWARD, RL_REWARD_MASK
from lerobot.transforms.core import *
from lerobot.utils.constants import HF_HOME, OBS_IMAGES


@DatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenize_state: bool = False
    max_prompt_length: int = 650
    mode: str = "train"
    chunk_size: int = 50
    use_fast_action_tokens: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    include_opd_inputs: bool = False
    opd_rollout_dataset: bool = False
    opd_num_views: int = 3
    include_rl_signals: bool = False
    rl_rollout_dataset: bool = False
    rl_advantage_key: str | None = None

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=InternVLAA15DatasetConfig.height,
                    width=InternVLAA15DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                ExtractVideoFramesTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                FASTInternVLAA15ActionTokenizerTransformFn(),
                LoadActionTextFromJsonlTransformFn(),
                InternVLAA15ChatProcessorTransformFn(),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15DatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15DatasetConfig.max_action_dim,
                ),
                ReorderStateActionTransform(),
                UnifyInternVLAA15InputsTransformFn(
                    num_video_frames=InternVLAA15DatasetConfig.num_video_frames,
                    video_height=InternVLAA15DatasetConfig.video_height,
                    video_width=InternVLAA15DatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        if self.opd_num_views < 1:
            raise ValueError("opd_num_views must be >= 1")
        inputs = list(self.data_transforms.inputs)
        inputs = [t for t in inputs if not isinstance(t, CaptureOPDInputsTransformFn)]
        if self.include_opd_inputs:
            inputs.insert(
                0,
                CaptureOPDInputsTransformFn(
                    is_on_policy=self.opd_rollout_dataset,
                    max_views=self.opd_num_views,
                ),
            )
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta" and not has_delta:
            logging.info("action_mode='delta' -> Adding DeltaActionTransformFn")
            inputs = [DeltaActionTransformFn(), *inputs]
        elif self.action_mode == "abs" and has_delta:
            logging.info("action_mode='abs' -> Removing DeltaActionTransformFn")
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]

        processor = InternVLAA15ChatProcessorTransformFn(
            tokenize_state=self.tokenize_state,
            max_state_dim=self.max_state_dim,
            max_length=self.max_prompt_length,
            use_fast_action_tokens=self.use_fast_action_tokens,
            mode=self.mode,
        )
        inputs = [t for t in inputs if not isinstance(t, InternVLAA15ChatProcessorTransformFn)]
        insert_idx = next(
            (i for i, t in enumerate(inputs) if isinstance(t, PadStateAndActionTransformFn)),
            len(inputs),
        )
        inputs.insert(insert_idx, processor)

        has_fast = any(isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn) for t in inputs)
        if self.use_fast_action_tokens and not has_fast:
            inputs.insert(insert_idx, FASTInternVLAA15ActionTokenizerTransformFn())
        elif not self.use_fast_action_tokens and has_fast:
            inputs = [t for t in inputs if not isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn)]

        for t in inputs:
            if isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn):
                t.chunk_size = self.chunk_size
                break

        for index, transform in enumerate(inputs):
            if isinstance(transform, UnifyInternVLAA15InputsTransformFn):
                inputs[index] = replace(
                    transform,
                    include_opd_inputs=self.include_opd_inputs,
                    include_rl_signals=self.include_rl_signals,
                    rl_is_on_policy=self.rl_rollout_dataset,
                    rl_advantage_key=self.rl_advantage_key,
                )

        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@DataTransformFn.register_subclass("unify_internvla_a1_5_inputs")
@dataclass
class UnifyInternVLAA15InputsTransformFn(DataTransformFn):
    """Unify robot samples and always include video_frames for WAN.

    Always outputs observation.video_frames so that robot and VQA samples
    have identical keys and can be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    include_opd_inputs: bool = False
    include_rl_signals: bool = False
    rl_is_on_policy: bool = False
    rl_advantage_key: str | None = None

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_NONE
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_NONE, dtype=torch.long))

        video_key = "observation.video_frames"
        if video_key in data:
            video_frames = data[video_key]
        else:
            video_frames = torch.zeros(
                self.num_video_frames + 1, 3, self.video_height, self.video_width
            )

        output = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": data["vqa_type"],
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            video_key: video_frames,
        }
        if self.include_opd_inputs:
            for key in (
                OPD_RAW_STATE,
                OPD_IMAGE_MASK,
                OPD_TASK,
                OPD_ROBOT_TYPE,
                OPD_IS_ON_POLICY,
            ):
                output[key] = data[key]
            for view_index in range(data[OPD_IMAGE_MASK].numel()):
                key = f"{OPD_IMAGE_PREFIX}{view_index}"
                output[key] = data[key]
        if self.include_rl_signals:
            from lerobot.utils.constants import REWARD

            if REWARD not in data:
                raise KeyError(
                    f"RL training requires {REWARD!r} in the student rollout dataset"
                )
            rewards = torch.as_tensor(data[REWARD], dtype=torch.float32)
            is_pad = data.get(f"{REWARD}_is_pad")
            if is_pad is None:
                reward_mask = torch.ones_like(rewards, dtype=torch.bool)
            else:
                reward_mask = ~torch.as_tensor(is_pad, dtype=torch.bool)
            output[RL_REWARD] = rewards
            output[RL_REWARD_MASK] = reward_mask
            output[RL_IS_ON_POLICY] = torch.tensor(self.rl_is_on_policy, dtype=torch.bool)
            if self.rl_advantage_key is not None:
                if self.rl_advantage_key not in data:
                    raise KeyError(
                        f"Configured RL advantage key {self.rl_advantage_key!r} is missing"
                    )
                output[RL_ADVANTAGE] = torch.as_tensor(
                    data[self.rl_advantage_key], dtype=torch.float32
                )
        return output


@DataTransformFn.register_subclass("unify_internvla_a1_5_vqa_inputs")
@dataclass
class UnifyInternVLAA15VQAInputsTransformFn(DataTransformFn):
    """VQA unify transform that includes a dummy video_frames tensor.

    Ensures VQA samples have the same keys as robot samples so they can
    be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_TEXT
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_TEXT, dtype=torch.long))

        video_frames = torch.zeros(
            self.num_video_frames + 1, 3, self.video_height, self.video_width
        )

        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": torch.tensor(1, dtype=torch.long),
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            "observation.video_frames": video_frames,
        }


@VQADatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15VQADatasetConfig(VQADatasetConfig):
    """VQA dataset config with dummy video_frames for shared batching."""

    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # Note: If you resize the VQA images, the coordinates in the VQA Data should be scaled accordingly. 
                # Please pre-process the VQA data offline based on its format.
                ResizeVQAImagesWithPadFn(
                    height=InternVLAA15VQADatasetConfig.height,
                    width=InternVLAA15VQADatasetConfig.width,
                ),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15VQADatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15VQADatasetConfig.max_action_dim,
                ),
                InternVLAA15VQAProcessorTransformFn(),
                UnifyInternVLAA15VQAInputsTransformFn(
                    num_video_frames=InternVLAA15VQADatasetConfig.num_video_frames,
                    video_height=InternVLAA15VQADatasetConfig.video_height,
                    video_width=InternVLAA15VQADatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        inputs = list(self.data_transforms.inputs)
        processor_type = InternVLAA15VQAProcessorTransformFn
        inputs = [t for t in inputs if not isinstance(t, processor_type)]
        insert_idx = next(
            (
                i
                for i, t in enumerate(inputs)
                if isinstance(t, UnifyInternVLAA15VQAInputsTransformFn)
            ),
            len(inputs),
        )
        inputs.insert(insert_idx, InternVLAA15VQAProcessorTransformFn())
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15Config(PreTrainedConfig):
    # VLM model selection - supports Qwen3.5-2B/4B/8B
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"

    # Action expert customization
    action_expert_hidden_size: int | None = 1024
    action_expert_intermediate_size: int | None = 3072

    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # VQA configurations
    enable_vqa_loss: bool = True
    lambda_vqa: float = 1.0
    tokenize_state: bool = True

    # FAST action tokens
    action_token_min: int = 248077
    action_token_max: int = 250124

    # Knowledge insulation
    knowledge_insulation: bool = False
    block_action_attend_fast_tokens: bool = True

    inference_action_type: str = "fm"  # "fm" for flow matching, "fast" for fast-token supervision
    inference_backend: str = "standard"  # "standard" or "optimized"

    # Use SDPA (scaled_dot_product_attention) instead of eager attention in training
    use_sdpa: bool = False

    num_learnable_tokens: int = 50

    wan_checkpoint_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    wan_config_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    vae_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    video_precision: str = "bfloat16"

    freeze_wan_dit: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    video_loss_weight: float = 1.0
    video_loss_only: bool = False
    action_loss_only: bool = False
    freeze_learnable_tokens: bool = False

    # Continuous-action OPD. These values are populated from TrainPipelineConfig.opd.
    opd_enabled: bool = False
    opd_loss_weight: float = 1.0
    opd_sft_loss_weight: float = 0.0
    opd_student_std: float = 0.10
    opd_max_kl_per_dim: float | None = 20.0

    # Reward-weighted flow matching reinforcement learning.
    rl_enabled: bool = False
    rl_gamma: float = 0.99
    rl_reward_horizon: int | None = None
    rl_normalize_advantage: bool = True
    rl_temperature: float = 1.0
    rl_min_weight: float = 0.05
    rl_max_weight: float = 20.0
    rl_loss_weight: float = 1.0
    rl_sft_loss_weight: float = 0.1
    rl_require_on_policy: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.lambda_vqa < 0:
            raise ValueError(f"lambda_vqa must be >= 0, got {self.lambda_vqa}")
        if self.action_token_min > self.action_token_max:
            raise ValueError(
                f"action_token_min ({self.action_token_min}) must be <= action_token_max ({self.action_token_max})"
            )
        if self.inference_backend not in {"standard", "optimized"}:
            raise ValueError(
                "inference_backend must be either 'standard' or 'optimized', "
                f"got {self.inference_backend!r}"
            )
        if self.inference_backend == "optimized" and not self.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")
        if self.opd_loss_weight < 0 or self.opd_sft_loss_weight < 0:
            raise ValueError("OPD loss weights must be non-negative")
        if self.opd_student_std <= 0:
            raise ValueError("opd_student_std must be > 0")
        if self.opd_enabled and self.video_loss_only:
            raise ValueError("OPD requires action prediction, so video_loss_only must be false")
        if self.opd_enabled and self.inference_backend == "optimized":
            raise ValueError("OPD training currently requires inference_backend='standard'")
        if self.rl_temperature <= 0:
            raise ValueError("rl_temperature must be > 0")
        if not 0 <= self.rl_min_weight <= self.rl_max_weight or self.rl_max_weight == 0:
            raise ValueError("Require 0 <= rl_min_weight <= rl_max_weight and rl_max_weight > 0")
        if self.rl_loss_weight < 0 or self.rl_sft_loss_weight < 0:
            raise ValueError("RL loss weights must be non-negative")
        if self.rl_enabled and self.video_loss_only:
            raise ValueError("RL requires action prediction, so video_loss_only must be false")
        if self.rl_enabled and self.inference_backend == "optimized":
            raise ValueError("RL training currently requires inference_backend='standard'")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
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
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        if not self.rl_enabled:
            return None
        horizon = self.rl_reward_horizon or self.chunk_size
        return list(range(horizon))

    @property
    def image_delta_indices(self) -> list | None:
        n = self.num_video_frames + 1
        return [self.chunk_size * i // (n - 1) for i in range(n)]
