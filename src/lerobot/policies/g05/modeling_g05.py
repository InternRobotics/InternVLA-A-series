"""LeRobot adapter for the official Galaxea G0.5 implementation.

The official package owns the Qwen3.5 model, visual-memory modules and
ActionCodec. This adapter owns only framework concerns: configuration,
LeRobot batch conversion, optimizer integration, action queuing and portable
checkpoint sidecars.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import Tensor

from lerobot.policies.g05.configuration_g05 import G05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


def _add_g05_source_to_path(source_path: Path | None) -> None:
    if source_path is not None:
        source = Path(source_path).expanduser().resolve()
        if not (source / "src" / "g05").is_dir():
            raise FileNotFoundError(
                f"g05_source_path must contain src/g05, got {source}. "
                "Clone https://github.com/OpenGalaxea/GalaxeaVLA and pass its path."
            )
        source_src = str(source / "src")
        if source_src not in sys.path:
            sys.path.insert(0, source_src)
    try:
        import g05  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The official GalaxeaVLA package is required for policy.type=g05. "
            "Install it or set --policy.g05_source_path=/path/to/GalaxeaVLA."
        ) from exc


class G05Policy(PreTrainedPolicy):
    """Expose official G0.5 through the common InternVLA policy contract."""

    config_class = G05Config
    name = "g05"

    def __init__(self, config: G05Config) -> None:
        super().__init__(config)
        config.validate_features()
        _add_g05_source_to_path(config.g05_source_path)

        self._resolve_sidecars()
        model_arch = self._load_official_model_arch()
        self.model = self._build_official_policy(model_arch)
        self._cast_official_policy()
        self.reset()

    def _resolve_sidecars(self) -> None:
        """Prefer sidecars stored beside a LeRobot checkpoint when resuming."""
        if self.config.pretrained_path is None:
            return
        checkpoint_dir = Path(self.config.pretrained_path)
        local_codec = checkpoint_dir / "action_tokenizer.pt"
        local_processor = checkpoint_dir / "hf_processor"
        if local_codec.is_file():
            self.config.action_tokenizer_path = local_codec
        if local_processor.is_dir():
            self.config.hf_processor_path = str(local_processor)

    def _official_root(self) -> Path:
        if self.config.g05_source_path is None:
            import g05

            return Path(g05.__file__).resolve().parents[2]
        return Path(self.config.g05_source_path).expanduser().resolve()

    def _load_official_model_arch(self):
        root = self._official_root()
        model_path = Path(self.config.official_model_config or root / "configs/model/g05.yaml")
        tokenizer_path = Path(
            self.config.official_tokenizer_config or root / "configs/tokenizer/actioncodec.yaml"
        )
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Missing official G0.5 config: model={model_path}, tokenizer={tokenizer_path}"
            )

        from g05.utils.config.config_resolvers import register_default_resolvers

        register_default_resolvers()
        model_cfg = OmegaConf.load(model_path)
        tokenizer_cfg = OmegaConf.load(tokenizer_path)
        root_cfg = OmegaConf.create(
            {
                "model": model_cfg,
                "data": {"action_size": self.config.chunk_size, "obs_size": self.config.n_obs_steps},
            }
        )
        root_cfg.model.tokenizer = tokenizer_cfg
        arch = root_cfg.model.model_arch

        arch._target_ = "g05.models.g05.g05_policy_qwen35.G05PolicyQwen35"
        arch.horizon_steps = self.config.chunk_size
        arch.cond_steps = self.config.n_obs_steps
        arch.num_obs_steps = self.config.n_obs_steps
        arch.action_dim = self.config.max_action_dim
        arch.proprio_dim = self.config.max_state_dim
        arch.num_input_images = self.config.num_cameras * self.config.n_obs_steps
        arch.camera_size_config = {
            "exterior": list(self.config.image_resolution),
            "wrist_left": list(self.config.image_resolution),
            "wrist_right": list(self.config.image_resolution),
        }
        arch.discrete_action = self.config.discrete_action
        arch.continuous_action = self.config.continuous_action
        arch.return_continuous_action = self.config.continuous_action
        arch.predict_cot = self.config.predict_cot
        arch.proprio_encoder = self.config.proprio_encoder
        arch.checkpoint_vision = self.config.gradient_checkpointing
        arch.checkpoint_vlm = self.config.gradient_checkpointing
        arch.checkpoint_action_expert = self.config.gradient_checkpointing and self.config.continuous_action
        arch.ar.use_fused_ce = self.config.use_fused_ce
        arch.ar.block_wise_autoregressive = self.config.block_wise_autoregressive
        arch.fm.horizon_steps = self.config.chunk_size
        arch.fm.action_dim = self.config.max_action_dim
        arch.action_expert.input_dim = self.config.max_action_dim
        arch.action_expert.output_dim = self.config.max_action_dim
        arch.hf_processor_path = self.config.hf_processor_path
        arch.pretrained_model_path = None

        tokenizer_cfg.vq_config.ckpt_dir = str(self.config.action_tokenizer_path)
        tokenizer_cfg.vq_config.parts_meta = dict(self.config.action_parts_meta)
        tokenizer_cfg.vq_config.model_arch.horizon = self.config.chunk_size
        tokenizer_cfg.vq_config.rule_based_key_patterns = list(self.config.gripper_parts)
        tokenizer_cfg.vq_config.num_residuals = self.config.action_codec_num_residuals
        tokenizer_cfg.vq_config.block_wise_autoregressive = self.config.block_wise_autoregressive
        tokenizer_cfg.vq_config.use_group_markers = self.config.use_group_markers
        tokenizer_cfg.vq_config.group_order_shuffle = False
        tokenizer_cfg.vq_config.dropout_noop_parts = self.config.dropout_noop_parts
        tokenizer_cfg.vq_config.absent_key_fill_value = self.config.absent_key_fill_value
        arch.action_tokenizer = tokenizer_cfg._target_
        arch.AT_CONFIG = tokenizer_cfg.vq_config

        if not arch.hf_processor_path:
            raise ValueError("hf_processor_path is required for G0.5")
        if not self.config.action_tokenizer_path:
            raise ValueError("action_tokenizer_path is required for G0.5")
        # Resolve only model_arch. Resolving the entire official config would
        # also evaluate processor-side ``oc.load`` paths relative to this repo.
        return OmegaConf.create(OmegaConf.to_container(arch, resolve=True))

    def _build_official_policy(self, model_arch):
        official_checkpoint = self.config.official_pretrained_checkpoint
        is_lerobot_reload = self.config.pretrained_path is not None
        if official_checkpoint is not None and not is_lerobot_reload:
            checkpoint = Path(official_checkpoint).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"G0.5 checkpoint not found: {checkpoint}")
            from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint

            return load_model_from_checkpoint(
                model_arch,
                str(checkpoint),
                device="cpu",
                use_meta_device=False,
                eval_mode=False,
            )

        from hydra.utils import instantiate

        if not is_lerobot_reload:
            raise ValueError(
                "official_pretrained_checkpoint is required for first-time G0.5 fine-tuning. "
                "Only a LeRobot checkpoint reload may omit it."
            )
        return instantiate(model_arch)

    def _cast_official_policy(self) -> None:
        if self.config.dtype == "bfloat16":
            self.model.to(torch.bfloat16)
        self.model.apply_fp32_params()

    def __str__(self) -> str:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return (
            f"G05Policy(total_parameters={total:,}, trainable_parameters={trainable:,}, "
            f"discrete_action={self.config.discrete_action}, "
            f"continuous_action={self.config.continuous_action})"
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        save_directory = Path(save_directory)
        codec = Path(self.config.action_tokenizer_path)
        if codec.is_file():
            shutil.copy2(codec, save_directory / "action_tokenizer.pt")
        processor = Path(str(self.config.hf_processor_path))
        if processor.is_dir():
            destination = save_directory / "hf_processor"
            destination.mkdir(exist_ok=True)
            allowlist = {
                "config.json",
                "configuration.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "generation_config.json",
                "chat_template.jinja",
                "video_preprocessor_config.json",
                "merges.txt",
                "vocab.json",
            }
            for filename in allowlist:
                source = processor / filename
                if source.is_file():
                    shutil.copy2(source, destination / filename)

    def state_dict(self, *args, **kwargs):
        """Exclude ActionCodec, which is persisted as an immutable sidecar.

        The official action tokenizer is not trained by the G0.5 policy and is
        already copied as ``action_tokenizer.pt``. Excluding its duplicate
        parameters keeps LeRobot checkpoints smaller and avoids alias warnings.
        """
        state = super().state_dict(*args, **kwargs)
        codec_prefix = "model.action_tokenizer.action_tokenizer."
        return {key: value for key, value in state.items() if not key.startswith(codec_prefix)}

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if hasattr(self, "model") and hasattr(self.model, "action_tokenizer"):
            device = next(self.parameters()).device
            # The official tokenizer is a plain Python wrapper rather than a
            # registered nn.Module. Its ``to`` method moves the codec and also
            # updates the device used when it constructs token tensors.
            self.model.action_tokenizer.to(device)
        return result

    def get_optim_params(self):
        return self.model.get_optim_param_groups(
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            backbone_lr_multiplier=self.config.backbone_lr_multiplier,
            vision_lr_multiplier=self.config.vision_lr_multiplier,
        )

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @staticmethod
    def _map_features(
        value: Tensor,
        target_dim: int,
        mapping: list[list[int]] | None,
    ) -> tuple[Tensor, Tensor]:
        output = value.new_zeros((*value.shape[:-1], target_dim))
        is_pad = torch.ones((value.shape[0], target_dim), dtype=torch.bool, device=value.device)
        if mapping is None:
            copied = min(value.shape[-1], target_dim)
            output[..., :copied] = value[..., :copied]
            is_pad[:, :copied] = False
            return output, is_pad

        for src_start, src_end, dst_start, dst_end in mapping:
            if src_end > value.shape[-1]:
                raise ValueError(
                    f"Mapping source [{src_start}:{src_end}] exceeds input dimension {value.shape[-1]}"
                )
            output[..., dst_start:dst_end] = value[..., src_start:src_end]
            is_pad[:, dst_start:dst_end] = False
        return output, is_pad

    def _unmap_actions(self, action: Tensor) -> Tensor:
        mapping = self.config.action_dim_mapping
        if mapping is None:
            raw_dim = self.config.environment_action_dim or action.shape[-1]
            return action[..., :raw_dim]
        raw_dim = self.config.environment_action_dim or max(entry[1] for entry in mapping)
        output = action.new_zeros((*action.shape[:-1], raw_dim))
        for src_start, src_end, dst_start, dst_end in mapping:
            output[..., src_start:src_end] = action[..., dst_start:dst_end]
        return output

    @staticmethod
    def _as_string_list(value: Any, batch_size: int) -> list[str]:
        if isinstance(value, str):
            return [value] * batch_size
        if value is None:
            return [""] * batch_size
        return [str(item) for item in value]

    def _make_official_batch(self, batch: dict[str, Any], *, training: bool) -> dict[str, Any]:
        state = batch[OBS_STATE]
        if state.ndim == 2:
            state = state.unsqueeze(1)
        state, state_dim_is_pad = self._map_features(
            state, self.config.max_state_dim, self.config.state_dim_mapping
        )
        if self.config.state_dim_mapping is None and self.config.active_state_dims is not None:
            state_dim_is_pad[:] = True
            offset = 0
            for part_size, active_size in zip(
                self.config.action_parts_meta.values(),
                self.config.active_state_dims,
                strict=True,
            ):
                state_dim_is_pad[:, offset : offset + min(part_size, active_size)] = False
                offset += part_size
        batch_size = state.shape[0]
        tasks = self._as_string_list(batch.get("task"), batch_size)

        images = []
        for index in range(self.config.num_cameras):
            key = f"{OBS_IMAGES}.image{index}"
            image = batch[key]
            if image.ndim == 5 and image.shape[1] == 1:
                image = image[:, 0]
            mask = batch.get(f"{key}_mask")
            if mask is not None:
                mask = torch.as_tensor(mask, device=image.device, dtype=torch.bool).reshape(batch_size)
                image = torch.where(mask[:, None, None, None], image, torch.zeros_like(image))
            images.append(image.to(torch.float32) * 2.0 - 1.0)
        pixel_values = torch.stack(images, dim=1)

        samples: list[dict[str, Any]] = []
        template_images = "".join(f"<image{index}_image_!>" for index in range(self.config.num_cameras))
        template = (
            "<chat_user_prefix>"
            + template_images
            + "<bos>Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
            "State: <proprio_proprio_!>;<chat_user_suffix><chat_assistant_prefix>"
            "Action: <EOV><EOC><action_action>|<eos>"
        )

        action = action_dim_is_pad = action_is_pad = None
        if training:
            action, action_dim_is_pad = self._map_features(
                batch[ACTION], self.config.max_action_dim, self.config.action_dim_mapping
            )
            if self.config.action_dim_mapping is None and self.config.active_action_dims is not None:
                action_dim_is_pad[:] = True
                offset = 0
                for part_size, active_size in zip(
                    self.config.action_parts_meta.values(),
                    self.config.active_action_dims,
                    strict=True,
                ):
                    action_dim_is_pad[:, offset : offset + min(part_size, active_size)] = False
                    offset += part_size
            action_is_pad = batch.get("action_is_pad")
            if action_is_pad is None:
                action_is_pad = torch.zeros(action.shape[:2], dtype=torch.bool, device=action.device)
            else:
                action_is_pad = torch.as_tensor(action_is_pad, device=action.device, dtype=torch.bool)

        for index in range(batch_size):
            sample = {
                "template": template,
                "command": tasks[index],
                "embodiment": self.config.embodiment,
                "proprio": {
                    "value": state[index],
                    "proprio_dim_is_pad": state_dim_is_pad[index],
                },
            }
            for camera in range(self.config.num_cameras):
                sample[f"image{camera}"] = tuple(self.config.image_resolution)
            if training:
                sample["action"] = {
                    "value": action[index],
                    "action_dim_is_pad": action_dim_is_pad[index],
                    "action_op_mask": ~action_dim_is_pad[index],
                    "parts_meta": dict(self.config.action_parts_meta),
                }
            samples.append(sample)

        result = {
            "samples": samples,
            "pixel_values": pixel_values,
            "proprio": state,
            "action_dim_is_pad": action_dim_is_pad,
        }
        if training:
            result["action"] = action
            result["action_is_pad"] = action_is_pad
        return result

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        official_batch = self._make_official_batch(batch, training=True)
        loss, official_metrics = self.model(official_batch, inference_mode=False)
        metrics = {
            name: value.detach() if torch.is_tensor(value) else value
            for name, value in official_metrics.items()
        }
        if "ce_loss" in metrics:
            metrics["loss_action"] = metrics["ce_loss"]
        elif "fm_loss" in metrics:
            metrics["loss_action"] = metrics["fm_loss"]
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> Tensor:
        del kwargs
        was_training = self.training
        self.eval()
        official_batch = self._make_official_batch(batch, training=False)
        predicted = self.model(official_batch, inference_mode=True)["action"]
        self.train(was_training)
        return self._unmap_actions(predicted)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
