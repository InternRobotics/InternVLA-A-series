"""LeRobot adapter for the official LingBot-VLA 2.0 implementation."""

from __future__ import annotations

import logging
import shutil
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor

from lerobot.policies.lingbot_vla_2.configuration_lingbot_vla_2 import (
    LingBotVLA2Config,
    preprocess_lingbot_vla_2_sample,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


def _apply_lingbot_vla_2_release_architecture(official_config) -> None:
    """Restore architecture values omitted from the public checkpoint config.

    The released ModelScope/Hugging Face ``config.json`` only declares the
    Qwen3-VL family. The actual V2 checkpoint uses the sparse-MoE/55D settings
    published in the official training YAML; leaving the Python defaults in
    place creates a dense 14D model whose parameter names and shapes cannot
    match the six released safetensor shards.
    """
    release_values = {
        "post_training": True,
        "adanorm_time": True,
        "vlm_causal": True,
        "use_moe": True,
        "token_moe_layers": list(range(36)),
        "token_num_experts": 32,
        "token_top_k": 4,
        "token_moe_intermediate_size": 512,
        "token_shared_intermediate_size": 704,
        "bias_update_speed": 0.0,
        "sequence_wise_mode": "per_sequence",
        "sequence_wise_loss_coeff": 1e-3,
        "router_z_loss_coeff": 1e-4,
        "router_activation": "sigmoid",
        "routed_scaling_factor": 4.0,
        "use_shared_expert_gate": False,
        "loss_type": "L1_fm",
    }
    for name, value in release_values.items():
        setattr(official_config, name, value)


def _set_lingbot_attention_implementations(model, text_impl: str, vision_impl: str) -> None:
    """Undo the official constructor's unconditional FlashAttention override."""
    wrapper = model.model.qwenvl_with_expert
    qwen = wrapper.qwenvl
    qwen.config._attn_implementation = text_impl
    qwen.config.text_config._attn_implementation = text_impl
    qwen.config.vision_config._attn_implementation = vision_impl
    wrapper.qwen_expert.config._attn_implementation = text_impl


def _add_lingbot_vla_source_to_path(source_path: Path | None) -> None:
    if source_path is not None:
        source = Path(source_path).expanduser().resolve()
        if not (source / "lingbotvla").is_dir():
            raise FileNotFoundError(
                f"lingbot_vla_source_path must contain lingbotvla/, got {source}. "
                "Clone https://github.com/Robbyant/lingbot-vla-v2 and pass its path."
            )
        source_string = str(source)
        if source_string not in sys.path:
            sys.path.insert(0, source_string)
    try:
        import lingbotvla  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The official lingbot-vla-v2 package is required for policy.type=lingbot_vla_2. "
            "Install it or set --policy.lingbot_vla_source_path=/path/to/lingbot-vla-v2."
        ) from exc


class LingBotVLA2Policy(PreTrainedPolicy):
    """Expose official LingBot-VLA 2.0 through the common InternVLA contract."""

    config_class = LingBotVLA2Config
    name = "lingbot_vla_2"

    def __init__(self, config: LingBotVLA2Config) -> None:
        super().__init__(config)
        config.validate_features()
        _add_lingbot_vla_source_to_path(config.lingbot_vla_source_path)
        self._resolve_sidecars()
        self.processor = self._load_processor()
        official_config = self._load_official_config()
        self.model = self._build_official_policy(official_config)
        self._cast_official_policy()
        self._configure_moe_load_balancing()
        self.reset()

    def _resolve_sidecars(self) -> None:
        if self.config.pretrained_path is None:
            return
        checkpoint_dir = Path(self.config.pretrained_path)
        processor_dir = checkpoint_dir / "qwen_processor"
        official_config_dir = checkpoint_dir / "official_model_config"
        if processor_dir.is_dir():
            self.config.qwen_processor_path = str(processor_dir)
        if official_config_dir.is_dir():
            self.config.official_config_path = str(official_config_dir)

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            self.config.qwen_processor_path,
            padding_side="right",
            trust_remote_code=True,
        )

    def _load_official_config(self):
        from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
            LingbotVLAV2Config,
        )

        config_source = self.config.official_config_path or self.config.official_pretrained_model
        if config_source is None:
            raise ValueError(
                "official_pretrained_model (or official_config_path when resuming) is required "
                "for LingBot-VLA 2.0"
            )
        official_config = LingbotVLAV2Config.from_pretrained(str(config_source))
        _apply_lingbot_vla_2_release_architecture(official_config)
        official_config.tokenizer_path = self.config.qwen_processor_path
        official_config.action_dim = self.config.max_action_dim
        official_config.max_action_dim = self.config.max_action_dim
        official_config.max_state_dim = self.config.max_state_dim
        official_config.chunk_size = self.config.chunk_size
        official_config.n_action_steps = self.config.chunk_size
        official_config.num_steps = self.config.num_inference_steps
        official_config.tokenizer_max_length = self.config.tokenizer_max_length
        official_config.return_image_grid_thw = True
        official_config.qwen3vl_use_vision_boundaries = True
        official_config.use_cache = False
        if self.config.attention_implementation is not None:
            official_config.attention_implementation = self.config.attention_implementation
        if self.config.vit_attn_implementation is not None:
            official_config.vit_attn_implementation = self.config.vit_attn_implementation
        if self.config.moe_implementation is not None:
            official_config.moe_implementation = self.config.moe_implementation
            official_config._moe_implementation = self.config.moe_implementation
        if self.config.disable_auxiliary_distillation:
            # Action-only post-training needs no MoGe/LingBot-Depth/DINO teachers.
            # The corresponding checkpoint tensors are intentionally ignored.
            official_config.align_params = {}
        return official_config

    def _build_official_policy(self, official_config):
        from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
            LingbotVlaV2Policy,
        )
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
            apply_lingbot_qwen3_vl_patch,
        )

        apply_lingbot_qwen3_vl_patch()
        is_lerobot_reload = self.config.pretrained_path is not None
        if is_lerobot_reload:
            model = LingbotVlaV2Policy(official_config, eval=False)
        else:
            model_source = self.config.official_pretrained_model
            if model_source is None:
                raise ValueError("official_pretrained_model is required for first-time post-training")
            text_impl = self.config.attention_implementation
            vision_impl = self.config.vit_attn_implementation
            use_non_flash_smoke_path = text_impl in {"eager", "sdpa"} or vision_impl in {
                "eager",
                "sdpa",
            }
            attention_dispatch = nullcontext()
            if use_non_flash_smoke_path:
                # The official Qwen wrapper hardcodes FlashAttention during
                # construction even when its public config requests eager.
                # No forward executes inside from_pretrained, so bypass the
                # availability guard just for construction, then restore every
                # runtime attention config below.
                from transformers import PreTrainedModel

                attention_dispatch = patch.object(
                    PreTrainedModel,
                    "_check_and_adjust_attn_implementation",
                    return_value="eager",
                )
            with attention_dispatch:
                model = LingbotVlaV2Policy.from_pretrained(
                    str(model_source),
                    config=official_config,
                    torch_dtype=getattr(torch, self.config.dtype),
                    low_cpu_mem_usage=True,
                )
            if use_non_flash_smoke_path:
                _set_lingbot_attention_implementations(
                    model,
                    text_impl or "eager",
                    vision_impl or text_impl or "eager",
                )
        model.config.use_cache = False
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model

    def _cast_official_policy(self) -> None:
        if self.config.dtype == "bfloat16":
            self.model.to(torch.bfloat16)

    def _configure_moe_load_balancing(self) -> None:
        """Reuse LeRobot's post-step update hook for official loss-free MoE routing."""
        self._moe_load_balance_hook = None
        if not getattr(self.model.config, "use_moe", False):
            return
        from lingbotvla.models.vla.lingbot_vla.moe_load_balance import (
            build_moe_load_balance_hook,
        )

        self._moe_load_balance_hook = build_moe_load_balance_hook(
            self.model,
            coeff=float(getattr(self.model.config, "bias_update_speed", 0.0)),
            bias_centering=bool(getattr(self.model.config, "bias_centering", False)),
            update_interval=int(getattr(self.model.config, "bias_update_interval", 1)),
        )

    @torch.no_grad()
    def update(self) -> None:
        """Update/snapshot MoE routing load after each optimizer step."""
        if self._moe_load_balance_hook is not None:
            self._moe_load_balance_hook(None)

    def __str__(self) -> str:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return (
            f"LingBotVLA2Policy(total_parameters={total:,}, trainable_parameters={trainable:,}, "
            f"canonical_dims={self.config.max_action_dim}, chunk_size={self.config.chunk_size})"
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        save_directory = Path(save_directory)
        official_config_dir = save_directory / "official_model_config"
        processor_dir = save_directory / "qwen_processor"
        self.model.config.save_pretrained(official_config_dir)
        self.processor.save_pretrained(processor_dir)

        # Some custom processors keep auxiliary files that save_pretrained does
        # not copy. Preserve the small, load-critical allowlist when local.
        source = Path(str(self.config.qwen_processor_path)).expanduser()
        if source.is_dir():
            processor_dir.mkdir(exist_ok=True)
            for filename in {
                "chat_template.jinja",
                "video_preprocessor_config.json",
                "merges.txt",
                "vocab.json",
            }:
                candidate = source / filename
                if candidate.is_file() and not (processor_dir / filename).exists():
                    shutil.copy2(candidate, processor_dir / filename)

    def get_optim_params(self):
        return self.model.get_optim_params()

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @staticmethod
    def _map_features(
        value: Tensor,
        target_dim: int,
        mapping: list[list[int]] | None,
    ) -> tuple[Tensor, Tensor]:
        output = value.new_zeros((*value.shape[:-1], target_dim))
        dim_is_pad = torch.ones((value.shape[0], target_dim), dtype=torch.bool, device=value.device)
        if mapping is None:
            copied = min(value.shape[-1], target_dim)
            output[..., :copied] = value[..., :copied]
            dim_is_pad[:, :copied] = False
            return output, dim_is_pad

        for src_start, src_end, dst_start, dst_end in mapping:
            if src_end > value.shape[-1]:
                raise ValueError(
                    f"Mapping source [{src_start}:{src_end}] exceeds input dimension {value.shape[-1]}"
                )
            output[..., dst_start:dst_end] = value[..., src_start:src_end]
            dim_is_pad[:, dst_start:dst_end] = False
        return output, dim_is_pad

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

    def _preprocess_raw_batch(self, batch: dict[str, Any], batch_size: int) -> dict[str, Tensor]:
        tasks = self._as_string_list(batch.get("task"), batch_size)
        samples = []
        for sample_index in range(batch_size):
            images = []
            masks = []
            for camera_index in range(self.config.num_cameras):
                key = f"{OBS_IMAGES}.image{camera_index}"
                images.append(batch[key][sample_index])
                mask = batch.get(f"{key}_mask", True)
                masks.append(mask[sample_index] if torch.is_tensor(mask) and mask.ndim else mask)
            samples.append(
                preprocess_lingbot_vla_2_sample(
                    self.processor,
                    images,
                    masks,
                    tasks[sample_index],
                    self.config.tokenizer_max_length,
                )
            )
        return {
            key: torch.stack([sample[key] for sample in samples]).to(batch[OBS_STATE].device)
            for key in samples[0]
        }

    def _make_official_batch(self, batch: dict[str, Any], *, training: bool) -> dict[str, Tensor]:
        state, _ = self._map_features(
            batch[OBS_STATE], self.config.max_state_dim, self.config.state_dim_mapping
        )
        batch_size = state.shape[0]
        if {"images", "img_masks", "image_grid_thw", "lang_tokens", "lang_masks"}.issubset(batch):
            model_inputs = {
                key: batch[key]
                for key in ("images", "img_masks", "image_grid_thw", "lang_tokens", "lang_masks")
            }
        else:
            model_inputs = self._preprocess_raw_batch(batch, batch_size)

        dtype = next(self.model.parameters()).dtype
        result = {
            **model_inputs,
            "images": model_inputs["images"].to(dtype=dtype),
            "img_masks": model_inputs["img_masks"].to(torch.bool),
            "image_grid_thw": model_inputs["image_grid_thw"].to(torch.long),
            "lang_tokens": model_inputs["lang_tokens"].to(torch.long),
            "lang_masks": model_inputs["lang_masks"].to(torch.bool),
            "state": state.to(dtype=dtype),
        }
        if training:
            actions, action_dim_is_pad = self._map_features(
                batch[ACTION], self.config.max_action_dim, self.config.action_dim_mapping
            )
            # The final four slots in the official 55D representation are
            # reserved padding, even when an upstream dataset tensor is 55D.
            action_dim_is_pad[:, 51:] = True
            action_is_pad = batch.get("action_is_pad")
            if action_is_pad is None:
                action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
            else:
                action_is_pad = torch.as_tensor(action_is_pad, device=actions.device, dtype=torch.bool)
            joint_mask = (~action_dim_is_pad)[:, None, :].expand_as(actions)
            joint_mask = joint_mask & (~action_is_pad[:, :, None])
            result.update(
                actions=actions.to(dtype=dtype),
                action_is_pad=action_is_pad,
                joint_mask=joint_mask,
            )
        return result

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        outputs = self.model(**self._make_official_batch(batch, training=True))
        total_loss, action_loss = outputs[0], outputs[1]
        metrics: dict[str, Any] = {"loss_action": action_loss.detach()}
        if len(outputs) > 6 and isinstance(outputs[6], dict):
            for name, value in outputs[6].items():
                if torch.is_tensor(value) and value.numel() == 1:
                    metrics[name] = value.detach()
                elif isinstance(value, (float, int)):
                    metrics[name] = value
        return total_loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> Tensor:
        del kwargs
        was_training = self.training
        self.eval()
        official_batch = self._make_official_batch(batch, training=False)
        previous_cache = bool(self.model.config.use_cache)
        self.model.config.use_cache = True
        try:
            predicted = self.model.sample_actions(
                official_batch["images"],
                official_batch["img_masks"],
                official_batch["lang_tokens"],
                official_batch["lang_masks"],
                official_batch["state"],
                image_grid_thw=official_batch["image_grid_thw"],
            )
        finally:
            self.model.config.use_cache = previous_cache
            self.train(was_training)
        return self._unmap_actions(predicted)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
