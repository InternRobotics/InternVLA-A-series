#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import logging
import math
from collections import deque
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
import torch._dynamo as dynamo
from torch import Tensor, nn

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextModel,
    Qwen3_5Config,
    Qwen3_5Tokenizer,
)

from lerobot.policies.internvla_a1_5.action_tokens import ensure_qwen35_action_tokens
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.wan_model import WanVideoModel
from lerobot.policies.internvla_a1_5.wan.modules.model import sinusoidal_embedding_1d
from lerobot.policies.internvla_a1_5.wan.utils.fm import FlowMatchScheduler
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    LABEL_MODE_FAST,
    LABEL_MODE_NONE,
    LABEL_MODE_TEXT,
)
from lerobot.opd import (
    OPD_TEACHER_ACTION_MASK,
    OPD_TEACHER_ACTION_MEAN,
    OPD_TEACHER_ACTION_STD,
)
from lerobot.opd.losses import diagonal_gaussian_reverse_kl
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl import RL_ADVANTAGE, RL_GROUP_ID, RL_IS_ON_POLICY, RL_REWARD, RL_REWARD_MASK
from lerobot.rl.flow_rl import (
    group_relative_flow_matching_loss,
    reward_weighted_flow_matching_loss,
)
from lerobot.utils.utils import format_big_number
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OBS_PREFIX,
    OBS_STR,
    OPENPI_ATTENTION_MASK_VALUE,
)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    qwen3_5,
    action_expert,
    prefix_len: int,
    knowledge_insulation: bool = False,
    use_sdpa: bool = False,
    linear_attn_mask: torch.Tensor | None = None,
):
    """Run one transformer layer jointly on [VLM prefix, action expert suffix].

    When ``knowledge_insulation`` is True, the suffix (action expert) queries
    attend to the prefix keys/values with gradient detached, so gradients from
    the action / suffix branch cannot flow back into the VLM.

    ``linear_attention`` layers are processed fully independently per model –
    their recurrent state cannot be shared – so KI is naturally enforced there.

    Args:
        linear_attn_mask: 2D [B, total_len] padding mask for linear attention
            layers (1=real, 0=padding/masked). If None, no masking is applied.
    """

    models = [qwen3_5.language_model, action_expert]
    layer_type = qwen3_5.language_model.layers[layer_idx].layer_type

    if layer_type == "linear_attention":
        if linear_attn_mask is not None:
            prefix_linear_mask = linear_attn_mask[:, :prefix_len]
            suffix_linear_mask = linear_attn_mask[:, prefix_len:]
            linear_masks_per_model = [prefix_linear_mask, suffix_linear_mask]
        else:
            linear_masks_per_model = [None, None]

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.linear_attn(
                hidden_states=hidden_states,
                cache_params=None,
                cache_position=None,
                # Linear attention expects a 2D padding mask here.
                attention_mask=linear_masks_per_model[i],
            )
            hidden_states = residual + hidden_states

            after_first_residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype=torch.bfloat16)
            hidden_states = layer.mlp(hidden_states)

            hidden_states = hidden_states + after_first_residual
            outputs_embeds.append(hidden_states)

        return outputs_embeds

    elif layer_type == "full_attention":
        # Compute Q/K/V/gate for each branch separately.
        query_states = []
        key_states = []
        value_states = []
        gates = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]

            q_gate = layer.self_attn.q_proj(hidden_states).view(
                *input_shape, -1, layer.self_attn.head_dim * 2
            )
            query_state, gate = torch.chunk(q_gate, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)

            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_state = layer.self_attn.q_norm(query_state.view(hidden_shape)).transpose(1, 2)
            key_state = layer.self_attn.k_norm(
                layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)
            gates.append(gate)

        prefix_query, suffix_query = query_states
        prefix_key, suffix_key = key_states
        prefix_value, suffix_value = value_states
        prefix_gate, suffix_gate = gates

        # Joint K/V/Q for applying rotary position embedding on the full sequence.
        joint_query = torch.cat(query_states, dim=2)
        joint_key = torch.cat(key_states, dim=2)
        joint_value = torch.cat(value_states, dim=2)

        dummy_tensor = torch.zeros(
            joint_query.shape[0],
            joint_query.shape[2],
            joint_query.shape[-1],
            device=joint_query.device,
            dtype=joint_query.dtype,
        )
        cos, sin = qwen3_5.language_model.rotary_emb(dummy_tensor, position_ids)
        joint_query, joint_key = modeling_qwen3_5.apply_rotary_pos_emb(
            joint_query, joint_key, cos, sin, unsqueeze_dim=1
        )

        # Split back after RoPE.
        prefix_query = joint_query[:, :, :prefix_len]
        suffix_query = joint_query[:, :, prefix_len:]
        prefix_key = joint_key[:, :, :prefix_len]
        suffix_key = joint_key[:, :, prefix_len:]
        prefix_value = joint_value[:, :, :prefix_len]
        suffix_value = joint_value[:, :, prefix_len:]

        scaling = qwen3_5.language_model.layers[layer_idx].self_attn.scaling
        attn_layer = qwen3_5.language_model.layers[layer_idx].self_attn

        batch_size = joint_query.shape[0]

        # --- prefix queries: attend only to prefix K/V (pad mask already
        # forbids prefix -> suffix). Gradient flows normally for VLM.
        prefix_attn_mask = attention_mask[:, :, :prefix_len, :prefix_len]
        if use_sdpa:
            prefix_key_expanded = modeling_qwen3_5.repeat_kv(prefix_key, attn_layer.num_key_value_groups)
            prefix_value_expanded = modeling_qwen3_5.repeat_kv(prefix_value, attn_layer.num_key_value_groups)
            prefix_att_output = F.scaled_dot_product_attention(
                prefix_query, prefix_key_expanded, prefix_value_expanded,
                attn_mask=prefix_attn_mask.to(prefix_query.dtype), scale=scaling,
            )
            prefix_att_output = prefix_att_output.transpose(1, 2).contiguous()
        else:
            prefix_att_output, _ = modeling_qwen3_5.eager_attention_forward(
                attn_layer,
                prefix_query,
                prefix_key,
                prefix_value,
                prefix_attn_mask,
                scaling,
            )

        # --- suffix queries: attend to [prefix (maybe-detached) K/V, suffix K/V].
        if knowledge_insulation:
            prefix_key_for_suffix = prefix_key.detach()
            prefix_value_for_suffix = prefix_value.detach()
        else:
            prefix_key_for_suffix = prefix_key
            prefix_value_for_suffix = prefix_value

        k_for_suffix = torch.cat([prefix_key_for_suffix, suffix_key], dim=2)
        v_for_suffix = torch.cat([prefix_value_for_suffix, suffix_value], dim=2)
        suffix_attn_mask = attention_mask[:, :, prefix_len:, :]

        if use_sdpa:
            k_for_suffix_expanded = modeling_qwen3_5.repeat_kv(k_for_suffix, attn_layer.num_key_value_groups)
            v_for_suffix_expanded = modeling_qwen3_5.repeat_kv(v_for_suffix, attn_layer.num_key_value_groups)
            suffix_att_output = F.scaled_dot_product_attention(
                suffix_query, k_for_suffix_expanded, v_for_suffix_expanded,
                attn_mask=suffix_attn_mask.to(suffix_query.dtype), scale=scaling,
            )
            suffix_att_output = suffix_att_output.transpose(1, 2).contiguous()
        else:
            suffix_att_output, _ = modeling_qwen3_5.eager_attention_forward(
                attn_layer,
                suffix_query,
                k_for_suffix,
                v_for_suffix,
                suffix_attn_mask,
                scaling,
            )

        att_output = torch.cat([prefix_att_output, suffix_att_output], dim=1)

        head_dim = qwen3_5.language_model.layers[layer_idx].self_attn.head_dim
        num_attention_heads = qwen3_5.language_model.layers[layer_idx].self_attn.config.num_attention_heads
        att_output = att_output.reshape(batch_size, -1, num_attention_heads * head_dim)

        gates_joint = torch.cat(gates, dim=1)

        outputs_embeds = []
        start_pos = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = models[i].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]

            att_out_slice = att_output[:, start_pos:end_pos]
            gate_slice = gates_joint[:, start_pos:end_pos]

            att_out_slice = att_out_slice * torch.sigmoid(gate_slice)

            if att_out_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                att_out_slice = att_out_slice.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_out_slice)

            out_emb = out_emb + hidden_states
            after_first_residual = out_emb.clone()
            out_emb = layer.post_attention_layernorm(out_emb)

            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)

            out_emb = out_emb + after_first_residual
            outputs_embeds.append(out_emb)
            start_pos = end_pos
        return outputs_embeds

    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")


class ActionExpertConfig:
    """Configuration for the action expert module.

    Note: head_dim is inherited from VLM for proper cross-attention computation.
    Only hidden_size and intermediate_size can be customized.
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        head_dim: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
    ):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


class InternVLAA15WithExpertModel(nn.Module):
    """Qwen3_5 model with action expert for InternVLAA15."""

    def __init__(
        self,
        vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B",
        action_expert_config: ActionExpertConfig | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.qwen3_5 = Qwen3_5ForConditionalGeneration.from_pretrained(vlm_model_name_or_path)
            tokenizer = Qwen3_5Tokenizer.from_pretrained(vlm_model_name_or_path)
            ensure_qwen35_action_tokens(tokenizer, self.qwen3_5)

        vlm_text_config = self.qwen3_5.config.text_config

        if action_expert_config is None:
            action_expert_config = ActionExpertConfig()

        if action_expert_config.head_dim is None:
            action_expert_config.head_dim = vlm_text_config.head_dim
        if action_expert_config.hidden_size is None:
            action_expert_config.hidden_size = vlm_text_config.hidden_size
        if action_expert_config.intermediate_size is None:
            action_expert_config.intermediate_size = vlm_text_config.intermediate_size

        action_expert_config.num_attention_heads = vlm_text_config.num_attention_heads
        action_expert_config.num_key_value_heads = vlm_text_config.num_key_value_heads

        action_expert_config_hf = CONFIG_MAPPING["qwen3_5_text"]()
        action_expert_config_hf.head_dim = action_expert_config.head_dim
        action_expert_config_hf.hidden_size = action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size = action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads = action_expert_config.num_attention_heads
        action_expert_config_hf.num_key_value_heads = action_expert_config.num_key_value_heads
        action_expert_config_hf.num_hidden_layers = vlm_text_config.num_hidden_layers
        action_expert_config_hf.max_position_embeddings = vlm_text_config.max_position_embeddings
        action_expert_config_hf.rope_parameters = vlm_text_config.rope_parameters
        action_expert_config_hf.rms_norm_eps = vlm_text_config.rms_norm_eps

        action_expert_config_hf.layer_types = vlm_text_config.layer_types
        action_expert_config_hf.linear_conv_kernel_dim = vlm_text_config.linear_conv_kernel_dim
        action_expert_config_hf.linear_key_head_dim = vlm_text_config.linear_key_head_dim
        action_expert_config_hf.linear_value_head_dim = vlm_text_config.linear_value_head_dim
        action_expert_config_hf.linear_num_key_heads = vlm_text_config.linear_num_key_heads
        action_expert_config_hf.linear_num_value_heads = vlm_text_config.linear_num_value_heads

        self.action_expert = Qwen3_5TextModel(config=action_expert_config_hf)
        self.action_expert.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        knowledge_insulation: bool = False,
        use_sdpa: bool = False,
        linear_attn_mask: torch.Tensor | None = None,
    ):
        if inputs_embeds[1] is None:
            prefix_output = self.qwen3_5.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.action_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.qwen3_5.language_model, self.action_expert]
            num_layers = self.qwen3_5.config.text_config.num_hidden_layers

            prefix_len = inputs_embeds[0].shape[1]

            use_gradient_checkpointing = (
                hasattr(self.action_expert, "gradient_checkpointing")
                and self.action_expert.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_5,
                        self.action_expert,
                        prefix_len,
                        knowledge_insulation,
                        use_sdpa,
                        linear_attn_mask,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        qwen3_5=self.qwen3_5,
                        action_expert=self.action_expert,
                        prefix_len=prefix_len,
                        knowledge_insulation=knowledge_insulation,
                        use_sdpa=use_sdpa,
                        linear_attn_mask=linear_attn_mask,
                    )

            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class InternVLAA15(nn.Module):

    def __init__(self, config: InternVLAA15Config):
        super().__init__()
        self.config = config

        action_expert_config = ActionExpertConfig(
            hidden_size=config.action_expert_hidden_size,
            intermediate_size=config.action_expert_intermediate_size,
        )

        self.qwen3_5_with_expert = InternVLAA15WithExpertModel(
            vlm_model_name_or_path=config.vlm_model_name_or_path,
            action_expert_config=action_expert_config,
            precision=config.dtype,
        )

        action_expert_hidden_size = self.qwen3_5_with_expert.action_expert.config.hidden_size

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_hidden_size)
        self.action_out_proj = nn.Linear(action_expert_hidden_size, config.max_action_dim)

        if not self.config.tokenize_state:
            self.state_proj = nn.Linear(config.max_state_dim, action_expert_hidden_size)

        self.action_time_mlp_in = nn.Linear(2 * action_expert_hidden_size, action_expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_hidden_size, action_expert_hidden_size)


        self.learnable_tokens = nn.Parameter(
            torch.zeros(config.num_learnable_tokens, action_expert_hidden_size)
        )
        nn.init.trunc_normal_(self.learnable_tokens, std=0.02)
        self.learnable_tokens_in_proj = nn.Linear(
            action_expert_hidden_size, action_expert_hidden_size
        )

        if not config.action_loss_only:
            self.wan_video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision,
            )
            wan_dim = self.wan_video_model.wan_model.dim
            self.learnable_to_wan_proj = nn.Linear(action_expert_hidden_size, wan_dim)
            self.fm_video_scheduler = FlowMatchScheduler(
                shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000,
            )
            self.fm_video_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            lat_T = 1 + config.num_video_frames // 4
            lat_H = config.video_height // 32
            lat_W = config.video_width // 32
            self.register_buffer(
                "_wan_grid_sizes", torch.tensor([lat_T, lat_H, lat_W], dtype=torch.long)
            )

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, dynamic=True, mode=config.compile_mode)

        self.set_requires_grad()
        self._setup_wan_grad()

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.qwen3_5_with_expert.qwen3_5.visual.eval()
            for params in self.qwen3_5_with_expert.qwen3_5.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwen3_5_with_expert.qwen3_5.eval()
            for params in self.qwen3_5_with_expert.qwen3_5.parameters():
                params.requires_grad = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.qwen3_5_with_expert.qwen3_5.language_model.gradient_checkpointing = True
        self.qwen3_5_with_expert.qwen3_5.visual.gradient_checkpointing = True
        self.qwen3_5_with_expert.action_expert.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for InternVLAA15 model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_5_with_expert.qwen3_5.language_model.gradient_checkpointing = False
        self.qwen3_5_with_expert.qwen3_5.visual.gradient_checkpointing = False
        self.qwen3_5_with_expert.action_expert.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for InternVLAA15 model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _compute_fast_token_mask(
        self, lang_tokens: Tensor, provided_mask: Tensor | None = None
    ) -> Tensor:
        """Return a bool mask marking fast-token positions in the prefix.

        Uses the transform-provided mask if available; otherwise derives it from
        the configured ``[action_token_min, action_token_max]`` range.
        """
        if provided_mask is not None:
            return provided_mask.to(device=lang_tokens.device, dtype=torch.bool)
        return (
            (lang_tokens >= self.config.action_token_min)
            & (lang_tokens <= self.config.action_token_max)
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    @dynamo.disable
    def embed_prefix(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks, labels=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_5_with_expert.qwen3_5.config.image_token_id
        D1 = pixel_values.shape[-1]
        pixel_values = pixel_values.view(-1, D1)
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs = self.qwen3_5_with_expert.qwen3_5.visual(pixel_values, image_grid_thw).pooler_output

        embs = self.qwen3_5_with_expert.qwen3_5.get_input_embeddings()(lang_tokens)
        B, L, D2 = embs.shape
        embs = embs.view(-1, D2)
        lang_tokens = lang_tokens.view(-1)
        embs[lang_tokens == image_token_id] = image_embs
        embs = embs.view(B, L, D2)

        pad_masks = lang_masks.to(torch.bool)
        # Force causal attention over the prefix to match Qwen3.5's pretrained
        # decoder-only LM regime. Each valid position starts its own block, so
        # cumsum becomes 1..L and `cumsum_kv <= cumsum_q` reduces to a standard
        # lower-triangular causal mask. The `labels` argument is kept for API
        # compatibility but no longer alters the prefix attention pattern.
        del labels  # noqa: F841 — intentionally unused
        att_masks = pad_masks.clone()

        return embs, pad_masks, att_masks

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks):
        L = lang_tokens.shape[1]
        pseudo_action_token_id = 777
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * pseudo_action_token_id
        padded_lang_tokens[:, :L] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)
        position_ids, rope_deltas = self.qwen3_5_with_expert.qwen3_5.model.get_rope_index(
            padded_lang_tokens,
            image_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def _block_suffix_attend_prefix_tokens(
        self,
        att_2d_masks: Tensor,
        prefix_len: int,
        blocked_prefix_mask: Tensor | None,
    ) -> Tensor:
        """Block suffix (action) queries from attending selected prefix keys."""
        if blocked_prefix_mask is None:
            return att_2d_masks
        if blocked_prefix_mask.dtype != torch.bool:
            blocked_prefix_mask = blocked_prefix_mask.bool()
        blocked_prefix_mask = blocked_prefix_mask.to(device=att_2d_masks.device)
        att_2d_masks[:, prefix_len:, :prefix_len] &= ~blocked_prefix_mask[:, None, :]
        return att_2d_masks

    @torch.no_grad()
    def generate_subtask_tokens(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        max_new_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        generated = self.qwen3_5_with_expert.qwen3_5.generate(
            input_ids=lang_tokens,
            attention_mask=lang_masks,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        new_tokens = generated[:, lang_tokens.shape[1] :]
        if new_tokens.numel() == 0:
            return lang_tokens, lang_masks

        appended_masks = torch.ones_like(new_tokens, dtype=lang_masks.dtype, device=lang_masks.device)
        lang_tokens = torch.cat([lang_tokens, new_tokens], dim=1)
        lang_masks = torch.cat([lang_masks, appended_masks], dim=1)
        return lang_tokens, lang_masks

    @torch.no_grad()
    def sample_actions(
        self,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        fast_token_mask: Tensor | None = None,
        noise=None,
        num_steps=None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, rope_deltas = self.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.qwen3_5_with_expert.qwen3_5.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.qwen3_5_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        if self.config.block_action_attend_fast_tokens:
            fast_mask = self._compute_fast_token_mask(lang_tokens, fast_token_mask)
        else:
            fast_mask = None

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                max_prefix_position_ids,
                x_t.to(dtype),
                expanded_time.to(dtype),
                fast_mask=fast_mask,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids,
        x_t,
        timestep,
        fast_mask: Tensor | None = None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Block suffix queries from attending to fast tokens at inference time too.
        if fast_mask is not None:
            mask_b = fast_mask.to(device=full_att_2d_masks.device, dtype=torch.bool)
            full_att_2d_masks[:, :, :prefix_len] &= ~mask_b[:, None, :]

        position_ids = (
            torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids)
            + max_prefix_position_ids
        )

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_5_with_expert.action_expert.config._attn_implementation = "eager"

        outputs_embeds, _ = self.qwen3_5_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def _setup_wan_grad(self):
        if self.config.action_loss_only or self.config.freeze_learnable_tokens:
            self.learnable_tokens.requires_grad = False
            for p in self.learnable_tokens_in_proj.parameters():
                p.requires_grad = False
        if self.config.action_loss_only:
            return  # WAN model not loaded; nothing else to configure
        for p in self.wan_video_model.vae.model.parameters():
            p.requires_grad = False
        if self.config.freeze_wan_dit:
            for p in self.wan_video_model.wan_model.parameters():
                p.requires_grad = False
        if self.config.freeze_learnable_tokens:
            for p in self.learnable_to_wan_proj.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)

        if self.config.freeze_vision_encoder:
            self.qwen3_5_with_expert.qwen3_5.visual.eval()

        if self.config.train_expert_only:
            self.qwen3_5_with_expert.qwen3_5.eval()
        if self.config.action_loss_only:
            return self
        self.wan_video_model.vae.model.eval()
        if self.config.freeze_wan_dit:
            self.wan_video_model.wan_model.eval()
        return self

    # ------------------------------------------------------------------
    # Suffix embedding with learnable tokens
    # ------------------------------------------------------------------

    def embed_suffix(self, state, noisy_actions, timestep):
        """Build suffix: [state(1)] [learnable(N)] [action_time(chunk_size)]."""
        embs = []
        pad_masks = []
        att_masks = []

        # State token
        if not self.config.tokenize_state:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self._apply_checkpoint(lambda s: self.state_proj(s), state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        bsize = state.shape[0]
        device = state.device

        # Learnable tokens
        num_lt = self.config.num_learnable_tokens
        lt_emb = self._apply_checkpoint(
            lambda t: self.learnable_tokens_in_proj(t), self.learnable_tokens
        )
        lt_emb = lt_emb[None].expand(bsize, -1, -1)
        embs.append(lt_emb)
        pad_masks.append(torch.ones(bsize, num_lt, dtype=torch.bool, device=device))
        att_masks += [1] + [0] * (num_lt - 1)

        # Action + time tokens
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(lambda a: self.action_in_proj(a), noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self._apply_checkpoint(
            lambda x: self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x))),
            action_time_emb,
        )

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + [0] * (self.config.chunk_size - 1)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_learnable_token_output(self, suffix_out):
        start = 1  # skip state token
        end = 1 + self.config.num_learnable_tokens
        return suffix_out[:, start:end]

    # ------------------------------------------------------------------
    # Inference: denoise step with full suffix output
    # ------------------------------------------------------------------

    def denoise_step_full(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids,
        x_t,
        timestep,
        fast_mask: Tensor | None = None,
    ):
        """Like denoise_step but returns (velocity, learnable_token_output)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        if fast_mask is not None:
            mask_b = fast_mask.to(device=full_att_2d_masks.device, dtype=torch.bool)
            full_att_2d_masks[:, :, :prefix_len] &= ~mask_b[:, None, :]

        position_ids = (
            torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids)
            + max_prefix_position_ids
        )

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_5_with_expert.action_expert.config._attn_implementation = "eager"

        outputs_embeds, _ = self.qwen3_5_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            knowledge_insulation=self.config.knowledge_insulation,
        )

        suffix_out = outputs_embeds[1]
        learnable_out = self.get_learnable_token_output(suffix_out)
        action_out = suffix_out[:, -self.config.chunk_size:]
        action_out = action_out.to(dtype=torch.float32)
        velocity = self.action_out_proj(action_out)
        return velocity, learnable_out

    # ------------------------------------------------------------------
    # Inference: video generation from learnable tokens
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_video(
        self,
        learnable_out: Tensor,
        cond_frame: Tensor,
        num_inference_steps: int = 50,
    ) -> Tensor:
        """Generate future video frames from learnable token outputs.

        Args:
            learnable_out: [B, N, hidden] learnable token hidden states
            cond_frame: [B, C, 1, H, W] first frame pixels in [-1, 1]
            num_inference_steps: number of denoising steps

        Returns:
            [B, T, C, H, W] generated video pixels in [-1, 1]
        """
        wan_device = next(self.wan_video_model.wan_model.parameters()).device
        wan_dtype = self.wan_video_model.precision

        proj_dtype = self.learnable_to_wan_proj.weight.dtype
        wan_context = self.learnable_to_wan_proj(learnable_out.to(proj_dtype))
        wan_context = wan_context.to(dtype=wan_dtype, device=wan_device)

        cond_frame = cond_frame.to(dtype=wan_dtype, device=wan_device)
        with torch.no_grad():
            cond_latent = self.wan_video_model.encode_video(cond_frame)

        # Ensure batch dimension exists
        if cond_latent.dim() == 4:
            cond_latent = cond_latent.unsqueeze(0)

        B = learnable_out.shape[0]
        C_lat = cond_latent.shape[1]
        lat_H = cond_latent.shape[3]
        lat_W = cond_latent.shape[4]
        lat_T = 1 + self.config.num_video_frames // 4
        latent = torch.randn(B, C_lat, lat_T, lat_H, lat_W, device=wan_device, dtype=wan_dtype)
        latent[:, :, 0:1] = cond_latent

        scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000,
        )
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            t_tensor = t.to(device=wan_device, dtype=wan_dtype)
            with torch.amp.autocast("cuda", dtype=wan_dtype):
                velocity = self.wan_dit_forward(latent, wan_context, t_tensor.expand(B))
            velocity[:, :, 0:1] = 0
            latent = scheduler.step(velocity, t_tensor, latent)
            latent[:, :, 0:1] = cond_latent

        video_pixels = self.wan_video_model.decode_video(latent)
        return video_pixels.permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        actions,
        labels: Tensor | None = None,
        fast_token_mask: Tensor | None = None,
        video_frames: Tensor | None = None,
        video_mask: Tensor | None = None,
        noise=None,
        time=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Training forward: action loss + VLM loss + video loss.

        Args:
            video_mask: [B] bool tensor, True for samples with real video frames.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, labels
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        if (
            self.qwen3_5_with_expert.qwen3_5.language_model.layers[0].mlp.up_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        if self.config.block_action_attend_fast_tokens:
            fast_mask = self._compute_fast_token_mask(lang_tokens, fast_token_mask)
            att_2d_masks = self._block_suffix_attend_prefix_tokens(
                att_2d_masks=att_2d_masks,
                prefix_len=prefix_pad_masks.shape[1],
                blocked_prefix_mask=fast_mask,
            )

        prefix_position_ids, rope_deltas = self.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )

        B = lang_tokens.shape[0]
        suffix_len = suffix_pad_masks.shape[1]

        if labels is not None:
            fast_mask = self._compute_fast_token_mask(lang_tokens, fast_token_mask)
            has_fast = fast_mask.any(dim=1)  # [B]
            first_fast_idx = fast_mask.long().argmax(dim=1)
            anchor_idx = (first_fast_idx - 1).clamp(min=0)

            batch_idx = torch.arange(B, device=lang_tokens.device)
            max_input_pos_before_fast = prefix_position_ids[:, batch_idx, anchor_idx].unsqueeze(-1)
            max_input_pos_all = prefix_position_ids.max(dim=-1, keepdim=True).values

            max_input_pos = torch.where(
                has_fast.view(1, -1, 1),
                max_input_pos_before_fast,
                max_input_pos_all,
            )
        else:
            max_input_pos = prefix_position_ids.max(dim=-1, keepdim=True).values

        suffix_position_ids = (
            torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_input_pos)
            + max_input_pos
        )
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (prefix_out, suffix_out), _ = self.qwen3_5_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                knowledge_insulation=self.config.knowledge_insulation,
                use_sdpa=self.config.use_sdpa,
                linear_attn_mask=pad_masks,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids
        )

        # VQA loss (per-token for detailed logging)
        if labels is not None:
            logits = self.qwen3_5_with_expert.qwen3_5.lm_head(prefix_out).to(dtype=torch.float32)
            labels = labels.to(device=logits.device, dtype=torch.long)
            bsize, seq_len, vocab_size = logits.shape
            logits_ar = logits[:, :-1, :].reshape(-1, vocab_size)
            labels_ar = labels[:, 1:].reshape(-1)
            loss_per_token = F.cross_entropy(
                logits_ar, labels_ar, reduction="none", ignore_index=-100
            ).reshape(bsize, seq_len - 1)
            token_mask = (labels[:, 1:] != -100)
            valid = token_mask.float()
            valid_per_sample = valid.sum(dim=1).clamp(min=1)
            loss_vqa = (loss_per_token * valid).sum(dim=1) / valid_per_sample
        else:
            bsize = actions.shape[0]
            loss_per_token = torch.zeros(bsize, 1, device=actions.device, dtype=torch.float32)
            token_mask = torch.zeros(bsize, 1, device=actions.device, dtype=torch.bool)
            loss_vqa = torch.zeros(bsize, device=actions.device, dtype=torch.float32)

        # Action loss
        if self.config.video_loss_only:
            loss_action = torch.zeros_like(u_t)
            clean_action_prediction = actions
        else:
            action_out = suffix_out[:, -self.config.chunk_size:]
            action_out = action_out.to(dtype=torch.float32)
            v_t = self._apply_checkpoint(lambda x: self.action_out_proj(x), action_out)
            loss_action = F.mse_loss(u_t, v_t, reduction="none")
            # The flow target is velocity from clean action to noise. Reversing
            # the sampled interpolation gives a differentiable clean-action mean.
            clean_action_prediction = x_t - time_expanded * v_t

        # Video loss — only computed for samples with real video frames
        if self.config.action_loss_only:
            video_loss = torch.tensor(0.0, device=actions.device)
        else:
            has_video = video_mask.any() if video_mask is not None else (video_frames is not None)
            if has_video:
                learnable_out = self.get_learnable_token_output(suffix_out).to(dtype=torch.float32)
                if video_mask is not None:
                    video_frames = video_frames[video_mask]
                    learnable_out = learnable_out[video_mask]
                video_loss = self._compute_video_loss(video_frames, learnable_out)
            else:
                video_loss = torch.tensor(0.0, device=actions.device)

        return (
            loss_action,
            loss_vqa,
            video_loss,
            loss_per_token,
            token_mask,
            clean_action_prediction,
        )

    # ------------------------------------------------------------------
    # WAN DiT forward (cross-attention conditioning)
    # ------------------------------------------------------------------

    def wan_dit_forward(
        self,
        noisy_video_latent: torch.Tensor,
        wan_context: torch.Tensor,
        video_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run WAN DiT with learnable token context as cross-attention K/V."""
        wan = self.wan_video_model.wan_model
        device = wan.patch_embedding.weight.device
        if wan.freqs.device != device:
            wan.freqs = wan.freqs.to(device)

        B = noisy_video_latent.shape[0]

        # Patch embedding
        x = wan.patch_embedding(noisy_video_latent)
        grid_sizes = self._wan_grid_sizes.unsqueeze(0).expand(B, -1).to(device)
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.shape[1]
        seq_lens = torch.full((B,), seq_len, dtype=torch.long, device=device)

        # Time embedding
        t_vid = video_timestep
        if t_vid.dim() == 1:
            t_vid = t_vid.unsqueeze(1).expand(B, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = wan.time_embedding(
                sinusoidal_embedding_1d(wan.freq_dim, t_vid.flatten())
                .unflatten(0, (B, seq_len))
                .float()
                .to(device)
            )
            e0 = wan.time_projection(e).unflatten(2, (6, wan.dim))

        # Use projected learnable tokens as context (skip text_embedding)
        context = wan_context

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=wan.freqs,
            context=context,
            context_lens=None,
        )

        for block in wan.blocks:
            x = block(x, **kwargs)

        x = wan.head(x, e)
        x = wan.unpatchify(x, grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    # ------------------------------------------------------------------
    # Video loss computation
    # ------------------------------------------------------------------

    def _compute_video_loss(
        self,
        video_frames: torch.Tensor,
        learnable_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WAN video prediction loss via flow matching.

        Args:
            video_frames: [B, T, C, H, W] in [-1, 1]
            learnable_out: [B, N, hidden] extracted learnable token outputs
        """
        B = video_frames.shape[0]
        wan_device = next(self.wan_video_model.wan_model.parameters()).device
        wan_dtype = self.wan_video_model.precision

        # Project learnable tokens to WAN context
        wan_context = self.learnable_to_wan_proj(learnable_out)
        wan_context = wan_context.to(dtype=wan_dtype, device=wan_device)

        # Encode video with frozen VAE: expects [B, C, T, H, W]
        video_bcthw = video_frames.permute(0, 2, 1, 3, 4).to(dtype=wan_dtype, device=wan_device)
        first_frame_bcthw = video_bcthw[:, :, 0:1]

        with torch.no_grad():
            clean_latent = self.wan_video_model.encode_video(video_bcthw)
            cond_latent = self.wan_video_model.encode_video(first_frame_bcthw)

        # Sample video flow-matching timestep
        timestep_id = torch.randint(
            0, self.fm_video_scheduler.num_train_timesteps, (B,)
        )
        sigma = self.fm_video_scheduler.sigmas[timestep_id].to(
            dtype=wan_dtype, device=wan_device
        ).view(B, 1, 1, 1, 1)
        video_t = self.fm_video_scheduler.timesteps[timestep_id].to(
            dtype=wan_dtype, device=wan_device
        )

        # Add noise (teacher forcing: keep frame 0 clean)
        video_noise = torch.randn_like(clean_latent)
        noisy_latent = clean_latent * (1 - sigma) + video_noise * sigma
        noisy_latent[:, :, 0:1] = cond_latent

        # Target velocity
        video_target = video_noise - clean_latent
        video_target[:, :, 0:1] = 0

        # WAN forward
        with torch.amp.autocast("cuda", dtype=wan_dtype):
            video_pred = self.wan_dit_forward(noisy_latent, wan_context, video_t)

        video_pred[:, :, 0:1] = 0
        return F.mse_loss(video_pred.float(), video_target.float(), reduction="mean")


# ======================================================================
# Policy wrapper
# ======================================================================

class InternVLAA15Policy(PreTrainedPolicy):
    """Qwen3.5 VLA policy with WAN video auxiliary supervision."""

    config_class = InternVLAA15Config
    name = "internvla_a1_5"
    _checkpoint_excluded_prefixes = ("model.wan_video_model.",)

    def __init__(self, config: InternVLAA15Config):
        super().__init__(config)
        config.validate_features()
        self.config = config

        if config.inference_backend == "optimized":
            from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5_optimized import (
                InternVLAA15Optimized,
            )

            self.model = InternVLAA15Optimized(config)
        else:
            self.model = InternVLAA15(config)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    def __str__(self) -> str:
        lines = ["=" * 60, f"Policy: {self.__class__.__name__}", ""]
        num_total = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_vlm = sum(p.numel() for p in self.model.qwen3_5_with_expert.qwen3_5.parameters())
        num_expert = sum(p.numel() for p in self.model.qwen3_5_with_expert.action_expert.parameters())
        num_wan = (
            sum(p.numel() for p in self.model.wan_video_model.parameters())
            if hasattr(self.model, "wan_video_model")
            else 0
        )
        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {format_big_number(num_total)}")
        lines.append(f"  - Trainable params    : {format_big_number(num_trainable)}")
        lines.append(f"  - Qwen3_5 params      : {format_big_number(num_vlm)}")
        lines.append(f"  - Action expert params: {format_big_number(num_expert)}")
        lines.append(f"  - WAN params          : {format_big_number(num_wan)}")
        lines.append(f"  - Learnable tokens    : {self.config.num_learnable_tokens}")
        lines.append(f"  - Knowledge insulation: {self.config.knowledge_insulation}")
        lines.append(f"  - Inference backend   : {self.config.inference_backend}")
        lines.append(f"  - Freeze WAN DiT      : {self.config.freeze_wan_dit}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.action_out_proj.to(torch.float32)
        if hasattr(self.model, "_cast_action_path_to_fp32"):
            self.model._cast_action_path_to_fp32()
        return self

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith(self._checkpoint_excluded_prefixes):
                del state[key]

        metadata = getattr(state, "_metadata", None)
        if metadata is not None:
            for key in list(metadata.keys()):
                if key == "model.wan_video_model" or key.startswith("model.wan_video_model."):
                    del metadata[key]
        return state

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def prepare_state(self, batch):
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        fast_token_mask = batch.get(f"{OBS_PREFIX}fast_token_mask")
        state = self.prepare_state(batch)

        actions = self.model.sample_actions(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
            fast_token_mask=fast_token_mask,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def predict_action_chunk_with_video(
        self, batch: dict[str, Tensor], num_video_steps: int = 50
    ) -> tuple[Tensor, Tensor]:
        """Predict actions and generate future video from learnable tokens.

        Returns:
            (actions [B, chunk, action_dim], video [B, T, C, H, W] in [-1, 1])
        """
        self.eval()
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        fast_token_mask = batch.get(f"{OBS_PREFIX}fast_token_mask")
        state = self.prepare_state(batch)

        model = self.model
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype
        num_steps = model.config.num_inference_steps

        actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
        noise = model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids, _ = model.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.qwen3_5_with_expert.qwen3_5.language_model.config._attn_implementation = "eager"

        _, past_key_values = model.qwen3_5_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            knowledge_insulation=model.config.knowledge_insulation,
        )

        dt = -1.0 / num_steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        if model.config.block_action_attend_fast_tokens:
            fast_mask = model._compute_fast_token_mask(lang_tokens, fast_token_mask)
        else:
            fast_mask = None

        # Denoising loop - use full step on last iteration to capture learnable outputs
        learnable_out = None
        while time >= -dt_tensor / 2:
            expanded_time = time.expand(bsize)
            is_last = (time + dt_tensor) < -dt_tensor / 2
            if is_last:
                v_t, learnable_out = model.denoise_step_full(
                    state, prefix_pad_masks, past_key_values,
                    max_prefix_position_ids, x_t.to(dtype), expanded_time.to(dtype),
                    fast_mask=fast_mask,
                )
            else:
                v_t = model.denoise_step(
                    state, prefix_pad_masks, past_key_values,
                    max_prefix_position_ids, x_t.to(dtype), expanded_time.to(dtype),
                    fast_mask=fast_mask,
                )
            x_t = x_t + dt_tensor * v_t
            time += dt_tensor

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = x_t[:, :, :original_action_dim]

        # Generate video from learnable token outputs
        video_frames = batch.get("observation.video_frames")
        if video_frames is not None:
            cond_frame = video_frames[:, 0:1].permute(0, 2, 1, 3, 4)
        else:
            cond_frame = torch.zeros(
                bsize, 3, 1, model.config.video_height, model.config.video_width,
                device=device, dtype=dtype,
            )

        generated_video = model.generate_video(
            learnable_out, cond_frame, num_inference_steps=num_video_steps
        )
        return actions, generated_video

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        fast_token_mask = batch.get(f"{OBS_PREFIX}fast_token_mask")
        video_frames = batch.get("observation.video_frames")

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        labels = batch["VQA.labels"] if self.config.enable_vqa_loss else None

        # video_mask: True for robot samples that have real video frames
        vqa_type = batch.get("vqa_type")
        if vqa_type is not None:
            video_mask = (vqa_type != 1)  # VQA-only samples have vqa_type=1
        else:
            video_mask = None

        losses, losses_vlm, video_loss, loss_per_token, token_mask, action_prediction = self.model.forward(
            pixel_values, image_grid_thw, lang_tokens, lang_masks,
            state, actions,
            labels=labels,
            fast_token_mask=fast_token_mask,
            video_frames=video_frames,
            video_mask=video_mask,
        )

        original_action_dim = batch[ACTION].shape[-1]
        losses = losses[:, :, :original_action_dim]
        zero = torch.tensor(0.0, device=losses.device)

        if self.config.opd_enabled:
            required_opd_keys = {
                OPD_TEACHER_ACTION_MEAN,
                OPD_TEACHER_ACTION_STD,
                OPD_TEACHER_ACTION_MASK,
            }
            missing_opd_keys = sorted(required_opd_keys.difference(batch))
            if missing_opd_keys:
                raise KeyError(f"OPD teacher labels are missing from the batch: {missing_opd_keys}")
            action_prediction = action_prediction[:, :, :original_action_dim]
            loss_opd = diagonal_gaussian_reverse_kl(
                action_prediction,
                batch[OPD_TEACHER_ACTION_MEAN][:, :, :original_action_dim],
                batch[OPD_TEACHER_ACTION_STD][:, :, :original_action_dim],
                student_std=self.config.opd_student_std,
                mask=batch[OPD_TEACHER_ACTION_MASK][:, :, :original_action_dim],
                max_kl_per_dim=self.config.opd_max_kl_per_dim,
            )
        else:
            loss_opd = zero

        rl_metrics = {}
        if self.config.rl_enabled:
            required_rl_keys = {RL_IS_ON_POLICY, RL_REWARD, RL_REWARD_MASK}
            if self.config.rl_algorithm == "grpo":
                required_rl_keys.add(RL_GROUP_ID)
            missing_rl_keys = sorted(required_rl_keys.difference(batch))
            if missing_rl_keys:
                raise KeyError(f"RL signals are missing from the batch: {missing_rl_keys}")
            if self.config.rl_require_on_policy and not bool(batch[RL_IS_ON_POLICY].all()):
                raise ValueError("Flow RL requires rollouts generated by the current student")
            per_sample_flow_loss = losses.mean(dim=(1, 2))
            if self.config.rl_algorithm == "grpo":
                rl_result = group_relative_flow_matching_loss(
                    per_sample_flow_loss,
                    batch[RL_REWARD],
                    batch[RL_REWARD_MASK],
                    batch[RL_GROUP_ID],
                    gamma=self.config.rl_gamma,
                    temperature=self.config.rl_temperature,
                    min_weight=self.config.rl_min_weight,
                    max_weight=self.config.rl_max_weight,
                    group_size=self.config.rl_group_size,
                )
            else:
                rl_result = reward_weighted_flow_matching_loss(
                    per_sample_flow_loss,
                    batch[RL_REWARD],
                    batch[RL_REWARD_MASK],
                    gamma=self.config.rl_gamma,
                    temperature=self.config.rl_temperature,
                    min_weight=self.config.rl_min_weight,
                    max_weight=self.config.rl_max_weight,
                    normalize_advantage=self.config.rl_normalize_advantage,
                    supplied_advantage=batch.get(RL_ADVANTAGE),
                )
            loss_rl = rl_result.loss
            rl_metrics = {
                "rl_return": rl_result.returns.mean().item(),
                "rl_advantage": rl_result.advantages.mean().item(),
                "rl_weight_max": rl_result.weights.max().item(),
            }
        else:
            loss_rl = zero

        # =====================================================================
        # Loss taxonomy:
        #
        # [VLM branch] prefix → lm_head → next-token cross-entropy
        #   loss_vlm       : mean CE over all valid label tokens
        #   loss_fast      : CE on fast action tokens (token_id ∈ [action_token_min, max])
        #   loss_subtask   : CE on subtask/text tokens (valid but not fast)
        #
        # [Action Expert branch] suffix → action_out_proj → flow matching MSE
        #   loss_fm_action : mean MSE of predicted velocity vs target
        #   loss_rl        : RWFM or GRPO-weighted per-sample flow MSE
        #   loss_opd       : continuous reverse-KL to the Kairos teacher
        #
        # [Video branch] suffix learnable tokens → WAN DiT → MSE
        #   loss_video     : video prediction flow matching loss
        # =====================================================================

        # --- VLM branch: per-token split ---
        if labels is not None and token_mask.any():
            labels_shifted = labels[:, 1:]
            fast_tok_mask = (
                (labels_shifted >= self.config.action_token_min)
                & (labels_shifted <= self.config.action_token_max)
            )
            subtask_tok_mask = token_mask & ~fast_tok_mask

            fast_sum = (loss_per_token * fast_tok_mask.float()).sum()
            fast_cnt = fast_tok_mask.float().sum()
            subtask_sum = (loss_per_token * subtask_tok_mask.float()).sum()
            subtask_cnt = subtask_tok_mask.float().sum()

            loss_fast = (fast_sum / fast_cnt) if fast_cnt > 0 else zero
            loss_subtask = (subtask_sum / subtask_cnt) if subtask_cnt > 0 else zero
        else:
            loss_fast = zero
            loss_subtask = zero

        # --- Per-sample aggregation by vqa_type ---
        if self.config.enable_vqa_loss:
            vqa_type = batch["vqa_type"]
            action_mask = (vqa_type == 0) | (vqa_type == 2)  # robot samples
            vlm_mask = (vqa_type == 1) | (vqa_type == 2)     # samples with VQA labels

            loss_fm_action = losses[action_mask].mean() if action_mask.any() else zero
            loss_vlm = losses_vlm[vlm_mask].mean() if vlm_mask.any() else zero

            if self.config.rl_enabled:
                action_loss_weight = self.config.rl_sft_loss_weight
            elif self.config.opd_enabled:
                action_loss_weight = self.config.opd_sft_loss_weight
            else:
                action_loss_weight = 1.0
            loss = (
                action_loss_weight * 10 * loss_fm_action
                + self.config.rl_loss_weight * 10 * loss_rl
                + self.config.lambda_vqa * loss_vlm
                + self.config.video_loss_weight * video_loss
                + self.config.opd_loss_weight * loss_opd
            )
            # --- Build loss_dict ---
            loss_dict = {
                "loss": loss.item(),
                # VLM branch
                "loss_vqa": loss_vlm.item(),
                "loss_fast": loss_fast.item(),
                "loss_subtask": loss_subtask.item(),
                # Action Expert branch
                "loss_action": loss_fm_action.item(),
                # Video branch
                "loss_video": video_loss.item(),
            }
        else:
            loss_fm_action = losses.mean()
            loss_vlm = zero
            if self.config.rl_enabled:
                action_loss_weight = self.config.rl_sft_loss_weight
            elif self.config.opd_enabled:
                action_loss_weight = self.config.opd_sft_loss_weight
            else:
                action_loss_weight = 1.0
            loss = (
                action_loss_weight * loss_fm_action
                + self.config.rl_loss_weight * loss_rl
                + self.config.video_loss_weight * video_loss
                + self.config.opd_loss_weight * loss_opd
            )
            loss_dict = {
                "loss": loss.item(),
                # Action Expert branch
                "loss_action": loss_fm_action.item(),
                # Video branch
                "loss_video": video_loss.item(),
            }
        if self.config.opd_enabled:
            loss_dict["loss_opd"] = loss_opd.item()
        if self.config.rl_enabled:
            loss_dict["loss_rl"] = loss_rl.item()
            loss_dict.update(rl_metrics)
        return loss, loss_dict
