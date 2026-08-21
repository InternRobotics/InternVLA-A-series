"""Kairos HTTP teacher adapter for continuous-action OPD.

The adapter follows Kairos' official WAM service protocol. The protocol uses
pickle, so remote endpoints are rejected unless explicitly allowed.
"""

from __future__ import annotations

import json
import pickle
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lerobot.dataset_schemas import get_schema
from lerobot.opd import (
    OPD_IMAGE_MASK,
    OPD_IMAGE_PREFIX,
    OPD_IS_ON_POLICY,
    OPD_RAW_STATE,
    OPD_ROBOT_TYPE,
    OPD_TASK,
    OPD_TEACHER_ACTION_MASK,
    OPD_TEACHER_ACTION_MEAN,
    OPD_TEACHER_ACTION_STD,
)
from lerobot.opd.config import OPDConfig


def _as_float_tensor(value: Any) -> Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device="cpu")


def _squeeze_stats(value: Any) -> Tensor:
    tensor = _as_float_tensor(value)
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


@dataclass(frozen=True)
class _ZScore:
    mean: Tensor
    std: Tensor

    def normalize(self, value: Tensor) -> Tensor:
        return (value.float().cpu() - self.mean) / (self.std + 1e-8)

    def denormalize(self, value: Tensor) -> Tensor:
        return value.float().cpu() * (self.std + 1e-8) + self.mean


class KairosDatasetStats:
    """Load the z-score schema published with Kairos benchmark checkpoints."""

    def __init__(self, path: str | Path, key: str = "default") -> None:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        try:
            state_stats = stats["state"][key]
            action_stats = stats["action"][key]
            self.state = _ZScore(
                _squeeze_stats(state_stats["global_mean"]),
                _squeeze_stats(state_stats["global_std"]),
            )
            self.action = _ZScore(
                _squeeze_stats(action_stats["global_mean"]),
                _squeeze_stats(action_stats["global_std"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Invalid Kairos dataset stats at {path}: missing {exc.args[0]!r}"
            ) from exc


class KairosWAMClient:
    """Minimal client for the official ``/health``, ``/load_engine``, ``/infer`` API."""

    def __init__(self, cfg: OPDConfig) -> None:
        self.cfg = cfg
        self.endpoint = cfg.teacher_endpoint.rstrip("/")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Invalid Kairos endpoint: {self.endpoint!r}")
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if parsed.hostname not in local_hosts and not cfg.allow_remote_pickle:
            raise ValueError(
                "Kairos' official service uses pickle. Refusing a non-local endpoint; "
                "set opd.allow_remote_pickle=true only for a trusted server."
            )
        self._get_json("/health", timeout=min(cfg.timeout_s, 30.0))
        if cfg.load_teacher_on_init:
            self._get_json("/load_engine", timeout=cfg.timeout_s)

    def _get_json(self, path: str, *, timeout: float) -> Any:
        try:
            with urllib.request.urlopen(f"{self.endpoint}{path}", timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Kairos request failed for {path}: {exc}") from exc

    def infer_action(
        self,
        *,
        input_images: Image.Image | list[Image.Image],
        robot_state: Tensor,
        action_horizon: int,
        prompts: str | list[str],
        seed: int,
    ) -> Tensor:
        first_image = input_images[0] if isinstance(input_images, list) else input_images
        width, height = first_image.size
        payload = {
            "save_path": "",
            "robot_state": robot_state.float().cpu(),
            "robot_state_mask": None,
            "robot_action_horizon": int(action_horizon),
            "wam_infer_mode": "action",
            "input_image": input_images,
            "prompt": prompts,
            "negative_prompt": self.cfg.negative_prompt,
            "seed": int(seed),
            "tiled": False,
            "height": int(height),
            "width": int(width),
            "num_frames": 1,
            "num_inference_steps": int(self.cfg.num_inference_steps),
            "cfg_scale": float(self.cfg.cfg_scale),
            "save_fps": 16,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/infer",
            data=pickle.dumps(payload),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout_s) as response:
                result = pickle.loads(response.read())  # noqa: S301 - trusted endpoint checked above
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Kairos inference returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Kairos inference failed: {exc}") from exc
        try:
            action = result["output"][1]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Kairos response did not contain output[1] action tensor") from exc
        if action is None:
            raise RuntimeError("Kairos returned no action tensor")
        return torch.as_tensor(action, dtype=torch.float32, device="cpu")


def _tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W] or [H,W,C], got {tuple(image.shape)}")
    if image.shape[0] in {1, 3, 4}:
        image = image.permute(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.expand(*image.shape[:-1], 3)
    if image.dtype.is_floating_point:
        image = (image.float().clamp(0, 1) * 255).round().to(torch.uint8)
    else:
        image = image.to(torch.uint8)
    return Image.fromarray(image.numpy()[..., :3], mode="RGB")


def compose_kairos_image(images: list[Image.Image], layout: str) -> Image.Image:
    if not images:
        raise ValueError("At least one valid OPD image is required")
    if layout == "auto":
        layout = "robotwin" if len(images) >= 3 else "first"
    if layout == "first" or len(images) == 1:
        return images[0]
    if layout == "robotwin":
        if len(images) < 3:
            raise ValueError("robotwin image layout requires head, left wrist, and right wrist views")
        head = images[0].resize((320, 256), Image.Resampling.BILINEAR)
        left = images[1].resize((160, 128), Image.Resampling.BILINEAR)
        right = images[2].resize((160, 128), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (320, 384))
        canvas.paste(head, (0, 0))
        canvas.paste(left, (0, 256))
        canvas.paste(right, (160, 256))
        return canvas
    if layout == "horizontal":
        target_height = min(image.height for image in images)
        resized = [
            image.resize(
                (max(1, round(image.width * target_height / image.height)), target_height),
                Image.Resampling.BILINEAR,
            )
            for image in images
        ]
        canvas = Image.new("RGB", (sum(image.width for image in resized), target_height))
        x = 0
        for image in resized:
            canvas.paste(image, (x, 0))
            x += image.width
        return canvas
    raise ValueError(f"Unsupported image layout: {layout!r}")


def _apply_reorder(value: Tensor, mapping: list[list[int]] | None) -> tuple[Tensor, Tensor]:
    if not mapping:
        return value, torch.ones(value.shape[-1], dtype=torch.bool)
    output_dim = max(item[3] for item in mapping)
    output = torch.zeros(*value.shape[:-1], output_dim, dtype=value.dtype)
    mask = torch.zeros(output_dim, dtype=torch.bool)
    for src_start, src_end, dst_start, dst_end in mapping:
        if src_end - src_start != dst_end - dst_start:
            raise ValueError(f"Invalid reorder entry: {[src_start, src_end, dst_start, dst_end]}")
        output[..., dst_start:dst_end] = value[..., src_start:src_end]
        mask[dst_start:dst_end] = True
    return output, mask


class StudentActionProjector:
    """Project raw teacher actions through the student's dataset normalization."""

    def __init__(
        self,
        data_stats: dict[str, dict[str, Any]],
        *,
        normalization_mode: str,
        action_mode: str,
        max_action_dim: int,
    ) -> None:
        self.data_stats = data_stats
        self.normalization_mode = normalization_mode
        self.action_mode = action_mode
        self.max_action_dim = max_action_dim

    def _normalize(self, value: Tensor, stats: dict[str, Any]) -> Tensor:
        eps = 1e-6
        if self.normalization_mode == "mean_std":
            mean, std = _as_float_tensor(stats["mean"]), _as_float_tensor(stats["std"])
            return (value - mean) / (std + eps)
        stat_min = "min" if self.normalization_mode == "min_max" else "q01"
        stat_max = "max" if self.normalization_mode == "min_max" else "q99"
        low, high = _as_float_tensor(stats[stat_min]), _as_float_tensor(stats[stat_max])
        return 2 * (value - low) / (high - low + eps) - 1

    def project(self, actions: Tensor, raw_state: Tensor, robot_type: str) -> tuple[Tensor, Tensor]:
        schema = get_schema(robot_type).resolve()
        stats_by_key = self.data_stats.get(robot_type)
        if stats_by_key is None:
            raise KeyError(f"No student dataset stats found for robot_type={robot_type!r}")
        action_keys = schema.get_action_keys()
        sizes = []
        for key in action_keys:
            if key not in stats_by_key:
                raise KeyError(f"Student stats for {robot_type!r} have no action key {key!r}")
            sizes.append(int(_as_float_tensor(stats_by_key[key]["mean"]).shape[-1]))
        compact_dim = sum(sizes)
        if actions.shape[-1] < compact_dim:
            raise ValueError(
                f"Kairos action dim {actions.shape[-1]} is smaller than student raw action dim {compact_dim}"
            )
        actions = actions[..., :compact_dim]

        if self.action_mode == "delta":
            if raw_state.shape[-1] < compact_dim:
                raise ValueError("Raw state is too small to convert Kairos absolute actions to delta actions")
            delta_mask = schema.action_mask
            if delta_mask.numel() == 0:
                delta_mask = torch.ones(compact_dim, dtype=torch.bool)
            if delta_mask.numel() != compact_dim:
                raise ValueError(
                    f"Action mask dim {delta_mask.numel()} does not match compact action dim {compact_dim}"
                )
            actions = actions - torch.where(delta_mask, raw_state[:compact_dim], 0.0)

        normalized_parts = []
        offset = 0
        for key, size in zip(action_keys, sizes, strict=True):
            normalized_parts.append(
                self._normalize(actions[..., offset : offset + size], stats_by_key[key])
            )
            offset += size
        normalized = torch.cat(normalized_parts, dim=-1)
        normalized, mask = _apply_reorder(normalized, schema.action_reorder)

        if normalized.shape[-1] > self.max_action_dim:
            raise ValueError(
                f"Projected teacher action dim {normalized.shape[-1]} exceeds "
                f"max_action_dim={self.max_action_dim}"
            )
        pad = self.max_action_dim - normalized.shape[-1]
        if pad:
            normalized = torch.nn.functional.pad(normalized, (0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad), value=False)
        return normalized, mask


class KairosOPDTeacher:
    """Label an on-policy batch with empirical Kairos action moments."""

    def __init__(
        self,
        cfg: OPDConfig,
        *,
        data_stats: dict[str, dict[str, Any]],
        action_mode: str,
        chunk_size: int,
        max_action_dim: int,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.kairos_stats = KairosDatasetStats(
            cfg.kairos_dataset_stats_path, key=cfg.kairos_stats_key
        )
        self.client = KairosWAMClient(cfg)
        self.projector = StudentActionProjector(
            data_stats,
            normalization_mode=cfg.student_normalization_mode,
            action_mode=action_mode,
            max_action_dim=max_action_dim,
        )
        self.chunk_size = chunk_size
        self.action_horizon = cfg.action_horizon or chunk_size
        self._request_index = 0

    def _batch_values(self, value: Any, batch_size: int) -> list[Any]:
        if isinstance(value, (list, tuple)):
            return list(value)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            return [value[index] for index in range(batch_size)]
        return [value] * batch_size

    def _build_images(self, batch: dict[str, Any], batch_size: int) -> list[Image.Image]:
        masks = batch[OPD_IMAGE_MASK].detach().cpu().bool()
        images = []
        for batch_index in range(batch_size):
            views = [
                _tensor_to_pil(batch[f"{OPD_IMAGE_PREFIX}{view_index}"][batch_index])
                for view_index in range(masks.shape[1])
                if bool(masks[batch_index, view_index])
            ]
            images.append(compose_kairos_image(views, self.cfg.image_layout))
        return images

    def _match_horizon(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        if actions.shape[-2] < 1:
            raise ValueError("Kairos returned an empty action horizon")
        horizon_mask = torch.zeros(self.chunk_size, dtype=torch.bool)
        horizon_mask[: min(actions.shape[-2], self.chunk_size)] = True
        if actions.shape[-2] >= self.chunk_size:
            return actions[..., : self.chunk_size, :], horizon_mask
        pad = self.chunk_size - actions.shape[-2]
        actions = torch.cat(
            [actions, actions[..., -1:, :].expand(*actions.shape[:-2], pad, -1)], dim=-2
        )
        return actions, horizon_mask

    def _query_samples(
        self, images: list[Image.Image], states: Tensor, prompts: list[str]
    ) -> tuple[Tensor, Tensor]:
        sample_batches = []
        horizon_masks = []
        batch_size = len(images)
        for sample_index in range(self.cfg.teacher_samples):
            seed = self.cfg.teacher_seed + self._request_index * self.cfg.teacher_samples + sample_index
            if self.cfg.batch_requests:
                action = self.client.infer_action(
                    input_images=images,
                    robot_state=states,
                    action_horizon=self.action_horizon,
                    prompts=prompts,
                    seed=seed,
                )
                if action.ndim == 2 and batch_size == 1:
                    action = action.unsqueeze(0)
                if action.ndim != 3 or action.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected batched Kairos actions [B,T,D], got {tuple(action.shape)}"
                    )
            else:
                rows = []
                for batch_index in range(batch_size):
                    row = self.client.infer_action(
                        input_images=images[batch_index],
                        robot_state=states[batch_index : batch_index + 1],
                        action_horizon=self.action_horizon,
                        prompts=prompts[batch_index],
                        seed=seed + batch_index,
                    )
                    if row.ndim == 3:
                        row = row[0]
                    if row.ndim != 2:
                        raise ValueError(
                            f"Expected Kairos action [T,D] for one sample, got {tuple(row.shape)}"
                        )
                    rows.append(row)
                action = torch.stack(rows)
            action, horizon_mask = self._match_horizon(action)
            sample_batches.append(action)
            horizon_masks.append(horizon_mask)
        self._request_index += 1
        if not all(torch.equal(horizon_masks[0], mask) for mask in horizon_masks[1:]):
            raise ValueError("Kairos returned inconsistent horizons across teacher samples")
        return torch.stack(sample_batches), horizon_masks[0]  # [K,B,T,D], [T]

    @torch.no_grad()
    def label_batch(self, batch: dict[str, Any], *, device: torch.device) -> dict[str, float]:
        required = {
            OPD_IMAGE_MASK,
            OPD_IS_ON_POLICY,
            OPD_RAW_STATE,
            OPD_ROBOT_TYPE,
            OPD_TASK,
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"OPD batch is missing inputs: {missing}")
        raw_states = batch[OPD_RAW_STATE].detach().float().cpu()
        if not torch.isfinite(raw_states).all():
            raise ValueError("OPD raw state contains non-finite values")
        batch_size = raw_states.shape[0]
        on_policy = batch[OPD_IS_ON_POLICY].detach().cpu().bool()
        if self.cfg.require_on_policy and not bool(on_policy.all()):
            raise ValueError(
                "OPD requires student-generated rollout states, but this batch is not marked on-policy"
            )

        state_mean = self.kairos_stats.state.mean
        if raw_states.shape[-1] != state_mean.shape[-1]:
            raise ValueError(
                f"Raw student state dim {raw_states.shape[-1]} does not match Kairos state dim "
                f"{state_mean.shape[-1]}. Use the matching Kairos embodiment checkpoint/stats."
            )
        kairos_states = self.kairos_stats.state.normalize(raw_states)
        images = self._build_images(batch, batch_size)
        tasks = [str(value) for value in self._batch_values(batch[OPD_TASK], batch_size)]
        prompts = [self.cfg.prompt_template.format(task=task) for task in tasks]
        robot_types = [str(value) for value in self._batch_values(batch[OPD_ROBOT_TYPE], batch_size)]

        start = time.perf_counter()
        samples, horizon_mask = self._query_samples(images, kairos_states, prompts)
        samples = self.kairos_stats.action.denormalize(samples)
        if not torch.isfinite(samples).all():
            raise ValueError("Kairos returned non-finite actions")

        projected_rows = []
        action_masks = []
        for batch_index, robot_type in enumerate(robot_types):
            projected, action_mask = self.projector.project(
                samples[:, batch_index], raw_states[batch_index], robot_type
            )
            projected_rows.append(projected)
            action_masks.append(action_mask)
        projected_samples = torch.stack(projected_rows, dim=1)  # [K,B,T,D]
        teacher_mean = projected_samples.mean(dim=0)
        empirical_std = projected_samples.std(dim=0, unbiased=False)
        uncertainty_mask = empirical_std <= self.cfg.teacher_std_max
        teacher_std = empirical_std.clamp(
            self.cfg.teacher_std_min, self.cfg.teacher_std_max
        )
        structural_mask = torch.stack(action_masks).unsqueeze(1).expand_as(teacher_mean)
        structural_mask = structural_mask & horizon_mask.view(1, -1, 1)
        action_mask = structural_mask & uncertainty_mask

        batch[OPD_TEACHER_ACTION_MEAN] = teacher_mean.to(device)
        batch[OPD_TEACHER_ACTION_STD] = teacher_std.to(device)
        batch[OPD_TEACHER_ACTION_MASK] = action_mask.to(device)
        valid_std = teacher_std[action_mask]
        return {
            "opd_teacher_s": time.perf_counter() - start,
            "opd_teacher_std": valid_std.mean().item() if valid_std.numel() else 0.0,
            "opd_teacher_valid_ratio": (
                action_mask.sum().float() / structural_mask.sum().clamp_min(1)
            ).item(),
        }
