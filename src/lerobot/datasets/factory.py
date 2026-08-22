#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import re
import logging
from pathlib import Path
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig

import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import load_json, cast_stats_to_numpy
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transformed_dataset import (
    TransformedLeRobotDataset, 
    TransformedStreamingLeRobotDataset, 
    MultiLeRobotDataset, 
    MultiStreamingLeRobotDataset, 
)
from lerobot.datasets.vqa_dataset import (
    VQADataset,
    TransformedVQADataset,
    MultiVQADataset,
    MixedMultimodalDataset,
)
from lerobot.datasets.sampler import (
    GroupedBatchSampler,
    MultiLeRobotWeightedSampler,
    MultiMixedWeightedSampler,
)

from lerobot.dataset_schemas import get_schema

from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD
from lerobot.utils.constants import HF_LEROBOT_HOME

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def _multimodal_collate(batch):
    """Collate robot/VQA batches with variable numbers of visual tokens."""
    if not batch or not isinstance(batch[0], dict):
        return default_collate(batch)

    ignored_keys = {"dataset_index", "repo_id"}
    visual_keys = (
        f"{OBS_PREFIX}pixel_values",
        f"{OBS_PREFIX}image_grid_thw",
    )

    collated = {}
    for key in batch[0]:
        if key in ignored_keys:
            continue
        values = [sample[key] for sample in batch]
        if key in visual_keys and all(torch.is_tensor(v) for v in values):
            collated[key] = torch.cat(values, dim=0)
        else:
            collated[key] = default_collate(values)
    return collated

def get_rank_and_world_size() -> tuple[int, int]:
    """Get the global rank and world_size.

    If torch.distributed is not initialized, fall back to (0, 1).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

def find_info_json_path_for_repo(cfg: TrainPipelineConfig, repo_id: str) -> Path | None:
    if cfg.dataset.root is not None: 
        root = Path(cfg.dataset.root)
        return root / repo_id / "meta" / "info.json"
    else:
        return HF_LEROBOT_HOME / repo_id / "meta" / "info.json"


def parse_repo_ids(repo_ids_cfg: str | list[str] | tuple[str, ...]) -> list[str]:
    """Parse repo ids from a YAML list or a whitespace-separated string."""
    if isinstance(repo_ids_cfg, str):
        repo_ids_cfg = repo_ids_cfg.strip()
        if not repo_ids_cfg:
            return []
        return repo_ids_cfg.split()

    if isinstance(repo_ids_cfg, (list, tuple)):
        return [str(rid).strip() for rid in repo_ids_cfg if str(rid).strip()]

    raise TypeError(
        f"Unsupported repo_id type: {type(repo_ids_cfg)}. "
        "Expected str, list[str], or tuple[str, ...]."
    )


def load_info_for_repos(
    cfg: TrainPipelineConfig,
    repo_ids: list[str],
) -> dict[str, int]:
    frames_map: dict[str, int] = {}
    episodes_map: dict[str, int] = {}

    for rid in repo_ids:
        info_path = find_info_json_path_for_repo(cfg, rid)
        info = load_json(info_path)
        frames_map[rid] = int(info["total_frames"])
        episodes_map[rid] = int(info["total_episodes"])

    return frames_map, episodes_map


def compute_balanced_repo_assignment(
    repo_ids: list[str],
    frames_map: dict[str, int],
    world_size: int,
) -> list[list[str]]:
    """
    Compute a balanced assignment of repo_ids to ranks based on total_frames.

    Goals:
    - Every rank gets at least one repo_id.
    - The total_frames sum per rank is as close as possible across ranks.
    - For len(repo_ids) >= world_size:
        * Each repo_id is used at most once (no duplication).
    - For len(repo_ids) < world_size:
        * Allow duplication of repo_ids across ranks to avoid empty ranks.

    Strategy:
    - Use a greedy LPT-style algorithm:
        1) Sort repo_ids by descending total_frames (ties broken by repo_id string).
        2) Repeatedly assign the next repo_id to the rank with the smallest current load.
        3) If len(repo_ids) < world_size, keep cycling over repo_ids until we have
           assigned at least one repo_id to every rank.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    n = len(repo_ids)
    if n == 0:
        raise ValueError("compute_balanced_repo_assignment: repo_ids is empty.")

    # Initialize per-rank containers
    rank_to_repos: list[list[str]] = [[] for _ in range(world_size)]
    rank_loads: list[int] = [0 for _ in range(world_size)]

    # Sort repo_ids by descending total_frames, tie-breaker by repo_id for determinism
    def frames_key(rid: str) -> tuple[int, str]:
        # Use negative so that larger total_frames come first
        return (-frames_map.get(rid, 0), rid)

    sorted_repos = sorted(repo_ids, key=frames_key)

    if n >= world_size:
        # Case A: enough repos to avoid duplication.
        # Greedy: always assign the next repo to the rank with the smallest current load.
        for rid in sorted_repos:
            min_rank = min(range(world_size), key=lambda r: (rank_loads[r], r))
            rank_to_repos[min_rank].append(rid)
            rank_loads[min_rank] += frames_map.get(rid, 0)
    else:
        # Case B: fewer repos than ranks -> we must allow duplication
        #
        # Strategy:
        # - Still use LPT greedily, but we keep "expanding" sorted_repos in cycles
        #   until every rank has at least one repo.
        # - This keeps total_frames per rank roughly balanced, even with repetition.
        assignments_done = 0
        idx = 0
        while assignments_done < world_size:
            rid = sorted_repos[idx % n]
            min_rank = min(range(world_size), key=lambda r: (rank_loads[r], r))
            rank_to_repos[min_rank].append(rid)
            rank_loads[min_rank] += frames_map.get(rid, 0)

            assignments_done += 1
            idx += 1

    logging.info("[dist_loading] total_frames=%d", sum(rank_loads))
    for r in range(world_size):
        logging.info(
            "[dist_loading] rank %d: num_repos=%d num_frames=%d",
            r,
            len(rank_to_repos[r]),
            rank_loads[r],
        )

    return rank_to_repos


def compute_repo_weights(
    repo_ids: list[str],
    frames_map: dict[str, int],
    episodes_map: dict[str, int],
    groups_cfg: DictConfig,
) -> dict[str, float]:
    """
    Compute global repo-level sampling weights from YAML group config.

    Returns:
        dict[str, float]: repo_id -> normalized weight (sum to 1)
    """
    repo_to_group = {}
    group_to_repos = defaultdict(list)

    for rid in repo_ids:
        matched = False
        for g in groups_cfg.groups:
            if re.search(g.match, rid):
                repo_to_group[rid] = g.name
                group_to_repos[g.name].append(rid)
                matched = True
                break
        if not matched:
            repo_to_group[rid] = "__default__"
            group_to_repos["__default__"].append(rid)

    group_budget = {}
    for g in groups_cfg.groups:
        group_budget[g.name] = float(g.total_weight)

    if "__default__" in group_to_repos:
        group_budget["__default__"] = float(groups_cfg.default.total_weight)

    repo_weights = {}

    for group_name, repos in group_to_repos.items():
        budget = group_budget[group_name]

        if group_name == "__default__":
            inside = groups_cfg.default.inside
            gamma = float(getattr(groups_cfg.default, "gamma", 1.0))
        else:
            g = next(x for x in groups_cfg.groups if x.name == group_name)
            inside = g.inside
            gamma = float(getattr(g, "gamma", 1.0))

        # compute raw scores
        scores = []
        for rid in repos:
            if inside == "frames_pow":
                s = frames_map[rid] ** gamma
            elif inside == "episodes_pow":
                s = episodes_map[rid] ** gamma
            elif inside == "uniform":
                s = 1.0
            else:
                raise ValueError(f"Unknown inside mode: {inside}")
            scores.append(s)

        total = sum(scores)
        for rid, s in zip(repos, scores):
            repo_weights[rid] = budget * (s / total)

    Z = sum(repo_weights.values())
    for rid in repo_weights:
        repo_weights[rid] /= Z

    return repo_weights


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    
    schema = get_schema(ds_meta.robot_type)
    action_keys = schema.get_action_keys()
    image_keys = list(schema.image_mapping.keys())
    
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        elif key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        elif key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        elif key in action_keys and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        
        if key in image_keys and hasattr(cfg, "image_delta_indices") and cfg.image_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.image_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _build_single_dataset(
    cfg: TrainPipelineConfig,
    repo_id: str,
    image_transforms,
    seed_offset: int, 
):
    """
    Build one dataset (single robot) including:
    - metadata
    - delta timestamps
    - LeRobotDataset (or streaming version)
    - ImageNet stats substitution
    - external stats loading (if enabled)
    - TransformedLeRobotDataset wrapping

    Returns:
        transformed_dataset,
        stats_copy,
        robot_type
    """

    # Load metadata + determine delta timestamps
    ds_meta = LeRobotDatasetMetadata(
        repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if cfg.dataset.streaming:
        root = cfg.dataset.root if cfg.dataset.root is not None else HF_LEROBOT_HOME / repo_id
        base_ds = StreamingLeRobotDataset(
            repo_id=repo_id,
            root=root,          
            episodes=cfg.dataset.episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            revision=cfg.dataset.revision,
            force_cache_sync=False,         
            streaming=True,                 
            buffer_size=cfg.dataset.buffer_size,               
            max_num_shards=cfg.num_workers,
            seed=cfg.seed + seed_offset,
            rng=None,
            shuffle=True,
        )
        transformed_ds = TransformedStreamingLeRobotDataset.from_base(
            base_ds,
            cfg.dataset.data_transforms.inputs,
        )
    else:

        # Create the actual LeRobot dataset (non-streaming recommended for multi-robot)
        base_ds = LeRobotDataset(
            repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            # tolerance_s=0.2, 
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        transformed_ds = TransformedLeRobotDataset.from_base(
            base_ds,
            cfg.dataset.data_transforms.inputs,
        )

    # Optional: override stats using ImageNet norm
    if cfg.dataset.use_imagenet_stats:
        for key in base_ds.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                base_ds.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    robot_type = base_ds.meta.robot_type

    # Optional: load aggregated external stats
    if cfg.dataset.use_external_stats:
        if cfg.dataset.external_stats_path is not None:
            stat_path = Path(cfg.dataset.external_stats_path)
        else:
            action_mode = cfg.dataset.action_mode
            stat_path = HF_LEROBOT_HOME / f"stats/{robot_type}/{action_mode}/stats.json"
        # stat_path = HF_LEROBOT_HOME / f"stats/{robot_type}/{action_mode}/{repo_id}/stats.json"

        if stat_path.exists():
            ext_stats = cast_stats_to_numpy(load_json(stat_path))
            logging.info(f"Using external stats from {stat_path}")
            base_ds.meta.stats.update(ext_stats)
        else:
            raise FileNotFoundError(
                f"use_external_stats=True but no file found at {stat_path}."
            )

    stats_copy = base_ds.meta.stats.copy()

    return transformed_ds, stats_copy, robot_type


def _make_vqa_dataset(cfg: TrainPipelineConfig):
    """Create default jsonl VQA dataset(s), sharded across ranks when needed."""
    if cfg.vqa_dataset is None:
        return None
    vqa_cfg = cfg.vqa_dataset
    if not vqa_cfg.repo_id:
        return None

    all_repo_ids = parse_repo_ids(vqa_cfg.repo_id)
    if not all_repo_ids:
        return None
    logging.info("[make_vqa_dataset] all_repo_ids=%s", all_repo_ids)

    if cfg.dataset.dist_loading:
        rank, world_size = get_rank_and_world_size()
        if len(all_repo_ids) >= world_size:
            repo_ids = [rid for i, rid in enumerate(all_repo_ids) if i % world_size == rank]
        else:
            repo_ids = [all_repo_ids[rank % len(all_repo_ids)]]
        logging.info("[rank=%d/%d] VQA repos: %s", rank, world_size, repo_ids)
    else:
        repo_ids = all_repo_ids

    transforms = vqa_cfg.data_transforms.inputs or None
    seed = getattr(vqa_cfg, "seed", cfg.seed)
    root = vqa_cfg.root or None

    datasets = [
        TransformedVQADataset.from_base(
            VQADataset(root=root, repo_id=rid, seed=seed),
            transforms=transforms,
        )
        for rid in repo_ids
    ]

    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return MultiVQADataset(datasets)


def make_dataset(cfg: TrainPipelineConfig):
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    all_data_stats = {}
    all_repo_ids = parse_repo_ids(cfg.dataset.repo_id)
    if not all_repo_ids:
        raise ValueError(
            "cfg.dataset.repo_id is empty. "
            "Please provide repo ids as a YAML list or a whitespace-separated string."
        )
    logging.info(
        f"[make_dataset] all_repo_ids={all_repo_ids}"
    )

    frames_map, episodes_map = load_info_for_repos(cfg, all_repo_ids)
    weight_cfg = None
    if cfg.dataset.weight_rules_path is not None:
        weight_cfg = OmegaConf.load(cfg.dataset.weight_rules_path)
        repo_weights_map = compute_repo_weights(
            all_repo_ids,
            frames_map,
            episodes_map,
            weight_cfg,
        )
    else:
        repo_weights_map = None

    rank, world_size = get_rank_and_world_size()
    if cfg.dataset.dist_loading:
        rank_to_repos = compute_balanced_repo_assignment(
            all_repo_ids,
            frames_map,
            world_size,
        )
        repo_ids = rank_to_repos[rank]
        logging.info("[make_dataset] dist_loading=True, using total_frames-balanced assignment.")
    else:
        repo_ids = all_repo_ids

    if repo_weights_map is not None:
        repo_summary = "\n".join(
            f"[rank {rank}] repo_id={rid}, weight={repo_weights_map[rid]:.6f}"
            for rid in repo_ids
        )
    else:
        repo_summary = "\n".join(f"[rank {rank}] repo_id={rid}" for rid in repo_ids)
    logging.info("[rank=%02d/%02d] repo_ids_for_this_rank:\n%s", rank, world_size, repo_summary)

    if len(repo_ids) == 1:
        repo_id = repo_ids[0]

        transformed_ds, stats_copy, robot_type = _build_single_dataset(
            cfg, repo_id, image_transforms, rank
        )
        all_data_stats[robot_type] = stats_copy

        robot_ds = transformed_ds
    else:
        transformed_datasets = []

        for rid, repo_id in enumerate(repo_ids):
            transformed_ds, stats_copy, robot_type = _build_single_dataset(
                cfg, repo_id, image_transforms, rank * 128 + rid
            )
            transformed_datasets.append(transformed_ds)
            all_data_stats[robot_type] = stats_copy

        dataset_weights = [
                repo_weights_map[ds.repo_id]
                for ds in transformed_datasets
            ] if repo_weights_map is not None else None

        if not cfg.dataset.streaming:
            robot_ds = MultiLeRobotDataset(transformed_datasets, dataset_weights=dataset_weights)
        else:
            robot_ds = MultiStreamingLeRobotDataset(transformed_datasets, dataset_weights=dataset_weights, seed=cfg.seed)

    # --- Optionally mix with VQA data ---
    vqa_ds = _make_vqa_dataset(cfg)
    logging.info(vqa_ds)
    if vqa_ds is not None:
        vqa_weight = cfg.vqa_dataset.weight
        mixed_ds = MixedMultimodalDataset(
            datasets=[robot_ds, vqa_ds],
            dataset_weights=[1.0 - vqa_weight, vqa_weight],
        )
        logging.info("Mixed dataset created:\n%s", mixed_ds)
        return mixed_ds, all_data_stats

    return robot_ds, all_data_stats


def _get_grpo_group_ids(dataset, group_id_key: str) -> list[int]:
    """Read lightweight group identifiers without running image/text transforms."""

    if isinstance(dataset, MultiLeRobotDataset):
        raise ValueError("GRPO currently requires exactly one rollout dataset")
    if not hasattr(dataset, "hf_dataset") or dataset.hf_dataset is None:
        raise ValueError("GRPO requires a map-style LeRobot dataset with loaded metadata")
    if group_id_key not in dataset.hf_dataset.features:
        raise KeyError(
            f"GRPO group id field {group_id_key!r} is missing from {dataset.repo_id!r}"
        )

    group_ids: list[int] = []
    for value in dataset.hf_dataset[group_id_key]:
        if hasattr(value, "item"):
            value = value.item()
        try:
            numeric_id = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"GRPO group ids must be integer-compatible, got {value!r}") from exc
        group_ids.append(numeric_id)
    return group_ids


def make_dataloader(
    cfg: TrainPipelineConfig,
    dataset,
) -> tuple[torch.utils.data.DataLoader, bool]:
    """Build the training DataLoader for *any* dataset returned by :func:`make_dataset`.

    Returns:
        (dataloader, self_managed)

        ``self_managed=True`` means the dataloader already handles its own
        sampling / distribution, so it should **not** be wrapped by
        ``accelerator.prepare`` and batches must be sent to device manually.
    """
    num_workers = cfg.num_workers
    prefetch = 2 if num_workers > 0 else None
    use_mm_collate = (
        getattr(cfg.policy, "type", None) == "internvla_a1_5"
        and bool(getattr(cfg.policy, "enable_vqa_loss", False))
    )
    collate_fn = _multimodal_collate # if use_mm_collate else None

    if cfg.rl.enable and cfg.rl.algorithm == "grpo":
        if hasattr(dataset, "dataset_weights") and dataset.dataset_weights is not None:
            raise ValueError("GRPO grouped batching is incompatible with weighted dataset sampling")
        group_ids = _get_grpo_group_ids(dataset, cfg.dataset.rl_group_id_key)
        batch_sampler = GroupedBatchSampler(
            group_ids,
            group_size=cfg.rl.group_size,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed or 0,
        )
        dl = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=prefetch,
        )
        return dl, False

    if (
        not cfg.dataset.streaming
        and hasattr(dataset, "dataset_weights")
        and dataset.dataset_weights is not None
    ):
        if isinstance(dataset, MixedMultimodalDataset):
            sampler = MultiMixedWeightedSampler(dataset=dataset)
        else:
            sampler = MultiLeRobotWeightedSampler(dataset=dataset)
        dl = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=prefetch,
        )
        return dl, False

    if cfg.dataset.streaming:
        dl = torch.utils.data.DataLoader(
            dataset,
            num_workers=1,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=4,
        )
        return dl, False

    dl = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=prefetch,
    )
    return dl, False
