#!/usr/bin/env python

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

AVAILABLE_POLICIES = (
    "pi0",
    "pi0_fast",
    "pi05",
    "internvla_a1_5",
    "g05",
    "lingbot_vla_2",
)

_CONFIG_MODULES = {
    "pi0": "lerobot.policies.pi0.configuration_pi0",
    "pi0_fast": "lerobot.policies.pi0_fast.configuration_pi0_fast",
    "pi05": "lerobot.policies.pi05.configuration_pi05",
    "internvla_a1_5": "lerobot.policies.internvla_a1_5.configuration_internvla_a1_5",
    "g05": "lerobot.policies.g05.configuration_g05",
    "lingbot_vla_2": "lerobot.policies.lingbot_vla_2.configuration_lingbot_vla_2",
}


def _arg_value(name: str) -> str | None:
    prefix = f"--{name}="
    return next((arg[len(prefix) :] for arg in sys.argv[1:] if arg.startswith(prefix)), None)


def _requested_config_types() -> set[str]:
    """Detect policy/dataset choices before draccus parses the full CLI.

    This lets G0.5 and InternVLA keep their incompatible Transformers versions
    in separate environments without eagerly importing the other model.
    """
    requested = {value for name in ("policy.type", "dataset.type") if (value := _arg_value(name)) is not None}
    for arg_name in ("policy.path", "config_path"):
        config_path = _arg_value(arg_name)
        if config_path is None:
            continue
        path = Path(config_path).expanduser()
        if path.is_dir():
            path = path / "config.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        policy_cfg = data.get("policy", data)
        dataset_cfg = data.get("dataset", {})
        if isinstance(policy_cfg, dict) and policy_cfg.get("type"):
            requested.add(policy_cfg["type"])
        if isinstance(dataset_cfg, dict) and dataset_cfg.get("type"):
            requested.add(dataset_cfg["type"])
    return requested


def _register_runtime_configs() -> None:
    requested = _requested_config_types()
    candidates = requested or set(AVAILABLE_POLICIES)
    for name in candidates:
        module = _CONFIG_MODULES.get(name)
        if module is None:
            continue
        try:
            importlib.import_module(module)
        except (ImportError, ModuleNotFoundError):
            if requested:
                raise


_register_runtime_configs()


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    if name == "lingbot_vla_2":
        from lerobot.policies.lingbot_vla_2.modeling_lingbot_vla_2 import LingBotVLA2Policy

        return LingBotVLA2Policy
    if name == "g05":
        from lerobot.policies.g05.modeling_g05 import G05Policy

        return G05Policy
    if name == "internvla_a1_5":
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy

        return InternVLAA15Policy
    if name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    if name == "pi0_fast":
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        return PI0FastPolicy
    if name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy

    raise ValueError(f"Policy type '{name}' is not available. Available policies: {AVAILABLE_POLICIES}.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "lingbot_vla_2":
        from lerobot.policies.lingbot_vla_2.configuration_lingbot_vla_2 import LingBotVLA2Config

        return LingBotVLA2Config(**kwargs)
    if policy_type == "g05":
        from lerobot.policies.g05.configuration_g05 import G05Config

        return G05Config(**kwargs)
    if policy_type == "internvla_a1_5":
        from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config

        return InternVLAA15Config(**kwargs)
    if policy_type == "pi0":
        from lerobot.policies.pi0.configuration_pi0 import PI0Config

        return PI0Config(**kwargs)
    if policy_type == "pi0_fast":
        from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig

        return PI0FastConfig(**kwargs)
    if policy_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        return PI05Config(**kwargs)

    raise ValueError(
        f"Policy type '{policy_type}' is not available. Available policies: {AVAILABLE_POLICIES}."
    )


def make_policy(cfg: PreTrainedConfig) -> PreTrainedPolicy:
    policy_cls = get_policy_class(cfg.type)

    kwargs = {"config": cfg}
    if cfg.pretrained_path:
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)
    return policy
