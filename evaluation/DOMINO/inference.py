#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
import tyro
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMINO_ROOT = Path(os.environ.get("DOMINO_ROOT", PROJECT_ROOT / "third_party" / "DOMINO")).expanduser()
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", PROJECT_ROOT / "third_party" / "RoboTwin")).expanduser()

if not DOMINO_ROOT.exists():
    raise RuntimeError(
        "DOMINO is not initialized. Put the DOMINO benchmark source at "
        f"{DOMINO_ROOT} or set DOMINO_ROOT=/path/to/DOMINO."
    )

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DOMINO_ROOT))
sys.path.insert(0, str(DOMINO_ROOT / "policy"))
sys.path.insert(0, str(DOMINO_ROOT / "description" / "utils"))
sys.path.insert(0, str(DOMINO_ROOT / "script"))
if (ROBOTWIN_ROOT / "envs" / "curobo" / "src").exists():
    sys.path.insert(0, str(ROBOTWIN_ROOT / "envs" / "curobo" / "src"))

# DOMINO imports some assets through paths relative to the process cwd.
os.chdir(DOMINO_ROOT)

from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from eval_metrics import AggregatedMetrics, EvalMetricsTracker  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.dataset_schemas import get_schema  # noqa: E402
from lerobot.datasets.utils import load_json  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (  # noqa: E402
    InternVLAA15Config,
)
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (  # noqa: E402
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (  # noqa: E402
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402


TASK_NAMES = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_can_basket",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
]


def _normalize_decoded_subtask(decoded_subtasks) -> str:
    if decoded_subtasks is None:
        return ""
    if isinstance(decoded_subtasks, (list, tuple)):
        text = "" if len(decoded_subtasks) == 0 else str(decoded_subtasks[0])
    else:
        text = str(decoded_subtasks)
    text = text.replace("\n", " ").strip()
    return " ".join(text.split())


def _swap_left_right_words(text: str) -> str:
    placeholder = "__SWAP_LEFT_RIGHT__"
    text = re.sub(r"\bleft\b", placeholder, text)
    text = re.sub(r"\bright\b", "left", text)
    text = re.sub(placeholder, "right", text)
    return text


def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.size == 0:
        return image.astype(np.uint8)
    if image.max() <= 1.0 and image.min() >= 0.0:
        return (image * 255.0).clip(0, 255).astype(np.uint8)
    return image.clip(0, 255).astype(np.uint8)


def _overlay_subtask_bottom(frame_hwc: np.ndarray, subtask_text: str) -> np.ndarray:
    frame_hwc = _convert_to_uint8(frame_hwc)
    image = Image.fromarray(frame_hwc)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    text = subtask_text if subtask_text else "Subtask: (empty)"
    max_width = max(64, width - 20)
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        candidate = word if not cur else f"{cur} {word}"
        text_width = draw.textbbox((0, 0), candidate, font=font)[2]
        if text_width <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]
    lines = lines[:3]

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 2
    box_h = line_height * len(lines) + 12
    y0 = max(0, height - box_h)
    draw.rectangle([(0, y0), (width, height)], fill=(0, 0, 0))

    y = y0 + 6
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    return np.asarray(image)


def _overlay_task_top(frame_hwc: np.ndarray, task_text: str) -> np.ndarray:
    frame_hwc = _convert_to_uint8(frame_hwc)
    image = Image.fromarray(frame_hwc)
    measure_draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    text = task_text if task_text else "Task: (empty)"
    max_width = max(64, width - 20)
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        candidate = word if not cur else f"{cur} {word}"
        text_width = measure_draw.textbbox((0, 0), candidate, font=font)[2]
        if text_width <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]
    lines = lines[:3]

    line_height = measure_draw.textbbox((0, 0), "Ag", font=font)[3] + 2
    box_h = line_height * len(lines) + 12
    canvas = Image.new(image.mode, (width, height + box_h), color=(0, 0, 0))
    canvas.paste(image, (0, box_h))
    draw = ImageDraw.Draw(canvas)

    y = 6
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    return np.asarray(canvas)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def get_embodiment_config(robot_file: str):
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return OmegaConf.load(f)


def class_decorator(task_name: str):
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _resolve_domino_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((DOMINO_ROOT / path).resolve())


def get_camera_config(camera_type: str):
    camera_config_path = DOMINO_ROOT / "task_config" / "_camera_config.yml"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    return camera_cfg[camera_type]


def build_task_args(task_config: str, task_name: str, output_dir: Path, save_env_video: bool):
    with open(DOMINO_ROOT / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        task_args = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return _resolve_domino_path(robot_file)

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = get_embodiment_file(embodiment_type[0])
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    task_args["eval_video_log"] = bool(save_env_video and task_args.get("eval_video_log", False))
    if task_args["eval_video_log"]:
        env_video_dir = output_dir / "domino_env_video"
        env_video_dir.mkdir(parents=True, exist_ok=True)
        task_args["eval_video_save_dir"] = env_video_dir
    return task_args


def _rewrite_config_local_paths(config):
    """Map checkpoint-saved model asset paths onto the local Hugging Face cache."""
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()
    replacements = {
        "models--Qwen--Qwen3.5-2B-Action": hub / "models--Qwen--Qwen3.5-2B-Action",
        "Wan2.2-TI2V-5B": hub / "Wan2.2-TI2V-5B",
    }

    for attr in (
        "vlm_model_name_or_path",
        "wan_checkpoint_path",
        "wan_config_path",
        "vae_path",
    ):
        value = getattr(config, attr, None)
        if not value:
            continue
        value_str = str(value)
        if Path(value_str).exists():
            continue
        for marker, replacement in replacements.items():
            if marker not in value_str:
                continue
            if attr == "vae_path":
                candidate = replacement / "Wan2.2_VAE.pth"
            else:
                candidate = replacement
            if candidate.exists():
                logging.info("Rewriting config.%s from %s to %s", attr, value_str, candidate)
                setattr(config, attr, str(candidate))
            break
    return config


def build_policy_and_transforms(
    ckpt_path: Path,
    stats_key: str,
    resize_size: int,
    dtype: torch.dtype,
    inference_backend: str,
    action_type: str,
):
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    if not isinstance(config, InternVLAA15Config):
        raise TypeError(
            "DOMINO release evaluation currently supports InternVLA-A1.5 checkpoints. "
            f"Got policy config type {type(config).__name__!r}."
        )
    config = _rewrite_config_local_paths(config)
    config.action_loss_only = True
    config.inference_backend = inference_backend
    if hasattr(config, "inference_action_type"):
        config.inference_action_type = action_type
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(ckpt_path, config=config)
    device = torch.device(config.device)
    policy.to(device=device, dtype=dtype)
    policy.eval()

    stats = load_json(ckpt_path / "stats.json")[stats_key]
    stat_keys = ["min", "max", "mean", "std"]

    schema = get_schema(stats_key)

    state_stat = {
        OBS_STATE: {
            k: np.asarray(stats["observation.state"][k])
            for k in stat_keys
        }
    }
    action_stat = {
        ACTION: {
            k: np.asarray(stats["action"][k])
            for k in stat_keys
        }
    }

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=[ACTION],
        mode="mean_std",
        norm_stats=action_stat,
    )

    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size, mapping=schema.image_mapping),
            RemapImageKeyTransformFn(mapping=schema.image_mapping),
            NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=state_stat),
            InternVLAA15ChatProcessorTransformFn(
                mode="eval",
                pretrained_model_name_or_path=getattr(config, "vlm_model_name_or_path", "Qwen/Qwen3.5-2B"),
                tokenize_state=getattr(config, "tokenize_state", True),
                max_state_dim=getattr(config, "max_state_dim", 32),
            ),
            PadStateAndActionTransformFn(
                max_state_dim=getattr(config, "max_state_dim", 32),
                max_action_dim=getattr(config, "max_action_dim", 32),
            ),
            ReorderStateActionTransform(
                state_reorder=schema.state_reorder,
                action_reorder=schema.action_reorder,
            ),
        ]
    )

    return policy, input_transforms, unnormalize_fn


@dataclass
class InferenceArgs:
    task_idx: int = 0
    task_name: str = ""
    task_config: str = "demo_clean_dynamic"
    instruction_type: str = "unseen"
    seed: int = 42
    ckpt_path: Path = Path("InternRobotics/InternVLA-A1.5-DOMINO")
    stats_key: str = "aloha"
    resize_size: int = 224
    image_history_interval: int = 15
    action_mode: str = "abs"
    dtype: str = "float32"
    video_dir: Path = Path("outputs/domino/internvla_a1_5")
    fps: int = 30
    save_replay_video: bool = True
    save_env_video: bool = True
    infer_horizon: int = 50
    execute_horizon: int = 10
    action_horizon_size: int = 50
    action_type: str = "fm"
    num_inference_steps: int = 0
    inference_backend: str = "standard"
    test_num: int = 100
    max_setup_attempts: int = 100000
    debug: bool = False
    log_level: str = "INFO"


def _select_task(args: InferenceArgs) -> str:
    if args.task_name:
        if args.task_name not in TASK_NAMES:
            logging.warning("Task name %s is not in the built-in DOMINO task list", args.task_name)
        return args.task_name
    if args.task_idx < 0 or args.task_idx >= len(TASK_NAMES):
        raise IndexError(f"task_idx must be in [0, {len(TASK_NAMES) - 1}], got {args.task_idx}")
    return TASK_NAMES[args.task_idx]


def _predict_action_chunk(
    policy,
    input_transforms,
    unnormalize_fn,
    observation,
    task: str,
    dtype: torch.dtype,
    args: InferenceArgs,
    image_buffers: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
):
    head_color_list, left_wrist_color_list, right_wrist_color_list = image_buffers

    img = observation["observation"]["head_camera"]["rgb"]
    left_wrist_img = observation["observation"]["left_camera"]["rgb"]
    right_wrist_img = observation["observation"]["right_camera"]["rgb"]

    head_color_list.append(torch.as_tensor(img).contiguous().cuda().to(dtype) / 255.0)
    left_wrist_color_list.append(torch.as_tensor(left_wrist_img).contiguous().cuda().to(dtype) / 255.0)
    right_wrist_color_list.append(torch.as_tensor(right_wrist_img).contiguous().cuda().to(dtype) / 255.0)

    while len(head_color_list) > args.image_history_interval + 1:
        head_color_list.pop(0)
        left_wrist_color_list.pop(0)
        right_wrist_color_list.pop(0)

    past_idx = max(len(head_color_list) - args.image_history_interval - 1, 0)
    image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
    image_hand_left_with_history = torch.stack(
        [left_wrist_color_list[past_idx], left_wrist_color_list[-1]], dim=0
    )
    image_hand_right_with_history = torch.stack(
        [right_wrist_color_list[past_idx], right_wrist_color_list[-1]], dim=0
    )

    init_action = torch.as_tensor(observation["joint_action"]["vector"][None]).contiguous()
    state = torch.from_numpy(observation["joint_action"]["vector"]).float().cuda()
    sample = {
        "observation.images.cam_high": image_head_with_history[-1],
        "observation.images.cam_left_wrist": image_hand_left_with_history[-1],
        "observation.images.cam_right_wrist": image_hand_right_with_history[-1],
        OBS_STATE: state,
        ACTION: torch.zeros(args.infer_horizon, 14, dtype=torch.float32),
        "task": task,
    }
    for key in list(sample.keys()):
        if OBS_IMAGES in key and "mask" not in key:
            sample[key] = sample[key].permute(2, 0, 1)

    sample = input_transforms(sample)
    transformed_image_to_save = sample[f"{OBS_IMAGES}.image0"]

    inputs = {}
    for key, value in sample.items():
        if key == "task":
            inputs[key] = [value]
        elif value.dtype == torch.int64 or value.dtype == torch.bool:
            inputs[key] = value[None].cuda()
        else:
            inputs[key] = value[None].cuda().to(dtype=dtype)

    with torch.no_grad():
        action_pred = policy.predict_action_chunk(inputs)

    action_pred = action_pred[0, : args.infer_horizon, :16]
    action_pred = torch.cat(
        [
            action_pred[:, :6],
            action_pred[:, 7:8],
            action_pred[:, 8:14],
            action_pred[:, 15:16],
        ],
        dim=1,
    )
    action_pred = unnormalize_fn({ACTION: action_pred})[ACTION]
    if args.action_mode == "delta":
        init_action[:, 6] = 0.0
        init_action[:, 13] = 0.0
        init_action = init_action.to(device=action_pred.device)
        action_pred += init_action

    execute_horizon = max(1, min(args.execute_horizon, action_pred.shape[0]))
    action_plan = action_pred[:execute_horizon].cpu().numpy()
    replay_frame = _convert_to_uint8(transformed_image_to_save.detach().float().cpu().numpy())
    return action_plan, replay_frame


def _save_replay_video(args: InferenceArgs, suffix: str, episode_idx: int, replay_images, replay_subtasks, replay_tasks):
    if not args.save_replay_video or not replay_images:
        return
    video_frames = []
    for frame_chw, subtask_text, task_text in zip(replay_images, replay_subtasks, replay_tasks):
        frame_hwc = np.asarray(frame_chw).transpose(1, 2, 0)
        frame_hwc = _overlay_task_top(frame_hwc, f"Task: {task_text}")
        frame_hwc = _overlay_subtask_bottom(frame_hwc, f"Subtask: {subtask_text}")
        video_frames.append(frame_hwc)
    imageio.mimwrite(
        args.video_dir / f"{suffix}_{episode_idx}.mp4",
        video_frames,
        fps=args.fps,
    )


def _write_metrics(args: InferenceArgs, task_name: str, task_args: dict, aggregated_metrics: AggregatedMetrics):
    summary = aggregated_metrics.get_summary()
    payload = {
        "task_name": task_name,
        "task_config": task_args["task_config"],
        "use_dynamic": task_args.get("use_dynamic", False),
        "dynamic_level": task_args.get("dynamic_level"),
        "dynamic_coefficient": task_args.get("dynamic_coefficient"),
        "summary": summary,
    }
    with open(args.video_dir / "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)
    with open(args.video_dir / "_episodes_detail.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(aggregated_metrics.get_all_episodes()), f, indent=2)
    with open(args.video_dir / "_metrics_report.txt", "w", encoding="utf-8") as f:
        f.write(aggregated_metrics.to_detailed_report())


def _close_env_safely(task_env, clear_cache: bool = False, release: bool = False):
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        pass
    if release and hasattr(task_env, "release_episode_resources"):
        try:
            task_env.release_episode_resources()
        except Exception:
            pass


def infer_once(args: InferenceArgs):
    task_name = _select_task(args)
    task_args = build_task_args(args.task_config, task_name, args.video_dir, args.save_env_video)
    task_env = class_decorator(task_args["task_name"])

    dtype_name = str(args.dtype).lower()
    if dtype_name in {"float32", "fp32"}:
        dtype = torch.float32
    elif dtype_name in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    else:
        raise ValueError(f"dtype must be float32/fp32/bfloat16/bf16, got {args.dtype}")
    policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
        args.ckpt_path,
        args.stats_key,
        args.resize_size,
        dtype,
        args.inference_backend,
        args.action_type,
    )
    if args.num_inference_steps > 0:
        policy.config.num_inference_steps = args.num_inference_steps
        if hasattr(policy, "model") and hasattr(policy.model, "config"):
            policy.model.config.num_inference_steps = args.num_inference_steps

    logging.info("=" * 80)
    logging.info("Starting DOMINO inference")
    logging.info("task=%s config=%s seed=%s", task_name, args.task_config, args.seed)
    logging.info(
        "use_dynamic=%s dynamic_level=%s dynamic_coefficient=%s",
        task_args.get("use_dynamic", False),
        task_args.get("dynamic_level"),
        task_args.get("dynamic_coefficient"),
    )
    logging.info("infer_horizon=%s execute_horizon=%s", args.infer_horizon, args.execute_horizon)
    logging.info("dtype=%s num_inference_steps=%s", args.dtype, policy.config.num_inference_steps)

    task_env.suc = 0
    task_env.test_num = 0
    expert_check = True

    now_id = 0
    succ_seed = 0
    np.random.seed(args.seed)
    st_seed = 100000 * (1 + args.seed)
    now_seed = st_seed
    clear_cache_freq = task_args["clear_cache_freq"]
    task_args["eval_mode"] = True
    setup_attempts = 0
    aggregated_metrics = AggregatedMetrics()

    camera_config = get_camera_config(task_args["camera"]["head_camera_type"])
    video_size = f"{camera_config['w']}x{camera_config['h']}"

    while succ_seed < args.test_num:
        setup_attempts += 1
        if args.max_setup_attempts > 0 and setup_attempts > args.max_setup_attempts:
            raise RuntimeError(
                f"Exceeded max_setup_attempts={args.max_setup_attempts} "
                f"while searching valid seeds for task={task_name}; last_seed={now_seed}"
            )

        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0
        episode_info = None

        if expert_check:
            try:
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
                print("TASK_ENV.setup_demo")
                episode_info = task_env.play_once()
                print("TASK_ENV.play_once")
                task_env.close_env()
            except UnStableError as e:
                print(f"[UnStableError] task={task_name} seed={now_seed} err={e}")
                _close_env_safely(task_env)
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(f"[Exception] task={task_name} seed={now_seed} err={repr(e)}")
                print(traceback.format_exc())
                _close_env_safely(task_env)
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

        if (not expert_check) or (task_env.plan_success and task_env.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        saved_dynamic_motion_info = getattr(task_env, "_saved_dynamic_motion_info", None)
        task_args["render_freq"] = render_freq

        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
            if saved_dynamic_motion_info is not None:
                task_env._saved_dynamic_motion_info = saved_dynamic_motion_info

            episode_info_list = [episode_info["info"]] if episode_info is not None else [{}]
            results = generate_episode_descriptions(task_name, episode_info_list, args.test_num)
            instruction = np.random.choice(results[0][args.instruction_type])
            task_env.set_instruction(instruction=instruction)

            if task_args.get("use_dynamic", False):
                dynamic_init_success = task_env.init_dynamic_motion_for_eval()
                if not dynamic_init_success:
                    print(f"Error: Failed to initialize dynamic motion for seed {now_seed}, skipping...")
                    _close_env_safely(task_env, release=True)
                    now_seed += 1
                    succ_seed -= 1
                    continue

            check_z_threshold = None
            target_actor_check = None
            start_z = None
            has_lifted = False
            stop_on_contact = True
            dynamic_config = getattr(task_env, "get_dynamic_motion_config", lambda: None)()
            if dynamic_config and "check_z_threshold" in dynamic_config:
                check_z_threshold = dynamic_config["check_z_threshold"]
                target_actor_check = dynamic_config.get("check_z_actor", dynamic_config["target_actor"])
                start_z = target_actor_check.get_pose().p[2]
            if dynamic_config and "stop_on_contact" in dynamic_config:
                stop_on_contact = dynamic_config["stop_on_contact"]

            metrics_tracker = EvalMetricsTracker(task_env, task_args)
            metrics_tracker.on_episode_start()
            task_env._metrics_tracker = metrics_tracker

            ffmpeg = None
            if task_env.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                task_env._set_eval_video_ffmpeg(ffmpeg)

            succ = False
            fail_reason = None
            policy.reset()
            action_plan = deque([], maxlen=max(args.action_horizon_size, args.execute_horizon, 1))
            replay_images = []
            replay_subtasks = []
            replay_tasks = []
            current_subtask_text = ""
            image_buffers: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] = ([], [], [])

            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                task = task_env.get_instruction()
                if not action_plan:
                    action_chunk, replay_frame = _predict_action_chunk(
                        policy,
                        input_transforms,
                        unnormalize_fn,
                        observation,
                        task,
                        dtype,
                        args,
                        image_buffers,
                    )
                    replay_images.append(replay_frame)
                    replay_subtasks.append(current_subtask_text)
                    replay_tasks.append(task)
                    decoded_subtasks = None
                    current_subtask_text = _normalize_decoded_subtask(decoded_subtasks)
                    replay_subtasks[-1] = current_subtask_text
                    action_plan.extend(action_chunk)

                action = action_plan.popleft()[:14]
                action[6] = np.clip(action[6], 0, 1)
                action[13] = np.clip(action[13], 0, 1)
                task_env.take_action(action, action_type="qpos")

                if check_z_threshold is not None and not has_lifted:
                    curr_z = target_actor_check.get_pose().p[2]
                    if curr_z - start_z > check_z_threshold:
                        has_lifted = True

                if task_env.eval_success:
                    if check_z_threshold is not None and not has_lifted:
                        succ = False
                        fail_reason = "object_not_lifted"
                        break
                    succ = True
                    break

                out_of_bounds = False
                if task_args.get("use_dynamic", False):
                    if task_env.check_gripper_contact_dynamic_object() and stop_on_contact:
                        task_env.stop_dynamic_object_motion()

                    if not getattr(task_env, "_dynamic_object_stopped", False):
                        if not task_env.check_dynamic_object_boundary():
                            task_env.eval_fail = True
                            fail_reason = "out_of_bounds"
                            out_of_bounds = True

                if out_of_bounds:
                    metrics_tracker.record_out_of_bounds()
                    break
                if task_env.eval_fail:
                    fail_reason = "eval_fail"
                    break

            if getattr(task_env, "eval_video_ffmpeg", None) is not None:
                task_env._del_eval_video_ffmpeg()

            if succ:
                task_env.suc += 1
                print("\033[92mSuccess!\033[0m")
            elif fail_reason == "out_of_bounds":
                print("\033[91mFail! (Object out of bounds)\033[0m")
            else:
                print("\033[91mFail!\033[0m")

            episode_metrics = metrics_tracker.get_episode_metrics(succ, fail_reason, seed=now_seed)
            aggregated_metrics.add_episode(episode_metrics)
            task_env._metrics_tracker = None

            print(
                f"  MS: \033[96m{episode_metrics.manipulation_score:.1f}\033[0m | "
                f"RC: \033[96m{episode_metrics.route_completion:.1f}%\033[0m"
            )
            if episode_metrics.penalty_events:
                penalty_str = ", ".join(
                    [f"{p.event_type}({p.penalty_factor})" for p in episode_metrics.penalty_events]
                )
                print(f"  Penalties: \033[93m{penalty_str}\033[0m")

            suffix = "success" if succ else "failure"
            _save_replay_video(args, suffix, succ_seed, replay_images, replay_subtasks, replay_tasks)

            now_id += 1
            clear_cache = (task_env.test_num + 1) % clear_cache_freq == 0
            task_env.close_env(clear_cache=clear_cache)
            if task_env.render_freq:
                task_env.viewer.close()
            task_env.release_episode_resources()

            task_env.test_num += 1
            summary = aggregated_metrics.get_summary()
            print(
                f"\033[93m{task_name}\033[0m | \033[92m{task_args['task_config']}\033[0m\n"
                f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
                f"\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m | "
                f"Avg MS: \033[95m{summary['manipulation_score_mean']:.1f}\033[0m | "
                f"current seed: \033[90m{now_seed}\033[0m\n"
            )
            _write_metrics(args, task_name, task_args, aggregated_metrics)
            now_seed += 1

        except Exception as e:
            print(f"[PolicyException] task={task_name} seed={now_seed} err={repr(e)}")
            print(traceback.format_exc())
            _close_env_safely(task_env, release=True)
            raise

    return aggregated_metrics


def _validate_output_dir(video_dir: Path) -> Path:
    return Path(video_dir).expanduser().resolve()


def main(args: InferenceArgs):
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("curobo").setLevel(logging.WARNING)

    args.video_dir = _validate_output_dir(args.video_dir)
    if args.video_dir.exists():
        shutil.rmtree(args.video_dir)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    if args.execute_horizon <= 0:
        raise ValueError(f"execute_horizon must be positive, got {args.execute_horizon}")

    logging.info("=" * 80)
    logging.info("Starting DOMINO evaluation")
    logging.info("task_idx=%s task_name=%s ckpt=%s", args.task_idx, args.task_name, args.ckpt_path)
    logging.info("output=%s", args.video_dir)
    sys.stdout.flush()

    try:
        infer_once(args)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.stdout.flush()
    except Exception as e:
        logging.error("DOMINO inference failed: %s", e, exc_info=True)
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    os.chdir(DOMINO_ROOT)
    tyro.cli(main)
