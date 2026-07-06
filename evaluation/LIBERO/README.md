# LIBERO Evaluation

Evaluate InternVLA-A1.5 policies on the LIBERO benchmark via a
websocket policy server and a LIBERO client that drives the simulator.

## Table of Contents

- [Installation](#installation)
- [Running Evaluation](#running-evaluation)
- [Viewing Results](#viewing-results)
- [Implementation Notes](#implementation-notes)

---

## Installation

### Step 1: Two Python environments

- **Server env** (default `lerobot_lab`): the policy server env with this repo installed via `pip install -e ".[all]"`.
- **Client env** (default `libero_venv`): LIBERO + robosuite + MuJoCo.

### Step 2: Clone & install LIBERO

**LIBERO Repository:** [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git <LIBERO_REPO_ROOT>
cd <LIBERO_REPO_ROOT>
pip install -e .
```

Set the env var so the eval script can find BDDL files / config root:

```bash
export LIBERO_HOME=<LIBERO_REPO_ROOT>
```

> **Note:** LIBERO renders offscreen via robosuite/MuJoCo. Set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU) in your client env if rendering fails. The eval script defaults to `egl`.

### Step 3: Verify dataset / stats compatibility

The default `STATS_KEY_MODE=panda` / `ROBOT_TYPE_MODE=panda` matches the public `lerobot/libero` dataset on HuggingFace. If your checkpoint was trained on a custom variant, override these env vars (or add a new schema YAML under `src/lerobot/dataset_schemas/configs/`).

## Running Evaluation

```bash
cd <PROJECT_ROOT>/lerobot_lab
export CONDA_ROOT=<MINICONDA_ROOT>
export LIBERO_HOME=<LIBERO_REPO_ROOT>
export CKPT_PATH=<CKPT_PATH>
export STATS_KEY_MODE=suite
export ROBOT_TYPE_MODE=panda
bash evaluation/LIBERO/run_eval_libero_server_client.sh
```

The script auto-detects free GPUs (≥ `GPU_MEM_FREE_THRESHOLD_MB` free, default 30 GiB), launches one policy server per GPU, and dispatches the four LIBERO task suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) from a shared queue.

Common env-var overrides:

- `CKPT_PATH` *(required)*: checkpoint path for the policy server, you can download our fine-tuned ckpt on https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero.
- `LIBERO_HOME` *(required)*: LIBERO repo root.
- `SERVER_ENV` / `CLIENT_ENV` *(defaults `lerobot_lab` / `libero_venv`)*: conda envs.
- `CONDA_ROOT` *(default `$HOME/miniconda3`)*.
- `GPU_IDS` *(e.g. `0,2,5`)*: skip auto-detection and use these GPUs.
- `GPU_MEM_FREE_THRESHOLD_MB` *(default `30000`)*: free-memory threshold for auto-detection.
- `BASE_PORT` *(default `5734`)*: first port; per-worker ports are `BASE_PORT + worker_idx`.
- `NUM_TRIALS_PER_TASK` *(default `50`)*, `MAX_TASKS` *(default `-1`)*, `SEED` *(default `7`)*, `NUM_STEPS_WAIT` *(default `10`)*, `REPLAN_STEPS` *(default `8`)*.
- `STATS_KEY_MODE` / `ROBOT_TYPE_MODE` *(default `panda`)*: use `suite` to bind stats / robot_type to the LIBERO suite name (for multi-subset stats.json checkpoints).
- `VLM_MODEL_PATH` *(InternVLA-A1.5 only)*: override the VLM path baked into the checkpoint.
- `WAN_MODEL_PATH` / `WAN_VAE_PATH`: only used when overriding `ACTION_LOSS_ONLY_FLAG=--no-action_loss_only`. The default `--action_loss_only` skips WAN weight loading, which matches the public InternVLA-A1.5 release.
- `INFERENCE_BACKEND` *(default `standard`)*: `optimized` requires `--action_loss_only`.
- `EVAL_LOG_DIR`: destination log dir (default `outputs/sim_eval/libero/<timestamp>_all`).

