# DOMINO Evaluation

DOMINO evaluation is provided as an optional benchmark entry next to RoboTwin. The code expects the DOMINO benchmark source to be available locally and keeps scheduler or cluster submission details outside the release scripts.

The bundled entry supports the `internvla_a1_5` policy included in this release. For other policy families, install the corresponding policy package first and extend `evaluation/DOMINO/inference.py` in the same pattern.

## Setup

```bash
# Put the DOMINO benchmark source here, or set DOMINO_ROOT=/path/to/DOMINO.
mkdir -p third_party
git clone https://github.com/H-EmbodVis/DOMINO.git third_party/DOMINO
pip install -r evaluation/DOMINO/requirements.txt
```

If PyTorch3D installation fails, follow the official PyTorch3D installation instructions for your local CUDA and PyTorch versions.

Install DOMINO assets and simulator dependencies following the official DOMINO instructions. If your DOMINO installation reuses RoboTwin's curobo source, initialize RoboTwin as usual:

```bash
git submodule update --init third_party/RoboTwin
```

Cluster-specific settings such as conda activation, `HF_HOME`, cache directories, or DLC submission commands should be configured in your launcher. The scripts accept `PYTHON_BIN`, `DOMINO_ROOT`, `ROBOTWIN_ROOT`, `HF_HOME`, `DOMINO_JOB_CACHE_ROOT`, `GPUS`, and `INFERENCE_BACKEND` as environment variables.

## Run

Single task:

```bash
bash evaluation/DOMINO/eval.sh \
  /path/to/checkpoint/pretrained_model \
  outputs/domino/internvla_a1_5 \
  demo_clean_dynamic \
  8 \
  fm \
  50 \
  100 \
  100000 \
  10 \
  abs \
  float32 \
  10
```

Arguments:

- `checkpoint`: checkpoint `pretrained_model` directory, you can download our fine-tuned ckpt on https://huggingface.co/InternRobotics/InternVLA-A1.5-DOMINO.
- `output_path`: directory where replay videos and metrics are written.
- `task_config`: DOMINO task config, such as `demo_clean_dynamic` or `demo_random_dynamic`.
- `task_idx`: index into `TASK_NAMES` in `evaluation/DOMINO/inference.py`.
- `action_type`: model action type; `fm` is the default flow-matching inference mode.
- `horizon`: number of predicted actions kept from each policy call.
- `test_num`: number of valid episodes to evaluate.
- `max_setup_attempts`: maximum seed retries while searching valid initial states.
- `exec_horizon`: number of predicted joint actions executed before the next policy call. The default is `10`.
- `action_mode`: `abs`, `delta`, or `auto`. The default is `abs`, which executes absolute joint targets. `auto` reads `dataset.action_mode` from `train_config.json` and falls back to `abs`.
- `dtype`: `float32` or `bfloat16`.
- `num_inference_steps`: diffusion or flow inference steps; `0` keeps the checkpoint default.

Run a task range across GPUs:

```bash
GPUS=0,1,2,3,4,5,6,7 bash evaluation/DOMINO/eval_ngpu.sh \
  /path/to/checkpoint/pretrained_model \
  outputs/domino/full_eval \
  demo_clean_dynamic \
  0 \
  34 \
  100 \
  100000 \
  10 \
  abs \
  float32 \
  10 \
  50
```

Use a job scheduler by wrapping `eval_ngpu.sh` or by launching smaller task ranges. Keep scheduler credentials, queue names, and machine paths in your deployment scripts rather than in this repository.

## Results

Each task output directory contains:

- `_metrics.json`: machine-readable success rate, manipulation score, route completion, and penalties.
- `_episodes_detail.json`: per-episode metrics.
- `_metrics_report.txt`: human-readable summary.
- `success_<id>.mp4` and `failure_<id>.mp4`: replay videos when `SAVE_REPLAY_VIDEO=true`.

To compare clean and random settings, run both `demo_clean_dynamic` and `demo_random_dynamic` with different output roots, then aggregate the `_metrics.json` files in each root.
