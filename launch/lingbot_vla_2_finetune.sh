#!/usr/bin/env bash
set -euo pipefail

# Post-train LingBot-VLA 2.0 through the same LeRobot/Accelerate entrypoint as
# the other policies. The official checkout supplies only model code/weights.
#
# Usage:
#   bash launch/lingbot_vla_2_finetune.sh <dataset_repo_id> [abs|delta]
#
# Required:
#   LINGBOT_VLA_2_SOURCE_PATH  official Robbyant/lingbot-vla-v2 checkout
#   LINGBOT_VLA_2_MODEL_PATH   local or Hugging Face native-depth model path
#   QWEN3_VL_PATH              Qwen3-VL-4B-Instruct processor/model path

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_repo_id> [abs|delta]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_REPO_ID="$1"
ACTION_MODE="${2:-abs}"

LINGBOT_VLA_2_CONDA_ENV="${LINGBOT_VLA_2_CONDA_ENV:-lingbotvla2}"
LINGBOT_VLA_2_CONDA_ROOT="${LINGBOT_VLA_2_CONDA_ROOT:-${_CONDA_ROOT:-}}"
if [[ -n "${LINGBOT_VLA_2_CONDA_ROOT}" && -f "${LINGBOT_VLA_2_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${LINGBOT_VLA_2_CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${LINGBOT_VLA_2_CONDA_ENV}"
fi

LINGBOT_VLA_2_SOURCE_PATH="${LINGBOT_VLA_2_SOURCE_PATH:-${PROJECT_ROOT}/third_party/lingbot-vla-v2}"
LINGBOT_VLA_2_MODEL_PATH="${LINGBOT_VLA_2_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-vla-v2-6b}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen3-VL-4B-Instruct}"

if [[ ! -d "${LINGBOT_VLA_2_SOURCE_PATH}/lingbotvla" ]]; then
    echo "Missing official LingBot-VLA 2.0 checkout: ${LINGBOT_VLA_2_SOURCE_PATH}" >&2
    exit 1
fi
if [[ ! -d "${LINGBOT_VLA_2_MODEL_PATH}" && "${LINGBOT_VLA_2_MODEL_PATH}" != */* ]]; then
    echo "Missing LingBot-VLA 2.0 model: ${LINGBOT_VLA_2_MODEL_PATH}" >&2
    exit 1
fi
if [[ "${QWEN3_VL_PATH}" = /* && ! -d "${QWEN3_VL_PATH}" ]]; then
    echo "Missing local Qwen3-VL model: ${QWEN3_VL_PATH}" >&2
    echo "Run scripts/download_lingbot_vla_2_modelscope.sh first." >&2
    exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6381}"
PROC_PER_NODE="${PROC_PER_NODE:-1}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
NUM_CAMERAS="${NUM_CAMERAS:-3}"
ENVIRONMENT_ACTION_DIM="${ENVIRONMENT_ACTION_DIM:-55}"
JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-lingbot-vla-2-${DATASET_REPO_ID//[\/ ]/_}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/lingbot_vla_2/${JOB_NAME}}"

export PYTHONPATH="${PROJECT_ROOT}/src:${LINGBOT_VLA_2_SOURCE_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ARGS=(
    --multi_gpu
    --mixed_precision=bf16
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    "${PROJECT_ROOT}/src/lerobot/scripts/lerobot_train.py"

    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
    --num_workers=4

    --policy.type=lingbot_vla_2
    --policy.repo_id=lerobot_lab/lingbot_vla_2
    --policy.push_to_hub=false
    --policy.lingbot_vla_source_path="${LINGBOT_VLA_2_SOURCE_PATH}"
    --policy.official_pretrained_model="${LINGBOT_VLA_2_MODEL_PATH}"
    --policy.qwen_processor_path="${QWEN3_VL_PATH}"
    --policy.num_cameras="${NUM_CAMERAS}"
    --policy.environment_action_dim="${ENVIRONMENT_ACTION_DIM}"
    --policy.disable_auxiliary_distillation=true
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16

    --dataset.type=lingbot_vla_2
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_MODE}"
    --dataset.num_cameras="${NUM_CAMERAS}"
    --dataset.qwen_processor_path="${QWEN3_VL_PATH}"
    --dataset.preprocess_in_dataset=true
    --dataset.use_imagenet_stats=false

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${TRAIN_STEPS}"
    --save_freq=5000
    --log_freq=50
    --wandb.enable=false
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # EXTRA_ARGS is intended for trusted local dimension mappings and overrides.
    read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
    ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

cd "${PROJECT_ROOT}"
accelerate launch "${ARGS[@]}"
