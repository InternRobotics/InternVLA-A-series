#!/usr/bin/env bash
set -euo pipefail

# Train Galaxea G0.5 through the same LeRobot/Accelerate entrypoint used by
# InternVLA policies. The official GalaxeaVLA checkout supplies only model code,
# ActionCodec, processor files and the pretrained checkpoint.
#
# Usage:
#   bash launch/g05_finetune.sh <dataset_repo_id> [abs|delta]
#
# Required environment variables:
#   G05_SOURCE_PATH   official OpenGalaxea/GalaxeaVLA checkout
#   G05_ASSET_DIR     directory containing g05-base/, action_tokenizer.pt and
#                     qwen3_5_2b_base_processor/
#
# Optional environment variables:
#   PROC_PER_NODE, NODE_COUNT, NODE_RANK, MASTER_ADDR, MASTER_PORT,
#   BATCH_SIZE, TRAIN_STEPS, EMBODIMENT, MAX_STATE_DIM, MAX_ACTION_DIM,
#   ENVIRONMENT_ACTION_DIM, NUM_CAMERAS, OUTPUT_DIR, G05_CONDA_ENV,
#   G05_CONDA_ROOT, EXTRA_ARGS

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dataset_repo_id> [abs|delta]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_REPO_ID="$1"
ACTION_MODE="${2:-abs}"

G05_CONDA_ENV="${G05_CONDA_ENV:-g05}"
G05_CONDA_ROOT="${G05_CONDA_ROOT:-${_CONDA_ROOT:-}}"
if [[ -n "${G05_CONDA_ROOT}" && -f "${G05_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${G05_CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${G05_CONDA_ENV}"
fi

G05_SOURCE_PATH="${G05_SOURCE_PATH:-${PROJECT_ROOT}/third_party/GalaxeaVLA}"
G05_ASSET_DIR="${G05_ASSET_DIR:-${PROJECT_ROOT}/checkpoints/g05}"
G05_BASE_CHECKPOINT="${G05_BASE_CHECKPOINT:-${G05_ASSET_DIR}/g05-base/checkpoints/model_state_dict.pt}"
G05_PROCESSOR_PATH="${G05_PROCESSOR_PATH:-${G05_ASSET_DIR}/qwen3_5_2b_base_processor}"
G05_ACTION_TOKENIZER="${G05_ACTION_TOKENIZER:-${G05_ASSET_DIR}/action_tokenizer.pt}"

for required_path in \
    "${G05_SOURCE_PATH}/src/g05" \
    "${G05_BASE_CHECKPOINT}" \
    "${G05_PROCESSOR_PATH}" \
    "${G05_ACTION_TOKENIZER}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required G0.5 asset: ${required_path}" >&2
        exit 1
    fi
done

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6380}"
PROC_PER_NODE="${PROC_PER_NODE:-1}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
MAX_STATE_DIM="${MAX_STATE_DIM:-27}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-27}"
ENVIRONMENT_ACTION_DIM="${ENVIRONMENT_ACTION_DIM:-${MAX_ACTION_DIM}}"
NUM_CAMERAS="${NUM_CAMERAS:-3}"
EMBODIMENT="${EMBODIMENT:-unknown}"
JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-g05-${DATASET_REPO_ID//[\/ ]/_}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/g05/${JOB_NAME}}"

if [[ "${MAX_ACTION_DIM}" -ne 27 ]]; then
    echo "The released G0.5 ActionCodec uses a 27D canonical layout; MAX_ACTION_DIM must remain 27." >&2
    echo "Use --policy.action_dim_mapping to map a narrower dataset layout into it." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${G05_SOURCE_PATH}/src:${PYTHONPATH:-}"
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
    --num_workers=8

    --policy.type=g05
    --policy.repo_id=lerobot_lab/g05
    --policy.push_to_hub=false
    --policy.g05_source_path="${G05_SOURCE_PATH}"
    --policy.official_pretrained_checkpoint="${G05_BASE_CHECKPOINT}"
    --policy.hf_processor_path="${G05_PROCESSOR_PATH}"
    --policy.action_tokenizer_path="${G05_ACTION_TOKENIZER}"
    --policy.embodiment="${EMBODIMENT}"
    --policy.num_cameras="${NUM_CAMERAS}"
    --policy.max_state_dim="${MAX_STATE_DIM}"
    --policy.max_action_dim="${MAX_ACTION_DIM}"
    --policy.environment_action_dim="${ENVIRONMENT_ACTION_DIM}"
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16

    --dataset.type=g05
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_MODE}"
    --dataset.num_cameras="${NUM_CAMERAS}"
    --dataset.use_imagenet_stats=false

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${TRAIN_STEPS}"
    --save_freq=5000
    --log_freq=50
    --wandb.enable=false
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # EXTRA_ARGS is intended for trusted local overrides such as dimension maps.
    read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
    ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

cd "${PROJECT_ROOT}"
accelerate launch "${ARGS[@]}"
