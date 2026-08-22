#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <student_rollout_dataset> <kairos_dataset_stats.json> [abs|delta]"
    exit 2
fi

DATASET_REPO_ID="$1"
KAIROS_STATS_PATH="$2"
ACTION_MODE="${3:-abs}"
EXTRA_ARGS=("${@:4}")
POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-InternRobotics/InternVLA-A1.5-base}"
KAIROS_ENDPOINT="${KAIROS_ENDPOINT:-http://127.0.0.1:8006}"
PROC_PER_NODE="${PROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-6379}"
TEACHER_SAMPLES="${TEACHER_SAMPLES:-4}"
STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RL_ENABLE="${RL_ENABLE:-false}"
RL_ALGORITHM="${RL_ALGORITHM:-rwfm}"
RL_GAMMA="${RL_GAMMA:-0.99}"
RL_TEMPERATURE="${RL_TEMPERATURE:-1.0}"
RL_GROUP_SIZE="${RL_GROUP_SIZE:-4}"
RL_GROUP_ID_KEY="${RL_GROUP_ID_KEY:-grpo_group_id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-kairos-opd"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${POLICY}/${JOB_NAME}}"

ACCELERATE_ARGS=()
if (( PROC_PER_NODE > 1 )); then
    ACCELERATE_ARGS+=(--multi_gpu)
fi

RL_ARGS=()
if [[ "${RL_ENABLE}" == "true" ]]; then
    if [[ "${RL_ALGORITHM}" != "rwfm" && "${RL_ALGORITHM}" != "grpo" ]]; then
        echo "RL_ALGORITHM must be rwfm or grpo, got ${RL_ALGORITHM}"
        exit 2
    fi
    RL_ARGS+=(
        --dataset.include_rl_signals=true
        --dataset.rl_rollout_dataset=true
        --rl.enable=true
        --rl.algorithm="${RL_ALGORITHM}"
        --rl.gamma="${RL_GAMMA}"
        --rl.temperature="${RL_TEMPERATURE}"
        --rl.loss_weight=1.0
        --rl.sft_loss_weight=0.1
    )
    if [[ "${RL_ALGORITHM}" == "grpo" ]]; then
        RL_ARGS+=(
            --dataset.rl_group_id_key="${RL_GROUP_ID_KEY}"
            --rl.group_size="${RL_GROUP_SIZE}"
        )
    fi
fi

accelerate launch "${ACCELERATE_ARGS[@]}" \
    --num_processes="${PROC_PER_NODE}" \
    --main_process_port="${MASTER_PORT}" \
    src/lerobot/scripts/lerobot_train.py \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --num_workers=4 \
    --policy.type="${POLICY}" \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.push_to_hub=false \
    --policy.dtype=bfloat16 \
    --policy.enable_vqa_loss=false \
    --policy.action_loss_only=true \
    --policy.video_loss_only=false \
    --dataset.type="${POLICY}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.action_mode="${ACTION_MODE}" \
    --dataset.include_opd_inputs=true \
    --dataset.opd_rollout_dataset=true \
    --dataset.use_fast_action_tokens=false \
    --opd.enable=true \
    --opd.teacher_type=kairos \
    --opd.teacher_endpoint="${KAIROS_ENDPOINT}" \
    --opd.kairos_dataset_stats_path="${KAIROS_STATS_PATH}" \
    --opd.teacher_samples="${TEACHER_SAMPLES}" \
    --opd.image_layout=auto \
    --opd.student_normalization_mode=mean_std \
    --opd.loss_weight=1.0 \
    --opd.sft_loss_weight=0.1 \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=10000 \
    --log_freq=50 \
    --wandb.enable=false \
    "${RL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
