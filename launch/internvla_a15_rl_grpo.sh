#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <grouped_student_rollout_dataset> [abs|delta] [extra lerobot-train args...]"
    exit 2
fi

DATASET_REPO_ID="$1"
ACTION_MODE="${2:-abs}"
EXTRA_ARGS=("${@:3}")
POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-InternRobotics/InternVLA-A1.5-base}"
PROC_PER_NODE="${PROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-6379}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RL_GAMMA="${RL_GAMMA:-0.99}"
RL_TEMPERATURE="${RL_TEMPERATURE:-1.0}"
RL_REWARD_HORIZON="${RL_REWARD_HORIZON:-50}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
GRPO_GROUP_ID_KEY="${GRPO_GROUP_ID_KEY:-grpo_group_id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-grpo-rl"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${POLICY}/${JOB_NAME}}"

ACCELERATE_ARGS=()
if (( PROC_PER_NODE > 1 )); then
    ACCELERATE_ARGS+=(--multi_gpu)
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
    --dataset.include_rl_signals=true \
    --dataset.rl_rollout_dataset=true \
    --dataset.rl_group_id_key="${GRPO_GROUP_ID_KEY}" \
    --dataset.use_fast_action_tokens=false \
    --rl.enable=true \
    --rl.algorithm=grpo \
    --rl.group_size="${GRPO_GROUP_SIZE}" \
    --rl.gamma="${RL_GAMMA}" \
    --rl.reward_horizon="${RL_REWARD_HORIZON}" \
    --rl.temperature="${RL_TEMPERATURE}" \
    --rl.loss_weight=1.0 \
    --rl.sft_loss_weight=0.1 \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=10000 \
    --log_freq=50 \
    --wandb.enable=false \
    "${EXTRA_ARGS[@]}"
