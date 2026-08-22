#!/usr/bin/env bash
set -euo pipefail

# Download the action-only LingBot-VLA 2.0 post-training assets from
# ModelScope. Rerunning this command resumes incomplete files.
#
# Usage:
#   MODELSCOPE_BIN=/path/to/modelscope \
#     bash scripts/download_lingbot_vla_2_modelscope.sh [checkpoint_root]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_ROOT="${1:-${PROJECT_ROOT}/checkpoints}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
MAX_WORKERS="${MAX_WORKERS:-4}"

if [[ "${MODELSCOPE_BIN}" = */* ]]; then
    if [[ ! -x "${MODELSCOPE_BIN}" ]]; then
        echo "modelscope executable not found: ${MODELSCOPE_BIN}" >&2
        exit 1
    fi
elif ! command -v "${MODELSCOPE_BIN}" >/dev/null 2>&1; then
    echo "modelscope executable not found: ${MODELSCOPE_BIN}" >&2
    echo "Install it with: python -m pip install modelscope" >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}"

"${MODELSCOPE_BIN}" download Robbyant/lingbot-vla-v2-6b \
    --local-dir "${CHECKPOINT_ROOT}/lingbot-vla-v2-6b" \
    --max-workers "${MAX_WORKERS}"

"${MODELSCOPE_BIN}" download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir "${CHECKPOINT_ROOT}/Qwen3-VL-4B-Instruct" \
    --max-workers "${MAX_WORKERS}"

test -s "${CHECKPOINT_ROOT}/lingbot-vla-v2-6b/model.safetensors.index.json"
test -s "${CHECKPOINT_ROOT}/Qwen3-VL-4B-Instruct/model.safetensors.index.json"

echo "ModelScope snapshots are ready under ${CHECKPOINT_ROOT}"
