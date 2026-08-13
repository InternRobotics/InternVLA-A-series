# Train LingBot-VLA 2.0 in the InternVLA framework

LingBot-VLA 2.0 is registered as another policy (`policy.type=lingbot_vla_2`).
It uses the same LeRobot dataset factory, transforms, Accelerate/DDP loop,
optimizer/scheduler, logging, and checkpoint format as InternVLA-A1.5 and G0.5.
The official checkout remains the source of the model architecture and initial
weights; its standalone training loop is not invoked.

## 1. Create the isolated runtime

The official release requires Python 3.12, PyTorch 2.8.0 and Transformers
4.57.3. Use a separate conda environment because these pins differ from the
InternVLA-A1.5 and G0.5 runtimes.

```bash
conda create -n lingbotvla2 python=3.12 -y
conda activate lingbotvla2

git clone https://github.com/Robbyant/lingbot-vla-v2 \
  third_party/lingbot-vla-v2

python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  torchdata==0.11.0 torchcodec==0.6.0
python -m pip install -e '.[lingbot-vla-2]'
python -m pip install -e third_party/lingbot-vla-v2 --no-deps
```

Install `flash-attn==2.8.3` when using the released flash-attention defaults.
For a smoke test without FlashAttention, select eager attention through
`EXTRA_ARGS`:

```bash
export EXTRA_ARGS="--policy.attention_implementation=eager --policy.vit_attn_implementation=eager"
```

## 2. Download model assets

ModelScope is recommended on networks where Hugging Face is unavailable:

```bash
python -m pip install modelscope

bash scripts/download_lingbot_vla_2_modelscope.sh
```

The two verified ModelScope IDs are
`Robbyant/lingbot-vla-v2-6b` and `Qwen/Qwen3-VL-4B-Instruct`. The snapshots are
about 28.2 GB and 8.9 GB respectively. Downloads can be resumed by rerunning
the same script. Set `MODELSCOPE_BIN` when ModelScope is installed in a
separate environment, for example:

```bash
MODELSCOPE_BIN=/root/code/.venvs/modelscope-download/bin/modelscope \
  bash scripts/download_lingbot_vla_2_modelscope.sh
```

The equivalent Hugging Face commands are:

```bash
hf download robbyant/lingbot-vla-v2-6b \
  --local-dir checkpoints/lingbot-vla-v2-6b
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir checkpoints/Qwen3-VL-4B-Instruct
```

The default integration is action-only post-training. It loads the released
VLM/action-expert weights and omits the auxiliary dual-query heads and
MoGe/LingBot-Depth/DINO-Video teacher losses, so those teacher checkpoints are
not needed. Use the official training stack if you need to reproduce
native-depth pre-training with auxiliary distillation.

## 3. Canonical 55D mapping

The policy always receives the official 55-dimensional layout:

```text
arm(14) | end-effector(14) | gripper(2) | waist(4) |
head(2) | base(3) | dexterous-hand(12) | reserved(4)
```

Datasets with fewer dimensions are padded automatically at the tail. For a
different ordering, pass `policy.action_dim_mapping` and
`policy.state_dim_mapping`. Each entry is
`[source_start, source_end, canonical_start, canonical_end]`. The resulting
dimension mask is combined with the action-horizon padding mask before the
flow-matching loss is computed.

## 4. Launch

```bash
export LINGBOT_VLA_2_CONDA_ROOT=/root/miniconda3
export LINGBOT_VLA_2_CONDA_ENV=lingbotvla2
export LINGBOT_VLA_2_SOURCE_PATH=$PWD/third_party/lingbot-vla-v2
export LINGBOT_VLA_2_MODEL_PATH=$PWD/checkpoints/lingbot-vla-v2-6b
export QWEN3_VL_PATH=$PWD/checkpoints/Qwen3-VL-4B-Instruct
export EXTRA_ARGS="--policy.attention_implementation=eager --policy.vit_attn_implementation=eager"

bash launch/lingbot_vla_2_finetune.sh <lerobot_dataset_repo_id> abs
```

Qwen3-VL image patching and `image_grid_thw` generation run inside the dataset
workers. Checkpoints use the standard LeRobot safetensors/config format and
include the official model config plus Qwen processor files as sidecars.

## License

The official LingBot-VLA 2.0 code is Apache-2.0. This adapter and the surrounding
InternVLA-A-series repository remain under CC BY-NC-SA 4.0. Preserve both
license and notice files when redistributing a combined environment.
