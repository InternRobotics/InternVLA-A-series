# Train Galaxea G0.5 in the InternVLA framework

G0.5 is integrated as another LeRobot policy (`policy.type=g05`). It uses the
same dataset factory, transforms, Accelerate/DDP loop, optimizer/scheduler,
logging and checkpoint format as InternVLA-A1.5. The official GalaxeaVLA
checkout is used as a model dependency for the Qwen3.5 backbone, visual-memory
modules and ActionCodec; its Hydra training loop is not used.

## 1. Install the model dependency

The official implementation currently pins Python `<3.11`, PyTorch 2.7.1 and
Transformers 4.57.1. Use a dedicated environment for G0.5; do not install these
pins into an existing InternVLA-A1.5 environment that uses Transformers 5.2.

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA third_party/GalaxeaVLA
python -m pip install -e '.[g05]'
python -m pip install -e ./third_party/GalaxeaVLA --no-deps
```

Install the CUDA attention kernels that match the environment, following the
official GalaxeaVLA installation guide.

## 2. Download G0.5 assets

```bash
hf download OpenGalaxea/G05 \
  --include 'g05-base/*' 'action_tokenizer.pt' 'qwen3_5_2b_base_processor/*' \
  --local-dir checkpoints/g05
```

The model repository is access-gated. Accept the G0.5 Community License on its
Hugging Face page and run `hf auth login` before downloading. In regions that
need the official mirror, set `HF_ENDPOINT=https://hf-mirror.com`.

Expected inputs:

```text
checkpoints/g05/
├── action_tokenizer.pt
├── qwen3_5_2b_base_processor/
└── g05-base/checkpoints/model_state_dict.pt
```

## 3. Launch training

```bash
export G05_SOURCE_PATH=$PWD/third_party/GalaxeaVLA
export G05_ASSET_DIR=$PWD/checkpoints/g05
export G05_CONDA_ROOT=/root/miniconda3
export G05_CONDA_ENV=g05
export PROC_PER_NODE=4
export EMBODIMENT=r1lite
bash launch/g05_finetune.sh <lerobot_dataset_repo_id> abs
```

The default canonical action layout is the 27-dimensional G0.5 layout:

```text
left_control(9) | left_gripper(1) | right_control(9) |
right_gripper(1) | lower_body(7)
```

The adapter keeps the released R1 Lite post-training tokenization defaults:
two residual ActionCodec rounds, per-part group markers, and omission of padded
or inactive embodiment parts. `chunk_size` is fixed at 32 by the released codec.

For a dataset already stored in that layout, no extra mapping is needed. For a
different raw layout, pass `policy.action_dim_mapping` and
`policy.state_dim_mapping`. Each entry is
`[source_start, source_end, canonical_start, canonical_end]`. Also set
`policy.environment_action_dim` so inference maps the canonical prediction back
to the robot's raw action width.

Example for a 14-dimensional bimanual layout consisting of two 6-DoF arms and
two grippers:

```bash
export ENVIRONMENT_ACTION_DIM=14
export EXTRA_ARGS="--policy.action_dim_mapping=[[0,6,0,6],[12,13,9,10],[6,12,10,16],[13,14,19,20]] --policy.state_dim_mapping=[[0,6,0,6],[12,13,9,10],[6,12,10,16],[13,14,19,20]]"
bash launch/g05_finetune.sh <dataset_repo_id> abs
```

Checkpoint directories contain the LeRobot config and safetensors weights, plus
copies of `action_tokenizer.pt` and the Hugging Face processor sidecars. Resume
through the standard `--policy.path=<checkpoint>/pretrained_model` mechanism.

## Licensing

InternVLA-A-series is CC BY-NC-SA 4.0. G0.5 code and weights are separately
licensed under the G0.5 Community License (non-commercial plus limited patent
license). Keep the official GalaxeaVLA `LICENSE-G0.5`, `NOTICE`, and third-party
notices with any redistribution, and review both licenses before use.
