# Unified VLA Training Framework

本仓库以 [InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series)
为基础，将多种 Vision-Language-Action 模型接入同一套 LeRobot 训练框架。目前支持：

- **InternVLA-A1.5**：仓库原生模型；
- **Galaxea G0.5**：通过 `policy.type=g05` 接入官方 GalaxeaVLA；
- **LingBot-VLA 2.0**：通过 `policy.type=lingbot_vla_2` 接入官方 Qwen3-VL + sparse-MoE 模型。

三种模型共享数据集工厂、数据变换、Accelerate/DDP 训练循环、优化器与调度器、日志和
LeRobot checkpoint 格式。G0.5 与 LingBot-VLA 2.0 的官方模型实现和权重不复制到本仓库，
而是作为隔离环境中的外部依赖加载。

> [!IMPORTANT]
> 三个模型的 Python、PyTorch 和 Transformers 版本不兼容，必须使用独立环境，不能把所有
> policy 依赖安装到同一个环境中。

## 支持矩阵

| 模型 | Policy / Dataset type | 动作空间 | Chunk | 相机 | 当前训练模式 | 官方依赖 |
|---|---|---:|---:|---:|---|---|
| InternVLA-A1.5 | `internvla_a1_5` | 由数据配置决定 | 由配置决定 | 多相机 | 预训练、微调、评测 | 仓库原生 |
| Galaxea G0.5 | `g05` | 27D canonical | 32 | 1–3 | G0.5 权重微调 | [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) |
| LingBot-VLA 2.0 | `lingbot_vla_2` | 55D canonical | 50 | 1–3 | action-only post-training | [lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2) |

当前 G0.5 和 LingBot 适配层支持绝对动作与 delta 动作、非标准机器人维度映射、多相机
mask、动作序列 padding，以及统一的保存和恢复流程。

## 统一架构

```text
LeRobot Dataset
      │
      ├── InternVLA transforms
      ├── G0.5 transforms + ActionCodec
      └── LingBot transforms + Qwen3-VL processor
      │
      ▼
src/lerobot/scripts/lerobot_train.py
      │
      ├── common optimizer / scheduler
      ├── Accelerate / DDP
      ├── logging and checkpointing
      └── policy factory
             ├── InternVLAA15Policy
             ├── G05Policy ───────────► official GalaxeaVLA model
             └── LingBotVLA2Policy ───► official LingBot-VLA 2.0 model
```

这里的“统一框架”指训练基础设施统一，并不改变各模型本身的 backbone、action expert、
tokenizer 或官方预训练参数。

## 主要适配内容

| 模块 | G0.5 | LingBot-VLA 2.0 |
|---|---|---|
| Policy 注册 | `policy.type=g05` | `policy.type=lingbot_vla_2` |
| 数据注册 | `dataset.type=g05` | `dataset.type=lingbot_vla_2` |
| 官方模型 | 包装 GalaxeaVLA 与 ActionCodec | 包装官方 Qwen3-VL + flow-matching policy |
| 标准动作布局 | 27D，chunk 32 | 55D，chunk 50；末尾 4D 保留 |
| 非标准机器人 | state/action 显式区间映射 | state/action 显式区间映射 |
| 视觉输入 | 1–3 相机与有效性 mask | 1–3 相机、Qwen3-VL patch 与 `image_grid_thw` |
| Loss mask | horizon 与无效维度联合 mask | horizon、无效维度及保留维度联合 mask |
| 模型特殊逻辑 | 两轮残差 ActionCodec | 36 层、32 experts、top-4 sparse MoE |
| Smoke attention | 按官方环境配置 | 支持 eager；正式训练可使用 FlashAttention |
| 辅助训练头 | 按 G0.5 官方模型配置 | 默认关闭 depth/video teacher，仅训练动作 |
| 权重下载 | 官方 gated Hugging Face 仓库 | ModelScope 断点续传脚本 |

## 仓库结构

```text
.
├── launch/
│   ├── internvla_a15_finetune.sh
│   ├── g05_finetune.sh
│   └── lingbot_vla_2_finetune.sh
├── scripts/
│   └── download_lingbot_vla_2_modelscope.sh
├── src/lerobot/policies/
│   ├── internvla_a1_5/
│   ├── g05/
│   └── lingbot_vla_2/
├── tests/
│   ├── test_g05_integration.py
│   └── test_lingbot_vla_2_integration.py
├── tutorials/
│   ├── finetune_g05_in_internvla.md
│   └── finetune_lingbot_vla_2_in_internvla.md
├── checkpoints/                 # 本地模型权重，Git 忽略
└── third_party/                 # 官方模型源码，Git 忽略
```

## 获取代码

```bash
git clone https://codeup.aliyun.com/ebkernel/VLA.git
cd VLA
```

Codeup 私有仓库需要已配置的 HTTPS 克隆凭据或 SSH key。

## 环境准备

### InternVLA-A1.5

InternVLA-A1.5 已在 Python 3.11、CUDA 12.8 和 PyTorch 2.10.0 环境中测试。
安装步骤与 Qwen3.5 patch 说明见
[InternVLA 安装教程](tutorials/installation.md)。

### Galaxea G0.5

G0.5 官方依赖使用 Python 3.10、PyTorch 2.7.1 和 Transformers 4.57.1。

```bash
conda create -n g05 python=3.10 -y
conda activate g05

git clone https://github.com/OpenGalaxea/GalaxeaVLA \
  third_party/GalaxeaVLA

python -m pip install -e '.[g05]'
python -m pip install -e third_party/GalaxeaVLA --no-deps
```

CUDA attention kernel 请按 GalaxeaVLA 官方文档安装。完整说明见
[G0.5 微调教程](tutorials/finetune_g05_in_internvla.md)。

### LingBot-VLA 2.0

LingBot 官方发布环境使用 Python 3.12、PyTorch 2.8.0 和 Transformers 4.57.3。

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

正式训练可按官方要求安装 `flash-attn==2.8.3`。仅做 smoke test 时可以使用本文后面的
eager attention 参数。完整说明见
[LingBot-VLA 2.0 微调教程](tutorials/finetune_lingbot_vla_2_in_internvla.md)。

## 模型权重

模型权重全部保存到 `checkpoints/`，该目录已被 Git 忽略。

### InternVLA-A1.5

启动脚本默认使用 `InternRobotics/InternVLA-A1.5-base`。官方模型也可从
[ModelScope](https://www.modelscope.cn/models/InternRobotics/InternVLA-A1.5-base)
或 [Hugging Face](https://huggingface.co/InternRobotics/InternVLA-A1.5-base)
下载到本地，并修改启动脚本中的 `PRETRAINED_PATH`。

### Galaxea G0.5

G0.5 权重受官方许可协议约束。接受模型页面协议并登录 Hugging Face 后执行：

```bash
hf download OpenGalaxea/G05 \
  --include 'g05-base/*' 'action_tokenizer.pt' \
            'qwen3_5_2b_base_processor/*' \
  --local-dir checkpoints/g05
```

期望目录：

```text
checkpoints/g05/
├── action_tokenizer.pt
├── qwen3_5_2b_base_processor/
└── g05-base/checkpoints/model_state_dict.pt
```

### LingBot-VLA 2.0：ModelScope

国内网络推荐使用已提供的断点续传脚本：

```bash
python -m pip install modelscope
bash scripts/download_lingbot_vla_2_modelscope.sh
```

脚本下载两个完整快照，并在结束时检查关键 safetensors 索引文件：

| ModelScope ID | 本地目录 | 约占用空间 |
|---|---|---:|
| `Robbyant/lingbot-vla-v2-6b` | `checkpoints/lingbot-vla-v2-6b` | 28.2 GB |
| `Qwen/Qwen3-VL-4B-Instruct` | `checkpoints/Qwen3-VL-4B-Instruct` | 8.9 GB |

重复运行脚本会复用已完成文件，并继续未完成的下载。如需严格发布校验，可再按
ModelScope 文件清单核对每个文件的大小。若 ModelScope 安装在独立环境：

```bash
MODELSCOPE_BIN=/path/to/venv/bin/modelscope \
  bash scripts/download_lingbot_vla_2_modelscope.sh
```

## 启动训练

所有入口最终调用同一个 `src/lerobot/scripts/lerobot_train.py`。

### InternVLA-A1.5

原启动脚本需要 `_CONDA_ROOT`、`HF_HOME` 和 `WANDB_TOKEN` 环境变量：

```bash
export _CONDA_ROOT=/root/miniconda3
export HF_HOME=/path/to/huggingface/cache
export WANDB_TOKEN=<your_wandb_token>

bash launch/internvla_a15_finetune.sh \
  <lerobot_dataset_repo_id> abs false
```

第三个参数控制是否读取 `HF_HOME/lerobot/stats/.../stats.json` 外部统计文件。

### Galaxea G0.5

```bash
export G05_CONDA_ROOT=/root/miniconda3
export G05_CONDA_ENV=g05
export G05_SOURCE_PATH=$PWD/third_party/GalaxeaVLA
export G05_ASSET_DIR=$PWD/checkpoints/g05
export PROC_PER_NODE=4
export EMBODIMENT=r1lite

bash launch/g05_finetune.sh <lerobot_dataset_repo_id> abs
```

常用覆盖项包括 `BATCH_SIZE`、`TRAIN_STEPS`、`NUM_CAMERAS`、
`ENVIRONMENT_ACTION_DIM` 和 `EXTRA_ARGS`。

### LingBot-VLA 2.0

不安装 FlashAttention 的 smoke 启动：

```bash
export LINGBOT_VLA_2_CONDA_ROOT=/root/miniconda3
export LINGBOT_VLA_2_CONDA_ENV=lingbotvla2
export LINGBOT_VLA_2_SOURCE_PATH=$PWD/third_party/lingbot-vla-v2
export LINGBOT_VLA_2_MODEL_PATH=$PWD/checkpoints/lingbot-vla-v2-6b
export QWEN3_VL_PATH=$PWD/checkpoints/Qwen3-VL-4B-Instruct
export EXTRA_ARGS="--policy.attention_implementation=eager --policy.vit_attn_implementation=eager"

bash launch/lingbot_vla_2_finetune.sh \
  <lerobot_dataset_repo_id> abs
```

正式 FlashAttention 环境中可不设置上述 `EXTRA_ARGS`。当前适配默认
`disable_auxiliary_distillation=true`，不会构造 MoGe/LingBot-Depth/DINO-Video teacher
loss；若要复现官方 native-depth 预训练，应使用官方训练栈。

## 非标准机器人维度映射

G0.5 和 LingBot 都接受以下形式的映射：

```text
[source_start, source_end, canonical_start, canonical_end]
```

例如，把 14D 双臂动作映射到 G0.5 的 27D canonical layout：

```bash
export ENVIRONMENT_ACTION_DIM=14
export EXTRA_ARGS="--policy.action_dim_mapping=[[0,6,0,6],[12,13,9,10],[6,12,10,16],[13,14,19,20]] --policy.state_dim_mapping=[[0,6,0,6],[12,13,9,10],[6,12,10,16],[13,14,19,20]]"

bash launch/g05_finetune.sh <dataset_repo_id> abs
```

映射外的 canonical 维度自动填零并从 loss 中屏蔽。LingBot 的 55D 布局为：

```text
arm(14) | end-effector(14) | gripper(2) | waist(4) |
head(2) | base(3) | dexterous-hand(12) | reserved(4)
```

## 测试与当前验证状态

适配层回归测试：

```bash
python -m pytest -q \
  tests/test_g05_integration.py \
  tests/test_lingbot_vla_2_integration.py
```

当前版本已验证：

- G0.5 + LingBot 集成测试：`16 passed`；
- LingBot ModelScope 快照逐文件、逐字节校验通过；
- LingBot 官方 6 个 safetensors 分片完整加载；
- 加载后的模型约 6.23B 参数，MoE 形状为 `[32, 512, 768]`，动作投影为
  `[768, 55]`；
- 单张 A100 80GB 上完成合成 batch 前向和反向，梯度有限值检查通过；
- 上述 LingBot eager smoke 的峰值显存约 23.6 GiB。

这些结果用于验证集成链路，不代表任务数据集上的收敛效果或基准成绩。

## Checkpoint 与恢复

三个 policy 均使用 LeRobot checkpoint 目录。G0.5 checkpoint 额外保存 ActionCodec 和
processor sidecar；LingBot checkpoint 额外保存官方模型配置和 Qwen processor sidecar。
恢复训练时使用统一的 `--policy.path=<checkpoint>/pretrained_model` 机制。

## 许可证

- 本仓库及适配层：[`CC BY-NC-SA 4.0`](LICENSE)；
- G0.5 官方源码与权重：G0.5 Community License，包含非商业用途限制；详见
  [`G05_INTEGRATION_NOTICE.md`](G05_INTEGRATION_NOTICE.md)；
- LingBot-VLA 2.0 官方源码：Apache-2.0，模型卡及第三方数据/模型条款仍适用；详见
  [`LINGBOT_VLA_2_INTEGRATION_NOTICE.md`](LINGBOT_VLA_2_INTEGRATION_NOTICE.md)。

分发组合环境时必须同时保留各上游项目的 LICENSE、NOTICE 和第三方声明。

## 上游项目与引用

本项目建立在以下开源项目之上：

- [InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series)
- [InternVLA-A1.5 paper](https://arxiv.org/pdf/2607.04988)
- [Galaxea G0.5 paper](https://opengalaxea.github.io/G05/Galaxea_G0_5.pdf)
- [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA)
- [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)
- [LeRobot](https://github.com/huggingface/lerobot)

使用相应模型时，请同时引用其官方论文和仓库。InternVLA-A1.5：

```bibtex
@article{internvla_a15,
  title={InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization},
  author={Ma, Haoxiang and Cai, Junhao and Xu, Xiaoxu and Li, Hao and Yang, Yuyin and Tian, Yang and Cao, Jiafei and Zhu, Hongrui and Qiu, Zherui and others},
  journal={arXiv preprint arXiv:2607.04988},
  year={2026}
}
```
