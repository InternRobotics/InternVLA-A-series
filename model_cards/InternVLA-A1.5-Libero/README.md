---
license: cc-by-nc-sa-4.0
base_model:
- InternRobotics/InternVLA-A1.5-base
- Qwen/Qwen3.5-2B
tags:
- robotics
- vision-language-action-model
- world-model
- manipulation
- libero
- zero-shot
datasets:
- InternRobotics/InternData-A1
---

# InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization

<div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
    <img src="https://raw.githubusercontent.com/InternRobotics/InternVLA-A-series/master/assets/teaser.png" alt="InternVLA-A1.5 Teaser Image" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
</div>

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2607.04988)
[![Code](https://img.shields.io/badge/GitHub-Code-800820?logo=github)](https://github.com/InternRobotics/InternVLA-A-series)
[![Models](https://img.shields.io/badge/Models-HuggingFace-blue?logo=huggingface)](https://huggingface.co/collections/InternRobotics/internvla-a15)
[![Data](https://img.shields.io/badge/Data-HuggingFace-blue?logo=huggingface)](https://huggingface.co/datasets/InternRobotics/InternData-A1)
[![Website](https://img.shields.io/badge/Website-Pages-blue.svg)](https://internrobotics.github.io/internvla-a15.github.io/)

<strong>InternVLA-A1.5</strong> unifies vision-language understanding, latent visual foresight, and action generation in one robot policy. It builds on a native Qwen3.5-2B VLM backbone, preserves semantic learning through VQA and subtask prediction, and attaches a lightweight unified action expert for continuous control.

This repository hosts <strong>InternVLA-A1.5-Libero</strong>, the checkpoint fine-tuned from InternVLA-A1.5-base on LIBERO. It is intended for LIBERO evaluation and zero-shot robustness evaluation on LIBERO-Plus, where the same LIBERO-tuned checkpoint is evaluated directly under camera, language, layout, lighting, background, noise, and robot perturbations.

Covering base and benchmark-specific checkpoints, we release the InternVLA-A1.5 series:

- [x] [InternVLA-A1.5-base](https://huggingface.co/InternRobotics/InternVLA-A1.5-base): base checkpoint for downstream fine-tuning and evaluation
- [x] [InternVLA-A1.5-RoboTwin](https://huggingface.co/InternRobotics/InternVLA-A1.5-RoboTwin): fine-tuned on RoboTwin 2.0
- [x] [InternVLA-A1.5-Libero](https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero): fine-tuned on LIBERO
- [x] [InternVLA-A1.5-DOMINO](https://huggingface.co/InternRobotics/InternVLA-A1.5-DOMINO): fine-tuned on DOMINO

## 🔑 Key Features

<div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
    <img src="https://raw.githubusercontent.com/InternRobotics/InternVLA-A-series/master/assets/model.png" alt="InternVLA-A1.5 Model" style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
</div>

- 🔮 *The Core: Attaches a lightweight unified action expert to a native Qwen3.5-2B VLM backbone through shared full-attention layers, while preserving modality-specific Gated DeltaNet processing.*
- 🚀 *The Foresight: Uses learnable foresight tokens to query task-relevant future dynamics from the shared multimodal context, supervised by a frozen WAN2.2-5B video generation model during training.*
- ⚡ *The Output: Discards the video branch at inference and predicts continuous action chunks through flow matching, keeping deployment latency practical.*

## Model Details

- **Model type:** Vision-Language-Action robot policy
- **Base checkpoint:** InternRobotics/InternVLA-A1.5-base
- **Backbone:** Qwen/Qwen3.5-2B
- **Policy type:** `internvla_a1_5`
- **Fine-tuning target:** LIBERO Spatial, Object, Goal, and Long suites
- **Evaluation scope:** LIBERO and LIBERO-Plus zero-shot
- **Action head:** unified action expert with flow-matching action generation
- **State/action dimension:** up to 32
- **Image resolution:** 224 x 224
- **License:** CC BY-NC-SA 4.0

## Usage

Please refer to our official repo [InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series) for installation, training, fine-tuning, and evaluation.

For LIBERO evaluation:

```bash
git clone https://github.com/InternRobotics/InternVLA-A-series.git
cd InternVLA-A-series
export CKPT_PATH=InternRobotics/InternVLA-A1.5-Libero
export LIBERO_HOME=<LIBERO_REPO_ROOT>
export STATS_KEY_MODE=suite
export ROBOT_TYPE_MODE=panda
bash evaluation/LIBERO/run_eval_libero_server_client.sh
```

For LIBERO-Plus zero-shot evaluation:

```bash
export CKPT_PATH=InternRobotics/InternVLA-A1.5-Libero
export LIBERO_HOME=<LIBERO_PLUS_REPO_ROOT>
export STATS_KEY_MODE=suite
export ROBOT_TYPE_MODE=panda
SHARDS_PER_SUITE=8 bash evaluation/LIBERO-plus/run_eval_libero_plus.sh
```

For benchmark workflows, please see:

- [LIBERO Evaluation](https://github.com/InternRobotics/InternVLA-A-series/tree/master/evaluation/LIBERO)
- [LIBERO-Plus Evaluation](https://github.com/InternRobotics/InternVLA-A-series/tree/master/evaluation/LIBERO-plus)

## Demonstrations

**InternVLA-A1.5-Libero** corresponds to the LIBERO benchmark result and the LIBERO-Plus zero-shot robustness result.

| LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO Long | LIBERO Avg. |
| :---: | :---: | :---: | :---: | :---: |
| **98.6** | **99.8** | **98.6** | **98.4** | **98.9** |

| LIBERO-Plus Camera | Robot | Language | Light | Background | Noise | Layout | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **83.1** | **55.1** | **86.9** | **96.4** | **98.2** | **95.6** | **85.2** | **84.8** |

## License and Citation

All code within this repo is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our project if it helps your research.

```BibTeX
@article{internvla_a15,
  title={InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization},
  author={Ma, Haoxiang and Cai, Junhao and Xu, Xiaoxu and Li, Hao and Yang, Yuyin and Tian, Yang and Cao, Jiafei and Zhu, Hongrui and Qiu, Zherui and Zhaxizhuoma and Yang, Yuqiang and Peng, Jiaqi and Wei, Xueyuan and Zhu, Yangkun and Jiang, Jiahao and Gao, Xing and Wang, Hanqing and Yuan, Feng and Li, Kailin and Zhu, Xueyue and Wang, Tai and Ding, Yan and Pang, Jiangmiao and Zeng, Jia and Zhang, Jingjing and Zhou, Bowen and Mu, Yao and Shen, Chunhua and Zhang, Weinan},
  journal={arXiv preprint arXiv:2607.04988},
  year={2026}
}
```

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [InternVLA-A1](https://github.com/InternRobotics/InternVLA-A1)
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Wan](https://github.com/Wan-Video/Wan2.2)
