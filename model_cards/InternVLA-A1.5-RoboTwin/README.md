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
- robotwin
- domino
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

This repository hosts <strong>InternVLA-A1.5-RoboTwin</strong>, the checkpoint fine-tuned from InternVLA-A1.5-base on RoboTwin 2.0. It is intended for RoboTwin 2.0 bimanual manipulation evaluation and for static-to-dynamic zero-shot evaluation on DOMINO, where the same RoboTwin-tuned checkpoint is evaluated without DOMINO training data.

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
- **Fine-tuning target:** RoboTwin 2.0, ALOHA-AgileX embodiment
- **Evaluation scope:** RoboTwin 2.0 and DOMINO zero-shot
- **Action head:** unified action expert with flow-matching action generation
- **State/action dimension:** up to 32
- **Image resolution:** 224 x 224
- **License:** CC BY-NC-SA 4.0

## Usage

Please refer to our official repo [InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series) for installation, training, fine-tuning, and evaluation.

For RoboTwin 2.0 evaluation:

```bash
git clone https://github.com/InternRobotics/InternVLA-A-series.git
cd InternVLA-A-series
bash evaluation/RoboTwin/eval.sh InternRobotics/InternVLA-A1.5-RoboTwin outputs/robotwin/internvla_a1_5 demo_clean 0 abs 50
```

For DOMINO zero-shot evaluation with the RoboTwin checkpoint:

```bash
bash evaluation/DOMINO/eval.sh \
  InternRobotics/InternVLA-A1.5-RoboTwin \
  outputs/domino/internvla_a1_5_robotwin_zero_shot \
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

For benchmark workflows, please see:

- [RoboTwin Evaluation](https://github.com/InternRobotics/InternVLA-A-series/tree/master/evaluation/RoboTwin)
- [DOMINO Evaluation](https://github.com/InternRobotics/InternVLA-A-series/tree/master/evaluation/DOMINO)

## Demonstrations

**InternVLA-A1.5-RoboTwin** corresponds to the RoboTwin 2.0 benchmark result and the DOMINO static-to-dynamic zero-shot result.

| RoboTwin 2.0 Clean | RoboTwin 2.0 Rand. | RoboTwin 2.0 Avg. |
| :---: | :---: | :---: |
| **93.3** | **93.0** | **93.2** |

| DOMINO Setting | SR (%) ↑ | MS ↑ |
| :--- | :---: | :---: |
| Zero-shot to dynamic manipulation | **27.7** | **39.8** |

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
