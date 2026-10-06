# FARM: Foundational Aerial Radio Map for Intelligent Low-Altitude Networking

## Overview

FARM is a foundation model for three-dimensional aerial radio map (ARM) construction. It combines a masked autoencoder-based radio encoder with a diffusion-based map decoder in a unified model that supports three input settings:

| Input setting | Available information |
| --- | --- |
| Condition-free | Sparse measurements |
| Condition-only | Radio environmental conditions, including building geometry, base station location, and transmission configuration |
| Condition-and-sample | Radio environmental conditions and sparse measurements |

## Dataset Preparation

The dataset used for pretraining and inference is available on Hugging Face: [FARM_training_test](https://huggingface.co/datasets/jliang097/FARM_training_test).

## Code and Model Weights

The source code and pretrained model weights will be made publicly available in this repository upon acceptance.

## Citation

If you find this repo or dataset helpful, please cite our paper:

[FARM: Foundational Aerial Radio Map for Intelligent Low-Altitude Networking](https://arxiv.org/abs/2604.17362)

```bibtex
@article{gao2026farm,
  author  = {Gao, Shijian and Liang, Jiahui and Yuan, Yifeng and Lu, Wenlihan and Shen, Guobin and Yang, Liuqing},
  title   = {{FARM}: Foundational Aerial Radio Map for Intelligent Low-Altitude Networking},
  journal = {arXiv},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.17362},
}
```
