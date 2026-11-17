<div align="center">

# 📡 FARM

**Foundational Aerial Radio Map for Intelligent Low-Altitude Networking**

[![Paper](https://img.shields.io/badge/arXiv-2604.17362-B31B1B?style=flat-square)](https://arxiv.org/abs/2604.17362)
[![Model Weights](https://img.shields.io/badge/Hugging%20Face-Model%20Weights-FFD21E?style=flat-square)](https://huggingface.co/jliang097/FARM)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/jliang097/FARM_training_test)

</div>

## 🎉 News

- FARM has been accepted by **npj Wireless Technology**. The code and model weights are now open source.

## 📖 Overview

FARM is a foundation model for three-dimensional aerial radio map (ARM) construction. It combines a masked autoencoder-based radio encoder with a diffusion-based map decoder in a unified model that supports three input settings:

| Input setting | Available information |
| --- | --- |
| Condition-free | Sparse measurements |
| Condition-only | Radio environmental conditions, including building geometry, base station location, and transmission configuration |
| Condition-and-sample | Radio environmental conditions and sparse measurements |

## 🛠️ Installation

Python 3.10 or later and PyTorch 2.6 or later are required.

```bash
git clone https://github.com/Jiahui-L/FARM.git
cd FARM
pip install -r requirements.txt
pip install -U huggingface_hub
```

## 📦 Dataset Preparation

The dataset used for pretraining and inference is available on Hugging Face: [FARM_training_test](https://huggingface.co/datasets/jliang097/FARM_training_test).

Download one ARM volume, its base station coordinates, and the height-layer metadata:

```bash
hf download jliang097/FARM_training_test --repo-type dataset \
  meta_ranges.json D6/52/tx_positions_512.csv \
  D6/52/freq21/pattern_120/tx0_yaw299.npy \
  --local-dir data/FARM_training_test
```

To use a complete scene, download its archive and base station coordinates, then extract the archive:

```bash
hf download jliang097/FARM_training_test --repo-type dataset \
  meta_ranges.json D1/7/tx_positions_512.csv D1/scene_7.tar.gz \
  --local-dir data/FARM_training_test
tar -xzf data/FARM_training_test/D1/scene_7.tar.gz -C data/FARM_training_test/D1
```

Use an extracted ARM `.npy` file as `--sample`. Keep the downloaded folder structure
so the script can locate the metadata for that sample.

## 🚀 Inference

Download the checkpoint for your chosen model from [Hugging Face](https://huggingface.co/jliang097/FARM).
For the Base model:

```bash
hf download jliang097/FARM farm_base.pth --local-dir weights
```

Use the matching configuration:

| Model | Weights | Configuration |
| --- | --- | --- |
| Small | `weights/farm_small.pth` | `configs/farm_small.json` |
| Base | `weights/farm_base.pth` | `configs/farm_base.json` |
| Large | `weights/farm_large.pth` | `configs/farm_large.json` |
| Extra large | `weights/farm_extra_large.pth` | `configs/farm_extra_large.json` |

### Condition-free

Use sparse received signal strength (RSS) measurements only. This example samples 5% of the spatial patches:

```bash
python infer.py --checkpoint weights/farm_base.pth --config configs/farm_base.json \
  --sample data/FARM_training_test/D6/52/freq21/pattern_120/tx0_yaw299.npy \
  --input-setting condition-free --sampling-rate 0.05 --device cuda --amp \
  --output outputs/condition-free.npz
```

### Condition-only

Use environmental conditions without RSS measurements:

```bash
python infer.py --checkpoint weights/farm_base.pth --config configs/farm_base.json \
  --sample data/FARM_training_test/D6/52/freq21/pattern_120/tx0_yaw299.npy \
  --input-setting condition-only --device cuda --amp \
  --output outputs/condition-only.npz
```

### Condition-and-sample

Use environmental conditions and RSS measurements from 5% of the spatial patches:

```bash
python infer.py --checkpoint weights/farm_base.pth --config configs/farm_base.json \
  --sample data/FARM_training_test/D6/52/freq21/pattern_120/tx0_yaw299.npy \
  --input-setting condition-and-sample --sampling-rate 0.05 --device cuda --amp \
  --output outputs/condition-and-sample.npz
```

## 📚 Citation

If you find this repo or dataset helpful, please cite our paper:

[FARM: Foundational Aerial Radio Map for Intelligent Low-Altitude Networking](https://arxiv.org/abs/2604.17362)

```bibtex
@article{gao2026farm,
  author  = {Gao, Shijian and Liang, Jiahui and Yuan, Yifeng and Lu, Wenlihan and Shen, Guobin and Yang, Liuqing},
  title   = {{FARM}: Foundational Aerial Radio Map for Intelligent Low-Altitude Networking},
  journal = {npj Wireless Technology},
  year    = {2026},
}
```
