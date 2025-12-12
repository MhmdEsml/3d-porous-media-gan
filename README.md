# 🏗️ 3D Porous Media GAN

### Conditional Generation of Digital Rock Microstructures with Porosity Control

[![GitHub stars](https://img.shields.io/github/stars/MhmdEsml/3d-porous-media-gan?style=social)](https://github.com/MhmdEsml/3d-porous-media-gan)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.5.2%2B-orange)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/MhmdEsml)

A high-performance **3D Generative Adversarial Network** for synthesizing realistic porous media microstructures with **controllable porosity**, powered by **JAX/Flax** for fast GPU/TPU execution.

---

## 🔬 Key Features

- 🎯 **Porosity Conditioning** (target porosity ∈ [0, 1])
- ⚡ **TPU/GPU Optimized** via `pmap`
- 🧠 **Advanced 3D GAN Architecture** (U-Net + residuals + attention)
- 📊 **Supports 11 Real Porous Media Datasets**
- 🤗 **Hugging Face Model Hub Integration**
- 💾 **Scientific Output** (`.raw` volumes for OpenPNM, PoreSpy, GeoDict, Avizo)

---

## 📁 Project Structure
```
3d-porous-media-gan/
├── src/
│   ├── models/
│   ├── training/
│   ├── data/
│   └── inference/
├── configs/
├── scripts/
│   ├── train.py
│   ├── inference.py
│   └── push_to_hf.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 📦 Installation
```bash
git clone https://github.com/MhmdEsml/3d-porous-media-gan.git
cd 3d-porous-media-gan
pip install -e .
```

### 📥 Download Dataset
```bash
python -c "from src.data.downloader import download_dataset; download_dataset(['Berea'])"
```

### 🏋️ Training
```bash
python scripts/train.py --dataset Berea
```

Custom training:
```bash
python scripts/train.py --dataset Bentheimer --num_steps 50000
```

### 🎛️ Inference
```bash
python scripts/inference.py \
  --checkpoint checkpoints/ \
  --porosity 0.25 \
  --num_samples 5 \
  --output generated_samples.raw
```

---

## 📜 License

MIT License.

---
