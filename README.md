# 🏗️ 3D Porous Media GAN

**Conditional Generation of Digital Rock Microstructures with Porosity Control**

[![GitHub stars](https://img.shields.io/github/stars/MhmdEsml/3d-porous-media-gan?style=social)](https://github.com/MhmdEsml/3d-porous-media-gan)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.5.2%2B-orange)](https://github.com/google/jax)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-🤗-yellow)](https://huggingface.co/MhmdEsml)

A high-performance 3D Generative Adversarial Network for synthesizing realistic porous media microstructures with precise porosity conditioning. Built with JAX/Flax for TPU/GPU acceleration, designed for scientific applications in materials science, geology, and digital rock physics.

![3D Porous Media Generation](https://via.placeholder.com/800x400.png?text=3D+Porous+Media+GAN+Visualization)

## 🔬 Key Features

- **🎯 Precise Porosity Control**: Generate 3D volumes with target porosity values (0.0-1.0)
- **⚡ TPU/GPU Optimized**: Multi-device parallel training with JAX's `pmap` for blazing speed
- **🏗️ Advanced Architecture**: 3D U-Net with residual connections, dilated convolutions, and self-attention
- **📊 Multi-Dataset Ready**: Pre-configured for 11 real porous media datasets (Berea, Bentheimer, Kirby, etc.)
- **🤗 Hugging Face Integration**: Easy model sharing and deployment
- **🔬 Scientific Output**: Binary .raw files compatible with pore-scale analysis tools (OpenPNM, PoreSpy, GeoDict)

## 📁 Project Structure
