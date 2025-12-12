🏗️ 3D Porous Media GAN: Conditional Generation of Digital Rock Microstructures with Porosity Control

A high-performance 3D Generative Adversarial Network (GAN) for synthesizing realistic porous media microstructures with precise porosity conditioning. Built with JAX/Flax for high-speed TPU/GPU acceleration, designed for scientific applications in materials science, geology, and digital rock physics.

💡 Scientific Motivation: The Need for Fast Digital Rock Generation

The Challenge: Obtaining high-resolution 3D microstructures for porous media (like rock or synthetic materials) is typically done through expensive $\text{X-ray}$ micro-CT scanning or complex, time-consuming physics simulations.

The Solution: This 3D Porous Media GAN leverages deep learning to offer instantaneous and precise generation of diverse, high-fidelity digital rocks. Researchers can rapidly generate thousands of conditioned samples for large-scale computational fluid dynamics (CFD) or transport simulations, accelerating scientific discovery.

🧠 Architecture & Methodology

The project uses a Conditional Generative Adversarial Network (cGAN) architecture, where the porosity value is used as a conditioning input to both the Generator and the Discriminator.

Component

Role

Key Features

Generator ($G$)

Creates realistic 3D microstructures (pore/solid mask).

3D U-Net backbone, Residual Blocks, Self-Attention.

Discriminator ($D$)

Distinguishes between real and fake 3D volumes.

3D Convolutional Network.

Training Paradigm

Ensures stable training and high image quality.

WGAN-GP (Wasserstein GAN with Gradient Penalty).

🔬 Key Features

🎯 Precise Porosity Control: Generate 3D volumes with any target porosity value in the range $[0.0, 1.0]$, enabling targeted property studies.

⚡ TPU/GPU Optimized: Leverages JAX's automatic vectorization and pmap for multi-device parallel training, achieving industry-leading speeds for large 3D tensor operations.

🏗️ Advanced Architecture: The 3D U-Net generator integrates residual connections, dilated convolutions, and a self-attention module to capture complex, multi-scale pore geometry.

📊 Multi-Dataset Ready: Pre-configured for 11 real porous media datasets (Berea, Bentheimer, Kirby, etc.), allowing for generalized model training.

🤗 Hugging Face Integration: Easily push, share, and deploy trained models and generated samples.

🔬 Scientific Output: Outputs standard binary .raw files, ready for immediate use in popular pore-scale analysis tools (OpenPNM, PoreSpy, GeoDict).

📁 Project Structure

3d-porous-media-gan/
├── src/
│ ├── models/       # JAX/Flax implementations of G and D
│ ├── training/     # Loss functions, optimizers, pmap setup
│ ├── data/         # Dataset loading, preprocessing, and conditioning logic
│ └── inference/    # Sampling and result saving scripts
├── configs/        # YAML configuration files for different datasets/hyperparameters
├── scripts/
│ ├── train.py      # Main training entry point
│ ├── inference.py  # Generation entry point
│ └── push_to_hf.py # Script for model sharing
├── requirements.txt
├── setup.py
└── README.md


🚀 Quick Start

1. Prerequisites & Installation

Ensure you have Python 3.8+ and a compatible JAX installation (with CUDA for GPU or configured for TPU).

# Clone the repository
git clone [https://github.com/MhmdEsml/3d-porous-media-gan.git](https://github.com/MhmdEsml/3d-porous-media-gan.git)
cd 3d-porous-media-gan

# Install dependencies (requires JAX to be set up first)
pip install -e .


2. Download Dataset

Use the built-in downloader to fetch the necessary training data.

# Download Berea sandstone dataset (e.g., 1000³ voxels at 2.25µm resolution)
python -c "from src.data.downloader import download_dataset; download_dataset(['Berea'])"


3. Training

Launch training on your available accelerator (TPU or GPU). Checkpoints will be saved automatically.

# Train on Berea sandstone using default config (TPU/GPU)
python scripts/train.py --dataset Berea

# Train with custom parameters
python scripts/train.py --dataset Bentheimer --num_steps 50000 --batch_size 16


4. Inference (Generation)

Generate new, conditioned samples using a trained checkpoint.

# Generate 5 samples with a precise target porosity of 25%
python scripts/inference.py \
    --checkpoint checkpoints/Berea_final/ \
    --porosity 0.25 \
    --num_samples 5 \
    --output generated_samples_25pct.raw

# Generate 3 samples with high (50%) porosity
python scripts/inference.py \
    --checkpoint checkpoints/Bentheimer_final/ \
    --porosity 0.50 \
    --num_samples 3 \
    --output generated_samples_50pct.raw


📖 Citation

If you use this code or the generated microstructures in your scientific work, please cite the corresponding paper (to be updated upon publication):

@article{esml2024porousmedia,
  title={Conditional Generation of 3D Porous Media Microstructures with Porosity Control using JAX/Flax},
  author={Esml, Mhmd and [Co-Author 1] and [Co-Author 2]},
  journal={Journal Name},
  year={2024},
  publisher={Publisher}
}
