\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{margin=1in}

\title{3D Porous Media GAN: Conditional Generation of Digital Rock Microstructures}
\author{Mohammad Esmaeili}
\date{\today}

\begin{document}

\maketitle

\section*{Overview}
A high-performance 3D Conditional Generative Adversarial Network built with JAX/Flax for synthesizing realistic porous media microstructures with precise porosity control. Designed for scientific applications in materials science, geology, and digital rock physics.

\section*{Quick Start}
\begin{lstlisting}[language=bash, backgroundcolor=\color{gray!10}]
git clone https://github.com/MhmdEsml/3d-porous-media-gan.git
cd 3d-porous-media-gan
pip install -e .
python -c "from src.data.downloader import download_dataset; download_dataset(['Berea'])"
python scripts/train.py --dataset Berea
python scripts/inference.py --checkpoint checkpoints/ --porosity 0.25 --output generated.raw
python scripts/push_to_hf.py --model_dir checkpoints/ --repo_id MhmdEsml/porous-media-gan-3d
\end{lstlisting}

\section*{Architecture}
\textbf{Generator:} 3D U-Net with residual connections, dilated convolutions, self-attention, porosity conditioning. \\
\textbf{Discriminator:} 3D PatchGAN with conditional input and feature matching. \\
\textbf{Losses:} LSGAN + Feature Matching + Porosity L1 with Straight-Through Estimator.

\section*{Features}
\begin{itemize}
    \item Precise porosity control (0.0-1.0)
    \item TPU/GPU multi-device training with JAX pmap
    \item 11 real porous media datasets (Berea, Bentheimer, etc.)
    \item Hugging Face integration
    \item Binary .raw output for pore-scale analysis
\end{itemize}

\section*{Performance}
\begin{tabular}{|l|l|}
\hline
Training speed & ~1000 steps/hour (TPU v3) \\
Generation speed & ~50ms/64×64×64 volume \\
Porosity accuracy & ±1\% \\
Output format & Binary .raw files \\
\hline
\end{tabular}

\section*{Applications}
\begin{itemize}
    \item Digital rock physics simulations
    \item Porous materials design
    \item Data augmentation for limited experimental data
    \item Educational tool for materials science
\end{itemize}

\section*{Project Structure}
\begin{verbatim}
3d-porous-media-gan/
├── src/ (models, training, data, inference)
├── configs/ (training configurations)
├── scripts/ (train.py, inference.py, push_to_hf.py)
├── requirements.txt
└── README.md
\end{verbatim}

\section*{Citation}
If using in research:
\begin{verbatim}
@software{esmaeili_2025_3d_porous_media_gan,
  author = {Esmaeili, Mohammad},
  title = {3D Porous Media GAN},
  year = {2025},
  url = {https://github.com/MhmdEsml/3d-porous-media-gan}
}
\end{verbatim}

\section*{License}
MIT License

\end{document}
