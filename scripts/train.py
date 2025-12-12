#!/usr/bin/env python
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.downloader import download_dataset
from src.data.dataset import load_volume
from src.training.trainer import Trainer
from configs.default_config import Config

def main():
    parser = argparse.ArgumentParser(description="Train 3D Porous Media GAN")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["Berea", "BanderaBrown", "BanderaGray", "Bentheimer",
                                "BSG", "BUG", "BuffBerea", "CastleGate", 
                                "Kirby", "Leopard", "Parker"],
                       help="Dataset to train on")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--num_steps", type=int, default=None,
                       help="Number of training steps")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    config = Config()
    if args.num_steps:
        config.num_steps = args.num_steps
    if not args.resume:
        config.resume = False
    
    download_dataset([args.dataset], config.data_path)
    
    volume = load_volume(
        os.path.join(config.data_path, f"{args.dataset}.raw"),
        tuple(config.volume_dims)
    )
    
    trainer = Trainer(config)
    trainer.train(volume, args.dataset)

if __name__ == "__main__":
    main()
