#!/usr/bin/env python
import argparse
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.inference import generate_from_hf, load_generator_from_checkpoint, generate_samples
from configs.default_config import Config
import jax
import jax.numpy as jnp

def main():
    parser = argparse.ArgumentParser(description="Generate 3D porous media samples")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to generator checkpoint")
    parser.add_argument("--hf_repo", type=str, default=None,
                       help="Hugging Face repository ID")
    parser.add_argument("--porosity", type=float, default=0.2,
                       help="Target porosity (0.0 to 1.0)")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="generated.raw",
                       help="Output file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.hf_repo:
        print("Error: Either --checkpoint or --hf_repo must be specified")
        sys.exit(1)
    
    config = Config()
    
    if args.hf_repo:
        print(f"Generating from Hugging Face: {args.hf_repo}")
        generated = generate_from_hf(
            args.hf_repo,
            config,
            num_samples=args.num_samples,
            porosity=args.porosity,
            seed=args.seed
        )
    else:
        print(f"Generating from checkpoint: {args.checkpoint}")
        generator_state = load_generator_from_checkpoint(args.checkpoint, config)
        
        key = jax.random.PRNGKey(args.seed)
        noise = jax.random.normal(
            key,
            (args.num_samples, config.img_size[0], config.img_size[1], 
             config.img_size[2], config.latent_dim)
        )
        
        porosity_condition = jnp.array([args.porosity] * args.num_samples, dtype=jnp.float32)
        
        generated = generate_samples(generator_state, noise, porosity_condition)
        generated = np.where(generated <= 0.0, -1.0, 1.0)
    
    generated_binary = ((generated + 1) / 2 * 255).astype(np.uint8)
    
    with open(args.output, 'wb') as f:
        generated_binary.tofile(f)
    
    print(f"Generated {args.num_samples} samples with porosity {args.porosity}")
    print(f"Saved to: {args.output}")
    
    actual_porosity = np.mean(generated == -1.0)
    print(f"Actual porosity: {actual_porosity:.4f}")

if __name__ == "__main__":
    main()
