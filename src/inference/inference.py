import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
import os
from huggingface_hub import hf_hub_download
from ..models.generator import UNetGenerator
from ..training.utils import TrainState
import optax
from flax.training import checkpoints
import flax

def generate_samples(generator_state, noise, porosity_condition):
    return generator_state.apply_fn(
        {'params': generator_state.params},
        noise,
        porosity_condition
    )

def load_generator_from_checkpoint(checkpoint_path, config):
    generator = UNetGenerator(
        features=config.g_features,
        kernel_sizes=config.g_kernel_sizes,
        use_dilation=config.g_use_dilation,
        use_attention=config.g_use_attention
    )
    
    dummy_noise = jnp.ones((
        1, config.img_size[0], config.img_size[1], 
        config.img_size[2], config.latent_dim
    ), dtype=jnp.float32)
    
    dummy_condition = jnp.ones((1,), dtype=jnp.float32)
    
    state = TrainState.create(
        apply_fn=generator.apply,
        params=generator.init(jax.random.PRNGKey(0), dummy_noise, dummy_condition)['params'],
        tx=optax.adam(0.001)
    )
    
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state,
        prefix='generator_'
    )
    
    return restored_state

def generate_from_hf(repo_id, config, num_samples=1, porosity=0.2, seed=42):
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="generator_checkpoint"
    )
    
    generator_state = load_generator_from_checkpoint(checkpoint_path, config)
    
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(
        key,
        (num_samples, config.img_size[0], config.img_size[1], 
         config.img_size[2], config.latent_dim)
    )
    
    porosity_condition = jnp.array([porosity] * num_samples, dtype=jnp.float32)
    
    generated = generate_samples(generator_state, noise, porosity_condition)
    
    generated_binary = np.where(generated <= 0.0, -1.0, 1.0)
    
    return generated_binary

def generate_and_visualize_samples(state_g, train_loader, step, key, config):
    num_devices = jax.local_device_count()
    samples_per_device = config.num_viz_samples // num_devices
    
    accumulated_images = []
    accumulated_porosities = []
    
    while len(accumulated_images) < config.num_viz_samples:
        try:
            batch = next(train_loader)
            batch_images = batch['images']
            batch_porosities = batch['porosity']
            
            num_needed = config.num_viz_samples - len(accumulated_images)
            num_to_take = min(num_needed, len(batch_images))
            
            accumulated_images.extend(batch_images[:num_to_take])
            accumulated_porosities.extend(batch_porosities[:num_to_take])
            
        except StopIteration:
            break
    
    real_samples_np = np.stack(accumulated_images)
    real_porosities_np = np.array(accumulated_porosities)
    
    total_samples = len(real_samples_np)
    if total_samples % num_devices != 0:
        total_samples = (total_samples // num_devices) * num_devices
        real_samples_np = real_samples_np[:total_samples]
        real_porosities_np = real_porosities_np[:total_samples]
    
    samples_per_device = total_samples // num_devices
    
    key, subkey = jax.random.split(key)
    sharded_keys = jax.random.split(subkey, num_devices)
    
    def create_sharded_noise(device_key):
        return jax.random.normal(
            device_key,
            (samples_per_device,
             config.img_size[0], config.img_size[1], config.img_size[2],
             config.latent_dim)
        )
    
    sharded_noise = jnp.stack([create_sharded_noise(k) for k in sharded_keys])
    sharded_porosity = jnp.array(real_porosities_np).reshape(
        (num_devices, samples_per_device)
    )
    
    @jax.pmap
    def generate_pmap(state_g, noise, porosity):
        return state_g.apply_fn(
            {'params': state_g.params},
            noise,
            porosity
        )
    
    fake_samples_sharded = generate_pmap(state_g, sharded_noise, sharded_porosity)
    
    fake_samples = fake_samples_sharded.reshape(
        (total_samples, config.img_size[0], config.img_size[1],
         config.img_size[2], 1)
    )
    
    fake_samples_continuous_np = np.array(fake_samples)
    fake_samples_binary_np = np.where(fake_samples_continuous_np <= 0.0, -1.0, 1.0)
    
    os.makedirs(config.viz_dir, exist_ok=True)
    
    fig, axes = plt.subplots(total_samples, 4, figsize=(16, 4 * total_samples))
    if total_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(total_samples):
        real_vol = real_samples_np[i, :, :, :, 0]
        fake_vol = fake_samples_binary_np[i, :, :, :, 0]
        
        d_mid = real_vol.shape[0] // 2
        h_mid = real_vol.shape[1] // 2
        
        axes[i, 0].imshow(real_vol[d_mid, :, :], cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Sample {i+1} - Real (XY)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(real_vol[:, h_mid, :], cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Sample {i+1} - Real (XZ)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(fake_vol[d_mid, :, :], cmap='gray', vmin=-1, vmax=1)
        axes[i, 2].set_title(f'Sample {i+1} - Fake (XY)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(fake_vol[:, h_mid, :], cmap='gray', vmin=-1, vmax=1)
        axes[i, 3].set_title(f'Sample {i+1} - Fake (XZ)')
        axes[i, 3].axis('off')
    
    plt.suptitle(f'3D Porous Media GAN - Step {step}', fontsize=16, y=0.98)
    plt.tight_layout()
    
    fig_path = os.path.join(config.viz_dir, f'cross_sections_step_{step:06d}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated visualization at step {step}")
