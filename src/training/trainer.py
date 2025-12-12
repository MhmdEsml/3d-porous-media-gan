import os
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state, checkpoints
import functools
from typing import Dict, Any, Tuple
from tqdm.auto import tqdm

from ..models.generator import UNetGenerator
from ..models.discriminator import Discriminator
from .losses import compute_generator_loss, compute_discriminator_loss
from .utils import create_train_state, load_checkpoint_if_exists
from ..data.dataset import create_train_generator

class Trainer:
    def __init__(self, config):
        self.config = config
        self.num_devices = jax.local_device_count()
        self.total_batch_size = config.batch_size_per_device * self.num_devices
        
    def initialize_models(self, key):
        generator = UNetGenerator(
            features=self.config.g_features,
            kernel_sizes=self.config.g_kernel_sizes,
            use_dilation=self.config.g_use_dilation,
            use_attention=self.config.g_use_attention
        )
        
        discriminator = Discriminator(features=self.config.d_features)
        
        dummy_noise = jnp.ones((
            1, self.config.img_size[0], self.config.img_size[1], 
            self.config.img_size[2], self.config.latent_dim
        ), dtype=jnp.float32)
        
        dummy_image = jnp.ones((
            1, self.config.img_size[0], self.config.img_size[1], 
            self.config.img_size[2], 1
        ), dtype=jnp.float32)
        
        dummy_condition = jnp.ones((1,), dtype=jnp.float32)
        
        key, g_key, d_key = jax.random.split(key, 3)
        
        state_g = create_train_state(
            generator, g_key, self.config.learning_rate_g,
            self.config.beta1, self.config.beta2,
            dummy_noise, dummy_condition
        )
        
        state_d = create_train_state(
            discriminator, d_key, self.config.learning_rate_d,
            self.config.beta1, self.config.beta2,
            dummy_image, dummy_condition
        )
        
        return state_g, state_d
    
    def create_train_step(self):
        @functools.partial(jax.pmap, axis_name='batch')
        def train_step(state_g, state_d, real_batch, noise):
            real_images = real_batch['images']
            real_porosity = real_batch['porosity']
            
            def d_loss_fn(g_params, d_params, real_images, real_porosity, noise):
                fake_images = state_g.apply_fn({'params': g_params}, noise, real_porosity)
                real_logits, _ = state_d.apply_fn({'params': d_params}, real_images, real_porosity)
                fake_logits, _ = state_d.apply_fn({'params': d_params}, fake_images, real_porosity)
                loss = compute_discriminator_loss(real_logits, fake_logits)
                return loss
            
            def g_loss_fn(g_params, d_params, real_images, real_porosity, noise):
                fake_images = state_g.apply_fn({'params': g_params}, noise, real_porosity)
                real_logits, real_features = state_d.apply_fn({'params': d_params}, real_images, real_porosity)
                fake_logits, fake_features = state_d.apply_fn({'params': d_params}, fake_images, real_porosity)
                
                loss_adv = compute_generator_loss(fake_logits, real_features, fake_features, 
                                                 real_images, fake_images, real_porosity,
                                                 self.config)
                return loss_adv
            
            d_grad_fn = jax.value_and_grad(d_loss_fn, argnums=1)
            g_grad_fn = jax.value_and_grad(g_loss_fn, argnums=0)
            
            d_loss, d_grads = d_grad_fn(
                state_g.params, state_d.params, real_images, real_porosity, noise
            )
            d_grads = jax.lax.pmean(d_grads, axis_name='batch')
            new_state_d = state_d.apply_gradients(grads=d_grads)
            
            g_loss, g_grads = g_grad_fn(
                state_g.params, new_state_d.params, real_images, real_porosity, noise
            )
            g_grads = jax.lax.pmean(g_grads, axis_name='batch')
            new_state_g = state_g.apply_gradients(grads=g_grads)
            
            metrics = {
                'd_loss': d_loss,
                'g_loss': g_loss
            }
            metrics = jax.lax.pmean(metrics, axis_name='batch')
            
            return new_state_g, new_state_d, metrics
        
        return train_step
    
    def train(self, volume, dataset_name):
        key = jax.random.PRNGKey(self.config.seed)
        
        state_g, state_d = self.initialize_models(key)
        
        start_step = 0
        if self.config.resume:
            state_g, g_step = load_checkpoint_if_exists(self.config.save_dir, state_g, 'generator_')
            state_d, d_step = load_checkpoint_if_exists(self.config.save_dir, state_d, 'discriminator_')
            start_step = max(g_step, d_step)
        
        train_loader = create_train_generator(
            volume,
            (self.config.img_size[0], self.config.img_size[1], self.config.img_size[2]),
            self.total_batch_size
        )
        
        state_g = flax.jax_utils.replicate(state_g)
        state_d = flax.jax_utils.replicate(state_d)
        
        p_train_step = self.create_train_step()
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        metrics_log = []
        pbar = tqdm(total=self.config.num_steps, desc=f"Training {dataset_name}", 
                   unit="step", initial=start_step)
        
        for step in range(start_step, self.config.num_steps):
            try:
                batch = next(train_loader)
            except StopIteration:
                break
            
            sharded_batch = jax.tree_util.tree_map(
                lambda x: x.reshape((self.num_devices, self.config.batch_size_per_device) + x.shape[1:]),
                batch
            )
            
            key, noise_key = jax.random.split(key)
            sharded_noise = jax.random.normal(
                noise_key,
                (self.num_devices, self.config.batch_size_per_device,
                 self.config.img_size[0], self.config.img_size[1], 
                 self.config.img_size[2], self.config.latent_dim)
            )
            
            state_g, state_d, metrics = p_train_step(
                state_g, state_d, sharded_batch, sharded_noise
            )
            
            current_step = step + 1
            pbar.update(1)
            
            if current_step % self.config.log_every == 0:
                metrics = flax.jax_utils.unreplicate(metrics)
                metrics_log.append(metrics)
                pbar.set_postfix({
                    'D_loss': f"{metrics['d_loss']:.4f}",
                    'G_loss': f"{metrics['g_loss']:.4f}"
                })
            
            if current_step % self.config.viz_every == 0:
                key, viz_key = jax.random.split(key)
                # Import here to avoid circular dependency
                from ..inference.inference import generate_and_visualize_samples
                generate_and_visualize_samples(
                    state_g, train_loader, current_step, viz_key, self.config
                )

            
            if current_step % self.config.save_every == 0:
                checkpoints.save_checkpoint(
                    self.config.save_dir,
                    flax.jax_utils.unreplicate(state_g),
                    step=current_step,
                    prefix='generator_',
                    overwrite=True
                )
                checkpoints.save_checkpoint(
                    self.config.save_dir,
                    flax.jax_utils.unreplicate(state_d),
                    step=current_step,
                    prefix='discriminator_',
                    overwrite=True
                )
        
        pbar.close()
        return state_g, state_d
