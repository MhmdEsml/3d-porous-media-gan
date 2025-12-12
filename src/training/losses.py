import jax.numpy as jnp
import jax
from typing import List

def l2_loss(x, y):
    return jnp.mean((x - y) ** 2)

def l1_loss(x, y):
    return jnp.mean(jnp.abs(x - y))

def compute_discriminator_loss(real_logits, fake_logits):
    real_loss = l2_loss(real_logits, 1.0)
    fake_loss = l2_loss(fake_logits, 0.0)
    return (real_loss + fake_loss) * 0.5

def compute_generator_loss(fake_logits, real_features, fake_features, 
                          real_images, fake_images, real_porosity, config):
    loss_adv = l2_loss(fake_logits, 1.0)
    
    loss_fm = 0.0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss_fm += l1_loss(real_feat, fake_feat)
    loss_fm = loss_fm / len(real_features)
    
    loss_por = loss_g_porosity(real_porosity, fake_images)
    
    total_loss = (
        loss_adv * config.lambda_adv +
        loss_fm * config.lambda_fm +
        loss_por * config.lambda_porosity
    )
    
    return total_loss

def loss_g_porosity(real_porosity, fake_images):
    soft_porosity_map = (1.0 - fake_images) / 2.0
    hard_porosity_map = jnp.round(soft_porosity_map)
    ste_porosity_map = jax.lax.stop_gradient(hard_porosity_map - soft_porosity_map) + soft_porosity_map
    fake_porosity = jnp.mean(ste_porosity_map, axis=(1, 2, 3, 4))
    return l1_loss(real_porosity, fake_porosity)
