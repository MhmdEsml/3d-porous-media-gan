import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence, List

class DiscriminatorBlock(nn.Module):
    features: int
    strides: Tuple[int, int, int] = (2, 2, 2)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            features=self.features,
            kernel_size=(4, 4, 4),
            strides=self.strides,
            padding='SAME',
            kernel_init=nn.initializers.xavier_normal(),
        )(x)
        x = nn.GroupNorm(num_groups=min(32, self.features))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        return x

class Discriminator(nn.Module):
    features: Sequence[int] = (64, 128, 256, 512)

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        c_reshaped = jnp.reshape(c, (c.shape[0], 1, 1, 1, 1))
        c_tiled = jnp.tile(c_reshaped, (1, x.shape[1], x.shape[2], x.shape[3], 1))
        x_cond = jnp.concatenate([x, c_tiled], axis=-1)
        intermediate_features = []
        x = nn.Conv(
            features=self.features[0],
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
            padding='SAME',
            kernel_init=nn.initializers.xavier_normal(),
        )(x_cond)
        x = nn.leaky_relu(x, negative_slope=0.2)
        intermediate_features.append(x)
        for f in self.features[1:]:
            x = DiscriminatorBlock(features=f)(x)
            intermediate_features.append(x)
        logits = nn.Conv(
            features=1,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='SAME',
            kernel_init=nn.initializers.xavier_normal(),
        )(x)
        return logits, intermediate_features
