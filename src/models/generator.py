import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, List, Sequence

class ConvBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    strides: Tuple[int, int, int] = (1, 1, 1)
    dilation: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
            kernel_dilation=self.dilation,
            kernel_init=nn.initializers.xavier_normal(),
        )(x)
        x = nn.GroupNorm(num_groups=min(32, self.features))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        return x

class ResidualConvBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    use_dilation: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        dilation = 2 if self.use_dilation else 1
        x = ConvBlock(
            features=self.features,
            kernel_size=self.kernel_size,
            dilation=dilation
        )(x)
        x = ConvBlock(
            features=self.features,
            kernel_size=self.kernel_size,
            dilation=1
        )(x)
        if residual.shape[-1] != self.features:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1, 1),
                padding='SAME',
                kernel_init=nn.initializers.xavier_normal(),
            )(residual)
        return x + residual

class SelfAttention3D(nn.Module):
    num_heads: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, D, H, W, C = x.shape
        x_flat = jnp.reshape(x, (B, D * H * W, C))
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=C,
            kernel_init=nn.initializers.xavier_normal(),
        )(x_flat, x_flat)
        attn_out = jnp.reshape(attn_out, (B, D, H, W, C))
        return attn_out + x

class UNetDownBlock(nn.Module):
    features: int
    pool: bool = True
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    use_dilation: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = ResidualConvBlock(
            features=self.features,
            kernel_size=self.kernel_size,
            use_dilation=self.use_dilation
        )(x)
        skip = x
        if self.pool:
            pool = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
            return pool, skip
        return x, skip

class UNetUpBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)

    @nn.compact
    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        x = jax.image.resize(
            x,
            shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3] * 2, x.shape[4]),
            method='nearest'
        )
        x = jnp.concatenate([x, skip], axis=-1)
        x = ResidualConvBlock(
            features=self.features,
            kernel_size=self.kernel_size
        )(x)
        return x

class UNetGenerator(nn.Module):
    features: Sequence[int] = (64, 128, 256, 512)
    kernel_sizes: Sequence[Tuple[int, int, int]] = ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3))
    use_dilation: Sequence[bool] = (False, False, False, False)
    use_attention: Sequence[bool] = (False, False, False, False)

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        skips = []
        for i, f in enumerate(self.features[:-1]):
            x, skip = UNetDownBlock(
                features=f,
                pool=True,
                kernel_size=self.kernel_sizes[i],
                use_dilation=self.use_dilation[i]
            )(x)
            if self.use_attention[i]:
                x = SelfAttention3D(num_heads=8)(x)
            skips.append(skip)
        bottleneck_idx = len(self.features) - 1
        x, _ = UNetDownBlock(
            features=self.features[-1],
            pool=False,
            kernel_size=self.kernel_sizes[bottleneck_idx],
            use_dilation=self.use_dilation[bottleneck_idx]
        )(x)
        if self.use_attention[bottleneck_idx]:
            x = SelfAttention3D(num_heads=8)(x)
        c_reshaped = jnp.reshape(c, (c.shape[0], 1, 1, 1, 1))
        c_tiled = jnp.tile(c_reshaped, (1, x.shape[1], x.shape[2], x.shape[3], 1))
        x = jnp.concatenate([x, c_tiled], axis=-1)
        x = nn.Conv(
            features=self.features[-1],
            kernel_size=(1, 1, 1),
            padding='SAME',
            kernel_init=nn.initializers.xavier_normal(),
        )(x)
        skips.reverse()
        for i, f in enumerate(reversed(self.features[:-1])):
            decoder_idx = len(self.features) - 2 - i
            x = UNetUpBlock(
                features=f,
                kernel_size=self.kernel_sizes[decoder_idx]
            )(x, skips[i])
        x = nn.Conv(
            features=1,
            kernel_size=(1, 1, 1),
            padding='SAME',
            kernel_init=nn.initializers.xavier_normal(),
        )(x)
        return nn.tanh(x)
