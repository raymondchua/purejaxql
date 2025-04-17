import jax
import jax.numpy as jnp
import chex
from typing import Any, Callable
from flax import linen as nn


def factorized_noise(shape, rng):
    """Generates factorized Gaussian noise for NoisyNet (Fortunato et al., 2017)."""
    noise = jax.random.normal(rng, shape)
    return jnp.sign(noise) * jnp.sqrt(jnp.abs(noise))


class NoisyLinear(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        # Base parameters (deterministic part)
        self.kernel = self.param("kernel", self.kernel_init, (self.features,))
        self.bias = self.param("bias", self.bias_init, (self.features,)) if self.use_bias else None

        # Noisy parameters
        self.sigma_kernel = self.param("sigma_kernel", nn.initializers.constant(0.5 / jnp.sqrt(self.features)), (self.features,))
        self.sigma_bias = self.param("sigma_bias", nn.initializers.constant(0.5 / jnp.sqrt(self.features)), (self.features,)) if self.use_bias else None

    @nn.compact
    def __call__(self, inputs, *, rng):
        """
        Args:
            inputs: (batch, in_features)
            rng: PRNGKey for noise sampling
        """
        in_features = inputs.shape[-1]

        # Sample factorized noise
        noise_in = factorized_noise((in_features,), rng)
        noise_out = factorized_noise((self.features,), rng)

        # Outer product for full noise matrix
        noise_matrix = jnp.outer(noise_out, noise_in)

        # Expand base parameters to full matrix
        w_mu = self.param("weight_mu", self.kernel_init, (self.features, in_features))
        w_sigma = self.param("weight_sigma", nn.initializers.constant(0.5 / jnp.sqrt(in_features)), (self.features, in_features))

        w_noisy = w_mu + w_sigma * noise_matrix

        out = jnp.dot(inputs, w_noisy.T)

        if self.use_bias:
            b_mu = self.bias
            b_sigma = self.sigma_bias
            b_noisy = b_mu + b_sigma * noise_out
            out += b_noisy

        return out.astype(self.dtype)