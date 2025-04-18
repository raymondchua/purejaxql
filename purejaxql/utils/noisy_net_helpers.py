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

    @nn.compact
    def __call__(self, inputs, *, rng):
        in_features = inputs.shape[-1]

        # Base parameters (mean)
        w_mu = self.param("weight_mu", self.kernel_init, (self.features, in_features))
        w_sigma = self.param("weight_sigma",
                             nn.initializers.constant(0.5 / jnp.sqrt(in_features)),
                             (self.features, in_features))

        if self.use_bias:
            b_mu = self.param("bias_mu", self.bias_init, (self.features,))
            b_sigma = self.param("bias_sigma",
                                 nn.initializers.constant(0.5 / jnp.sqrt(self.features)),
                                 (self.features,))
        else:
            b_mu, b_sigma = None, None

        # Sample factorized noise
        noise_in = factorized_noise((in_features,), rng)
        noise_out = factorized_noise((self.features,), rng)
        noise_matrix = jnp.outer(noise_out, noise_in)

        # Add noise to weights and biases
        w_noisy = w_mu + w_sigma * noise_matrix
        out = jnp.dot(inputs, w_noisy.T)

        if self.use_bias:
            b_noisy = b_mu + b_sigma * noise_out
            out += b_noisy

        return out.astype(self.dtype)