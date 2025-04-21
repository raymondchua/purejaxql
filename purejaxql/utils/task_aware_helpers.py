import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple


class TaskModulatedDense(nn.Module):
    num_tasks: int
    features: int

    @nn.compact
    def __call__(self, x, task_id: int):
        layer = nn.Dense(features=self.features, use_bias=False)
        y = layer(x)

        # Initialize all task-specific gains and biases at once
        gain_shape = (self.num_tasks, self.features)
        bias_shape = (self.num_tasks, self.features)
        gains = self.param("gains", nn.initializers.ones, gain_shape)
        biases = self.param("biases", nn.initializers.zeros, bias_shape)

        # Use the task_id to index into the gains and biases
        gain = gains[task_id]
        bias = biases[task_id]

        y = gain * y + bias
        return y


class TaskModulatedConv(nn.Module):
    num_tasks: int
    features: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str = "VALID"
    kernel_init: Callable = nn.initializers.he_normal()

    @nn.compact
    def __call__(self, x, task_id: int):
        layer = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            use_bias=False,
            strides=self.strides,
            padding=self.padding,
            kernel_init=self.kernel_init,
        )
        y = layer(x)

        print("y shape: ", y.shape)

        gain_shape = (self.num_tasks, self.features, 1, 1)
        bias_shape = (self.num_tasks, self.features, 1, 1)
        gains = self.param('gains', nn.initializers.ones, gain_shape)
        biases = self.param('biases', nn.initializers.zeros, bias_shape)

        gain = jnp.take(gains, task_id, axis=0)  # Shape (features, 1, 1)
        bias = jnp.take(biases, task_id, axis=0)  # Shape (features, 1, 1)

        # Reshape gain and bias to be compatible with (batch_size, height, width, features)
        gain = jnp.expand_dims(gain, axis=(0, 1))  # Now shape (1, 1, 1, features)
        bias = jnp.expand_dims(bias, axis=(0, 1))  # Now shape (1, 1, 1, features)

        print("gain shape: ", gain.shape)

        y = gain * y + bias
        return y
