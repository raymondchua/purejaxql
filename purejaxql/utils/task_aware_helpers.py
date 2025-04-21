import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Tuple


class TaskModulatedLayer(nn.Module):
    """
    A layer that modulates its output with gains and biases based on the task ID. This is done in both
    "Continual Reinforcement Learning with Complex Synapses" paper and "Overcoming catastrophic forgetting in neural
    networks"
    """

    features: int
    task_id: int
    is_conv: bool = True
    kernel_size: Tuple[int, int] = (4, 4)
    strides: Tuple[int, int] = (2, 2)
    padding: str = "VALID"
    kernel_init: Callable = nn.initializers.he_normal()

    def setup(self):
        if self.is_conv:
            self.core = nn.Conv(
                self.features,
                kernel_size=self.kernel_size,
                padding=self.padding,
                strides=self.strides,
                kernel_init=self.kernel_init,
                use_bias=False,
                dtype=jnp.float32,
            )
        else:
            self.core = nn.Dense(self.features, kernel_init=self.kernel_init, use_bias=False, dtype=jnp.float32)
        self.task_bias = self.param(
            f"bias_task{self.task_id}", nn.initializers.zeros, (self.features,), dtype=jnp.float32
        )
        self.task_gain = self.param(
            f"gain_task{self.task_id}", nn.initializers.ones, (self.features,), dtype=jnp.float32
        )

    def __call__(self, x):
        y = self.core(x)
        y = y + self.task_bias
        y = y * self.task_gain
        return y
