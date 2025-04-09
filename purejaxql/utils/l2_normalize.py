import jax.numpy as jnp
from typing import Any, Callable

NetworkFn = Callable[..., Any]

def l2_normalize(
    p: int = 2,
    dim: int = 1,
    eps: float = 1e-12,
) -> NetworkFn:
    def net_fn(inputs):
        denominator = jnp.clip(
            jnp.linalg.norm(inputs, ord=p, axis=dim, keepdims=True), a_min=eps
        )
        return inputs / denominator

    return net_fn