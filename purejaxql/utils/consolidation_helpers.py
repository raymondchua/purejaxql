import jax
import jax.numpy as jnp
import chex
from flax.core import FrozenDict

Params = FrozenDict

def update_and_accumulate_tree(p1: Params, p2: Params, scale: float, loss: float, max_norm: float = 10.0, delta_t_consolidation: int = 1, mask: int = 1) -> [Params, float]:
    def update_fn(a, b):
        delta = scale * (b - a) * delta_t_consolidation * mask
        # return a + delta, jnp.sum(jnp.square(delta))
        norm = jnp.linalg.norm(delta)
        # Clip the norm if it's too large
        clipped_delta = jnp.where(norm > max_norm, delta * (max_norm / norm), delta)
        return a + clipped_delta, jnp.sum(jnp.square(clipped_delta))

    # Flatten the PyTrees
    flat_p1, tree_def = jax.tree_util.tree_flatten(p1)
    flat_p2, _ = jax.tree_util.tree_flatten(p2)

    new_flat = []
    loss_terms = []

    for a, b in zip(flat_p1, flat_p2):
        updated, l = update_fn(a, b)
        new_flat.append(updated)
        loss_terms.append(l)

    # Rebuild the tree
    updates = jax.tree_util.tree_unflatten(tree_def, new_flat)
    total_loss = loss + jnp.sum(jnp.stack(loss_terms))
    return updates, total_loss

def update_and_accumulate_task(p1: chex.Array, p2: chex.Array, scale: float, loss: float, max_norm: float = 10.0, delta_t_consolidation: int = 1, mask: int = 1) -> [chex.Array, float]:
    def update_fn(a, b):
        delta = scale * (b - a) * delta_t_consolidation * mask
        # return a + delta, jnp.sum(jnp.square(delta))
        norm = jnp.linalg.norm(delta)
        # Clip the norm if it's too large
        clipped_delta = jnp.where(norm > max_norm, delta * (max_norm / norm), delta)
        return a + clipped_delta, jnp.sum(jnp.square(clipped_delta))

    # for a, b in zip(p1, p2):
    #     updated, l = update_fn(a, b)
    #     new_flat.append(updated)
    #     loss_terms.append(l)

    updated, l = update_fn(p1, p2)

    total_loss = loss + l

    # use chex to ensure that updated and p1 have the same shape
    chex.assert_equal_shape([updated, p1])

    return updated, total_loss