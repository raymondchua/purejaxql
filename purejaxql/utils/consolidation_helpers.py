import jax
import jax.numpy as jnp
import chex
from flax.core import FrozenDict

Params = FrozenDict

def update_and_accumulate_tree(p1: Params, p2: Params, scale: float, loss: float) -> [Params, float]:
    def update_fn(a, b):
        delta = scale * (b - a)
        return a + delta, jnp.sum(jnp.square(delta))

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