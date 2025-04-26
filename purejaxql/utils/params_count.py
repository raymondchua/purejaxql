import jax
import jax.numpy as jnp

def count_parameters(params):
    """Return the total number of scalars in a Flax parameter PyTree."""
    return sum(jnp.prod(jnp.array(p.shape))            # size of each leaf array
               for p in jax.tree_util.tree_leaves(params))