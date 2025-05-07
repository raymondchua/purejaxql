import jax.numpy as jnp

"""
Source: Craftax-Foraging project
"""

def compute_score(state, done):
    # repeat done for each achievement
    done_repeat = jnp.reshape(done, (1, done.shape[0]))
    done_repeat = jnp.repeat(done_repeat, state.env_state.achievements.shape[1], axis=0)
    achievements = state.env_state.achievements * jnp.swapaxes(done_repeat, 0, 1)  * 100.0
    # Geometric mean with an offset of 1%
    return jnp.exp(jnp.mean(jnp.log(1 + achievements), keepdims=True)) - 1.0
