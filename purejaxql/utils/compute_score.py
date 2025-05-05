import jax.numpy as jnp

"""
Source: Craftax-Foraging project
"""

def compute_score(state, done):
    # repeat done for each achievement
    done_repeat = jnp.repeat(done, state.env_state.achievements.shape[1], axis=0)
    print("done_repeat shape: ", done_repeat.shape)
    achievements = state.env_state.achievements * done * 100.0
    info = {}
    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + achievements))) - 1.0
    return info
