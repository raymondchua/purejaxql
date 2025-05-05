import jax.numpy as jnp

"""
Source: Craftax-Foraging project
"""

def compute_score(state, done):
    print("achievements: ", state.env_state.achievements)
    print("done: ", done)
    achievements = state.achievements * done * 100.0
    info = {}
    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(1 + achievements))) - 1.0
    return info
