import jax.numpy as jnp

def cosine_similarity(a, b, eps=1e-8):
    # a, b: shape (batch_size, num_beakers-1, feature_dim)
    numerator = jnp.sum(a * b, axis=-1)  # (batch_size, num_beakers-1)
    denominator = jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + eps  # (batch_size, num_beakers-1)
    return numerator / denominator