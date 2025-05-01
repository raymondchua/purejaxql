import jax.numpy as jnp

def cosine_similarity(a, b, eps=1e-8):
    # a, b: shape (batch_size, num_beakers-1, feature_dim)
    numerator = jnp.sum(a * b, axis=-1)  # (batch_size, num_beakers-1)
    denominator = jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + eps  # (batch_size, num_beakers-1)
    return numerator / denominator


def rbf_similarity(a, b, sigma=1.0):
    squared_distance = jnp.sum((a - b) ** 2, axis=-1)  # (batch_size, num_beakers-1)
    return jnp.exp(-squared_distance / (2 * sigma ** 2))  # (batch_size, num_beakers-1)