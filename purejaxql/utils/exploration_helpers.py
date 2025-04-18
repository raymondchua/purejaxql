import jax
import jax.numpy as jnp

def compute_action_entropy_probs(q_values: jnp.ndarray, tau: float = 10.0) -> jnp.ndarray:
    """
    Computes entropy and probs of the softmax policy induced by Q-values.

    Args:
        q_values: jnp.ndarray of shape (batch_size, num_actions)
        tau: temperature for softmax. Should be higher if Q-values are large (e.g. 10–50 for Q in [0,100]).

    Returns:
        Scalar: average entropy across the batch (in nats).
    """
    logits = q_values / tau
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
    return entropy, probs
