import jax
import jax.numpy as jnp
from jax import random

def init_params(vocab_size, hidden_size, key):
    k1, k2, k3 = random.split(key, 3)

    scale = 0.1
    return {
        'Wxh': random.normal(k1, (hidden_size, vocab_size)) * scale,
        'Whh': random.normal(k2, (hidden_size, hidden_size)) * scale,
        'Why': random.normal(k3, (vocab_size, hidden_size)) * scale,
        'bh': jnp.zeros((hidden_size, 1)),
        'by': jnp.zeros((vocab_size, 1))
    }