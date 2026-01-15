import jax.nn as nn
import jax.numpy as jnp
import jax
from src.model import forward_pass

def loss_fn(params, h_init, inputs, targets):
    logits = forward_pass(params, h_init, inputs)

    log_probs = nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1)
    return -jnp.mean(target_log_probs)

@jax.jit
def update_step(params, h_init, inputs, targets, learning_rate):
    loss, grads = jax.value_and_grad(loss_fn)(params, h_init, inputs, targets)

    new_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads
    )
    return new_params, loss