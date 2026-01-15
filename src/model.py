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

def rnn_step(params, h_prev, x_indexed):
    x_t = jax.nn.one_hot(x_indexed, params['Wxh'].shape[1]).reshape(-1, 1)

    #We chose the tanh activation function for the hidden state and a linear activation for the output
    # RNN formula 1:
    # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h) 
    h_next = jnp.tanh(jnp.dot(params['Wxh'], x_t) + jnp.dot(params['Whh'], h_prev) + params['bh'])
    
    # RNN formula 2:
    # y_t = W_hy * h_t + b_y
    y_t = jnp.dot(params['Why'], h_next) + params['by']

    return h_next, y_t.flatten()

def forward_pass(params, h_init, input_indices):
    # jax.lax.scan 
    def scan_fn(h, x):
        h_next, y_t = rnn_step(params, h, x)
        return h_next, y_t
    
    h_final, logits_sequence = jax.lax.scan(scan_fn, h_init, input_indices)
    return logits_sequence