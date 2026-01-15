import jax
import jax.numpy as jnp
import pickle
from src.model import rnn_step

def complete_text(prompt, length=200):
    with open("rnn_montaigne.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    proc = checkpoint['proc']
    hidden_size = params['Wxh'].shape[0]
    
    h = jnp.zeros((hidden_size, 1))
    
    indices = proc.encode(prompt)
    for idx in indices[:-1]:
        h, _ = rnn_step(params, h, idx)
    
    current_idx = indices[-1]
    result = [current_idx]
    
    key = jax.random.PRNGKey(0)
    for _ in range(length):
        h, y_t = rnn_step(params, h, current_idx)
        
        # temp
        probs = jax.nn.softmax(y_t / 0.8) 
        
        key, subkey = jax.random.split(key)
        current_idx = jax.random.choice(subkey, proc.vocab_size, p=probs)
        result.append(int(current_idx))
    
    return proc.decode(result)


input_prompt = input("Enter a prompt: ")
print(f"Prompt: {input_prompt}")
print(f"Completion: {complete_text(input_prompt)}")