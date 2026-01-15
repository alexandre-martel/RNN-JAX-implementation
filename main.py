from src.model import init_params
from src.train import update_step
from src.utils import TextProcessor
import jax.numpy as jnp
from jax import random
import os

file_path = "data\essais_montaigne.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

proc = TextProcessor(text)
data = proc.encode(text)

seq_length = 50  
hidden_size = 256 
learning_rate = 0.001
epochs = 10000

key = random.PRNGKey(1)
params = init_params(proc.vocab_size, hidden_size=hidden_size, key=key)
h_init = jnp.zeros((hidden_size, 1))

p = 0 
for epoch in range(epochs):
    if p + seq_length + 1 >= len(data):
        p = 0
    
    inputs = data[p : p + seq_length]
    targets = data[p + 1 : p + seq_length + 1]
    
    params, loss = update_step(params, h_init, inputs, targets, learning_rate)
    p += seq_length
    
    if epoch % 500 == 0:
        print(f"Étape {epoch} | Position dans le texte: {p} | Loss: {loss:.4f}")