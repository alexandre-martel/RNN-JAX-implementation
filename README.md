# RNN-JAX from scratch

Implementation of a discrete RNN for text generation, designed just with JAX.

The model is trained on Montaigne's *Essays* to capture the structures of Old French, in order to try to do carachter prediction.

We need a lot of epochs for traning (500.000 at least) but with Jax and especially jit and lax.scan, it is very fast.

## Structure du Projet
- `data/essais_montaigne.txt` : Training data
- `src/model.py` : Sampling and RNN maths
- `src/train.py` : Optimisation 
- `src/utils.py` : Token processor
- `main.py` : Training and weights saving
- `inference.py` : Inference 

## Installation
```bash
pip install -r requirement.txt
```