import jnp

class TextProcessor:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return jnp.array([self.char_to_ix[ch] for ch in text])

    def decode(self, ix_list):
        return "".join([self.ix_to_char[int(i)] for i in ix_list])