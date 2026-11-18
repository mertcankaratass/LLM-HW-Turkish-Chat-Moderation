"""Technique area A: our own word2vec (trained on the training messages) + a feedforward network.

A message is represented by the average of its word2vec vectors (words come from lexicon.tokenize,
i.e. normalized + asciified text). The feedforward network is trained with SGD + backpropagation.
"""
import numpy as np
import torch
from torch import nn

from chatmod.lexicon import tokenize


def embed(texts, w2v) -> np.ndarray:
    """Average word2vec vector per message (zero vector when no word is in the vocabulary)."""
    dim = w2v.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        vecs = [w2v.wv[w] for w in tokenize(text) if w in w2v.wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
