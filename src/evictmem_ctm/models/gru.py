"""Single-layer GRU baseline for event-token sequences."""

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int = 16, embedding_dim: int = 32,
                 hidden_size: int = 64, num_classes: int = 2,
                 num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if num_layers != 1 or dropout != 0:
            raise ValueError("This baseline uses one GRU layer and no dropout")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, final_hidden = self.gru(self.embedding(sequence))
        return self.classifier(final_hidden[-1])
