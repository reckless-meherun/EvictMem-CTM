"""CTM adapted to receive one system-event token per recurrent tick."""

import torch
from torch import nn

from evictmem_ctm.models.ctm_components import (
    DeepNeuronLevelModel,
    DepthOneSynapse,
    InitialCTMState,
    RandomPairSynchronisation,
    SuperLinear,
    advance_pre_activation_trace,
)


class SequenceCTM(nn.Module):
    """Classify a sequence from final-tick random-pair synchronization."""

    def __init__(self, vocab_size: int = 16, embedding_dim: int = 32,
                 d_model: int = 64, memory_length: int = 5,
                 memory_hidden_dim: int = 16, n_synch_out: int = 64,
                 num_classes: int = 2, dropout: float = 0.0,
                 deep_nlm: bool = True, pairing_seed: int = 42,
                 evicted_memory: bool = False, alpha: float = 0.95) -> None:
        super().__init__()
        if min(vocab_size, embedding_dim, d_model, memory_length,
               memory_hidden_dim, n_synch_out, num_classes) < 1:
            raise ValueError("All model dimensions must be positive")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.memory_length = memory_length
        self.evicted_memory = evicted_memory
        self.alpha = alpha
        nlm_input_length = memory_length + int(evicted_memory)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.initial_state = InitialCTMState(d_model, memory_length)
        self.synapse = DepthOneSynapse(embedding_dim + d_model, d_model, dropout)
        if deep_nlm:
            self.nlm = DeepNeuronLevelModel(d_model, nlm_input_length,
                                            memory_hidden_dim, dropout=dropout)
        else:
            self.nlm = nn.Sequential(
                SuperLinear(nlm_input_length, 2, d_model, dropout=dropout),
                nn.GLU(dim=-1),
                nn.Flatten(start_dim=1),
            )
        self.synchronisation = RandomPairSynchronisation(
            d_model, n_synch_out, pairing_seed=pairing_seed)
        self.classifier = nn.Linear(n_synch_out, num_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1] == 0:
            raise ValueError("Expected token IDs with shape [batch, sequence_length >= 1]")
        embeddings = self.embedding(sequence)
        activated, trace = self.initial_state(sequence.shape[0])
        if self.evicted_memory:
            compressed_memory = torch.zeros_like(activated)

        # As upstream does for output synchronization, include the learned start state.
        _, alpha, beta = self.synchronisation(activated)
        for tick in range(sequence.shape[1]):
            synapse_input = torch.cat((embeddings[:, tick, :], activated), dim=-1)
            pre_activation = self.synapse(synapse_input)
            if self.evicted_memory and tick >= self.memory_length:
                compressed_memory = (self.alpha * compressed_memory
                                     + (1 - self.alpha) * trace[:, :, 0])
            trace = advance_pre_activation_trace(trace, pre_activation)
            nlm_input = (torch.cat((trace, compressed_memory.unsqueeze(-1)), dim=-1)
                         if self.evicted_memory else trace)
            activated = self.nlm(nlm_input)
            synchronised, alpha, beta = self.synchronisation(activated, alpha, beta)

        return self.classifier(synchronised)
