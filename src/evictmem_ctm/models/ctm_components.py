"""Minimal CTM primitives for a future one-token-per-tick sequence model.

Adapted and minimized from SakanaAI Continuous Thought Machines:
https://github.com/SakanaAI/continuous-thought-machines
Reference commit: 4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466
Upstream code is licensed under Apache-2.0; see the repository LICENSE.
"""

import math

import torch
from torch import nn


class SuperLinear(nn.Module):
    """Apply an independent linear transform to each neuron's last dimension."""

    def __init__(self, in_dims: int, out_dims: int, n_neurons: int,
                 temperature: float = 1.0, do_norm: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if min(in_dims, out_dims, n_neurons) < 1 or temperature <= 0:
            raise ValueError("Dimensions and temperature must be positive")
        limit = 1 / math.sqrt(in_dims + out_dims)
        self.weight = nn.Parameter(torch.empty(in_dims, out_dims, n_neurons).uniform_(-limit, limit))
        self.bias = nn.Parameter(torch.zeros(1, n_neurons, out_dims))
        self.temperature = nn.Parameter(torch.tensor([temperature]))
        self.norm = nn.LayerNorm(in_dims) if do_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        values = self.norm(self.dropout(history))
        output = torch.einsum("bdm,mhd->bdh", values, self.weight) + self.bias
        return output.squeeze(-1) / self.temperature


class DeepNeuronLevelModel(nn.Module):
    """Map each neuron's finite pre-activation history to one activation."""

    def __init__(self, n_neurons: int, memory_length: int, hidden_dims: int,
                 do_norm: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            SuperLinear(memory_length, 2 * hidden_dims, n_neurons,
                        do_norm=do_norm, dropout=dropout),
            nn.GLU(dim=-1),
            SuperLinear(hidden_dims, 2, n_neurons,
                        do_norm=do_norm, dropout=dropout),
            nn.GLU(dim=-1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, pre_activation_trace: torch.Tensor) -> torch.Tensor:
        return self.layers(pre_activation_trace)


class InitialCTMState(nn.Module):
    """Learned initial activated state and pre-activation FIFO trace."""

    def __init__(self, n_neurons: int, memory_length: int) -> None:
        super().__init__()
        if n_neurons < 1 or memory_length < 1:
            raise ValueError("State dimensions must be positive")
        self.activated = nn.Parameter(torch.empty(n_neurons).uniform_(
            -1 / math.sqrt(n_neurons), 1 / math.sqrt(n_neurons)))
        self.pre_activation_trace = nn.Parameter(torch.empty(n_neurons, memory_length).uniform_(
            -1 / math.sqrt(n_neurons + memory_length),
            1 / math.sqrt(n_neurons + memory_length)))

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.activated.unsqueeze(0).expand(batch_size, -1),
                self.pre_activation_trace.unsqueeze(0).expand(batch_size, -1, -1))


def advance_pre_activation_trace(trace: torch.Tensor,
                                 pre_activation: torch.Tensor) -> torch.Tensor:
    """Drop the oldest value and append one new pre-activation per neuron."""
    return torch.cat((trace[:, :, 1:], pre_activation.unsqueeze(-1)), dim=-1)


class DepthOneSynapse(nn.Module):
    """Upstream depth-one synapse: dropout, linear, GLU, LayerNorm."""

    def __init__(self, input_dim: int, n_neurons: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, 2 * n_neurons),
            nn.GLU(dim=-1), nn.LayerNorm(n_neurons),
        )

    def forward(self, synapse_input: torch.Tensor) -> torch.Tensor:
        return self.layers(synapse_input)


class RandomPairSynchronisation(nn.Module):
    """Fixed sampled neuron pairs with learned recurrent synchronization decay."""

    def __init__(self, n_neurons: int, n_pairs: int, n_self_pairs: int = 0,
                 pairing_seed: int = 42) -> None:
        super().__init__()
        if n_neurons < 1 or n_pairs < 1 or not 0 <= n_self_pairs < n_pairs or pairing_seed < 0:
            raise ValueError("Require positive dimensions, valid self-pair count, and nonnegative seed")
        generator = torch.Generator().manual_seed(pairing_seed)
        left = torch.randint(n_neurons, (n_pairs,), generator=generator)
        right = torch.cat((left[:n_self_pairs],
                           torch.randint(n_neurons, (n_pairs - n_self_pairs,), generator=generator)))
        self.register_buffer("left_indices", left)
        self.register_buffer("right_indices", right)
        self.decay_params = nn.Parameter(torch.zeros(n_pairs))

    def forward(self, activated_state: torch.Tensor,
                alpha: torch.Tensor | None = None,
                beta: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return synchronization and its updated numerator/normalizer state."""
        if (alpha is None) != (beta is None):
            raise ValueError("alpha and beta must both be present or both be absent")
        pair_product = (activated_state[:, self.left_indices] *
                        activated_state[:, self.right_indices])
        if alpha is None:
            alpha = pair_product
            beta = torch.ones_like(pair_product)
        else:
            # Project decay parameters to the upstream range before using them.
            with torch.no_grad():
                self.decay_params.clamp_(0, 15)
            retention = torch.exp(-self.decay_params).unsqueeze(0)
            alpha = retention * alpha + pair_product
            beta = retention * beta + 1
        return alpha / torch.sqrt(beta), alpha, beta
