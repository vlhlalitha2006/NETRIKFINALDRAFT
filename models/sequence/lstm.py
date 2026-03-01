from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2
EMBEDDING_SIZE = 32


class LSTMSequenceEncoder(nn.Module):
    def __init__(self, feature_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(p=DROPOUT)
        self.projection = nn.Linear(HIDDEN_SIZE, EMBEDDING_SIZE)

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequences.dim() != 3:
            raise ValueError("Expected input shape (batch_size, seq_len, feature_dim).")

        if lengths is not None:
            packed = pack_padded_sequence(
                sequences,
                lengths=lengths.to("cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden_state, _) = self.lstm(packed)
        else:
            _, (hidden_state, _) = self.lstm(sequences)

        last_hidden = hidden_state[-1]
        return self.projection(self.dropout(last_hidden))


def pad_or_truncate_sequences(
    sequences: Sequence[np.ndarray],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("No sequences provided.")

    first_sequence = sequences[0]
    if first_sequence.ndim != 2:
        raise ValueError("Each sequence must have shape (seq_len, feature_dim).")

    feature_dim = first_sequence.shape[1]
    batch_size = len(sequences)
    batch = np.zeros((batch_size, max_seq_len, feature_dim), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for idx, sequence in enumerate(sequences):
        if sequence.ndim != 2 or sequence.shape[1] != feature_dim:
            raise ValueError("All sequences must share the same feature_dim.")
        current_len = min(sequence.shape[0], max_seq_len)
        batch[idx, :current_len, :] = sequence[:current_len]
        lengths[idx] = current_len

    return batch, lengths
