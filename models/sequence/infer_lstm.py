from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.sequence.lstm import LSTMSequenceEncoder, pad_or_truncate_sequences


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def load_lstm_encoder(
    model_path: str | Path,
    feature_dim: int,
    max_seq_len: int,
) -> tuple[LSTMSequenceEncoder, torch.device]:
    device = get_device()
    model = LSTMSequenceEncoder(feature_dim=feature_dim, max_seq_len=max_seq_len)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def encode_sequences(
    model: LSTMSequenceEncoder,
    device: torch.device,
    sequences: list[np.ndarray],
    max_seq_len: int,
    batch_size: int = 32,
) -> np.ndarray:
    padded_sequences, lengths = pad_or_truncate_sequences(
        sequences=sequences,
        max_seq_len=max_seq_len,
    )

    dataset = TensorDataset(
        torch.tensor(padded_sequences, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.int64),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch_sequences, batch_lengths in dataloader:
            batch_sequences = batch_sequences.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_embeddings = model(batch_sequences, batch_lengths)
            all_embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)
