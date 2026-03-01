from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.sequence.lstm import EMBEDDING_SIZE, LSTMSequenceEncoder


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def _split_train_validation(
    sequences: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    validation_split: float,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    return train_test_split(
        sequences,
        lengths,
        labels,
        test_size=validation_split,
        random_state=random_state,
        stratify=labels,
    )


def _infer_lengths_from_padded_sequences(sequences: np.ndarray, max_seq_len: int) -> np.ndarray:
    non_padding_mask = np.any(sequences != 0.0, axis=2)
    lengths = non_padding_mask.sum(axis=1).astype(np.int64)
    lengths = np.clip(lengths, 1, max_seq_len)
    return lengths


def _build_dataloaders(
    train_sequences: np.ndarray,
    train_lengths: np.ndarray,
    train_labels: np.ndarray,
    val_sequences: np.ndarray,
    val_lengths: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.tensor(train_sequences, dtype=torch.float32),
        torch.tensor(train_lengths, dtype=torch.int64),
        torch.tensor(train_labels, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_sequences, dtype=torch.float32),
        torch.tensor(val_lengths, dtype=torch.int64),
        torch.tensor(val_labels, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_lstm_encoder(
    sequences: np.ndarray,
    labels: np.ndarray,
    max_seq_len: int,
    model_path: str | Path,
    batch_size: int = 32,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    patience: int = 3,
    random_state: int = 42,
) -> dict[str, float | tuple[int, ...] | bool]:
    if sequences.ndim != 3:
        raise ValueError("Expected sequences shape (batch_size, seq_len, feature_dim).")
    if labels.ndim != 1:
        raise ValueError("Expected labels shape (batch_size,).")
    if epochs > 15:
        raise ValueError("epochs must be <= 15.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    feature_dim = sequences.shape[2]
    lengths = _infer_lengths_from_padded_sequences(sequences, max_seq_len=max_seq_len)
    labels = labels.astype(np.float32)

    (
        train_sequences,
        val_sequences,
        train_lengths,
        val_lengths,
        train_labels,
        val_labels,
    ) = _split_train_validation(
        sequences=sequences,
        lengths=lengths,
        labels=labels,
        validation_split=validation_split,
        random_state=random_state,
    )

    train_loader, val_loader = _build_dataloaders(
        train_sequences=train_sequences,
        train_lengths=train_lengths,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_lengths=val_lengths,
        val_labels=val_labels,
        batch_size=batch_size,
    )

    device = get_device()
    encoder = LSTMSequenceEncoder(feature_dim=feature_dim, max_seq_len=max_seq_len).to(device)
    classifier_head = nn.Linear(EMBEDDING_SIZE, 1).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        list(encoder.parameters()) + list(classifier_head.parameters()),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(epochs):
        epoch_index = _ + 1
        encoder.train()
        classifier_head.train()
        total_train_loss = 0.0
        total_train_count = 0
        for batch_sequences, batch_lengths, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            embedding = encoder(batch_sequences, batch_lengths)
            logits = classifier_head(embedding).squeeze(-1)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_sequences.size(0)
            total_train_loss += float(loss.item()) * batch_size_actual
            total_train_count += batch_size_actual

        encoder.eval()
        classifier_head.eval()
        total_val_loss = 0.0
        total_count = 0
        val_labels_all: list[np.ndarray] = []
        val_preds_all: list[np.ndarray] = []
        with torch.no_grad():
            for batch_sequences, batch_lengths, batch_labels in val_loader:
                batch_sequences = batch_sequences.to(device)
                batch_lengths = batch_lengths.to(device)
                batch_labels = batch_labels.to(device)

                embedding = encoder(batch_sequences, batch_lengths)
                logits = classifier_head(embedding).squeeze(-1)
                batch_loss = loss_fn(logits, batch_labels)
                batch_size_actual = batch_sequences.size(0)
                total_val_loss += float(batch_loss.item()) * batch_size_actual
                total_count += batch_size_actual
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                val_labels_all.append(batch_labels.cpu().numpy())
                val_preds_all.append(predictions.cpu().numpy())

        avg_train_loss = total_train_loss / max(total_train_count, 1)
        avg_val_loss = total_val_loss / max(total_count, 1)
        val_f1 = f1_score(
            np.concatenate(val_labels_all).astype(int),
            np.concatenate(val_preds_all).astype(int),
            zero_division=0,
        )
        print(
            f"Epoch {epoch_index}/{epochs} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"val_f1={val_f1:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(encoder.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    sample_sequences, sample_lengths, _ = next(iter(val_loader))
    with torch.no_grad():
        sample_embeddings = encoder(sample_sequences.to(device), sample_lengths.to(device))
    embedding_shape = tuple(sample_embeddings.shape)

    return {
        "best_val_loss": float(best_val_loss),
        "model_saved": model_path.exists(),
        "sample_embedding_shape": embedding_shape,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM sequence encoder.")
    parser.add_argument(
        "--sequences-path",
        type=Path,
        default=Path("data/processed/sequence_features.npy"),
        help="NumPy file with shape (batch_size, seq_len, feature_dim).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/processed/sequence_labels.npy"),
        help="NumPy file with binary labels shape (batch_size,).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/sequence/lstm_encoder.pt"),
        help="Output path for encoder state dict.",
    )
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=2000,
        help="Use first N samples for quick sanity training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sequences = np.load(args.sequences_path).astype(np.float32)
    labels = np.load(args.labels_path).astype(np.float32)

    if args.subset_size > 0 and sequences.shape[0] > args.subset_size:
        sequences = sequences[: args.subset_size]
        labels = labels[: args.subset_size]

    if sequences.shape[1] != args.max_seq_len:
        sequences = sequences[:, : args.max_seq_len, :]
        if sequences.shape[1] < args.max_seq_len:
            pad_len = args.max_seq_len - sequences.shape[1]
            padding = np.zeros(
                (sequences.shape[0], pad_len, sequences.shape[2]), dtype=np.float32
            )
            sequences = np.concatenate([sequences, padding], axis=1)

    metrics = train_lstm_encoder(
        sequences=sequences,
        labels=labels,
        max_seq_len=args.max_seq_len,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        patience=args.patience,
    )
    print(f"Best validation loss: {metrics['best_val_loss']:.6f}")
    print(f"Saved encoder to: {args.model_path}")
    print(f"Model saved successfully: {metrics['model_saved']}")
    print(f"Sample embedding shape: {metrics['sample_embedding_shape']}")


if __name__ == "__main__":
    main()

