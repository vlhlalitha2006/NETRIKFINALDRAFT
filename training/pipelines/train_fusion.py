from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.fusion.fusion_mlp import FusionMLP, count_parameters, get_device
from models.fusion.infer_fusion import build_fusion_input_matrix, load_tabular_pipeline


def _encode_binary_target(y_series: pd.Series) -> np.ndarray:
    unique_values = sorted(pd.Series(y_series).dropna().unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(f"Expected binary labels; found classes: {unique_values}")
    mapping = {unique_values[0]: 0.0, unique_values[1]: 1.0}
    return y_series.map(mapping).astype(np.float32).to_numpy()


def _load_fusion_training_inputs(
    data_csv_path: Path,
    target_column: str,
    id_column: str,
    tabular_pipeline_path: Path,
    sequence_features_path: Path,
    sequence_ids_path: Path,
    lstm_model_path: Path,
    graph_embeddings_path: Path,
    graph_index_path: Path,
    graph_node_ids_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(data_csv_path)
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' missing in {data_csv_path}")

    labels = _encode_binary_target(dataframe[target_column])
    tabular_input = dataframe.drop(columns=[target_column])
    applicant_ids = (
        dataframe[id_column].astype(str).tolist()
        if id_column in dataframe.columns
        else dataframe.index.astype(str).tolist()
    )

    tabular_pipeline = load_tabular_pipeline(tabular_pipeline_path)
    fusion_inputs = build_fusion_input_matrix(
        tabular_pipeline=tabular_pipeline,
        tabular_dataframe=tabular_input,
        sequence_features_path=sequence_features_path,
        sequence_ids_path=sequence_ids_path,
        lstm_model_path=lstm_model_path,
        graph_embeddings_path=graph_embeddings_path,
        graph_index_path=graph_index_path,
        applicant_ids=applicant_ids,
        node_ids_path=graph_node_ids_path if graph_node_ids_path.exists() else None,
        batch_size=batch_size,
    )
    return fusion_inputs, labels


def train_fusion_model(
    fusion_inputs: np.ndarray,
    labels: np.ndarray,
    model_path: Path,
    batch_size: int = 32,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    patience: int = 4,
    random_state: int = 42,
) -> dict[str, float | int]:
    if fusion_inputs.ndim != 2:
        raise ValueError("fusion_inputs must have shape (num_samples, input_dim).")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (num_samples,).")
    if epochs > 20:
        raise ValueError("epochs must be <= 20.")
    if np.isnan(fusion_inputs).any():
        raise ValueError("Fusion input contains NaN values.")
    if int(fusion_inputs.shape[1]) != 65:
        raise ValueError(
            f"Expected concatenated input dimension 65, got {int(fusion_inputs.shape[1])}."
        )

    x_train, x_val, y_train, y_val = train_test_split(
        fusion_inputs,
        labels,
        test_size=validation_split,
        random_state=random_state,
        stratify=labels.astype(int),
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    device = get_device()
    model = FusionMLP(input_dim=int(fusion_inputs.shape[1])).to(device)
    model_param_count = int(count_parameters(model))
    print(f"Model parameter count: {model_param_count}")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    final_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_size_actual = batch_x.size(0)
            train_loss_sum += float(loss.item()) * batch_size_actual
            train_count += batch_size_actual

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_true: list[np.ndarray] = []
        val_pred: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int()

                batch_size_actual = batch_x.size(0)
                val_loss_sum += float(loss.item()) * batch_size_actual
                val_count += batch_size_actual
                val_true.append(batch_y.int().cpu().numpy())
                val_pred.append(preds.cpu().numpy())

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_f1 = f1_score(
            np.concatenate(val_true).astype(int),
            np.concatenate(val_pred).astype(int),
            zero_division=0,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_f1={val_f1:.4f}"
        )
        final_val_f1 = float(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = float(val_f1)
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return {
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "final_val_f1": float(final_val_f1),
        "input_dim": int(fusion_inputs.shape[1]),
        "param_count": model_param_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shallow multimodal fusion MLP.")
    parser.add_argument("--data-csv-path", type=Path, default=Path("data/raw/TRAIN.csv"))
    parser.add_argument("--target-column", type=str, default="Loan_Status")
    parser.add_argument("--id-column", type=str, default="Loan_ID")
    parser.add_argument(
        "--tabular-pipeline-path",
        type=Path,
        default=Path("artifacts/tabular/sklearn_xgb_pipeline.joblib"),
    )
    parser.add_argument(
        "--lstm-model-path",
        type=Path,
        default=Path("artifacts/sequence/lstm_encoder.pt"),
    )
    parser.add_argument(
        "--sequence-features-path",
        type=Path,
        default=Path("data/processed/sequence_features.npy"),
    )
    parser.add_argument(
        "--sequence-ids-path",
        type=Path,
        default=Path("data/processed/sequence_ids.npy"),
    )
    parser.add_argument(
        "--graph-embeddings-path",
        type=Path,
        default=Path("artifacts/graph/precomputed_node_embeddings.npy"),
    )
    parser.add_argument(
        "--graph-index-path",
        type=Path,
        default=Path("artifacts/graph/node_embedding_index.pkl"),
    )
    parser.add_argument(
        "--graph-node-ids-path",
        type=Path,
        default=Path("data/graph/node_ids.npy"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/fusion/fusion_mlp.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion_inputs, labels = _load_fusion_training_inputs(
        data_csv_path=args.data_csv_path,
        target_column=args.target_column,
        id_column=args.id_column,
        tabular_pipeline_path=args.tabular_pipeline_path,
        sequence_features_path=args.sequence_features_path,
        sequence_ids_path=args.sequence_ids_path,
        lstm_model_path=args.lstm_model_path,
        graph_embeddings_path=args.graph_embeddings_path,
        graph_index_path=args.graph_index_path,
        graph_node_ids_path=args.graph_node_ids_path,
        batch_size=args.batch_size,
    )

    metrics = train_fusion_model(
        fusion_inputs=fusion_inputs,
        labels=labels,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        patience=args.patience,
    )
    print(f"Saved fusion model to: {args.model_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Expected saved model at: {args.model_path}")
    model_size_mb = args.model_path.stat().st_size / (1024 * 1024)
    print(f"Best validation F1: {metrics['best_val_f1']:.4f}")
    print(f"Final validation F1: {metrics['final_val_f1']:.4f}")
    print(f"Final input dimension: {metrics['input_dim']}")
    print(f"Saved model size (MB): {model_size_mb:.4f}")


if __name__ == "__main__":
    main()

