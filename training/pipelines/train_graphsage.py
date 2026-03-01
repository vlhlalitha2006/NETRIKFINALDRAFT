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
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from models.graph.graphsage import GraphSAGENodeClassifier
from models.graph.precompute_embeddings import save_precomputed_embeddings


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def load_graph_data(
    node_features_path: Path,
    edge_index_path: Path,
    node_ids_path: Path,
) -> tuple[Data, np.ndarray]:
    x = np.load(node_features_path).astype(np.float32)
    edge_index = np.load(edge_index_path).astype(np.int64)
    node_ids = np.load(node_ids_path, allow_pickle=True)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index.npy must have shape (2, num_edges).")
    if x.ndim != 2:
        raise ValueError("node_features.npy must have shape (num_nodes, feature_dim).")
    if node_ids.ndim != 1 or node_ids.shape[0] != x.shape[0]:
        raise ValueError("node_ids.npy must have shape (num_nodes,).")

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.zeros(x.shape[0], dtype=torch.float32),
    )
    return data, node_ids


def build_train_labels_and_masks(
    train_csv_path: Path,
    node_ids: np.ndarray,
    validation_split: float,
    random_state: int,
    id_column: str = "Loan_ID",
    target_column: str = "Loan_Status",
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    train_df = pd.read_csv(train_csv_path)
    if id_column not in train_df.columns:
        raise ValueError(f"Missing ID column '{id_column}' in train CSV.")
    if target_column not in train_df.columns:
        raise ValueError(f"Missing target column '{target_column}' in train CSV.")

    unique_values = sorted(train_df[target_column].dropna().unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(f"Expected binary target labels, found: {unique_values}")
    target_mapping = {unique_values[0]: 0.0, unique_values[1]: 1.0}

    label_lookup = {
        str(row[id_column]): float(target_mapping[row[target_column]])
        for _, row in train_df.iterrows()
    }
    labels_for_all = np.zeros(len(node_ids), dtype=np.float32)
    train_node_indices: list[int] = []
    for idx, node_id in enumerate(node_ids.tolist()):
        key = str(node_id)
        if key in label_lookup:
            labels_for_all[idx] = label_lookup[key]
            train_node_indices.append(idx)

    if not train_node_indices:
        raise ValueError("No TRAIN Loan_ID values matched graph node_ids.")

    train_node_indices_np = np.array(train_node_indices, dtype=np.int64)
    train_labels_np = labels_for_all[train_node_indices_np].astype(int)
    train_ids, val_ids = train_test_split(
        train_node_indices_np,
        test_size=validation_split,
        random_state=random_state,
        stratify=train_labels_np,
    )
    train_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    val_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    train_mask[torch.tensor(train_ids)] = True
    val_mask[torch.tensor(val_ids)] = True
    return labels_for_all, train_mask, val_mask


def _train_full_batch(
    model: GraphSAGENodeClassifier,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    optimizer.zero_grad()
    logits, _ = model(data.x.to(device), data.edge_index.to(device))
    labels = data.y.to(device)
    train_loss = loss_fn(logits[train_mask.to(device)], labels[train_mask.to(device)])
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x.to(device), data.edge_index.to(device))
        val_logits = logits[val_mask.to(device)]
        val_labels = labels[val_mask.to(device)]
        val_loss = loss_fn(val_logits, val_labels)
        val_predictions = (torch.sigmoid(val_logits) >= 0.5).int().cpu().numpy()
        val_truth = val_labels.int().cpu().numpy()
        val_f1 = f1_score(val_truth, val_predictions, zero_division=0)

    return float(train_loss.item()), float(val_loss.item()), float(val_f1)


def _build_neighbor_loader(
    data: Data,
    input_nodes: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> NeighborLoader:
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _train_neighbor_sampled(
    model: GraphSAGENodeClassifier,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: Adam,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float]:
    train_loader = _build_neighbor_loader(data, train_mask, batch_size=batch_size, shuffle=True)
    val_loader = _build_neighbor_loader(data, val_mask, batch_size=batch_size, shuffle=False)

    model.train()
    total_train_loss = 0.0
    total_train_count = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch.x, batch.edge_index)
        seed_logits = logits[: batch.batch_size]
        seed_labels = batch.y[: batch.batch_size]
        loss = loss_fn(seed_logits, seed_labels)
        loss.backward()
        optimizer.step()

        total_train_loss += float(loss.item()) * int(batch.batch_size)
        total_train_count += int(batch.batch_size)

    model.eval()
    total_val_loss = 0.0
    total_val_count = 0
    val_preds: list[np.ndarray] = []
    val_truth: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits, _ = model(batch.x, batch.edge_index)
            seed_logits = logits[: batch.batch_size]
            seed_labels = batch.y[: batch.batch_size]
            loss = loss_fn(seed_logits, seed_labels)

            probabilities = torch.sigmoid(seed_logits)
            predictions = (probabilities >= 0.5).int().cpu().numpy()
            labels_np = seed_labels.int().cpu().numpy()
            val_preds.append(predictions)
            val_truth.append(labels_np)

            total_val_loss += float(loss.item()) * int(batch.batch_size)
            total_val_count += int(batch.batch_size)

    mean_train_loss = total_train_loss / max(total_train_count, 1)
    mean_val_loss = total_val_loss / max(total_val_count, 1)
    val_f1 = f1_score(
        np.concatenate(val_truth).astype(int),
        np.concatenate(val_preds).astype(int),
        zero_division=0,
    )
    return float(mean_train_loss), float(mean_val_loss), float(val_f1)


def train_graphsage_offline(
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    model_path: Path,
    epochs: int = 15,
    learning_rate: float = 0.005,
    patience: int = 4,
    batch_size: int = 256,
    full_batch_threshold: int = 15000,
) -> GraphSAGENodeClassifier:
    if epochs > 20:
        raise ValueError("epochs must be <= 20.")
    if learning_rate > 0.01:
        raise ValueError("learning_rate must be <= 0.01.")

    device = get_device()
    model = GraphSAGENodeClassifier(in_dim=int(data.x.shape[1])).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if int(data.num_nodes) <= full_batch_threshold:
            train_loss, val_loss, val_f1 = _train_full_batch(
                model=model,
                data=data,
                train_mask=train_mask,
                val_mask=val_mask,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
        else:
            train_loss, val_loss, val_f1 = _train_neighbor_sampled(
                model=model,
                data=data,
                train_mask=train_mask,
                val_mask=val_mask,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                batch_size=batch_size,
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GraphSAGE training pipeline.")
    parser.add_argument(
        "--node-features-path",
        type=Path,
        default=Path("data/graph/node_features.npy"),
        help="NumPy array with shape (num_nodes, feature_dim).",
    )
    parser.add_argument(
        "--edge-index-path",
        type=Path,
        default=Path("data/graph/edge_index.npy"),
        help="NumPy array with shape (2, num_edges).",
    )
    parser.add_argument(
        "--node-ids-path",
        type=Path,
        default=Path("data/graph/node_ids.npy"),
        help="Optional external node IDs aligned with row index.",
    )
    parser.add_argument(
        "--train-csv-path",
        type=Path,
        default=Path("data/raw/TRAIN.csv"),
        help="TRAIN CSV path for labels (Loan_Status).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/graph/graphsage_model.pt"),
        help="Saved GraphSAGE classifier state_dict path.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv_path}")
    if not args.node_ids_path.exists():
        raise FileNotFoundError(f"Node IDs file not found: {args.node_ids_path}")

    data, node_ids = load_graph_data(
        node_features_path=args.node_features_path,
        edge_index_path=args.edge_index_path,
        node_ids_path=args.node_ids_path,
    )
    labels_for_all, train_mask, val_mask = build_train_labels_and_masks(
        train_csv_path=args.train_csv_path,
        node_ids=node_ids,
        validation_split=args.validation_split,
        random_state=42,
    )
    data.y = torch.tensor(labels_for_all, dtype=torch.float32)

    model = train_graphsage_offline(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        model_path=args.model_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    from models.graph.precompute_embeddings import precompute_all_node_embeddings

    embeddings = precompute_all_node_embeddings(encoder=model.encoder, data=data)
    embedding_path, index_path = save_precomputed_embeddings(
        embeddings=embeddings,
        node_ids=node_ids,
        embedding_path="artifacts/graph/precomputed_node_embeddings.npy",
        index_path="artifacts/graph/node_embedding_index.pkl",
    )
    print(f"Saved model to: {args.model_path}")
    print(f"Saved embeddings to: {embedding_path}")
    print(f"Saved embedding index to: {index_path}")
    save_success = embedding_path.exists() and index_path.exists()
    print(f"embedding_shape: {embeddings.shape}")
    print(f"dtype: {embeddings.dtype}")
    print(f"total_nodes: {embeddings.shape[0]}")
    print(f"save_success: {save_success}")


if __name__ == "__main__":
    main()

