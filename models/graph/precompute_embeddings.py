from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from models.graph.graphsage import OUTPUT_DIM, GraphSAGEEncoder


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def precompute_all_node_embeddings(
    encoder: GraphSAGEEncoder,
    data: Data,
    batch_size: int = 512,
    num_neighbors: list[int] | None = None,
    full_batch_threshold: int = 20000,
) -> np.ndarray:
    if num_neighbors is None:
        num_neighbors = [15, 10]

    device = get_device()
    encoder = encoder.to(device)
    encoder.eval()

    num_nodes = int(data.num_nodes)
    if num_nodes <= full_batch_threshold:
        with torch.no_grad():
            embeddings = encoder(
                data.x.to(device),
                data.edge_index.to(device),
            )
        return embeddings.cpu().numpy().astype(np.float32)

    embedding_store = np.zeros((num_nodes, OUTPUT_DIM), dtype=np.float32)
    loader = NeighborLoader(
        data,
        input_nodes=torch.arange(num_nodes),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_embeddings = encoder(batch.x, batch.edge_index)
            seed_embeddings = batch_embeddings[: batch.batch_size]
            seed_node_ids = batch.n_id[: batch.batch_size].cpu().numpy()
            embedding_store[seed_node_ids] = seed_embeddings.cpu().numpy()

    return embedding_store


def save_precomputed_embeddings(
    embeddings: np.ndarray,
    node_ids: np.ndarray | None = None,
    embedding_path: str | Path = "artifacts/graph/precomputed_node_embeddings.npy",
    index_path: str | Path = "artifacts/graph/node_embedding_index.pkl",
) -> tuple[Path, Path]:
    if embeddings.ndim != 2 or embeddings.shape[1] != OUTPUT_DIM:
        raise ValueError(f"Expected embeddings with shape (num_nodes, {OUTPUT_DIM}).")

    embedding_path = Path(embedding_path)
    index_path = Path(index_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embedding_path, embeddings.astype(np.float32))
    if node_ids is None:
        index_mapping = {int(i): int(i) for i in range(embeddings.shape[0])}
    else:
        index_mapping = {str(node_id): int(i) for i, node_id in enumerate(node_ids.tolist())}

    with open(index_path, "wb") as file_obj:
        pickle.dump(index_mapping, file_obj)

    return embedding_path, index_path


def load_embedding_store(
    embedding_path: str | Path = "artifacts/graph/precomputed_node_embeddings.npy",
    index_path: str | Path = "artifacts/graph/node_embedding_index.pkl",
) -> tuple[np.ndarray, dict[str | int, int]]:
    embeddings = np.load(embedding_path)
    with open(index_path, "rb") as file_obj:
        index_mapping = pickle.load(file_obj)
    return embeddings, index_mapping


def lookup_node_embedding(
    node_id: int | str,
    embeddings: np.ndarray,
    index_mapping: dict[str | int, int],
) -> np.ndarray:
    if node_id in index_mapping:
        row_idx = index_mapping[node_id]
    elif str(node_id) in index_mapping:
        row_idx = index_mapping[str(node_id)]
    else:
        raise KeyError(f"Node id {node_id} not found in embedding index.")
    return embeddings[row_idx]
