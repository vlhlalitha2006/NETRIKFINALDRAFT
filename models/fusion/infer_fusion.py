from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models.fusion.fusion_mlp import FusionMLP, get_device
from models.graph.precompute_embeddings import load_embedding_store
from models.sequence.infer_lstm import encode_sequences, load_lstm_encoder


def _safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def load_tabular_pipeline(path: str | Path):
    return joblib.load(path)


def compute_tabular_logits(
    pipeline,
    tabular_dataframe: pd.DataFrame,
) -> np.ndarray:
    probabilities = pipeline.predict_proba(tabular_dataframe)[:, 1]
    logits = _safe_logit(probabilities).reshape(-1, 1).astype(np.float32)
    return logits


def build_graph_lookup_index(
    index_mapping: dict,
    node_ids_path: str | Path | None = None,
) -> dict[str | int, int]:
    lookup: dict[str | int, int] = {}
    for key, value in index_mapping.items():
        lookup[key] = int(value)
        lookup[str(key)] = int(value)

    if node_ids_path is not None and Path(node_ids_path).exists():
        node_ids = np.load(node_ids_path, allow_pickle=True)
        for row_idx, node_id in enumerate(node_ids.tolist()):
            lookup[node_id] = int(row_idx)
            lookup[str(node_id)] = int(row_idx)
    return lookup


def load_sequence_artifacts(
    sequence_features_path: str | Path,
    sequence_ids_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_features = np.load(sequence_features_path).astype(np.float32)
    sequence_ids = np.load(sequence_ids_path, allow_pickle=True)
    if sequence_features.ndim != 3:
        raise ValueError("sequence_features must have shape (num_samples, seq_len, feature_dim).")
    if sequence_ids.ndim != 1:
        raise ValueError("sequence_ids must be a 1D array.")
    if sequence_features.shape[0] != sequence_ids.shape[0]:
        raise ValueError("sequence_features and sequence_ids row counts must match.")
    return sequence_features, sequence_ids


def build_sequence_lookup_index(sequence_ids: np.ndarray) -> dict[str | int, int]:
    lookup: dict[str | int, int] = {}
    for row_idx, sequence_id in enumerate(sequence_ids.tolist()):
        lookup[sequence_id] = int(row_idx)
        lookup[str(sequence_id)] = int(row_idx)
    return lookup


def select_sequences_by_applicant_ids(
    applicant_ids: list[str | int],
    sequence_features: np.ndarray,
    sequence_lookup: dict[str | int, int],
) -> np.ndarray:
    missing_ids: list[str] = []
    selected_rows: list[np.ndarray] = []
    for applicant_id in applicant_ids:
        if applicant_id in sequence_lookup:
            row_idx = sequence_lookup[applicant_id]
        elif str(applicant_id) in sequence_lookup:
            row_idx = sequence_lookup[str(applicant_id)]
        else:
            missing_ids.append(str(applicant_id))
            continue
        selected_rows.append(sequence_features[row_idx])

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            "Missing Loan_ID values in sequence_ids.npy. "
            f"Count={len(missing_ids)}. Examples: {preview}"
        )

    return np.asarray(selected_rows, dtype=np.float32)


def lookup_graph_embeddings(
    applicant_ids: list[str | int],
    embeddings: np.ndarray,
    lookup_index: dict[str | int, int],
) -> np.ndarray:
    resolved_embeddings = np.zeros((len(applicant_ids), embeddings.shape[1]), dtype=np.float32)
    for row_idx, applicant_id in enumerate(applicant_ids):
        if applicant_id in lookup_index:
            embedding_idx = lookup_index[applicant_id]
        elif str(applicant_id) in lookup_index:
            embedding_idx = lookup_index[str(applicant_id)]
        else:
            raise KeyError(f"Graph embedding not found for applicant id: {applicant_id}")
        resolved_embeddings[row_idx] = embeddings[embedding_idx]
    return resolved_embeddings


def build_fusion_input_matrix(
    tabular_pipeline,
    tabular_dataframe: pd.DataFrame,
    sequence_features_path: str | Path,
    sequence_ids_path: str | Path,
    lstm_model_path: str | Path,
    graph_embeddings_path: str | Path,
    graph_index_path: str | Path,
    applicant_ids: list[str | int],
    node_ids_path: str | Path | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    tabular_logits = compute_tabular_logits(tabular_pipeline, tabular_dataframe)

    sequence_features, sequence_ids = load_sequence_artifacts(
        sequence_features_path=sequence_features_path,
        sequence_ids_path=sequence_ids_path,
    )
    sequence_lookup = build_sequence_lookup_index(sequence_ids)
    aligned_sequences = select_sequences_by_applicant_ids(
        applicant_ids=applicant_ids,
        sequence_features=sequence_features,
        sequence_lookup=sequence_lookup,
    )

    sequence_list = [aligned_sequences[i] for i in range(aligned_sequences.shape[0])]
    feature_dim = int(sequence_features.shape[2])
    max_seq_len = int(sequence_features.shape[1])
    lstm_model, lstm_device = load_lstm_encoder(
        model_path=lstm_model_path,
        feature_dim=feature_dim,
        max_seq_len=max_seq_len,
    )
    lstm_embeddings = encode_sequences(
        model=lstm_model,
        device=lstm_device,
        sequences=sequence_list,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    ).astype(np.float32)

    graph_embeddings, index_mapping = load_embedding_store(
        embedding_path=graph_embeddings_path,
        index_path=graph_index_path,
    )
    lookup_index = build_graph_lookup_index(index_mapping=index_mapping, node_ids_path=node_ids_path)
    selected_graph_embeddings = lookup_graph_embeddings(
        applicant_ids=applicant_ids,
        embeddings=graph_embeddings.astype(np.float32),
        lookup_index=lookup_index,
    )

    if not (
        tabular_logits.shape[0]
        == lstm_embeddings.shape[0]
        == selected_graph_embeddings.shape[0]
    ):
        raise ValueError(
            "Branch outputs have inconsistent sample counts after ID-based alignment."
        )

    fusion_inputs = np.concatenate(
        [tabular_logits, lstm_embeddings, selected_graph_embeddings],
        axis=1,
    ).astype(np.float32)
    return fusion_inputs


def load_fusion_model(
    model_path: str | Path,
    input_dim: int,
) -> tuple[FusionMLP, torch.device]:
    device = get_device()
    model = FusionMLP(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def predict_fusion_logits(
    model: FusionMLP,
    device: torch.device,
    fusion_inputs: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, fusion_inputs.shape[0], batch_size):
            batch = fusion_inputs[start : start + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            logits = model(batch_tensor)
            logits_all.append(logits.cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def predict_fusion_probabilities(
    model: FusionMLP,
    device: torch.device,
    fusion_inputs: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    logits = predict_fusion_logits(
        model=model,
        device=device,
        fusion_inputs=fusion_inputs,
        batch_size=batch_size,
    )
    return 1.0 / (1.0 + np.exp(-logits))
