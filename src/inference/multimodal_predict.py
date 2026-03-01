from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from models.fusion.fusion_mlp import FusionMLP
from models.fusion.infer_fusion import (
    build_graph_lookup_index,
    build_sequence_lookup_index,
    compute_tabular_logits,
    load_sequence_artifacts,
    lookup_graph_embeddings,
)
from models.graph.precompute_embeddings import load_embedding_store
from models.sequence.lstm import LSTMSequenceEncoder, pad_or_truncate_sequences
from scripts.build_sequence_features import build_financial_progression_sequence


CPU_DEVICE = torch.device("cpu")
TABULAR_PIPELINE_PATH = Path("artifacts/tabular/sklearn_xgb_pipeline.joblib")
LSTM_MODEL_PATH = Path("artifacts/sequence/lstm_encoder.pt")
FUSION_MODEL_PATH = Path("artifacts/fusion/fusion_mlp.pt")
SEQUENCE_FEATURES_PATH = Path("data/processed/sequence_features.npy")
SEQUENCE_IDS_PATH = Path("data/processed/sequence_ids.npy")
GRAPH_EMBEDDINGS_PATH = Path("artifacts/graph/precomputed_node_embeddings.npy")
GRAPH_INDEX_PATH = Path("artifacts/graph/node_embedding_index.pkl")
GRAPH_NODE_IDS_PATH = Path("data/graph/node_ids.npy")

STRUCTURED_INPUT_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

PIPELINE_INPUT_COLUMNS = ["Loan_ID", *STRUCTURED_INPUT_COLUMNS]

_TABULAR_PIPELINE = None
_TABULAR_PREPROCESSOR = None
_EXPECTED_TABULAR_DIM: int | None = None
_SEQUENCE_FEATURES: np.ndarray | None = None
_SEQUENCE_LOOKUP: dict[str | int, int] | None = None
_GRAPH_EMBEDDINGS: np.ndarray | None = None
_GRAPH_LOOKUP: dict[str | int, int] | None = None
_LSTM_MODEL: LSTMSequenceEncoder | None = None
_FUSION_MODEL: FusionMLP | None = None
_LSTM_MAX_SEQ_LEN: int | None = None
_LSTM_FEATURE_DIM: int | None = None


def _get_cached(name: str, value):
    if value is None:
        raise RuntimeError(f"{name} is not initialized.")
    return value


def _initialize_caches() -> None:
    global _TABULAR_PIPELINE
    global _TABULAR_PREPROCESSOR
    global _EXPECTED_TABULAR_DIM
    global _SEQUENCE_FEATURES
    global _SEQUENCE_LOOKUP
    global _GRAPH_EMBEDDINGS
    global _GRAPH_LOOKUP
    global _LSTM_MODEL
    global _FUSION_MODEL
    global _LSTM_MAX_SEQ_LEN
    global _LSTM_FEATURE_DIM

    if _TABULAR_PIPELINE is None:
        _TABULAR_PIPELINE = joblib.load(TABULAR_PIPELINE_PATH)
        _TABULAR_PREPROCESSOR = _TABULAR_PIPELINE.named_steps["preprocessor"]
        _EXPECTED_TABULAR_DIM = int(len(_TABULAR_PREPROCESSOR.get_feature_names_out()))

    if _SEQUENCE_FEATURES is None or _SEQUENCE_LOOKUP is None:
        sequence_features, sequence_ids = load_sequence_artifacts(
            sequence_features_path=SEQUENCE_FEATURES_PATH,
            sequence_ids_path=SEQUENCE_IDS_PATH,
        )
        _SEQUENCE_FEATURES = sequence_features.astype(np.float32)
        _SEQUENCE_LOOKUP = build_sequence_lookup_index(sequence_ids)
        _LSTM_MAX_SEQ_LEN = int(_SEQUENCE_FEATURES.shape[1])
        _LSTM_FEATURE_DIM = int(_SEQUENCE_FEATURES.shape[2])

    if _GRAPH_EMBEDDINGS is None or _GRAPH_LOOKUP is None:
        graph_embeddings, graph_index = load_embedding_store(
            embedding_path=GRAPH_EMBEDDINGS_PATH,
            index_path=GRAPH_INDEX_PATH,
        )
        _GRAPH_EMBEDDINGS = graph_embeddings.astype(np.float32)
        _GRAPH_LOOKUP = build_graph_lookup_index(
            index_mapping=graph_index,
            node_ids_path=GRAPH_NODE_IDS_PATH if GRAPH_NODE_IDS_PATH.exists() else None,
        )

    if _LSTM_MODEL is None:
        max_seq_len = int(_get_cached("lstm_max_seq_len", _LSTM_MAX_SEQ_LEN))
        feature_dim = int(_get_cached("lstm_feature_dim", _LSTM_FEATURE_DIM))
        model = LSTMSequenceEncoder(feature_dim=feature_dim, max_seq_len=max_seq_len)
        state_dict = torch.load(LSTM_MODEL_PATH, map_location=CPU_DEVICE)
        model.load_state_dict(state_dict)
        model.to(CPU_DEVICE)
        model.eval()
        _LSTM_MODEL = model

    if _FUSION_MODEL is None:
        model = FusionMLP(input_dim=65)
        state_dict = torch.load(FUSION_MODEL_PATH, map_location=CPU_DEVICE)
        model.load_state_dict(state_dict)
        model.to(CPU_DEVICE)
        model.eval()
        _FUSION_MODEL = model


def _ordered_row(df_row: pd.DataFrame, loan_id: str | None) -> pd.DataFrame:
    if len(df_row) != 1:
        raise ValueError("multimodal_predict expects exactly one input row.")
    for col in STRUCTURED_INPUT_COLUMNS:
        if col not in df_row.columns:
            raise ValueError(f"Missing required input column: {col}")

    row = df_row.iloc[0]
    resolved_loan_id = str(loan_id).strip() if loan_id else "MANUAL_INPUT"
    ordered = pd.DataFrame(
        [
            {
                "Loan_ID": resolved_loan_id,
                "Gender": str(row["Gender"]).strip(),
                "Married": str(row["Married"]).strip(),
                "Dependents": str(row["Dependents"]).strip(),
                "Education": str(row["Education"]).strip(),
                "Self_Employed": str(row["Self_Employed"]).strip(),
                "ApplicantIncome": float(row["ApplicantIncome"]),
                "CoapplicantIncome": float(row["CoapplicantIncome"]),
                "LoanAmount": float(row["LoanAmount"]),
                "Loan_Amount_Term": float(row["Loan_Amount_Term"]),
                "Credit_History": float(row["Credit_History"]),
                "Property_Area": str(row["Property_Area"]).strip(),
            }
        ],
        columns=PIPELINE_INPUT_COLUMNS,
    )
    return ordered


def _resolve_sequence(ordered: pd.DataFrame, loan_id: str | None) -> np.ndarray:
    sequence_features = _get_cached("sequence_features", _SEQUENCE_FEATURES)
    sequence_lookup = _get_cached("sequence_lookup", _SEQUENCE_LOOKUP)
    if loan_id is not None and (loan_id in sequence_lookup or str(loan_id) in sequence_lookup):
        key = loan_id if loan_id in sequence_lookup else str(loan_id)
        row_idx = int(sequence_lookup[key])
        return sequence_features[row_idx].astype(np.float32)

    generated_sequence, _ = build_financial_progression_sequence(ordered.copy())
    return generated_sequence[0].astype(np.float32)


def _resolve_graph_embedding(loan_id: str | None) -> np.ndarray:
    graph_embeddings = _get_cached("graph_embeddings", _GRAPH_EMBEDDINGS)
    graph_lookup = _get_cached("graph_lookup", _GRAPH_LOOKUP)
    if loan_id is not None and (loan_id in graph_lookup or str(loan_id) in graph_lookup):
        selected = lookup_graph_embeddings(
            applicant_ids=[loan_id],
            embeddings=graph_embeddings,
            lookup_index=graph_lookup,
        )
        return selected[0].astype(np.float32)
    return np.zeros((32,), dtype=np.float32)


def _encode_sequence(sequence_raw: np.ndarray) -> np.ndarray:
    lstm_model = _get_cached("lstm_model", _LSTM_MODEL)
    max_seq_len = int(_get_cached("lstm_max_seq_len", _LSTM_MAX_SEQ_LEN))
    padded, lengths = pad_or_truncate_sequences([sequence_raw], max_seq_len=max_seq_len)
    seq_tensor = torch.tensor(padded, dtype=torch.float32, device=CPU_DEVICE)
    length_tensor = torch.tensor(lengths, dtype=torch.int64, device=CPU_DEVICE)
    with torch.no_grad():
        embedding = lstm_model(seq_tensor, length_tensor).cpu().numpy()[0].astype(np.float32)
    if embedding.shape != (32,):
        raise ValueError(f"Expected sequence embedding shape (32,), got {embedding.shape}")
    return embedding


def multimodal_predict(
    df_row: pd.DataFrame,
    loan_id: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    _initialize_caches()
    ordered = _ordered_row(df_row, loan_id=loan_id)
    resolved_loan_id = str(loan_id).strip() if loan_id else None

    tabular_pipeline = _get_cached("tabular_pipeline", _TABULAR_PIPELINE)
    tabular_preprocessor = _get_cached("tabular_preprocessor", _TABULAR_PREPROCESSOR)
    expected_dim = int(_get_cached("expected_tabular_dim", _EXPECTED_TABULAR_DIM))

    transformed = tabular_preprocessor.transform(ordered)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    tabular_vector = np.asarray(transformed_dense, dtype=np.float32)[0]
    if tabular_vector.shape != (expected_dim,):
        raise ValueError(
            f"Unexpected transformed tabular shape {tabular_vector.shape}; expected {(expected_dim,)}"
        )

    tabular_logit = float(compute_tabular_logits(tabular_pipeline, ordered)[0, 0])
    sequence_raw = _resolve_sequence(ordered, loan_id=resolved_loan_id)
    sequence_embedding = _encode_sequence(sequence_raw)
    graph_embedding = _resolve_graph_embedding(loan_id=resolved_loan_id)

    fusion_input = np.concatenate(
        [
            np.array([tabular_logit], dtype=np.float32),
            sequence_embedding.astype(np.float32),
            graph_embedding.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)[None, :]
    if fusion_input.shape != (1, 65):
        raise ValueError(f"Expected fusion_input shape (1, 65), got {fusion_input.shape}")

    fusion_model = _get_cached("fusion_model", _FUSION_MODEL)
    fusion_model.eval()
    with torch.no_grad():
        tensor = torch.tensor(fusion_input, dtype=torch.float32, device=CPU_DEVICE)
        logit = float(fusion_model(tensor).item())
        probability = float(torch.sigmoid(torch.tensor(logit, dtype=torch.float32)).item())

    prediction = "Approved" if probability >= 0.5 else "Rejected"

    if debug:
        print(f"[DEBUG] tabular_logit={tabular_logit:.10f}")
        print(f"[DEBUG] sequence_embedding[:5]={sequence_embedding[:5]}")
        print(f"[DEBUG] graph_embedding[:5]={graph_embedding[:5]}")
        print(f"[DEBUG] fusion_input[:10]={fusion_input[0, :10]}")
        print(f"[DEBUG] probability={probability:.10f}")

    return {
        "loan_id": resolved_loan_id,
        "tabular_vector": tabular_vector,
        "tabular_logit": tabular_logit,
        "sequence_raw": sequence_raw,
        "sequence_embedding": sequence_embedding,
        "graph_embedding": graph_embedding,
        "fusion_input": fusion_input,
        "logit": logit,
        "approval_probability": probability,
        "prediction": prediction,
    }

