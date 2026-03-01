from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import shap  # type: ignore

from models.fusion.fusion_mlp import FusionMLP, get_device
from models.fusion.infer_fusion import (
    build_graph_lookup_index,
    build_sequence_lookup_index,
    compute_tabular_logits,
    load_sequence_artifacts,
    lookup_graph_embeddings,
    select_sequences_by_applicant_ids,
)
from models.graph.precompute_embeddings import load_embedding_store
from models.sequence.infer_lstm import encode_sequences, load_lstm_encoder


DATA_CSV = Path("data/raw/TRAIN.csv")
TABULAR_PIPELINE_PATH = Path("artifacts/tabular/sklearn_xgb_pipeline.joblib")
LSTM_MODEL_PATH = Path("artifacts/sequence/lstm_encoder.pt")
SEQUENCE_FEATURES_PATH = Path("data/processed/sequence_features.npy")
SEQUENCE_IDS_PATH = Path("data/processed/sequence_ids.npy")
GRAPH_EMBEDDINGS_PATH = Path("artifacts/graph/precomputed_node_embeddings.npy")
GRAPH_INDEX_PATH = Path("artifacts/graph/node_embedding_index.pkl")
GRAPH_NODE_IDS_PATH = Path("data/graph/node_ids.npy")
FUSION_MODEL_PATH = Path("artifacts/fusion/fusion_mlp.pt")

TOP_K = 3
IG_STEPS = 16
SEQUENCE_FEATURE_NAMES = [
    "applicant_income_state",
    "coapplicant_income_state",
    "total_income_state",
    "loan_amount_state",
    "loan_term_state",
    "credit_history_state",
    "dependents_state",
    "leverage_state",
]

_DEVICE = get_device()
_DATAFRAME: pd.DataFrame | None = None
_TABULAR_PIPELINE = None
_TABULAR_PREPROCESSOR = None
_TABULAR_CLASSIFIER = None
_TABULAR_FEATURE_NAMES: list[str] | None = None
_SHAP_EXPLAINER = None
_SEQUENCE_FEATURES: np.ndarray | None = None
_SEQUENCE_LOOKUP: dict[str | int, int] | None = None
_LSTM_MODEL = None
_LSTM_MAX_SEQ_LEN: int | None = None
_LSTM_FEATURE_DIM: int | None = None
_GRAPH_EMBEDDINGS: np.ndarray | None = None
_GRAPH_LOOKUP: dict[str | int, int] | None = None
_FUSION_MODEL = None


def _require_initialized(name: str, value):
    if value is None:
        raise RuntimeError(f"{name} cache is not initialized.")
    return value


def _initialize_caches() -> None:
    global _DATAFRAME
    global _TABULAR_PIPELINE
    global _TABULAR_PREPROCESSOR
    global _TABULAR_CLASSIFIER
    global _TABULAR_FEATURE_NAMES
    global _SHAP_EXPLAINER
    global _SEQUENCE_FEATURES
    global _SEQUENCE_LOOKUP
    global _LSTM_MODEL
    global _LSTM_MAX_SEQ_LEN
    global _LSTM_FEATURE_DIM
    global _GRAPH_EMBEDDINGS
    global _GRAPH_LOOKUP
    global _FUSION_MODEL

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_CSV}")
    _DATAFRAME = pd.read_csv(DATA_CSV)
    if "Loan_ID" not in _DATAFRAME.columns:
        raise ValueError("Loan_ID column missing in TRAIN.csv")

    _TABULAR_PIPELINE = joblib.load(TABULAR_PIPELINE_PATH)
    _TABULAR_PREPROCESSOR = _TABULAR_PIPELINE.named_steps["preprocessor"]
    _TABULAR_CLASSIFIER = _TABULAR_PIPELINE.named_steps["classifier"]
    _TABULAR_FEATURE_NAMES = _TABULAR_PREPROCESSOR.get_feature_names_out().tolist()
    # Cached once at module load to avoid per-request initialization overhead.
    _SHAP_EXPLAINER = shap.TreeExplainer(_TABULAR_CLASSIFIER)

    _SEQUENCE_FEATURES, sequence_ids = load_sequence_artifacts(
        sequence_features_path=SEQUENCE_FEATURES_PATH,
        sequence_ids_path=SEQUENCE_IDS_PATH,
    )
    _SEQUENCE_LOOKUP = build_sequence_lookup_index(sequence_ids)
    _LSTM_MAX_SEQ_LEN = int(_SEQUENCE_FEATURES.shape[1])
    _LSTM_FEATURE_DIM = int(_SEQUENCE_FEATURES.shape[2])
    _LSTM_MODEL, _ = load_lstm_encoder(
        model_path=LSTM_MODEL_PATH,
        feature_dim=_LSTM_FEATURE_DIM,
        max_seq_len=_LSTM_MAX_SEQ_LEN,
    )
    _LSTM_MODEL = _LSTM_MODEL.to(_DEVICE).eval()

    _GRAPH_EMBEDDINGS, graph_index_mapping = load_embedding_store(
        embedding_path=GRAPH_EMBEDDINGS_PATH,
        index_path=GRAPH_INDEX_PATH,
    )
    _GRAPH_LOOKUP = build_graph_lookup_index(
        index_mapping=graph_index_mapping,
        node_ids_path=GRAPH_NODE_IDS_PATH if GRAPH_NODE_IDS_PATH.exists() else None,
    )

    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing fusion model artifact: {FUSION_MODEL_PATH}")
    _FUSION_MODEL = FusionMLP(input_dim=65)
    state_dict = torch.load(FUSION_MODEL_PATH, map_location=_DEVICE)
    _FUSION_MODEL.load_state_dict(state_dict)
    _FUSION_MODEL = _FUSION_MODEL.to(_DEVICE).eval()


def _find_applicant_row(loan_id: str) -> pd.DataFrame:
    dataframe = _require_initialized("dataframe", _DATAFRAME)
    if "Loan_ID" not in dataframe.columns:
        raise ValueError("Loan_ID column missing in TRAIN.csv")
    matched = dataframe[dataframe["Loan_ID"].astype(str) == str(loan_id)]
    if matched.empty:
        raise KeyError(f"Loan_ID not found: {loan_id}")
    return matched.iloc[[0]].copy()


def _load_graph_embedding_for_id(loan_id: str) -> np.ndarray:
    graph_embeddings = _require_initialized("graph_embeddings", _GRAPH_EMBEDDINGS)
    graph_lookup = _require_initialized("graph_lookup", _GRAPH_LOOKUP)
    selected = lookup_graph_embeddings(
        applicant_ids=[loan_id],
        embeddings=graph_embeddings.astype(np.float32),
        lookup_index=graph_lookup,
    )
    return selected[0]


def _load_sequence_for_id(loan_id: str) -> np.ndarray:
    sequence_features = _require_initialized("sequence_features", _SEQUENCE_FEATURES)
    sequence_lookup = _require_initialized("sequence_lookup", _SEQUENCE_LOOKUP)
    aligned = select_sequences_by_applicant_ids(
        applicant_ids=[loan_id],
        sequence_features=sequence_features,
        sequence_lookup=sequence_lookup,
    )
    return aligned[0]


def _compute_lstm_embedding(sequence: np.ndarray) -> np.ndarray:
    lstm_model = _require_initialized("lstm_model", _LSTM_MODEL)
    max_seq_len = int(_require_initialized("lstm_max_seq_len", _LSTM_MAX_SEQ_LEN))
    with torch.no_grad():
        embedding = encode_sequences(
            model=lstm_model,
            device=_DEVICE,
            sequences=[sequence],
            max_seq_len=max_seq_len,
            batch_size=1,
        )
    return embedding[0].astype(np.float32)


def _compute_tabular_shap(
    tabular_row: pd.DataFrame,
) -> list[dict[str, Any]]:
    preprocessor = _require_initialized("tabular_preprocessor", _TABULAR_PREPROCESSOR)
    feature_names = _require_initialized("tabular_feature_names", _TABULAR_FEATURE_NAMES)
    explainer = _require_initialized("shap_explainer", _SHAP_EXPLAINER)

    transformed = preprocessor.transform(tabular_row)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    shap_values = explainer.shap_values(transformed_dense)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])[0]
    else:
        values = np.asarray(shap_values)[0]

    top_indices = np.argsort(np.abs(values))[::-1][:TOP_K]
    top_items: list[dict[str, Any]] = []
    for idx in top_indices:
        impact = float(values[idx])
        top_items.append(
            {
                "feature": str(feature_names[idx]),
                "impact": impact,
                "direction": "increase_risk" if impact > 0 else "decrease_risk",
            }
        )
    return top_items


def _ig_sequence_attributions(
    sequence: np.ndarray,
    tabular_logit: float,
    graph_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lstm_model = _require_initialized("lstm_model", _LSTM_MODEL)
    fusion_model = _require_initialized("fusion_model", _FUSION_MODEL)
    max_seq_len = int(_require_initialized("lstm_max_seq_len", _LSTM_MAX_SEQ_LEN))

    input_tensor = torch.tensor(sequence[None, :, :], dtype=torch.float32, device=_DEVICE)
    baseline = torch.zeros_like(input_tensor)
    lengths = torch.tensor([max_seq_len], dtype=torch.int64, device=_DEVICE)
    graph_tensor = torch.tensor(graph_embedding[None, :], dtype=torch.float32, device=_DEVICE)
    tabular_tensor = torch.tensor([[tabular_logit]], dtype=torch.float32, device=_DEVICE)

    total_grad = torch.zeros_like(input_tensor)
    alphas = torch.linspace(0.0, 1.0, steps=IG_STEPS, device=_DEVICE)
    for alpha in alphas:
        lstm_model.zero_grad(set_to_none=True)
        fusion_model.zero_grad(set_to_none=True)
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        lstm_embed = lstm_model(interpolated, lengths)
        fusion_input = torch.cat([tabular_tensor, lstm_embed, graph_tensor], dim=1)
        output_logit = fusion_model(fusion_input).sum()
        output_logit.backward()
        total_grad += interpolated.grad.detach()

    avg_grad = total_grad / float(IG_STEPS)
    attributions = (input_tensor - baseline) * avg_grad
    attribution_np = attributions.squeeze(0).detach().cpu().numpy()

    timestep_importance = np.abs(attribution_np).sum(axis=1)
    feature_importance = np.abs(attribution_np).sum(axis=0)
    return timestep_importance, feature_importance


def _top_sequence_features(feature_importance: np.ndarray) -> list[dict[str, Any]]:
    names = (
        SEQUENCE_FEATURE_NAMES
        if len(SEQUENCE_FEATURE_NAMES) == len(feature_importance)
        else [f"sequence_feature_{i}" for i in range(len(feature_importance))]
    )
    top_indices = np.argsort(feature_importance)[::-1][:TOP_K]
    return [
        {"feature": str(names[idx]), "impact": float(feature_importance[idx])}
        for idx in top_indices
    ]


def _graph_influence_score(graph_embedding: np.ndarray) -> float:
    model = _require_initialized("fusion_model", _FUSION_MODEL)
    first_layer = model.net[0]
    if not isinstance(first_layer, torch.nn.Linear):
        return float(np.linalg.norm(graph_embedding))
    weights = first_layer.weight.detach().cpu().numpy()
    graph_weights = np.mean(np.abs(weights[:, 33:65]), axis=0)
    contribution = np.abs(graph_embedding.astype(np.float32)) * graph_weights.astype(np.float32)
    return float(np.linalg.norm(contribution, ord=2))


def explain_applicant(loan_id: str) -> dict[str, Any]:
    applicant_row = _find_applicant_row(loan_id=loan_id)
    tabular_row = applicant_row.drop(columns=["Loan_Status"], errors="ignore")
    tabular_pipeline = _require_initialized("tabular_pipeline", _TABULAR_PIPELINE)

    tabular_logit = float(compute_tabular_logits(tabular_pipeline, tabular_row)[0, 0])
    sequence = _load_sequence_for_id(loan_id=str(loan_id))
    lstm_embedding = _compute_lstm_embedding(sequence=sequence)
    graph_embedding = _load_graph_embedding_for_id(loan_id=str(loan_id))

    fusion_input = np.concatenate(
        [
            np.array([tabular_logit], dtype=np.float32),
            lstm_embedding.astype(np.float32),
            graph_embedding.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if fusion_input.shape[0] != 65:
        raise ValueError(f"Unexpected fusion input dimension: {fusion_input.shape[0]}")

    fusion_model = _require_initialized("fusion_model", _FUSION_MODEL)
    with torch.no_grad():
        fusion_tensor = torch.tensor(fusion_input[None, :], dtype=torch.float32, device=_DEVICE)
        approval_logit = fusion_model(fusion_tensor).item()
    approval_prob = float(1.0 / (1.0 + np.exp(-approval_logit)))
    confidence = float(abs(approval_prob - 0.5) * 2.0)
    decision = "Approved" if approval_prob >= 0.5 else "Rejected"

    tabular_explanations = _compute_tabular_shap(tabular_row=tabular_row)
    _, feature_importance = _ig_sequence_attributions(
        sequence=sequence,
        tabular_logit=tabular_logit,
        graph_embedding=graph_embedding,
    )
    sequence_explanations = _top_sequence_features(feature_importance)
    graph_influence_score = _graph_influence_score(graph_embedding)

    return {
        "approval_probability": approval_prob,
        "decision": decision,
        "confidence": confidence,
        "tabular_explanations": tabular_explanations,
        "sequence_explanations": sequence_explanations,
        "graph_influence_score": graph_influence_score,
    }


_initialize_caches()
