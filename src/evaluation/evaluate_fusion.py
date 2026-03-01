from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from models.fusion.infer_fusion import (
    build_fusion_input_matrix,
    load_fusion_model,
    load_tabular_pipeline,
    predict_fusion_logits,
)


def _encode_binary_target(y_series: pd.Series) -> np.ndarray:
    unique_values = sorted(pd.Series(y_series).dropna().unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(f"Expected binary target, found classes: {unique_values}")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return y_series.map(mapping).astype(int).to_numpy()


def _build_validation_fusion_inputs(
    validation_df: pd.DataFrame,
    target_column: str,
    id_column: str,
    tabular_pipeline_path: Path,
    sequence_features_path: Path,
    sequence_ids_path: Path,
    lstm_model_path: Path,
    graph_embeddings_path: Path,
    graph_index_path: Path,
    graph_node_ids_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = _encode_binary_target(validation_df[target_column])
    tabular_input = validation_df.drop(columns=[target_column])
    applicant_ids = (
        validation_df[id_column].astype(str).tolist()
        if id_column in validation_df.columns
        else validation_df.index.astype(str).tolist()
    )

    tabular_pipeline = load_tabular_pipeline(tabular_pipeline_path)
    x_fused = build_fusion_input_matrix(
        tabular_pipeline=tabular_pipeline,
        tabular_dataframe=tabular_input,
        sequence_features_path=sequence_features_path,
        sequence_ids_path=sequence_ids_path,
        lstm_model_path=lstm_model_path,
        graph_embeddings_path=graph_embeddings_path,
        graph_index_path=graph_index_path,
        applicant_ids=applicant_ids,
        node_ids_path=graph_node_ids_path if graph_node_ids_path.exists() else None,
        batch_size=32,
    )
    return x_fused, y_true


def evaluate_fusion_holdout(
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
    fusion_model_path: Path,
) -> dict[str, Any]:
    if not data_csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_csv_path}")

    dataframe = pd.read_csv(data_csv_path)
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    labels = _encode_binary_target(dataframe[target_column])
    _, validation_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    validation_df = validation_df.reset_index(drop=True)

    x_val, y_val = _build_validation_fusion_inputs(
        validation_df=validation_df,
        target_column=target_column,
        id_column=id_column,
        tabular_pipeline_path=tabular_pipeline_path,
        sequence_features_path=sequence_features_path,
        sequence_ids_path=sequence_ids_path,
        lstm_model_path=lstm_model_path,
        graph_embeddings_path=graph_embeddings_path,
        graph_index_path=graph_index_path,
        graph_node_ids_path=graph_node_ids_path,
    )

    fusion_model, device = load_fusion_model(
        model_path=fusion_model_path,
        input_dim=int(x_val.shape[1]),
    )
    logits = predict_fusion_logits(
        model=fusion_model,
        device=device,
        fusion_inputs=x_val,
        batch_size=32,
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)

    cm = confusion_matrix(y_val, predictions)
    report = {
        "validation_size": int(len(y_val)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_val, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "confusion_matrix": cm.tolist(),
    }
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("Hold-out validation is used to estimate generalization on unseen applicants.")
    print("ROC-AUC is important for loan approval systems because it measures ranking quality")
    print("across thresholds, not just one fixed approval cutoff.\n")
    print("=== Fusion Validation Summary ===")
    print(f"Validation Set Size: {report['validation_size']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"F1: {report['f1_score']:.4f}")
    print(f"ROC-AUC: {report['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print("                Pred 0    Pred 1")
    print(
        f"Actual 0      {report['confusion_matrix'][0][0]:>7}  {report['confusion_matrix'][0][1]:>7}"
    )
    print(
        f"Actual 1      {report['confusion_matrix'][1][0]:>7}  {report['confusion_matrix'][1][1]:>7}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained fusion model on hold-out split.")
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
        "--fusion-model-path",
        type=Path,
        default=Path("artifacts/fusion/fusion_mlp.pt"),
    )
    parser.add_argument(
        "--output-report-path",
        type=Path,
        default=Path("artifacts/evaluation_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_fusion_holdout(
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
        fusion_model_path=args.fusion_model_path,
    )
    _print_summary(report)

    args.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    print(f"\nSaved evaluation report to: {args.output_report_path}")


if __name__ == "__main__":
    main()
