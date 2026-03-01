from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.fusion.infer_fusion import (
    build_fusion_input_matrix,
    load_fusion_model,
    load_tabular_pipeline,
    predict_fusion_logits,
)


GROUP_COLUMNS = ["Gender", "Married", "Dependents", "Property_Area"]


def _encode_binary_target(y_series: pd.Series) -> np.ndarray:
    unique_values = sorted(pd.Series(y_series).dropna().unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(f"Expected binary target; found classes: {unique_values}")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return y_series.map(mapping).astype(int).to_numpy()


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true_bin = y_true.astype(int)
    y_pred_bin = y_pred.astype(int)
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    counts = _confusion_counts(y_true=y_true, y_pred=y_pred)
    approval_rate = _safe_rate(int(np.sum(y_pred == 1)), int(len(y_pred)))
    rejection_rate = _safe_rate(int(np.sum(y_pred == 0)), int(len(y_pred)))
    fpr = _safe_rate(counts["fp"], counts["fp"] + counts["tn"])
    fnr = _safe_rate(counts["fn"], counts["fn"] + counts["tp"])
    return {
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def _build_fusion_inputs(
    dataframe: pd.DataFrame,
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
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    if id_column not in dataframe.columns:
        raise ValueError(f"ID column '{id_column}' not found in dataset.")

    y_true = _encode_binary_target(dataframe[target_column])
    tabular_input = dataframe.drop(columns=[target_column])
    applicant_ids = dataframe[id_column].astype(str).tolist()

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
        batch_size=32,
    )
    return fusion_inputs, y_true


def run_fairness_analysis(
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
    alert_threshold: float = 0.15,
) -> dict[str, Any]:
    if not data_csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_csv_path}")
    dataframe = pd.read_csv(data_csv_path)

    fusion_inputs, y_true = _build_fusion_inputs(
        dataframe=dataframe,
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
        input_dim=int(fusion_inputs.shape[1]),
    )
    logits = predict_fusion_logits(
        model=fusion_model,
        device=device,
        fusion_inputs=fusion_inputs,
        batch_size=32,
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probabilities >= 0.5).astype(int)

    global_metrics = _compute_group_metrics(y_true=y_true, y_pred=y_pred)

    group_results: dict[str, list[dict[str, Any]]] = {}
    alerts: list[dict[str, Any]] = []

    for group_col in GROUP_COLUMNS:
        if group_col not in dataframe.columns:
            raise ValueError(f"Required fairness group column '{group_col}' is missing.")

        rows_for_group: list[dict[str, Any]] = []
        grouped_values = dataframe[group_col].fillna("Unknown").astype(str)
        unique_values = sorted(grouped_values.unique().tolist())

        for group_value in unique_values:
            mask = grouped_values == group_value
            indices = np.where(mask.to_numpy())[0]
            if len(indices) == 0:
                continue

            metrics = _compute_group_metrics(y_true=y_true[indices], y_pred=y_pred[indices])
            row = {
                "group": group_col,
                "value": group_value,
                "sample_size": int(len(indices)),
                **metrics,
            }
            rows_for_group.append(row)

            for metric_name in ("approval_rate", "false_positive_rate", "false_negative_rate"):
                metric_diff = abs(float(metrics[metric_name]) - float(global_metrics[metric_name]))
                if metric_diff > alert_threshold:
                    alerts.append(
                        {
                            "group": group_col,
                            "value": group_value,
                            "metric": metric_name,
                            "group_value_metric": float(metrics[metric_name]),
                            "global_metric": float(global_metrics[metric_name]),
                            "absolute_difference": float(metric_diff),
                            "threshold": float(alert_threshold),
                        }
                    )

        group_results[group_col] = rows_for_group

    return {
        "dataset": str(data_csv_path),
        "num_samples": int(len(dataframe)),
        "global_metrics": global_metrics,
        "group_metrics": group_results,
        "alerts": alerts,
        "alert_threshold": float(alert_threshold),
    }


def _print_table(report: dict[str, Any]) -> None:
    print("=== Fairness Comparison Table ===")
    print(f"{'Group':<35} {'Approval Rate':>14} {'FPR':>10} {'FNR':>10}")
    print("-" * 72)
    for group_col in GROUP_COLUMNS:
        for row in report["group_metrics"].get(group_col, []):
            group_name = f"{row['group']}={row['value']}"
            print(
                f"{group_name:<35} "
                f"{row['approval_rate']:>14.4f} "
                f"{row['false_positive_rate']:>10.4f} "
                f"{row['false_negative_rate']:>10.4f}"
            )

    print("\nGlobal averages:")
    global_metrics = report["global_metrics"]
    print(
        f"Approval Rate={global_metrics['approval_rate']:.4f}, "
        f"FPR={global_metrics['false_positive_rate']:.4f}, "
        f"FNR={global_metrics['false_negative_rate']:.4f}"
    )

    if report["alerts"]:
        print("\nAlerts (>15% absolute difference from global average):")
        for alert in report["alerts"]:
            print(
                f"- {alert['group']}={alert['value']} | {alert['metric']} "
                f"| group={alert['group_value_metric']:.4f} "
                f"| global={alert['global_metric']:.4f} "
                f"| diff={alert['absolute_difference']:.4f}"
            )
    else:
        print("\nNo fairness alerts above threshold.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fairness evaluation for trained fusion model.")
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
        default=Path("artifacts/fairness_report.json"),
    )
    parser.add_argument("--alert-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_fairness_analysis(
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
        alert_threshold=args.alert_threshold,
    )
    _print_table(report)

    args.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    print(f"\nSaved fairness report to: {args.output_report_path}")


if __name__ == "__main__":
    main()
