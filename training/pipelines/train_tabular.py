from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from models.tabular.xgboost_model import build_tabular_pipeline
from training.data.loaders import load_training_dataframe, split_features_target


def detect_feature_types(
    x_features,
) -> tuple[list[str], list[str]]:
    numeric_features = x_features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = x_features.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def encode_binary_target(y_target: pd.Series) -> pd.Series:
    unique_values = sorted(pd.Series(y_target).dropna().unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(
            f"Expected binary target for F1 optimization, found classes: {unique_values}"
        )
    target_mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return y_target.map(target_mapping).astype(int)


def run_cross_validated_training(
    data_path: Path,
    target_column: str,
    output_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    dataframe = load_training_dataframe(data_path)
    x_features, y_target = split_features_target(dataframe, target_column)
    y_encoded = encode_binary_target(y_target)
    numeric_features, categorical_features = detect_feature_types(x_features)

    pipeline = build_tabular_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        pipeline,
        x_features,
        y_encoded,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )

    pipeline.fit(x_features, y_encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    return {
        "f1_mean": float(np.mean(cv_scores)),
        "f1_std": float(np.std(cv_scores)),
        "f1_min": float(np.min(cv_scores)),
        "f1_max": float(np.max(cv_scores)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tabular XGBoost pipeline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Dataset 1/TRAIN.csv"),
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Loan_Status",
        help="Name of target column in dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/tabular/sklearn_xgb_pipeline.joblib"),
        help="Path to save trained sklearn pipeline.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of StratifiedKFold splits.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = run_cross_validated_training(
        data_path=args.data_path,
        target_column=args.target_column,
        output_path=args.output_path,
        n_splits=args.n_splits,
    )
    print(
        "Cross-validated F1 -> "
        f"mean: {metrics['f1_mean']:.4f}, std: {metrics['f1_std']:.4f}, "
        f"min: {metrics['f1_min']:.4f}, max: {metrics['f1_max']:.4f}"
    )
    print(f"Saved trained pipeline to: {args.output_path}")


if __name__ == "__main__":
    main()

