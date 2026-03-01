from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_training_dataframe(csv_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training data not found at: {dataset_path}")
    return pd.read_csv(dataset_path)


def split_features_target(
    dataframe: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    x_features = dataframe.drop(columns=[target_column])
    y_target = dataframe[target_column]
    return x_features, y_target

