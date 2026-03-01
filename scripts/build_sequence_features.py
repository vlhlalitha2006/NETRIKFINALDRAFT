from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_TRAIN_CSV = Path("data/raw/TRAIN.csv")
DEFAULT_TEST_CSV = Path("data/raw/TEST.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
SEQUENCE_STAGES = 5


def _to_numeric_dependents(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("+", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _impute_numeric(array_2d: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(array_2d)


def _zscore(array_2d: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(array_2d)


def build_financial_progression_sequence(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Dependents",
    ]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns for sequence construction: {missing}")

    node_ids = (
        dataframe["Loan_ID"].astype(str).to_numpy()
        if "Loan_ID" in dataframe.columns
        else dataframe.index.astype(str).to_numpy()
    )

    app_income = pd.to_numeric(dataframe["ApplicantIncome"], errors="coerce").to_numpy()
    co_income = pd.to_numeric(dataframe["CoapplicantIncome"], errors="coerce").to_numpy()
    loan_amount = pd.to_numeric(dataframe["LoanAmount"], errors="coerce").to_numpy()
    loan_term = pd.to_numeric(dataframe["Loan_Amount_Term"], errors="coerce").to_numpy()
    credit_history = pd.to_numeric(dataframe["Credit_History"], errors="coerce").to_numpy()
    dependents = _to_numeric_dependents(dataframe["Dependents"]).to_numpy()

    total_income = app_income + co_income
    loan_to_income = loan_amount / (total_income + 1.0)

    base_numeric = np.column_stack(
        [
            app_income,
            co_income,
            total_income,
            loan_amount,
            loan_term,
            credit_history,
            dependents,
            loan_to_income,
        ]
    )
    base_numeric = _impute_numeric(base_numeric)

    # Stage 1: Raw normalized financial state from static tabular columns.
    stage1 = _zscore(base_numeric)

    app_income_i = base_numeric[:, 0]
    co_income_i = base_numeric[:, 1]
    total_income_i = base_numeric[:, 2]
    loan_amount_i = base_numeric[:, 3]
    loan_term_i = base_numeric[:, 4]
    credit_history_i = np.clip(base_numeric[:, 5], 0.0, 1.0)
    dependents_i = np.maximum(base_numeric[:, 6], 0.0)
    dti = loan_amount_i / (total_income_i + 1.0)
    dti_clip = np.clip(dti, 0.0, 2.0)

    # Stage 2: Debt-to-income adjusted state (income carrying capacity after debt burden).
    income_capacity = np.clip(1.0 - 0.5 * dti_clip, 0.1, 1.0)
    stage2_raw = np.column_stack(
        [
            app_income_i * income_capacity,
            co_income_i * income_capacity,
            total_income_i * income_capacity,
            loan_amount_i,
            loan_term_i,
            credit_history_i,
            dependents_i,
            dti_clip,
        ]
    )
    stage2 = _zscore(stage2_raw)

    # Stage 3: Credit-weighted state (credit score influences effective financial strength).
    credit_weight = 0.5 + 0.5 * credit_history_i
    stage3_raw = np.column_stack(
        [
            stage2_raw[:, 0] * credit_weight,
            stage2_raw[:, 1] * credit_weight,
            stage2_raw[:, 2] * credit_weight,
            stage2_raw[:, 3] * (2.0 - credit_weight),
            stage2_raw[:, 4],
            credit_history_i,
            dependents_i,
            stage2_raw[:, 7] * (2.0 - credit_weight),
        ]
    )
    stage3 = _zscore(stage3_raw)

    # Stage 4: Risk interaction features from burden, support, and credit pressure interactions.
    emi_proxy = loan_amount_i / (loan_term_i + 1.0)
    burden_ratio = emi_proxy / (total_income_i + 1.0)
    co_support_ratio = co_income_i / (total_income_i + 1.0)
    dependent_pressure = dependents_i / (dependents_i + 1.0)
    credit_penalty = (1.0 - credit_history_i) * burden_ratio
    stability_proxy = (1.0 - np.clip(burden_ratio, 0.0, 1.0)) * (0.5 + 0.5 * credit_history_i)
    dti_credit_interaction = dti_clip * (1.0 - 0.5 * credit_history_i)
    loan_to_primary_income = loan_amount_i / (app_income_i + 1.0)

    stage4_raw = np.column_stack(
        [
            emi_proxy,
            burden_ratio,
            co_support_ratio,
            dependent_pressure,
            credit_penalty,
            stability_proxy,
            dti_credit_interaction,
            loan_to_primary_income,
        ]
    )
    stage4 = _zscore(stage4_raw)

    # Stage 5: Consolidated state blending credit-aware and interaction-aware representations.
    stage5 = (0.6 * stage3) + (0.4 * stage4)

    sequence = np.stack([stage1, stage2, stage3, stage4, stage5], axis=1).astype(np.float32)
    if sequence.shape[1] != SEQUENCE_STAGES:
        raise ValueError("Unexpected number of sequence stages generated.")
    return sequence, node_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic pseudo-temporal sequence features from TRAIN+TEST."
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    sequence_features, sequence_ids = build_financial_progression_sequence(combined_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_features_path = args.output_dir / "sequence_features.npy"
    sequence_ids_path = args.output_dir / "sequence_ids.npy"
    np.save(sequence_features_path, sequence_features)
    np.save(sequence_ids_path, sequence_ids)

    print(f"Saved sequence features: {sequence_features_path}")
    print(f"Saved sequence ids: {sequence_ids_path}")
    print(f"total_ids: {len(sequence_ids)}")
    print(f"train_count: {len(train_df)}")
    print(f"test_count: {len(test_df)}")
    print(f"sequence_shape: {sequence_features.shape}")


if __name__ == "__main__":
    main()
