from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from src.explainability import explainer_service as es
from src.inference.multimodal_predict import multimodal_predict


TEST_CSV_PATH = Path("data/raw/TEST.csv")
OUTPUT_CSV_PATH = Path("artifacts/test_predictions_with_explanations.csv")


def clean_feature_name(raw_name: str) -> str:
    explicit_map = {
        "num__Credit_History": "Credit History",
        "num__LoanAmount": "Loan Amount",
        "num__ApplicantIncome": "Applicant Income",
        "num__CoapplicantIncome": "Co-applicant Income",
        "num__Loan_Amount_Term": "Loan Amount Term",
        "coapplicant_income_state": "Co-applicant Income Pattern",
        "applicant_income_state": "Applicant Income Pattern",
        "total_income_state": "Total Income Pattern",
        "dependents_state": "Dependents Impact",
    }
    if raw_name in explicit_map:
        return explicit_map[raw_name]

    if raw_name.startswith("cat__Property_Area_"):
        suffix = raw_name.replace("cat__Property_Area_", "").replace("_", " ")
        return f"Property Area ({suffix.title()})"

    if raw_name.startswith("cat__"):
        label = raw_name.replace("cat__", "").replace("_", " ")
        return label.title()
    if raw_name.startswith("num__"):
        label = raw_name.replace("num__", "").replace("_", " ")
        return label.title()

    return raw_name.replace("_", " ").title()


def _build_explanation_text(
    decision: str,
    approval_probability: float,
    top_tabular_features: list[str],
) -> str:
    f1, f2, f3 = top_tabular_features[:3]
    if decision == "Rejected":
        return (
            f"Loan rejected because {f1}, {f2}, and {f3} reduced the approval likelihood. "
            "Behavioral sequence patterns also contributed to lower approval confidence. "
            f"Overall Approval Probability was {approval_probability:.4f}."
        )
    return (
        f"Loan approved because {f1}, {f2}, and {f3} positively influenced approval. "
        "Financial behavior indicators were stable. "
        f"Overall Approval Probability was {approval_probability:.4f}."
    )


def _ensure_test_dataset() -> pd.DataFrame:
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing TEST dataset at: {TEST_CSV_PATH}")
    dataframe = pd.read_csv(TEST_CSV_PATH)
    if "Loan_ID" not in dataframe.columns:
        raise ValueError("TEST.csv must contain Loan_ID column.")
    return dataframe


def _validate_artifact_alignment(test_df: pd.DataFrame) -> None:
    sequence_lookup = es._require_initialized("sequence_lookup", es._SEQUENCE_LOOKUP)
    graph_lookup = es._require_initialized("graph_lookup", es._GRAPH_LOOKUP)

    missing_sequence_ids: list[str] = []
    missing_graph_ids: list[str] = []
    for loan_id in test_df["Loan_ID"].astype(str).tolist():
        if loan_id not in sequence_lookup and str(loan_id) not in sequence_lookup:
            missing_sequence_ids.append(loan_id)
        if loan_id not in graph_lookup and str(loan_id) not in graph_lookup:
            missing_graph_ids.append(loan_id)

    if missing_sequence_ids:
        preview = ", ".join(missing_sequence_ids[:10])
        raise ValueError(
            "Some TEST Loan_ID values are missing in sequence artifacts. "
            f"Count={len(missing_sequence_ids)}. Examples: {preview}"
        )
    if missing_graph_ids:
        preview = ", ".join(missing_graph_ids[:10])
        raise ValueError(
            "Some TEST Loan_ID values are missing in graph embeddings. "
            f"Count={len(missing_graph_ids)}. Examples: {preview}"
        )


def _explain_single_applicant(
    tabular_row: pd.DataFrame,
    tabular_logit: float,
    sequence_raw: np.ndarray,
    graph_embedding: np.ndarray,
) -> tuple[list[str], list[str], float]:
    tabular_explanations = es._compute_tabular_shap(tabular_row=tabular_row)
    _, feature_importance = es._ig_sequence_attributions(
        sequence=sequence_raw,
        tabular_logit=tabular_logit,
        graph_embedding=graph_embedding,
    )
    sequence_explanations = es._top_sequence_features(feature_importance)
    graph_influence = es._graph_influence_score(graph_embedding)

    top_tabular = [item["feature"] for item in tabular_explanations[:3]]
    top_sequence = [item["feature"] for item in sequence_explanations[:3]]
    return top_tabular, top_sequence, float(graph_influence)


def run_batch_inference_with_explanations() -> Path:
    es._initialize_caches()
    test_df = _ensure_test_dataset()
    _validate_artifact_alignment(test_df)

    results: list[dict[str, Any]] = []

    for _, row in test_df.iterrows():
        loan_id = str(row["Loan_ID"])
        tabular_row = pd.DataFrame([row.to_dict()])
        prediction = multimodal_predict(df_row=tabular_row, loan_id=loan_id, debug=False)
        tabular_logit = float(prediction["tabular_logit"])
        sequence_raw = prediction["sequence_raw"]
        graph_embedding = prediction["graph_embedding"]
        approval_probability = float(prediction["approval_probability"])
        decision = str(prediction["prediction"])

        top_tabular_raw, top_sequence_raw, graph_influence = _explain_single_applicant(
            tabular_row=tabular_row,
            tabular_logit=tabular_logit,
            sequence_raw=sequence_raw,
            graph_embedding=graph_embedding,
        )
        top_tabular_clean = [clean_feature_name(name) for name in top_tabular_raw]
        top_sequence_clean = [clean_feature_name(name) for name in top_sequence_raw]
        explanation_text = _build_explanation_text(
            decision=decision,
            approval_probability=approval_probability,
            top_tabular_features=top_tabular_clean,
        )

        results.append(
            {
                "Loan_ID": loan_id,
                "Predicted_Loan_Status": decision,
                "Approval_Probability": approval_probability,
                "Top_Tabular_Features": ", ".join(top_tabular_clean),
                "Top_Sequence_Features": ", ".join(top_sequence_clean),
                "Graph_Influence_Score": graph_influence,
                "Explanation_Text": explanation_text,
            }
        )

    output_df = pd.DataFrame(results)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    approved_count = int((output_df["Predicted_Loan_Status"] == "Approved").sum())
    rejected_count = int((output_df["Predicted_Loan_Status"] == "Rejected").sum())
    print(f"Total applicants processed: {len(output_df)}")
    print(f"Number Approved: {approved_count}")
    print(f"Number Rejected: {rejected_count}")
    print(f"Output file path: {OUTPUT_CSV_PATH}")
    preview_columns = ["Loan_ID", "Predicted_Loan_Status", "Approval_Probability", "Explanation_Text"]
    print("\nFirst 5 rows:")
    print(output_df[preview_columns].head(5).to_string(index=False))
    return OUTPUT_CSV_PATH


if __name__ == "__main__":
    run_batch_inference_with_explanations()
