from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.explainability import explainer_service as es
from src.inference.multimodal_predict import multimodal_predict


@dataclass
class InferenceServiceError(Exception):
    status_code: int
    detail: str


class InferenceService:
    def __init__(self) -> None:
        # Caches are initialized in explainer_service at import/startup.
        self._device = es._DEVICE

    @property
    def device_name(self) -> str:
        return str(self._device)

    @property
    def applicant_count(self) -> int:
        dataframe = es._require_initialized("dataframe", es._DATAFRAME)
        return int(len(dataframe))

    @property
    def artifacts_cached(self) -> bool:
        return (
            es._TABULAR_PIPELINE is not None
            and es._LSTM_MODEL is not None
            and es._GRAPH_EMBEDDINGS is not None
            and es._FUSION_MODEL is not None
            and es._SHAP_EXPLAINER is not None
        )

    def score_applicant(self, loan_id: str) -> dict[str, Any]:
        try:
            applicant_row = es._find_applicant_row(loan_id=loan_id)
            tabular_row = applicant_row.drop(columns=["Loan_Status"], errors="ignore")
            prediction = multimodal_predict(df_row=tabular_row, loan_id=str(loan_id), debug=False)
            approval_probability = float(prediction["approval_probability"])
            confidence = float(abs(approval_probability - 0.5) * 2.0)
            return {
                "approval_probability": approval_probability,
                "decision": str(prediction["prediction"]),
                "confidence": confidence,
            }
        except KeyError as exc:
            raise InferenceServiceError(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise InferenceServiceError(
                status_code=500,
                detail=f"Unexpected scoring error: {exc}",
            ) from exc

    def explain_applicant(self, loan_id: str) -> dict[str, Any]:
        try:
            return es.explain_applicant(loan_id=loan_id)
        except KeyError as exc:
            raise InferenceServiceError(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise InferenceServiceError(
                status_code=500,
                detail=f"Unexpected explanation error: {exc}",
            ) from exc
