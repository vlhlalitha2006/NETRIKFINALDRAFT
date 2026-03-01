from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.dependencies import get_current_user
from src.serving.services.inference_service import InferenceServiceError


router = APIRouter(tags=["scoring"])


class ScoreRequest(BaseModel):
    loan_id: str = Field(min_length=1)


class ScoreResponse(BaseModel):
    approval_probability: float
    decision: str
    confidence: float


@router.post("/score", response_model=ScoreResponse)
async def score(
    payload: ScoreRequest,
    request: Request,
    current_user: dict[str, str] = Depends(get_current_user),
) -> ScoreResponse:
    service = request.app.state.inference_service
    executor = request.app.state.score_executor
    loop = asyncio.get_running_loop()
    try:
        _ = current_user
        result = await loop.run_in_executor(executor, service.score_applicant, payload.loan_id)
        return ScoreResponse(**result)
    except InferenceServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
