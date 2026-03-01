from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.dependencies import require_admin
from src.serving.services.inference_service import InferenceServiceError


router = APIRouter(tags=["explanations"])


class ExplainRequest(BaseModel):
    loan_id: str = Field(min_length=1)


@router.post("/explain")
async def explain(
    payload: ExplainRequest,
    request: Request,
    current_user: dict[str, str] = Depends(require_admin),
) -> dict[str, Any]:
    service = request.app.state.inference_service
    executor = request.app.state.explain_executor
    loop = asyncio.get_running_loop()
    try:
        _ = current_user
        result = await loop.run_in_executor(executor, service.explain_applicant, payload.loan_id)
        return result
    except InferenceServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
