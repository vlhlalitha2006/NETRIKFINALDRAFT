from __future__ import annotations

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import time

from fastapi import FastAPI, Request

from src.db.base import Base
from src.db import models as _db_models  # noqa: F401
from src.db.repositories import AuditLogRepository
from src.db.session import init_db
from src.db.session import session_scope
from src.explainability import explainer_service as es
from src.serving.dependencies import parse_actor_from_authorization_header
from src.serving.routers.auth import router as auth_router
from src.serving.routers.explain import router as explain_router
from src.serving.routers.health import router as health_router
from src.serving.routers.score import router as score_router
from src.serving.services.inference_service import InferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dedicated pools isolate lightweight scoring from heavier explanations.
    score_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="risk-score")
    explain_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="risk-explain")

    # Ensure explainability/model caches are initialized once at startup.
    es._initialize_caches()
    try:
        init_db(Base.metadata)
        db_ready = True
    except Exception:
        db_ready = False
    inference_service = InferenceService()

    app.state.score_executor = score_executor
    app.state.explain_executor = explain_executor
    app.state.inference_service = inference_service

    print(f"[startup] device: {inference_service.device_name}")
    print(f"[startup] applicants_loaded: {inference_service.applicant_count}")
    print(f"[startup] artifacts_cached: {inference_service.artifacts_cached}")
    print(f"[startup] db_audit_logging_ready: {db_ready}")
    print("[startup] routes_registered: /health, /login, /score, /explain")
    print("[startup] dependencies_attached: /score->get_current_user, /explain->require_admin")
    try:
        yield
    finally:
        score_executor.shutdown(wait=True)
        explain_executor.shutdown(wait=True)


app = FastAPI(
    title="Financial Risk Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Keep existing exception behavior while still auditing 500 path.
        response = None
        raise
    finally:
        ended = time.perf_counter()
        latency_ms = (ended - started) * 1000.0
        normalized_endpoint = request.url.path.lstrip("/").split("/")[0] or "root"

        actor_username = None
        actor_role = None
        try:
            actor_username, actor_role = parse_actor_from_authorization_header(
                request.headers.get("authorization")
            )
        except Exception:
            actor_username, actor_role = None, None

        loan_id_value = "unknown"
        if normalized_endpoint in {"score", "explain"}:
            try:
                raw = await request.body()
                if raw:
                    payload = json.loads(raw.decode("utf-8"))
                    if isinstance(payload, dict) and payload.get("loan_id"):
                        loan_id_value = str(payload["loan_id"])
            except Exception:
                loan_id_value = "unknown"

        status_code = response.status_code if response is not None else 500
        error_message = None if status_code < 400 else f"http_{status_code}"

        def _write_audit() -> None:
            try:
                with session_scope() as session:
                    AuditLogRepository().create_audit_log(
                        session,
                        endpoint=normalized_endpoint,
                        loan_id=loan_id_value,
                        actor_username=actor_username,
                        actor_role=actor_role,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        error_message=error_message,
                    )
            except Exception:
                return

        # Offload DB write so middleware does not block event loop.
        asyncio.create_task(asyncio.to_thread(_write_audit))

    return response


app.include_router(health_router)
app.include_router(auth_router)
app.include_router(score_router)
app.include_router(explain_router)
