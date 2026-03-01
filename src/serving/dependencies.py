from __future__ import annotations

from datetime import datetime, timedelta, timezone
from os import getenv
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
JWT_SECRET = getenv("JWT_SECRET", "change-this-secret-in-production")

_security = HTTPBearer(auto_error=False)
_users = {
    "admin_user": {"password": "admin123", "role": "admin"},
    "applicant_user": {"password": "applicant123", "role": "applicant"},
}


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    user = _users.get(username)
    if user is None or user["password"] != password:
        return None
    return {"username": username, "role": user["role"]}


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        if "sub" not in payload or "role" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload.",
            )
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict[str, str]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    payload = _decode_token(credentials.credentials)
    return {"username": str(payload["sub"]), "role": str(payload["role"])}


async def require_admin(
    current_user: dict[str, str] = Depends(get_current_user),
) -> dict[str, str]:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


def parse_actor_from_authorization_header(
    authorization_header: str | None,
) -> tuple[str | None, str | None]:
    if not authorization_header:
        return None, None
    prefix = "Bearer "
    if not authorization_header.startswith(prefix):
        return None, None
    token = authorization_header[len(prefix) :].strip()
    if not token:
        return None, None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return str(payload.get("sub")), str(payload.get("role"))
    except Exception:
        return None, None
