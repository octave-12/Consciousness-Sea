from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Header, HTTPException

from consciousness_sea.infrastructure.config import API_AUTH_ENABLED, API_KEY

log = logging.getLogger(__name__)

_CONSCIOUSNESS_SEA_API_KEY = os.environ.get("CONSCIOUSNESS_SEA_API_KEY", "") or API_KEY
_AUTH_ENABLED = API_AUTH_ENABLED or bool(_CONSCIOUSNESS_SEA_API_KEY)
_JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", "24"))


class APIKeyAuth:
    _api_key: str
    _enabled: bool

    def __init__(self, api_key: str = _CONSCIOUSNESS_SEA_API_KEY, enabled: bool = _AUTH_ENABLED) -> None:
        self._api_key = api_key
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def validate(self, api_key: str | None) -> bool:
        if not self._enabled:
            return True
        if api_key is None:
            return False
        return hmac.compare_digest(api_key, self._api_key)


_auth = APIKeyAuth()


def create_access_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=_JWT_EXPIRATION_HOURS))
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if not _auth.enabled:
        return "anonymous"
    if x_api_key and _auth.validate(x_api_key):
        return f"apikey:{x_api_key[:8]}"
    raise HTTPException(
        status_code=401,
        detail={"error": "unauthorized", "message": "Invalid or missing API key"},
    )


async def verify_jwt(authorization: str | None = Header(default=None)) -> str:
    if not _auth.enabled:
        return "anonymous"
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Missing Authorization header"},
        )
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Invalid Authorization header format"},
        )
    payload = decode_access_token(parts[1])
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Invalid or expired token"},
        )
    return payload.get("sub", "unknown")
