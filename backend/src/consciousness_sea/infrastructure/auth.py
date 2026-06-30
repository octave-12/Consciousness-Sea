from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException

from consciousness_sea.infrastructure.config import API_AUTH_ENABLED, API_KEY

log = logging.getLogger(__name__)

_CONSCIOUSNESS_SEA_API_KEY = os.environ.get("CONSCIOUSNESS_SEA_API_KEY", "") or API_KEY
_AUTH_ENABLED = API_AUTH_ENABLED or bool(_CONSCIOUSNESS_SEA_API_KEY)


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


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not _auth.enabled:
        return
    if not _auth.validate(x_api_key):
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Invalid or missing API key"},
        )
