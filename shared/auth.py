"""Static API-key authentication for the pipeline's public HTTP endpoints.

Each service reads API_KEY from its own environment (see docker-compose.yml)
and checks it independently — there's no shared auth service. If API_KEY is
unset, the dependency is a no-op, so local dev / unit tests without a
configured key keep working; the key becomes mandatory the moment it's set.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

API_KEY = os.environ.get("API_KEY")


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: require `Authorization: Bearer <API_KEY>`.

    Raises:
        HTTPException: 401 if API_KEY is configured and the request's bearer
            token is missing or doesn't match.
    """
    if not API_KEY:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
