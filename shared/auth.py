"""Static API-key authentication for the pipeline's public HTTP endpoints.

Each service reads API_KEY from its own environment (see docker-compose.yml)
and checks it independently — there's no shared auth service. If API_KEY is
unset, the dependency is a no-op, so local dev / unit tests without a
configured key keep working; the key becomes mandatory the moment it's set.

Uses fastapi.security.HTTPBearer (not a plain Header() param) so this
registers as a real OpenAPI security scheme — Swagger UI then shows an
"Authorize" lock button that applies the token to every request, instead of
a per-endpoint text field that's easy to miss and leaves the token out of
Swagger's generated curl entirely.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_KEY = os.environ.get("API_KEY")

_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """FastAPI dependency: require `Authorization: Bearer <API_KEY>`.

    Raises:
        HTTPException: 401 if API_KEY is configured and the request's bearer
            token is missing or doesn't match.
    """
    if not API_KEY:
        return

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(credentials.credentials, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
