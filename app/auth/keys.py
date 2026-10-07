import hashlib
import secrets
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.config import Settings
from app.db.models import ApiKey
from app.errors import GatewayError

bearer = HTTPBearer(
    auto_error=False,
    scheme_name="BearerKey",
    description="Admin key for key creation and metrics; client key for chat completions.",
)
CredentialsDep = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]


def hash_key(raw_key: str) -> str:
    # A fast hash is appropriate for random 256-bit keys, unlike human-selected passwords.
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    return "lgw_" + secrets.token_urlsafe(32)


async def require_api_key(request: Request, credentials: CredentialsDep) -> ApiKey:
    if credentials is None:
        raise GatewayError(401, "A valid bearer API key is required", "invalid_api_key")
    async with request.app.state.session_factory() as session:
        key = await session.scalar(
            select(ApiKey).where(
                ApiKey.key_hash == hash_key(credentials.credentials), ApiKey.active.is_(True)
            )
        )
    if key is None:
        raise GatewayError(401, "A valid bearer API key is required", "invalid_api_key")
    request.state.api_key_id = key.id
    return key


async def require_admin(request: Request, credentials: CredentialsDep) -> None:
    settings: Settings = request.app.state.settings
    configured = settings.admin_api_key
    if not configured:
        raise GatewayError(503, "Admin credentials are not configured", "admin_unavailable")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"), configured.get_secret_value().encode("utf-8")
    ):
        raise GatewayError(401, "A valid admin bearer key is required", "invalid_admin_key")
