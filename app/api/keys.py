from uuid import uuid4

from fastapi import APIRouter, Depends

from app.api.schemas import CreateApiKeyRequest, CreatedApiKey
from app.auth.keys import generate_key, hash_key, require_admin
from app.db.models import ApiKey
from app.deps import RateLimiterDep, SessionDep

router = APIRouter()


@router.post(
    "/admin/keys",
    response_model=CreatedApiKey,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
async def create_api_key(
    body: CreateApiKeyRequest, session: SessionDep, limiter: RateLimiterDep
) -> CreatedApiKey:
    raw_key = generate_key()
    async with session.begin():
        key = ApiKey(
            id=uuid4(),
            name=body.name,
            key_hash=hash_key(raw_key),
            active=True,
            requests_per_minute=body.requests_per_minute,
            daily_token_budget=body.daily_token_budget,
            bucket_tokens=float(body.requests_per_minute),
            bucket_updated_at=await limiter.now(session),
        )
        session.add(key)
    # There is intentionally no endpoint to retrieve the raw key after this response.
    return CreatedApiKey(
        id=key.id,
        name=key.name,
        api_key=raw_key,
        requests_per_minute=key.requests_per_minute,
        daily_token_budget=key.daily_token_budget,
    )
