import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import Select, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.cache.embedder import DIMENSIONS, MODEL_NAME, Embedder
from app.config import Settings
from app.db.models import CacheEntry
from app.providers.base import GenerationParams, ProviderResponse
from app.ratelimit.token_bucket import utc


def fingerprint(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def valid_vector(vector: Sequence[float]) -> bool:
    return (
        len(vector) == DIMENSIONS
        and all(math.isfinite(value) for value in vector)
        and any(value != 0 for value in vector)
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """SQLite test equivalent of pgvector cosine distance; reject unusable embeddings."""
    if not valid_vector(left) or not valid_vector(right):
        raise ValueError("Expected a finite, nonzero 384-dimensional embedding")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return max(-1.0, min(1.0, dot / norms))


@dataclass(frozen=True)
class CacheQuery:
    api_key_id: UUID
    namespace: str
    model: str
    params_hash: str
    prompt_hash: str
    context_hash: str
    semantic_allowed: bool
    provider: str


@dataclass(frozen=True)
class CacheLookup:
    response: ProviderResponse | None
    embedding: list[float] | None
    model: str | None = None
    provider: str | None = None
    tier: str | None = None


class SemanticCache:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.sessions = sessions
        self.embedder = embedder
        self.settings = settings
        self.clock = clock or (lambda: datetime.now(UTC))

    def query(
        self,
        key_id: UUID,
        provider: str,
        model: str,
        params: GenerationParams,
        messages: list[dict[str, str]],
        policy: str | None = None,
    ) -> CacheQuery:
        endpoint = (
            self.settings.openai_compat_base_url
            if provider == "openai_compat"
            else self.settings.ollama_base_url
        )
        return CacheQuery(
            api_key_id=key_id,
            namespace=fingerprint(["cache-v2", MODEL_NAME, provider, endpoint, policy]),
            model=model,
            params_hash=fingerprint(params.model_dump(mode="json")),
            prompt_hash=fingerprint(messages),
            # Only the final user question may vary; prior turns and instructions must match.
            context_hash=fingerprint(messages[:-1]),
            semantic_allowed=messages[-1]["role"] == "user",
            provider=provider,
        )

    def candidates(self, query: CacheQuery) -> Select[tuple[CacheEntry]]:
        statement = select(CacheEntry).where(
            CacheEntry.namespace == query.namespace,
            CacheEntry.model == query.model,
            CacheEntry.params_hash == query.params_hash,
            CacheEntry.context_hash == query.context_hash,
            CacheEntry.expires_at > utc(self.clock()),
        )
        if not self.settings.cache_share_across_keys:
            statement = statement.where(CacheEntry.api_key_id == query.api_key_id)
        return statement

    @staticmethod
    def response(entry: CacheEntry) -> ProviderResponse:
        # Usage tracks new inference, not historical work that produced the cached answer.
        return ProviderResponse(
            text=entry.response_text,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            finish_reason=entry.finish_reason,
        )

    async def lookup(self, query: CacheQuery, prompt: str) -> CacheLookup:
        return await self.lookup_many([query], prompt)

    def hit(self, entry: CacheEntry, vector: list[float] | None) -> CacheLookup:
        return CacheLookup(self.response(entry), vector, entry.model, entry.provider, entry.tier)

    async def lookup_many(self, queries: list[CacheQuery], prompt: str) -> CacheLookup:
        # All exact candidates are checked before any embedding work, even across fallback tiers.
        for query in queries:
            exact = await self.exact(query)
            if exact is not None:
                return self.hit(exact, None)
        if not queries[0].semantic_allowed:
            return CacheLookup(None, None)
        vector = await self.embedder.embed(prompt)
        if vector is None:
            return CacheLookup(None, None)
        if not valid_vector(vector):
            raise ValueError("Embedder returned an invalid vector")
        for query in queries:
            entry = await self.nearest(query, vector)
            if entry is not None:
                return self.hit(entry, vector)
        return CacheLookup(None, vector)

    async def exact(self, query: CacheQuery) -> CacheEntry | None:
        async with self.sessions() as session:
            exact = await session.scalar(
                self.candidates(query)
                .where(CacheEntry.prompt_hash == query.prompt_hash)
                .order_by(CacheEntry.created_at.desc(), CacheEntry.id)
                .limit(1)
            )
        return exact

    async def nearest(self, query: CacheQuery, vector: list[float]) -> CacheEntry | None:
        async with self.sessions() as session:
            statement = self.candidates(query).where(CacheEntry.embedding.is_not(None))
            if session.get_bind().dialect.name == "postgresql":
                distance = CacheEntry.embedding.cosine_distance(vector)
                entry = await session.scalar(
                    statement.where(distance <= 1 - self.settings.cache_similarity_threshold)
                    .order_by(distance, CacheEntry.created_at.desc(), CacheEntry.id)
                    .limit(1)
                )
            else:
                # SQLite exists only for key-free CI; production performs the search in PostgreSQL.
                entries = (await session.scalars(statement)).all()
                entry = max(
                    entries,
                    key=lambda item: cosine_similarity(vector, item.embedding),
                    default=None,
                )
                if entry is not None and cosine_similarity(vector, entry.embedding) < (
                    self.settings.cache_similarity_threshold
                ):
                    entry = None
        return entry

    async def store(
        self,
        query: CacheQuery,
        result: ProviderResponse,
        embedding: list[float] | None,
        tier: str = "direct",
    ) -> None:
        if not result.text.strip():
            return
        now = utc(self.clock())
        values = {
            "api_key_id": query.api_key_id,
            "namespace": query.namespace,
            "model": query.model,
            "params_hash": query.params_hash,
            "prompt_hash": query.prompt_hash,
            "context_hash": query.context_hash,
            "embedding": embedding,
            "response_text": result.text,
            "finish_reason": result.finish_reason,
            "provider": query.provider,
            "tier": tier,
            "created_at": now,
            "expires_at": now + timedelta(seconds=self.settings.cache_ttl_seconds),
        }
        async with self.sessions() as session, session.begin():
            # TTL does not slide on hits. Lazy cleanup bounds expired rows when traffic continues.
            await session.execute(delete(CacheEntry).where(CacheEntry.expires_at <= now))
            insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
            statement = insert(CacheEntry).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=["api_key_id", "namespace", "model", "params_hash", "prompt_hash"],
                set_=values,
            )
            await session.execute(statement)
