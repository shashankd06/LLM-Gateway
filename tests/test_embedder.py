import asyncio
import sys
import threading
from types import ModuleType
from unittest.mock import Mock

import pytest

from app.cache.embedder import SentenceTransformerEmbedder
from tests.test_cache import vector


async def test_lazy_model_initialization_and_length_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("sentence_transformers")
    model = Mock()
    model.max_seq_length = 4
    model.tokenizer.side_effect = [{"input_ids": [1, 2]}, {"input_ids": [1, 2, 3, 4, 5]}]
    model.encode.return_value = vector()
    factory = Mock(return_value=model)
    module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    embedder = SentenceTransformerEmbedder("model-cache")
    try:
        factory.assert_not_called()
        assert await embedder.embed("short") == vector()
        assert await embedder.embed("too long") is None
        factory.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_folder="model-cache"
        )
        model.encode.assert_called_once_with(
            "short", normalize_embeddings=True, show_progress_bar=False
        )
        assert model.tokenizer.call_args.kwargs["truncation"] is False
    finally:
        await embedder.close()


async def test_cpu_work_does_not_block_event_loop_and_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, release = threading.Event(), threading.Event()
    worker_ids: list[int] = []

    def blocking_encode(prompt: str) -> list[float]:
        worker_ids.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=10)
        return vector()

    embedder = SentenceTransformerEmbedder()
    monkeypatch.setattr(embedder, "_encode", blocking_encode)
    tasks = [asyncio.create_task(embedder.embed("one")), asyncio.create_task(embedder.embed("two"))]
    try:
        assert await asyncio.to_thread(started.wait, 5)
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        async with asyncio.timeout(2):
            await heartbeat.wait()
        assert len(worker_ids) == 1 and worker_ids[0] != threading.get_ident()
        release.set()
        assert await asyncio.gather(*tasks) == [vector(), vector()]
        assert len(set(worker_ids)) == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await embedder.close()
