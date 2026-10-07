import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384


class Embedder(Protocol):
    async def embed(self, prompt: str) -> list[float] | None:
        """Return None when truncation would make a semantic match unsafe."""
        ...


class SentenceTransformerEmbedder:
    """Lazy loading, tokenization, and inference all run on one dedicated worker.

    Serializing access avoids concurrent model initialization and overlapping inference.
    Exact cache hits never initialize or call the model.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = cache_dir
        self._model: SentenceTransformer | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embeddings")
        self._gate = asyncio.Semaphore(1)

    def _encode(self, prompt: str) -> list[float] | None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=self.cache_dir)
        tokens = self._model.tokenizer(prompt, truncation=False, verbose=False)["input_ids"]
        if len(tokens) > self._model.max_seq_length:
            # MiniLM truncates long inputs: two different suffixes could otherwise look identical.
            return None
        vector = self._model.encode(prompt, normalize_embeddings=True, show_progress_bar=False)
        return [float(value) for value in vector]

    async def embed(self, prompt: str) -> list[float] | None:
        async with self._gate:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._encode, prompt
            )

    async def close(self) -> None:
        # A cancelled coroutine cannot stop a running CPU operation; wait without blocking asyncio.
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
