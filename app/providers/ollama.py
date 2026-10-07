from time import perf_counter

import httpx

from app.providers.base import (
    GenerationParams,
    ProviderAdapter,
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)


class OllamaAdapter(ProviderAdapter):
    name = "ollama"

    def __init__(self, client: httpx.AsyncClient, base_url: str, health_timeout: float = 5) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.health_timeout = health_timeout

    async def generate(self, prompt: str, model: str, params: GenerationParams) -> ProviderResponse:
        options = params.model_dump(exclude_none=True, exclude={"json_mode", "max_tokens"})
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens
        payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
        if params.json_mode:
            payload["format"] = "json"
        started = perf_counter()
        try:
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            body = response.json()
            if body.get("done") is not True:
                raise ValueError("Expected a complete non-streaming response")
            return ProviderResponse(
                text=body["response"],
                input_tokens=body["prompt_eval_count"],
                output_tokens=body["eval_count"],
                latency_ms=(perf_counter() - started) * 1000,
                finish_reason="length" if body.get("done_reason") == "length" else "stop",
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama timed out") from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError("Ollama returned an invalid response or is unavailable") from exc

    async def health(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url}/api/tags", timeout=self.health_timeout
            )
            response.raise_for_status()
            return isinstance(response.json()["models"], list)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False
