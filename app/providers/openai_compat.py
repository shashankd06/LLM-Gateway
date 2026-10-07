from time import perf_counter

import httpx

from app.providers.base import (
    GenerationParams,
    ProviderAdapter,
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)


class OpenAICompatibleAdapter(ProviderAdapter):
    name = "openai_compat"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str | None = None,
        health_timeout: float = 5,
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.health_timeout = health_timeout

    async def generate(self, prompt: str, model: str, params: GenerationParams) -> ProviderResponse:
        payload = params.model_dump(exclude_none=True, exclude={"json_mode"})
        payload.update(model=model, messages=[{"role": "user", "content": prompt}], stream=False)
        if params.json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = perf_counter()
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions", headers=self.headers, json=payload
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            # Missing usage is an error: silently reporting zero would corrupt budget accounting.
            return ProviderResponse(
                text=choice["message"]["content"],
                input_tokens=body["usage"]["prompt_tokens"],
                output_tokens=body["usage"]["completion_tokens"],
                latency_ms=(perf_counter() - started) * 1000,
                finish_reason=choice["finish_reason"],
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("OpenAI-compatible provider timed out") from exc
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI-compatible provider returned an invalid response") from exc

    async def health(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url}/models", headers=self.headers, timeout=self.health_timeout
            )
            response.raise_for_status()
            return isinstance(response.json()["data"], list)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False
