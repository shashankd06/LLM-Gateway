import json
from collections.abc import Callable

import httpx
import pytest

from app.providers.base import GenerationParams, ProviderAdapter, ProviderError, ProviderTimeout
from app.providers.ollama import OllamaAdapter
from app.providers.openai_compat import OpenAICompatibleAdapter

AdapterFactory = Callable[[httpx.AsyncClient], ProviderAdapter]


def ollama(client: httpx.AsyncClient) -> ProviderAdapter:
    return OllamaAdapter(client, "http://provider/")


def compatible(client: httpx.AsyncClient) -> ProviderAdapter:
    return OpenAICompatibleAdapter(client, "http://provider/v1/", "test-only-key")


OLLAMA_RESPONSE = {"response": "Paris", "done": True, "prompt_eval_count": 12, "eval_count": 3}
COMPAT_RESPONSE = {
    "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


@pytest.mark.parametrize(
    "factory,reply", [(ollama, OLLAMA_RESPONSE), (compatible, COMPAT_RESPONSE)]
)
async def test_generate_contract(factory: AdapterFactory, reply: dict[str, object]) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await factory(client).generate(
            "Question", "model", GenerationParams(max_tokens=10, seed=7, stop="END", json_mode=True)
        )
    assert result.text == "Paris" and result.latency_ms >= 0
    assert (result.input_tokens, result.output_tokens, result.finish_reason) == (12, 3, "stop")
    payload = json.loads(requests[0].content)
    assert payload["model"] == "model" and payload["stream"] is False
    if factory is ollama:
        assert requests[0].url.path == "/api/generate"
        assert payload["prompt"] == "Question"
        assert payload["format"] == "json"
        assert payload["options"] == {
            "temperature": 0.7,
            "top_p": 1.0,
            "num_predict": 10,
            "seed": 7,
            "stop": "END",
        }
        assert "Authorization" not in requests[0].headers
    else:
        assert requests[0].url.path == "/v1/chat/completions"
        assert requests[0].headers["Authorization"] == "Bearer test-only-key"
        assert payload["messages"] == [{"role": "user", "content": "Question"}]
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["max_tokens"] == 10 and "json_mode" not in payload


@pytest.mark.parametrize("factory", [ollama, compatible])
@pytest.mark.parametrize(
    "failure", ["timeout", "connection", "http", "json", "shape", "negative", "string_tokens"]
)
async def test_provider_errors(factory: AdapterFactory, failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("secret", request=request)
        if failure == "connection":
            raise httpx.ConnectError("secret", request=request)
        if failure == "http":
            return httpx.Response(500, text="secret")
        if failure == "json":
            return httpx.Response(200, text="invalid-json-secret")
        if failure == "shape":
            return httpx.Response(200, json={})
        count = -1 if failure == "negative" else "12"
        body = (
            {**OLLAMA_RESPONSE, "prompt_eval_count": count}
            if factory is ollama
            else {**COMPAT_RESPONSE, "usage": {"prompt_tokens": count, "completion_tokens": 3}}
        )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderTimeout if failure == "timeout" else ProviderError) as exc:
            await factory(client).generate("Question", "model", GenerationParams())
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("factory", [ollama, compatible])
@pytest.mark.parametrize("state", ["healthy", "http", "timeout", "json", "shape"])
async def test_health(factory: AdapterFactory, state: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if state == "timeout":
            raise httpx.ConnectTimeout("unavailable", request=request)
        if state == "http":
            return httpx.Response(503)
        if state == "json":
            return httpx.Response(200, text="{")
        if state == "shape":
            return httpx.Response(200, json={})
        assert request.url.path == ("/api/tags" if factory is ollama else "/v1/models")
        return httpx.Response(200, json={"models" if factory is ollama else "data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await factory(client).health() is (state == "healthy")


async def test_ollama_incomplete_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**OLLAMA_RESPONSE, "done": False})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError):
            await ollama(client).generate("Question", "model", GenerationParams())


async def test_finish_reason_and_default_parameters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "format" not in body and "num_predict" not in body["options"]
        return httpx.Response(200, json={**OLLAMA_RESPONSE, "done_reason": "length"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ollama(client).generate("Question", "model", GenerationParams())
        assert result.finish_reason == "length"


async def test_compatible_without_auth_or_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        assert "response_format" not in json.loads(request.content)
        return httpx.Response(200, json={"choices": COMPAT_RESPONSE["choices"]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError):
            await OpenAICompatibleAdapter(client, "http://provider/v1").generate(
                "Question", "model", GenerationParams()
            )
