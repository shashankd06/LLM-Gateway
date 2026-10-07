from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.providers.base import GenerationOptions, GenerationParams


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(GenerationOptions):
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: Literal[False] = False
    n: Literal[1] = 1
    cache: bool = Field(default=True, strict=True)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)

    def generation_params(self) -> GenerationParams:
        return GenerationParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=self.stop,
            json_mode=self.response_format.type == "json_object",
        )


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class CreateApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    requests_per_minute: int = Field(default=60, ge=1, le=10000, strict=True)
    daily_token_budget: int = Field(default=100000, ge=1, le=10**12, strict=True)


class CreatedApiKey(BaseModel):
    id: UUID
    name: str
    api_key: str
    requests_per_minute: int
    daily_token_budget: int
