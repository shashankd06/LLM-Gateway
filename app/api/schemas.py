from typing import Literal

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
