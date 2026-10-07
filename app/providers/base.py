from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationOptions(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1, ge=0, le=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: str | list[str] | None = None


class GenerationParams(GenerationOptions):
    json_mode: bool = False


class ProviderResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    finish_reason: Literal["stop", "length"] = "stop"


class ProviderError(Exception):
    """A sanitized upstream failure; never pass response bodies or credentials to clients."""


class ProviderTimeout(ProviderError):
    """The configured upstream deadline elapsed."""


class ProviderAdapter(ABC):
    name: str

    @abstractmethod
    async def generate(self, prompt: str, model: str, params: GenerationParams) -> ProviderResponse:
        """Return generated text, measured wall latency, and provider-reported token counts."""

    @abstractmethod
    async def health(self) -> bool:
        """Check reachability without running inference."""
