from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TierConfig(StrictConfig):
    provider: Literal["ollama", "openai_compat"] = "ollama"
    model: str = Field(min_length=1, max_length=200)
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)


class Tiers(StrictConfig):
    cheap: TierConfig = Field(default_factory=lambda: TierConfig(model="qwen2.5:1.5b"))
    strong: TierConfig = Field(default_factory=lambda: TierConfig(model="qwen2.5:7b"))


class RetryConfig(StrictConfig):
    max_attempts: int = Field(default=2, ge=1, le=5, strict=True)
    backoff_base_seconds: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    backoff_max_seconds: float = Field(default=2, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_backoff(self) -> "RetryConfig":
        if self.backoff_base_seconds > self.backoff_max_seconds:
            raise ValueError("backoff_base_seconds must not exceed backoff_max_seconds")
        return self


class CircuitConfig(StrictConfig):
    failure_threshold: int = Field(default=5, ge=1, strict=True)
    recovery_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)


class ConfidenceConfig(StrictConfig):
    min_response_chars: int = Field(default=12, ge=1, strict=True)


class RoutingConfig(StrictConfig):
    tiers: Tiers = Field(default_factory=Tiers)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitConfig = Field(default_factory=CircuitConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)


def load_routing_config(path: str) -> RoutingConfig:
    """Safe YAML parsing happens once at startup; invalid configuration fails early."""
    with Path(path).open(encoding="utf-8") as source:
        return RoutingConfig.model_validate(yaml.safe_load(source))
