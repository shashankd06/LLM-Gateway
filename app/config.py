from decimal import Decimal
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: SecretStr
    ollama_base_url: str = "http://host.docker.internal:11434"
    default_model: str = Field(default="qwen2.5:1.5b", min_length=1)
    provider_timeout_seconds: float = Field(default=120, gt=0)
    health_timeout_seconds: float = Field(default=5, gt=0)
    default_provider: Literal["ollama", "openai_compat"] = "ollama"
    openai_compat_enabled: bool = False
    openai_compat_base_url: str | None = None
    openai_compat_api_key: SecretStr | None = None
    input_cost_per_1k: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    output_cost_per_1k: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> "Settings":
        if self.openai_compat_enabled and not self.openai_compat_base_url:
            raise ValueError("OPENAI_COMPAT_BASE_URL is required when the adapter is enabled")
        if self.default_provider == "openai_compat" and not self.openai_compat_enabled:
            raise ValueError("Enable OPENAI_COMPAT_ENABLED before selecting openai_compat")
        return self
