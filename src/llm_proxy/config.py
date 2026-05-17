from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelCapability(BaseModel):
    supports_thinking: bool = False
    thinking_field: str = ""
    supports_vision: bool = False
    max_tokens: int = 8192


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8082


class BackendConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    api_key_env: str = ""


class LoggingConfig(BaseModel):
    level: str = "info"
    log_requests: bool = False
    log_responses: bool = False


class ProxyConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    backend: BackendConfig = BackendConfig()
    model_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "claude-sonnet-4-20250514": "deepseek-chat",
            "claude-opus-4-20250514": "deepseek-reasoner",
            "claude-haiku-3-5-20241022": "deepseek-chat",
            "*": "deepseek-chat",
        }
    )
    model_capabilities: dict[str, ModelCapability] = Field(
        default_factory=lambda: {
            "deepseek-chat": ModelCapability(
                supports_thinking=False,
                supports_vision=False,
                max_tokens=8192,
            ),
            "deepseek-reasoner": ModelCapability(
                supports_thinking=True,
                thinking_field="reasoning_content",
                supports_vision=False,
                max_tokens=8192,
            ),
        }
    )
    parameters: dict = Field(default_factory=dict)
    logging: LoggingConfig = LoggingConfig()

    def get_backend_api_key(self) -> str:
        if self.backend.api_key:
            return self.backend.api_key
        if self.backend.api_key_env:
            return os.environ.get(self.backend.api_key_env, "")
        return os.environ.get("OPENAI_API_KEY", "")

    def map_model(self, anthropic_model: str) -> str:
        if anthropic_model in self.model_mapping:
            return self.model_mapping[anthropic_model]
        if "*" in self.model_mapping:
            return self.model_mapping["*"]
        return anthropic_model

    def get_model_capability(self, model: str) -> ModelCapability:
        if model in self.model_capabilities:
            return self.model_capabilities[model]
        return ModelCapability()


def load_config(config_path: Optional[str] = None) -> ProxyConfig:
    data: dict = {}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    env_overrides: dict = {}

    if v := os.environ.get("PROXY_HOST"):
        env_overrides.setdefault("server", {})["host"] = v
    if v := os.environ.get("PROXY_PORT"):
        env_overrides.setdefault("server", {})["port"] = int(v)
    if v := os.environ.get("OPENAI_BASE_URL"):
        env_overrides.setdefault("backend", {})["base_url"] = v
    if v := os.environ.get("OPENAI_API_KEY"):
        env_overrides.setdefault("backend", {})["api_key"] = v
    if v := os.environ.get("DEFAULT_MODEL"):
        env_overrides.setdefault("model_mapping", {})["*"] = v
    if v := os.environ.get("LOG_LEVEL"):
        env_overrides.setdefault("logging", {})["level"] = v

    if env_overrides:
        _deep_merge(data, env_overrides)

    return ProxyConfig(**data)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
