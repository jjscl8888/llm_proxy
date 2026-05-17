from __future__ import annotations

from llm_proxy.config import ProxyConfig


def map_model(anthropic_model: str, config: ProxyConfig) -> str:
    return config.map_model(anthropic_model)


def get_backend_url(config: ProxyConfig) -> str:
    base = config.backend.base_url.rstrip("/")
    return base
