from __future__ import annotations

from llm_proxy.config import ProxyConfig


def translate_auth(headers: dict, config: ProxyConfig) -> dict:
    backend_key = config.get_backend_api_key()
    if backend_key:
        return {"Authorization": f"Bearer {backend_key}"}

    api_key = headers.get("x-api-key", "") or headers.get(
        "authorization", ""
    ).replace("Bearer ", "")

    if api_key:
        return {"Authorization": f"Bearer {api_key}"}

    return {}
