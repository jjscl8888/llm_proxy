from __future__ import annotations

from llm_proxy.config import ProxyConfig
from llm_proxy.translator.content import (
    extract_text,
    translate_assistant_content,
    translate_content,
    translate_tool_result_blocks,
)
from llm_proxy.translator.tools import translate_tool_choice, translate_tools


def translate_request(anthropic_req: dict, config: ProxyConfig) -> dict:
    openai_req: dict = {}

    backend_model = config.map_model(
        anthropic_req.get("model", "claude-sonnet-4-20250514")
    )
    openai_req["model"] = backend_model
    capability = config.get_model_capability(backend_model)
    thinking_field = capability.thinking_field if capability.supports_thinking else ""

    messages: list[dict] = []

    if "system" in anthropic_req:
        system_content = extract_text(anthropic_req["system"])
        if system_content:
            messages.append({"role": "system", "content": system_content})

    for msg in anthropic_req.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            tool_results = []
            if isinstance(content, list):
                tool_results = translate_tool_result_blocks(content)

            if tool_results:
                messages.extend(tool_results)
                text_content = translate_content(content)
                if text_content:
                    messages.append({"role": "user", "content": text_content})
            else:
                messages.append(
                    {"role": "user", "content": translate_content(content)}
                )

        elif role == "assistant":
            openai_msg = translate_assistant_content(content, thinking_field)
            messages.append(openai_msg)

    openai_req["messages"] = messages

    if "max_tokens" in anthropic_req:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]

    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]

    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]

    if "stop_sequences" in anthropic_req:
        openai_req["stop"] = anthropic_req["stop_sequences"]

    openai_req["stream"] = anthropic_req.get("stream", False)

    if "tools" in anthropic_req:
        openai_req["tools"] = translate_tools(anthropic_req["tools"])

    if "tool_choice" in anthropic_req:
        tc = translate_tool_choice(anthropic_req["tool_choice"])
        if tc is not None:
            openai_req["tool_choice"] = tc

    for key in ("temperature", "top_p", "max_tokens"):
        if key in config.parameters:
            openai_req[key] = config.parameters[key]

    if capability.fixed_temperature is not None:
        openai_req["temperature"] = capability.fixed_temperature

    for field in capability.ignore_fields:
        openai_req.pop(field, None)

    return openai_req
