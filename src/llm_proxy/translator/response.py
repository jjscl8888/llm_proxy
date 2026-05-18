from __future__ import annotations

import json


STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def translate_response(
    openai_resp: dict, original_model: str, thinking_field: str = ""
) -> dict:
    choice = openai_resp["choices"][0]
    message = choice.get("message", {})

    content_blocks: list[dict] = []

    if thinking_field and message.get(thinking_field):
        content_blocks.append(
            {"type": "thinking", "thinking": message[thinking_field]}
        )

    if message.get("content"):
        content_blocks.append(
            {"type": "text", "text": message["content"]}
        )

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            func = tc.get("function", {})
            input_data = {}
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_data = {"raw_arguments": func.get("arguments", "")}

            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": input_data,
                }
            )

    usage = openai_resp.get("usage", {})

    return {
        "id": openai_resp.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": STOP_REASON_MAP.get(
            choice.get("finish_reason"), "end_turn"
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
