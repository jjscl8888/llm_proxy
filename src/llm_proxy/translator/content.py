from __future__ import annotations

from typing import Union


def translate_content(content: Union[str, list], flatten: bool = False) -> Union[str, list]:
    if isinstance(content, str):
        return content

    parts: list[dict] = []
    for block in content:
        block_type = block.get("type", "text")

        if block_type == "text":
            parts.append({"type": "text", "text": block.get("text", "")})

        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                parts.append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
            elif source.get("type") == "url":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": source.get("url", "")},
                    }
                )

        elif block_type == "tool_result":
            continue

        elif block_type == "tool_use":
            continue

        elif block_type == "thinking":
            continue

    if flatten:
        text_parts = [p["text"] for p in parts if p["type"] == "text"]
        return "\n".join(text_parts)

    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    if not parts:
        return ""
    return parts


def extract_text(content: Union[str, list]) -> str:
    if isinstance(content, str):
        return content

    texts: list[str] = []
    for block in content:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts)


def translate_assistant_content(content: Union[str, list], thinking_field: str = "") -> dict:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    reasoning_parts: list[str] = []

    for block in content:
        block_type = block.get("type", "text")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _serialize_input(block.get("input", {})),
                    },
                }
            )

        elif block_type == "thinking":
            if thinking_field:
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    reasoning_parts.append(thinking_text)

    msg: dict = {"role": "assistant"}
    if reasoning_parts:
        msg[thinking_field] = "\n".join(reasoning_parts)

    if text_parts:
        msg["content"] = "\n".join(text_parts)
    else:
        msg["content"] = None

    if tool_calls:
        msg["tool_calls"] = tool_calls

    return msg


def translate_tool_result_blocks(content: Union[str, list]) -> list[dict]:
    if isinstance(content, str):
        return []

    results: list[dict] = []
    for block in content:
        if block.get("type") == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = extract_text(tool_content)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(tool_content),
                }
            )
    return results


def _serialize_input(input_data) -> str:
    import json

    if isinstance(input_data, str):
        return input_data
    return json.dumps(input_data, ensure_ascii=False)
