from __future__ import annotations


def translate_tools(tools: list[dict]) -> list[dict]:
    openai_tools: list[dict] = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
        openai_tools.append(openai_tool)
    return openai_tools


def translate_tool_choice(tool_choice: dict | None) -> dict | str | None:
    if tool_choice is None:
        return None

    tc_type = tool_choice.get("type")

    if tc_type == "auto":
        return "auto"
    elif tc_type == "any":
        return "auto"
    elif tc_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    elif tc_type == "none":
        return "none"

    return None
