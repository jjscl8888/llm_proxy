import pytest

from llm_proxy.config import ProxyConfig
from llm_proxy.translator.content import (
    extract_text,
    translate_assistant_content,
    translate_content,
    translate_tool_result_blocks,
)
from llm_proxy.translator.request import translate_request
from llm_proxy.translator.response import translate_response
from llm_proxy.translator.stream import StreamConverter
from llm_proxy.translator.tools import translate_tool_choice, translate_tools


class TestTranslateContent:
    def test_string_passthrough(self):
        assert translate_content("hello") == "hello"

    def test_single_text_block(self):
        result = translate_content([{"type": "text", "text": "hello"}])
        assert result == "hello"

    def test_multiple_text_blocks(self):
        result = translate_content(
            [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_image_block(self):
        result = translate_content(
            [
                {"type": "text", "text": "see image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc123",
                    },
                },
            ]
        )
        assert isinstance(result, list)
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_empty_blocks(self):
        result = translate_content([])
        assert result == ""

    def test_tool_use_skipped(self):
        result = translate_content(
            [{"type": "tool_use", "id": "t1", "name": "fn", "input": {}}]
        )
        assert result == ""

    def test_tool_result_skipped(self):
        result = translate_content(
            [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
        )
        assert result == ""


class TestExtractText:
    def test_string(self):
        assert extract_text("hello") == "hello"

    def test_blocks(self):
        result = extract_text(
            [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        )
        assert result == "hello\nworld"


class TestTranslateAssistantContent:
    def test_string(self):
        result = translate_assistant_content("hello")
        assert result == {"role": "assistant", "content": "hello"}

    def test_text_and_tool_use(self):
        result = translate_assistant_content(
            [
                {"type": "text", "text": "let me check"},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Beijing"},
                },
            ]
        )
        assert result["role"] == "assistant"
        assert result["content"] == "let me check"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_use_only(self):
        result = translate_assistant_content(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_456",
                    "name": "search",
                    "input": {"q": "test"},
                }
            ]
        )
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1


class TestTranslateToolResultBlocks:
    def test_extract_tool_results(self):
        result = translate_tool_result_blocks(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "72°F, sunny",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_123"

    def test_no_tool_results(self):
        result = translate_tool_result_blocks(
            [{"type": "text", "text": "hello"}]
        )
        assert result == []


class TestTranslateTools:
    def test_basic_tool(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        result = translate_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["parameters"] == tools[0]["input_schema"]


class TestTranslateToolChoice:
    def test_auto(self):
        assert translate_tool_choice({"type": "auto"}) == "auto"

    def test_none(self):
        assert translate_tool_choice({"type": "none"}) == "none"

    def test_specific_tool(self):
        result = translate_tool_choice({"type": "tool", "name": "get_weather"})
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"

    def test_null(self):
        assert translate_tool_choice(None) is None


class TestTranslateRequest:
    def setup_method(self):
        self.config = ProxyConfig(
            backend=ProxyConfig.__fields__["backend"].default.__class__(
                base_url="https://api.deepseek.com",
                api_key="sk-test",
            ),
            model_mapping={
                "claude-sonnet-4-20250514": "deepseek-chat",
                "*": "deepseek-chat",
            },
        )

    def test_basic_request(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        }
        result = translate_request(anthropic_req, self.config)
        assert result["model"] == "deepseek-chat"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 1024

    def test_system_message(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are a helpful assistant",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
        }
        result = translate_request(anthropic_req, self.config)
        assert result["messages"][0] == {
            "role": "system",
            "content": "You are a helpful assistant",
        }

    def test_system_message_blocks(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Be helpful"}],
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
        }
        result = translate_request(anthropic_req, self.config)
        assert result["messages"][0] == {"role": "system", "content": "Be helpful"}

    def test_stream_flag(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(anthropic_req, self.config)
        assert result["stream"] is True

    def test_tools_conversion(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "calc",
                    "description": "Calculate",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "messages": [{"role": "user", "content": "Calculate 1+1"}],
        }
        result = translate_request(anthropic_req, self.config)
        assert "tools" in result
        assert result["tools"][0]["function"]["name"] == "calc"

    def test_assistant_with_tool_use(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "72°F sunny",
                        }
                    ],
                },
            ],
        }
        result = translate_request(anthropic_req, self.config)
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check"
        assert len(assistant_msg["tool_calls"]) == 1

        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_1"

    def test_stop_sequences(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "stop_sequences": ["END", "STOP"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(anthropic_req, self.config)
        assert result["stop"] == ["END", "STOP"]


class TestTranslateResponse:
    def test_basic_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = translate_response(openai_resp, "claude-sonnet-4-20250514")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_use_response(self):
        openai_resp = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = translate_response(openai_resp, "claude-sonnet-4-20250514")
        assert result["stop_reason"] == "tool_use"
        tool_block = result["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "NYC"}

    def test_max_tokens_stop(self):
        openai_resp = {
            "id": "chatcmpl-789",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "truncated..."},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100},
        }
        result = translate_response(openai_resp, "claude-sonnet-4-20250514")
        assert result["stop_reason"] == "max_tokens"


class TestStreamConverter:
    def test_basic_text_stream(self):
        converter = StreamConverter("claude-sonnet-4-20250514", input_tokens=10)

        chunk1 = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
        }
        chunk2 = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        }
        chunk3 = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {"content": " world"}, "finish_reason": None}],
        }
        chunk4 = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }

        result1 = converter.convert_chunk(chunk1)
        assert "message_start" in result1

        result2 = converter.convert_chunk(chunk2)
        assert "content_block_start" in result2
        assert "content_block_delta" in result2
        assert "text_delta" in result2

        result3 = converter.convert_chunk(chunk3)
        assert "content_block_delta" in result3

        result4 = converter.convert_chunk(chunk4)
        assert "content_block_stop" in result4
        assert "message_delta" in result4
        assert "message_stop" in result4

    def test_thinking_stream(self):
        converter = StreamConverter("claude-sonnet-4-20250514", input_tokens=10)

        chunk1 = {
            "id": "chatcmpl-1",
            "choices": [
                {"delta": {"reasoning_content": "thinking..."}, "finish_reason": None}
            ],
        }
        chunk2 = {
            "id": "chatcmpl-1",
            "choices": [
                {"delta": {"content": "answer"}, "finish_reason": None}
            ],
        }
        chunk3 = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }

        result1 = converter.convert_chunk(chunk1)
        assert "thinking" in result1

        result2 = converter.convert_chunk(chunk2)
        assert "text_delta" in result2

        result3 = converter.convert_chunk(chunk3)
        assert "message_stop" in result3

    def test_tool_calls_stream(self):
        converter = StreamConverter("claude-sonnet-4-20250514", input_tokens=10)

        chunk1 = {
            "id": "chatcmpl-1",
            "choices": [
                {"delta": {"content": "Let me"}, "finish_reason": None}
            ],
        }
        chunk2 = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "function": {"name": "search", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        chunk3 = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"q": "test"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        chunk4 = {
            "id": "chatcmpl-1",
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
        }

        converter.convert_chunk(chunk1)
        result2 = converter.convert_chunk(chunk2)
        assert "tool_use" in result2

        result3 = converter.convert_chunk(chunk3)
        assert "input_json_delta" in result3

        result4 = converter.convert_chunk(chunk4)
        assert "message_stop" in result4


class TestConfig:
    def test_load_defaults(self):
        config = ProxyConfig()
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8082

    def test_model_mapping(self):
        config = ProxyConfig(
            model_mapping={
                "claude-sonnet-4-20250514": "deepseek-chat",
                "*": "deepseek-chat",
            }
        )
        assert config.map_model("claude-sonnet-4-20250514") == "deepseek-chat"
        assert config.map_model("unknown-model") == "deepseek-chat"

    def test_backend_api_key(self):
        config = ProxyConfig(
            backend=ProxyConfig.__fields__["backend"].default.__class__(
                api_key="sk-test"
            )
        )
        assert config.get_backend_api_key() == "sk-test"
