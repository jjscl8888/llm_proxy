from __future__ import annotations

import json
import uuid
from typing import AsyncIterator


class StreamConverter:
    def __init__(self, original_model: str, input_tokens: int = 0):
        self.model = original_model
        self.input_tokens = input_tokens
        self.output_tokens = 0
        self.block_index = 0
        self.started = False
        self.current_block_type: str | None = None
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._tool_call_buffers: dict[int, dict] = {}
        self._finished = False

    def _sse_event(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _message_start_event(self) -> str:
        return self._sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 0,
                    },
                },
            },
        )

    def _content_block_start(self, block_type: str, **extra) -> str:
        block: dict = {"type": block_type}
        if block_type == "text":
            block["text"] = ""
        block.update(extra)
        return self._sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": block,
            },
        )

    def _content_block_delta(self, delta: dict) -> str:
        return self._sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": delta,
            },
        )

    def _content_block_stop(self) -> str:
        return self._sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.block_index},
        )

    def _ping(self) -> str:
        return self._sse_event("ping", {"type": "ping"})

    def _message_delta(self, stop_reason: str) -> str:
        return self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": self.output_tokens},
            },
        )

    def _message_stop(self) -> str:
        return self._sse_event("message_stop", {"type": "message_stop"})

    def convert_chunk(self, chunk: dict) -> str:
        events: list[str] = []

        choices = chunk.get("choices", [])
        if not choices:
            return ""

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        if not self.started:
            events.append(self._message_start_event())
            events.append(self._ping())
            self.started = True

        if delta.get("reasoning_content"):
            if self.current_block_type != "thinking":
                if self.current_block_type is not None:
                    events.append(self._content_block_stop())
                    self.block_index += 1
                events.append(
                    self._content_block_start("thinking", thinking="")
                )
                self.current_block_type = "thinking"

            events.append(
                self._content_block_delta(
                    {
                        "type": "thinking_delta",
                        "thinking": delta["reasoning_content"],
                    }
                )
            )
            self.output_tokens += 1

        if delta.get("content") is not None and delta["content"] != "":
            if self.current_block_type != "text":
                if self.current_block_type is not None:
                    events.append(self._content_block_stop())
                    self.block_index += 1
                events.append(self._content_block_start("text"))
                self.current_block_type = "text"

            events.append(
                self._content_block_delta(
                    {"type": "text_delta", "text": delta["content"]}
                )
            )
            self.output_tokens += 1

        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})

                if func.get("name"):
                    if self.current_block_type is not None:
                        events.append(self._content_block_stop())
                        self.block_index += 1

                    tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    self._tool_call_buffers[tc_index] = {
                        "id": tool_id,
                        "name": func["name"],
                        "arguments": "",
                    }

                    events.append(
                        self._content_block_start(
                            "tool_use",
                            id=tool_id,
                            name=func["name"],
                            input={},
                        )
                    )
                    self.current_block_type = "tool_use"

                if func.get("arguments"):
                    buf = self._tool_call_buffers.get(tc_index, {})
                    buf["arguments"] = buf.get("arguments", "") + func[
                        "arguments"
                    ]
                    events.append(
                        self._content_block_delta(
                            {
                                "type": "input_json_delta",
                                "partial_json": func["arguments"],
                            }
                        )
                    )

        if finish_reason:
            if self.current_block_type is not None:
                events.append(self._content_block_stop())

            stop_reason = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "content_filter": "end_turn",
            }.get(finish_reason, "end_turn")

            events.append(self._message_delta(stop_reason))
            events.append(self._message_stop())
            self._finished = True

        return "".join(events)


async def convert_stream(
    stream: AsyncIterator[bytes], original_model: str, input_tokens: int = 0
) -> AsyncIterator[str]:
    converter = StreamConverter(original_model, input_tokens)
    buffer = ""

    async for chunk_bytes in stream:
        chunk_text = chunk_bytes.decode("utf-8") if isinstance(chunk_bytes, bytes) else chunk_bytes
        buffer += chunk_text

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]

                if data_str.strip() == "[DONE]":
                    if not converter._finished:
                        if converter.started and converter.current_block_type is not None:
                            yield converter._content_block_stop()
                        yield converter._message_delta("end_turn")
                        yield converter._message_stop()
                    return

                try:
                    chunk_data = json.loads(data_str)
                    result = converter.convert_chunk(chunk_data)
                    if result:
                        yield result
                except json.JSONDecodeError:
                    continue
