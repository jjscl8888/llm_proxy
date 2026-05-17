# LLM Proxy — Anthropic-to-OpenAI 协议代理

## 1. 项目概述

### 1.1 背景

Claude Code CLI 是 Anthropic 官方提供的命令行 AI 编程助手，原生仅支持 Anthropic API 协议（`/v1/messages`）。然而许多用户拥有的是兼容 OpenAI API 协议的模型服务（如 DeepSeek、GLM、Qwen、Ollama 本地模型等），无法直接在 Claude Code CLI 中使用。

### 1.2 目标

构建一个**本地代理服务**，在 Claude Code CLI 与 OpenAI 兼容模型之间充当协议翻译层：

```
Claude Code CLI  ──Anthropic协议──▶  LLM Proxy  ──OpenAI协议──▶  目标模型
                  ◀──Anthropic协议──  LLM Proxy  ◀──OpenAI协议──  目标模型
```

### 1.3 核心价值

- 无需修改 Claude Code CLI 源码，通过配置 `ANTHROPIC_BASE_URL` 即可接入
- 支持任意 OpenAI 兼容模型（商业 API / 本地 Ollama / 自部署 vLLM 等）
- 流式响应实时转换，用户体验无损
- 轻量级，单二进制部署，零外部依赖

---

## 2. 协议差异分析

### 2.1 请求格式对比

| 维度 | Anthropic (`/v1/messages`) | OpenAI (`/v1/chat/completions`) |
|------|---------------------------|--------------------------------|
| **端点** | `POST /v1/messages` | `POST /v1/chat/completions` |
| **模型字段** | `model` | `model` |
| **消息格式** | `content` 为 `str` 或 `content_block[]` | `content` 为 `str` |
| **系统提示** | 独立 `system` 字段（顶层） | 嵌入 `messages[]` 中 `role=system` |
| **多模态** | `content_block` 含 `type: image` | `content` 含 `type: image_url` |
| **工具调用** | `tool_use` content block + `tool_result` | `tool_calls` 字段 + `role=tool` |
| **停止原因** | `stop_reason` | `finish_reason` |
| **流式协议** | SSE `event: message_start/content_block_delta/...` | SSE `data: {"choices":[{"delta":...}]}` |
| **Token统计** | `usage.input_tokens / output_tokens` | `usage.prompt_tokens / completion_tokens` |

### 2.2 关键转换难点

1. **Content Block ↔ Plain Text**：Anthropic 的 `content` 可以是 `[{"type":"text","text":"..."}]`，OpenAI 直接用字符串
2. **System Message 位置**：Anthropic 独立字段 → OpenAI 首条 system message
3. **Tool Use 映射**：Anthropic 的 `tool_use`/`tool_result` ↔ OpenAI 的 `tool_calls`/`role=tool`
4. **流式事件格式**：两套完全不同的 SSE 事件结构
5. **Thinking/Extended Thinking**：Claude 的思维链功能需特殊处理（OpenAI 无对应概念）
6. **Image 格式**：Anthropic 用 `source.type=base64` → OpenAI 用 `image_url.url`

---

## 3. 系统架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code CLI                       │
│          (ANTHROPIC_BASE_URL=http://localhost:8082)      │
└──────────────────────┬──────────────────────────────────┘
                       │ Anthropic Protocol
                       ▼
┌─────────────────────────────────────────────────────────┐
│                      LLM Proxy                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  HTTP Server │  │  Translator  │  │  Config Mgr   │  │
│  │  (Port 8082) │  │  (Req/Resp)  │  │  (YAML/ENV)   │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────┘  │
│         │                │                               │
│  ┌──────┴──────┐  ┌──────┴───────┐  ┌───────────────┐  │
│  │  Auth Handler│  │ Stream Conv  │  │  Model Router │  │
│  │  (API Key)   │  │ (SSE↔SSE)   │  │  (Multi-Model)│  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │ OpenAI Protocol
                       ▼
┌─────────────────────────────────────────────────────────┐
│              OpenAI-Compatible Model Service             │
│    (DeepSeek / GLM / Qwen / Ollama / vLLM / ...)       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心模块

| 模块 | 职责 |
|------|------|
| **HTTP Server** | 监听 Anthropic 协议端点，路由请求 |
| **Request Translator** | Anthropic 请求 → OpenAI 请求 |
| **Response Translator** | OpenAI 响应 → Anthropic 响应 |
| **Stream Converter** | OpenAI SSE → Anthropic SSE 实时转换 |
| **Auth Handler** | API Key 验证与透传 |
| **Config Manager** | 模型映射、端点配置、参数覆盖 |
| **Model Router** | 多模型路由，按规则分发到不同后端 |

---

## 4. 详细设计

### 4.1 请求转换（Anthropic → OpenAI）

#### 4.1.1 端点映射

```
POST /v1/messages           →  POST {openai_base}/v1/chat/completions
GET  /v1/models             →  GET  {openai_base}/v1/models
```

#### 4.1.2 请求体转换规则

```python
def translate_request(anthropic_req: dict) -> dict:
    openai_req = {}

    # 1. 模型映射
    openai_req["model"] = map_model(anthropic_req.get("model", "claude-sonnet-4-20250514"))

    # 2. 消息转换
    openai_req["messages"] = []
    
    # 2a. System → 首条 system message
    if "system" in anthropic_req:
        system_content = extract_text(anthropic_req["system"])
        openai_req["messages"].append({
            "role": "system",
            "content": system_content
        })

    # 2b. 普通消息转换
    for msg in anthropic_req.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            openai_req["messages"].append({
                "role": "user",
                "content": translate_content(content)  # content_block[] → str/image
            })
        elif role == "assistant":
            openai_msg = translate_assistant_message(content)
            openai_req["messages"].append(openai_msg)

    # 3. 参数映射
    if "max_tokens" in anthropic_req:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]
    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]
    if "stop_sequences" in anthropic_req:
        openai_req["stop"] = anthropic_req["stop_sequences"]

    # 4. 流式标记
    openai_req["stream"] = anthropic_req.get("stream", False)

    # 5. 工具定义转换
    if "tools" in anthropic_req:
        openai_req["tools"] = translate_tools(anthropic_req["tools"])

    return openai_req
```

#### 4.1.3 Content Block 转换

```
Anthropic:                          OpenAI:
[                                   "Hello! Here is the image:
  {"type":"text","text":"Hello!"},   [image url] and some more text"
  {"type":"image","source":{         
    "type":"base64",                 
    "media_type":"image/png",        
    "data":"..."}},                  
  {"type":"text","text":"more"}      
]                                   
```

对于多模态内容，转换为 OpenAI 的 `content` 数组格式：

```python
def translate_content(content) -> Union[str, list]:
    if isinstance(content, str):
        return content
    
    parts = []
    for block in content:
        if block["type"] == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image":
            data_url = f"data:{block['source']['media_type']};base64,{block['source']['data']}"
            parts.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
    
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts
```

#### 4.1.4 Tool Use 转换

```
Anthropic:                              OpenAI:
tools: [{                                tools: [{
  "name": "get_weather",                   "type": "function",
  "description": "...",                    "function": {
  "input_schema": { ... }                    "name": "get_weather",
}]                                           "description": "...",
                                             "parameters": { ... }
                                           }
                                         }]

assistant content:                       assistant message:
[{                                       
  "type": "text",                        content: "Let me check...",
  "text": "Let me check..."             tool_calls: [{
},                                         "id": "toolu_xxx",
{                                          "type": "function",
  "type": "tool_use",                      "function": {
  "id": "toolu_xxx",                         "name": "get_weather",
  "name": "get_weather",                     "arguments": "{...}"
  "input": {...}                           }
}]                                       }]

user content (tool_result):             assistant message:
[{                                       
  "type": "tool_result",                (合并到下一条 user 消息前)
  "tool_use_id": "toolu_xxx",           role: "tool",
  "content": "72°F, sunny"              tool_call_id: "toolu_xxx",
}]                                       content: "72°F, sunny"
```

### 4.2 响应转换（OpenAI → Anthropic）

#### 4.2.1 非流式响应

```python
def translate_response(openai_resp: dict, original_model: str) -> dict:
    choice = openai_resp["choices"][0]
    message = choice["message"]
    
    # 构建 content blocks
    content_blocks = []
    
    # 文本内容
    if message.get("content"):
        content_blocks.append({
            "type": "text",
            "text": message["content"]
        })
    
    # 工具调用
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            content_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"])
            })
    
    # 停止原因映射
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    
    return {
        "id": openai_resp["id"],
        "type": "message",
        "role": "assistant",
        "model": original_model,  # 返回 Claude 模型名以兼容 CLI
        "content": content_blocks,
        "stop_reason": stop_reason_map.get(choice["finish_reason"], "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_resp["usage"]["prompt_tokens"],
            "output_tokens": openai_resp["usage"]["completion_tokens"],
        }
    }
```

#### 4.2.2 流式响应转换

这是最复杂的部分。需要将 OpenAI 的增量格式转换为 Anthropic 的事件流格式。

**OpenAI SSE 格式：**
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

**Anthropic SSE 格式：**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_xxx","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
```

**转换状态机：**

```
[初始] ──收到首个delta──▶ [发送 message_start + content_block_start]
[content_block_start] ──收到content delta──▶ [发送 content_block_delta]
[content_block_delta] ──收到finish_reason──▶ [发送 content_block_stop + message_delta + message_stop]
[content_block_delta] ──收到tool_calls──▶ [发送 content_block_stop + content_block_start(tool_use) + ...]
```

```python
class StreamConverter:
    def __init__(self, original_model: str, input_tokens: int):
        self.model = original_model
        self.input_tokens = input_tokens
        self.output_tokens = 0
        self.block_index = 0
        self.started = False
        self.current_block_type = None
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    async def convert_chunk(self, chunk: dict) -> list[str]:
        events = []
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        # 首个 chunk：发送 message_start
        if not self.started:
            events.append(self._message_start_event())
            self.started = True
        
        # 文本内容 delta
        if delta.get("content"):
            if self.current_block_type != "text":
                if self.current_block_type is not None:
                    events.append(self._content_block_stop())
                    self.block_index += 1
                events.append(self._content_block_start("text"))
                self.current_block_type = "text
            
            events.append(self._content_block_delta(delta["content"]))
            self.output_tokens += 1
        
        # 工具调用 delta
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                if tc.get("function", {}).get("name"):
                    if self.current_block_type is not None:
                        events.append(self._content_block_stop())
                        self.block_index += 1
                    events.append(self._tool_use_start(tc))
                    self.current_block_type = "tool_use"
                if tc.get("function", {}).get("arguments"):
                    events.append(self._tool_use_delta(tc))
        
        # 结束
        if finish_reason:
            if self.current_block_type is not None:
                events.append(self._content_block_stop())
            events.append(self._message_delta(finish_reason))
            events.append(self._message_stop())
        
        return events
```

### 4.3 模型映射

```yaml
model_mapping:
  # Claude 模型名 → 实际 OpenAI 兼容模型
  claude-sonnet-4-20250514: deepseek-chat
  claude-sonnet-4-20250514: glm-4-plus
  claude-haiku-3-5-20241022: qwen-turbo
  claude-opus-4-20250514: deepseek-reasoner
  
  # 通配符：未匹配的模型使用默认
  "*": deepseek-chat
```

### 4.4 认证处理

```python
# Claude Code CLI 发送: x-api-key: sk-ant-xxx
# 代理需要: Authorization: Bearer sk-xxx

def translate_auth(anthropic_headers: dict) -> dict:
    api_key = anthropic_headers.get("x-api-key", "")
    
    # 如果配置了固定的后端 API Key，使用配置的
    if config.backend_api_key:
        return {"Authorization": f"Bearer {config.backend_api_key}"}
    
    # 否则透传原始 Key
    return {"Authorization": f"Bearer {api_key}"}
```

### 4.5 特殊处理：Thinking / Extended Thinking

Claude Code CLI 可能启用 `thinking` 功能（扩展思维链）。OpenAI 兼容模型中，仅部分模型支持类似功能（如 DeepSeek Reasoner 的 `reasoning_content`）。

**策略：**
- 如果后端模型支持推理（如 `deepseek-reasoner`），将 `reasoning_content` 映射为 `thinking` content block
- 如果后端模型不支持推理，**静默忽略** thinking 参数，仅返回文本响应
- 在配置中标记每个模型是否支持 thinking

```yaml
models:
  deepseek-reasoner:
    supports_thinking: true
    thinking_field: reasoning_content  # OpenAI 侧的字段名
  deepseek-chat:
    supports_thinking: false
  glm-4-plus:
    supports_thinking: false
```

---

## 5. 配置设计

### 5.1 配置文件格式 (`config.yaml`)

```yaml
server:
  host: "127.0.0.1"
  port: 8082

backend:
  base_url: "https://api.deepseek.com"
  api_key: "sk-xxxxxxxx"          # 后端模型的 API Key
  # api_key_env: "DEEPSEEK_API_KEY"  # 或从环境变量读取

model_mapping:
  claude-sonnet-4-20250514: "deepseek-chat"
  claude-opus-4-20250514: "deepseek-reasoner"
  claude-haiku-3-5-20241022: "deepseek-chat"
  "*": "deepseek-chat"

model_capabilities:
  deepseek-chat:
    supports_thinking: false
    supports_vision: false
    max_tokens: 8192
  deepseek-reasoner:
    supports_thinking: true
    thinking_field: "reasoning_content"
    supports_vision: false
    max_tokens: 8192
  glm-4-plus:
    supports_thinking: false
    supports_vision: true
    max_tokens: 4096

parameters:
  temperature: 0.7
  top_p: 1.0
  # 对所有请求强制覆盖的参数（可选）

logging:
  level: "info"          # debug / info / warn / error
  log_requests: false    # 是否记录完整请求体
  log_responses: false   # 是否记录完整响应体
```

### 5.2 环境变量支持

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `PROXY_HOST` | 监听地址 | `127.0.0.1` |
| `PROXY_PORT` | 监听端口 | `8082` |
| `OPENAI_BASE_URL` | 后端 API 地址 | 必填 |
| `OPENAI_API_KEY` | 后端 API Key | 必填 |
| `DEFAULT_MODEL` | 默认模型 | `deepseek-chat` |
| `LOG_LEVEL` | 日志级别 | `info` |

---

## 6. Claude Code CLI 接入方式

### 6.1 配置环境变量

```bash
# 指向本地代理
export ANTHROPIC_BASE_URL="http://127.0.0.1:8082"

# API Key（代理会透传或替换为后端 Key）
export ANTHROPIC_API_KEY="sk-any-value-works"

# 启动 Claude Code CLI
claude
```

### 6.2 一键启动脚本

```bash
#!/bin/bash
# start_with_proxy.sh

# 1. 启动代理（后台）
python -m llm_proxy --config config.yaml &
PROXY_PID=$!

# 2. 等待代理就绪
sleep 2

# 3. 配置 Claude Code CLI 环境变量
export ANTHROPIC_BASE_URL="http://127.0.0.1:8082"
export ANTHROPIC_API_KEY="sk-proxy-pass-through"

# 4. 启动 Claude Code CLI
claude

# 5. 退出时清理
kill $PROXY_PID 2>/dev/null
```

---

## 7. 项目结构

```
llm_proxy/
├── design.md                  # 本设计文档
├── config.yaml                # 配置文件
├── pyproject.toml              # 项目依赖
├── src/
│   └── llm_proxy/
│       ├── __init__.py
│       ├── __main__.py         # CLI 入口
│       ├── server.py           # HTTP 服务器（FastAPI/Starlette）
│       ├── config.py           # 配置加载
│       ├── translator/
│       │   ├── __init__.py
│       │   ├── request.py      # Anthropic → OpenAI 请求转换
│       │   ├── response.py     # OpenAI → Anthropic 响应转换
│       │   ├── stream.py       # 流式 SSE 转换
│       │   ├── content.py      # Content Block 转换
│       │   └── tools.py        # Tool Use 转换
│       ├── auth.py             # 认证处理
│       ├── model_router.py     # 模型映射与路由
│       └── logging.py          # 请求/响应日志
├── tests/
│   ├── test_request_translator.py
│   ├── test_response_translator.py
│   ├── test_stream_converter.py
│   └── test_integration.py
└── scripts/
    └── start_with_proxy.sh     # 一键启动脚本
```

---

## 8. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| **语言** | Python 3.11+ | 生态成熟，异步支持好，开发效率高 |
| **Web框架** | FastAPI | 原生 async，SSE 支持好，自动文档 |
| **HTTP客户端** | httpx | 异步 HTTP，流式支持好 |
| **配置** | Pydantic + YAML | 类型安全，校验完善 |
| **日志** | structlog | 结构化日志，便于调试 |
| **测试** | pytest + httpx.AsyncClient | 异步测试支持 |

---

## 9. 关键流程时序图

### 9.1 非流式请求

```
Claude Code CLI          LLM Proxy              OpenAI API
     │                       │                       │
     │  POST /v1/messages    │                       │
     │  (Anthropic格式)      │                       │
     │──────────────────────▶│                       │
     │                       │  转换请求格式          │
     │                       │  POST /v1/chat/comp   │
     │                       │  (OpenAI格式)          │
     │                       │──────────────────────▶│
     │                       │                       │
     │                       │  200 OK (OpenAI格式)   │
     │                       │◀──────────────────────│
     │                       │  转换响应格式          │
     │  200 OK (Anthropic)   │                       │
     │◀──────────────────────│                       │
     │                       │                       │
```

### 9.2 流式请求

```
Claude Code CLI          LLM Proxy              OpenAI API
     │                       │                       │
     │  POST /v1/messages    │                       │
     │  stream: true         │                       │
     │──────────────────────▶│                       │
     │                       │  POST stream: true    │
     │                       │──────────────────────▶│
     │                       │                       │
     │                       │  SSE: delta(chunk1)   │
     │                       │◀──────────────────────│
     │  SSE: message_start   │  转换 →               │
     │  SSE: content_block_  │  Anthropic格式        │
     │        start          │                       │
     │◀──────────────────────│                       │
     │                       │                       │
     │                       │  SSE: delta(chunk2)   │
     │                       │◀──────────────────────│
     │  SSE: content_block_  │                       │
     │        delta          │                       │
     │◀──────────────────────│                       │
     │          ...          │          ...          │
     │                       │  SSE: [DONE]          │
     │                       │◀──────────────────────│
     │  SSE: message_stop    │                       │
     │◀──────────────────────│                       │
```

---

## 10. 错误处理

### 10.1 错误映射

| OpenAI 错误 | Anthropic 错误 |
|-------------|---------------|
| `invalid_api_key` (401) | `authentication_error` (401) |
| `model_not_found` (404) | `not_found_error` (404) |
| `rate_limit_exceeded` (429) | `rate_limit_error` (429) |
| `context_length_exceeded` (400) | `invalid_request_error` (400) |
| `server_error` (500) | `api_error` (500) |

### 10.2 代理自身错误

```json
{
  "type": "error",
  "error": {
    "type": "proxy_error",
    "message": "Failed to connect to backend: Connection refused"
  }
}
```

---

## 11. 性能与限制

### 11.1 性能目标

- 请求转发延迟增加 < 50ms（不含模型推理时间）
- 流式首 token 延迟增加 < 20ms
- 内存占用 < 100MB

### 11.2 已知限制

| 限制 | 说明 | 应对策略 |
|------|------|---------|
| **Thinking 功能** | 大部分 OpenAI 兼容模型不支持 | 静默忽略，或映射到 reasoning_content |
| **Vision 多模态** | 部分模型不支持图片输入 | 配置中标记，返回错误提示 |
| **Token 计数** | OpenAI 的 usage 可能不准确 | 透传后端返回值，或自行估算 |
| **Caching** | Anthropic 支持 prompt caching | OpenAI 兼容端点通常不支持，忽略相关参数 |
| **Bash/Read 工具** | Claude Code 特有工具 | 透传工具定义，由模型决定是否调用 |

---

## 12. 安全考虑

1. **API Key 保护**：后端 API Key 仅存储在配置文件或环境变量中，不记录到日志
2. **本地监听**：默认仅监听 `127.0.0.1`，不暴露到公网
3. **请求日志脱敏**：日志中不记录完整请求/响应体（可配置开启，用于调试）
4. **输入校验**：对传入请求进行基本校验，防止注入攻击

---

## 13. 后续扩展

- [ ] **多后端负载均衡**：同一模型配置多个后端，轮询/加权分发
- [ ] **请求缓存**：对相同请求缓存响应，减少重复调用
- [ ] **Token 用量统计**：记录每次调用的 token 消耗
- [ ] **Web 管理界面**：可视化配置和监控
- [ ] **Docker 部署**：提供 Dockerfile 和 docker-compose.yml
- [ ] **Prompt 适配层**：针对不同模型自动调整 system prompt 格式
