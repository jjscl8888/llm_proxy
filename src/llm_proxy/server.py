from __future__ import annotations

import json
import logging
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.auth import translate_auth
from llm_proxy.config import ProxyConfig, load_config
from llm_proxy.model_router import get_backend_url
from llm_proxy.translator.request import translate_request
from llm_proxy.translator.response import translate_response
from llm_proxy.translator.stream import convert_stream

logger = logging.getLogger("llm_proxy")

app = FastAPI(title="LLM Proxy", version="0.1.0")

config: ProxyConfig = ProxyConfig()
http_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None


def set_config(cfg: ProxyConfig):
    global config
    config = cfg


@app.post("/v1/messages")
async def messages(request: Request):
    start = time.time()
    body = await request.json()
    original_model = body.get("model", "claude-sonnet-4-20250514")

    if config.logging.log_requests:
        logger.info("Request: %s", json.dumps(body, ensure_ascii=False)[:2000])

    openai_req = translate_request(body, config)
    is_stream = openai_req.get("stream", False)
    backend_model = openai_req.get("model", "")
    capability = config.get_model_capability(backend_model)
    thinking_field = capability.thinking_field if capability.supports_thinking else ""

    backend_url = get_backend_url(config)
    url = f"{backend_url}/chat/completions"

    auth_headers = translate_auth(dict(request.headers), config)
    req_headers = {
        **auth_headers,
        "Content-Type": "application/json",
    }

    if is_stream:
        return await _handle_stream(url, req_headers, openai_req, original_model, thinking_field, start)
    else:
        return await _handle_non_stream(url, req_headers, openai_req, original_model, thinking_field, start)


async def _handle_stream(url, headers, openai_req, original_model, thinking_field, start):
    try:
        assert http_client is not None
        req = http_client.build_request(
            "POST", url, json=openai_req, headers=headers
        )
        resp = await http_client.send(req, stream=True)

        if resp.status_code != 200:
            error_body = await resp.aread()
            await resp.aclose()
            return _translate_error(resp.status_code, error_body)

        input_tokens = 0

        async def generate():
            try:
                async for chunk in convert_stream(
                    resp.aiter_bytes(), original_model, input_tokens, thinking_field
                ):
                    yield chunk
            except Exception as e:
                logger.error("Stream error: %s", e)
                error_event = f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
                yield error_event
            finally:
                await resp.aclose()
                elapsed = time.time() - start
                logger.info(
                    "Stream completed: model=%s elapsed=%.2fs",
                    original_model,
                    elapsed,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except httpx.ConnectError as e:
        logger.error("Backend connection failed: %s", e)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "proxy_error",
                    "message": f"Failed to connect to backend: {e}",
                },
            },
        )


async def _handle_non_stream(url, headers, openai_req, original_model, thinking_field, start):
    try:
        assert http_client is not None
        resp = await http_client.post(url, json=openai_req, headers=headers)

        if resp.status_code != 200:
            return _translate_error(resp.status_code, resp.content)

        openai_resp = resp.json()
        anthropic_resp = translate_response(openai_resp, original_model, thinking_field)

        if config.logging.log_responses:
            logger.info(
                "Response: %s",
                json.dumps(anthropic_resp, ensure_ascii=False)[:2000],
            )

        elapsed = time.time() - start
        logger.info(
            "Request completed: model=%s elapsed=%.2fs",
            original_model,
            elapsed,
        )

        return JSONResponse(content=anthropic_resp)

    except httpx.ConnectError as e:
        logger.error("Backend connection failed: %s", e)
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "proxy_error",
                    "message": f"Failed to connect to backend: {e}",
                },
            },
        )


@app.get("/v1/models")
async def models(request: Request):
    backend_url = get_backend_url(config)
    url = f"{backend_url}/models"
    auth_headers = translate_auth(dict(request.headers), config)

    try:
        assert http_client is not None
        resp = await http_client.get(url, headers=auth_headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError as e:
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "proxy_error",
                    "message": f"Failed to connect to backend: {e}",
                },
            },
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


def _translate_error(status_code: int, body: bytes) -> JSONResponse:
    try:
        error_data = json.loads(body)
    except json.JSONDecodeError:
        error_data = {"error": {"message": body.decode("utf-8", errors="replace")}}

    error_type_map = {
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        400: "invalid_request_error",
        500: "api_error",
    }

    openai_error = error_data.get("error", {})
    if isinstance(openai_error, str):
        openai_error = {"message": openai_error}

    anthropic_type = error_type_map.get(status_code, "api_error")

    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": anthropic_type,
                "message": openai_error.get("message", str(error_data)),
            },
        },
    )
