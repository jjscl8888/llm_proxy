from __future__ import annotations

import argparse
import logging

import uvicorn

from llm_proxy.config import load_config
from llm_proxy.server import set_config


def main():
    parser = argparse.ArgumentParser(description="LLM Proxy - Anthropic to OpenAI protocol translator")
    parser.add_argument("--config", "-c", help="Path to config.yaml", default=None)
    parser.add_argument("--host", help="Host to bind", default=None)
    parser.add_argument("--port", "-p", help="Port to bind", type=int, default=None)
    parser.add_argument("--log-level", help="Log level", choices=["debug", "info", "warn", "error"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_config(cfg)

    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    log_level = (args.log_level or cfg.logging.level).upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger("llm_proxy")
    logger.info("Starting LLM Proxy on %s:%d", host, port)
    logger.info("Backend: %s", cfg.backend.base_url)
    logger.info("Default model mapping: %s", cfg.model_mapping.get("*", "none"))

    uvicorn.run(
        "llm_proxy.server:app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
