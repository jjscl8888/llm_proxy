#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-$PROJECT_DIR/config.yaml}"

cd "$PROJECT_DIR"

uv run python -m llm_proxy --config "$CONFIG" &
PROXY_PID=$!

sleep 2

if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo "Failed to start LLM Proxy"
    exit 1
fi

echo "LLM Proxy started (PID: $PROXY_PID)"
echo ""
echo "To use with Claude Code CLI, run:"
echo ""
echo "  export ANTHROPIC_BASE_URL=http://127.0.0.1:8082"
echo "  export ANTHROPIC_API_KEY=sk-proxy-pass-through"
echo "  claude"
echo ""

wait $PROXY_PID
