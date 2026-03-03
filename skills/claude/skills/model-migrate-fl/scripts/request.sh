#!/bin/bash
# Step 10.2: Send test request to verify serving
# Usage: bash scripts/request.sh <MODEL_DISPLAY_NAME> [PORT]
set -euo pipefail

MODEL="${1:?Usage: request.sh <MODEL_DISPLAY_NAME> [PORT]}"
PORT="${2:-8121}"

curl "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"/models/${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"介绍一下 vLLM 的核心优势\"}],\"max_tokens\":10000}"
