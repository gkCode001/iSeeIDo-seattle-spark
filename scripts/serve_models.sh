#!/usr/bin/env bash
# Start the one model process the whole system talks to (CLAUDE.md invariant 1).
#
# SPEC §10 D1 and D3 both resolve to gemma-4-E2B-it (Q4_K_XL + mmproj-F16, ~4 GB): it is
# multimodal, so ONE process serves the live captioner, the deep worker AND the ask LLM.
# That is stricter than SPEC §7's two-process design, not looser — a second model
# instance is what OOMs this box, and one process makes that impossible.
#
# The model is fetched by scripts/fetch_models.sh if it is not already on disk.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${SPARK_VLM_PORT:-8000}"

# --- the server binary ----------------------------------------------------------------
# llama.cpp built for ARM64 + CUDA 13 (sm_121). LM Studio ships one, which is why this
# project needs no NGC account, no docker and no compile step. A standalone llama-server
# on PATH is used if present.
find_server() {
    if [ -n "${SPARK_LLAMA_SERVER:-}" ]; then printf '%s\n' "$SPARK_LLAMA_SERVER"; return; fi
    local cuda
    cuda="$(ls -d "$HOME"/.lmstudio/extensions/backends/llama.cpp-linux-arm64-nvidia-cuda*/ 2>/dev/null | sort -V | tail -1)"
    if [ -n "$cuda" ] && [ -x "${cuda}llama-server" ]; then printf '%s\n' "${cuda}llama-server"; return; fi
    local plain
    plain="$(ls -d "$HOME"/.lmstudio/extensions/backends/llama.cpp-linux-arm64-*/ 2>/dev/null | sort -V | tail -1)"
    if [ -n "$plain" ] && [ -x "${plain}llama-server" ]; then printf '%s\n' "${plain}llama-server"; return; fi
    command -v llama-server 2>/dev/null || true
}

SERVER="$(find_server)"
if [ -z "$SERVER" ]; then
    cat >&2 <<'EOF'
No llama-server found.

This project serves its model with llama.cpp built for ARM64 + CUDA (sm_121). The
easiest source on a DGX Spark is LM Studio, which ships exactly that binary:

    1. Install LM Studio        https://lmstudio.ai/download   (ARM64 Linux)
    2. Open it once and let it install its llama.cpp backend
    3. Re-run this script

Already have a llama-server elsewhere?  SPARK_LLAMA_SERVER=/path/to/llama-server
EOF
    exit 1
fi

MODEL_DIR="$("$HERE/fetch_models.sh")"
WEIGHTS="${SPARK_MODEL_WEIGHTS:-gemma-4-E2B-it-UD-Q4_K_XL.gguf}"
MMPROJ="${SPARK_MODEL_MMPROJ:-mmproj-F16.gguf}"

echo "  server : $SERVER" >&2
echo "  model  : $MODEL_DIR/$WEIGHTS" >&2
echo "  port   : $PORT" >&2

# Context size is sized from a MEASUREMENT, not a guess: a native-resolution 1080p frame
# costs ~261 prompt tokens through this model's vision encoder, so the deep path's 4 fps
# (SPEC §5) buys ~30 s of footage inside 32k. At the llama.cpp default of 8192 a deep
# request of only 10 s returns HTTP 400 "exceeds the available context size" — which
# surfaces in the UI as a deep job that never completes.
#
# --reasoning off is LOAD-BEARING, not a preference. This is a thinking model: left on,
# it spends the entire 80-token live budget inside reasoning_content and returns an EMPTY
# caption with finish_reason=length. That is CLAUDE.md invariant 6 expressed as a server
# flag — `enable_reasoning: false` in settings.yaml is NIM/Cosmos vocabulary that
# llama-server does not speak, so the switch has to be made here.
exec "$SERVER" \
    -m "$MODEL_DIR/$WEIGHTS" \
    --mmproj "$MODEL_DIR/$MMPROJ" \
    --host 127.0.0.1 --port "$PORT" \
    -c "${SPARK_CTX:-32768}" -ngl 99 --parallel 1 \
    --reasoning off --reasoning-budget 0
