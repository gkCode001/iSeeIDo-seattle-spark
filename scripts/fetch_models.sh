#!/usr/bin/env bash
# Ensure the model this system needs is on disk, downloading it only if it is not.
#
# SPEC §10 D1/D3 resolve to gemma-4-E2B-it (Q4_K_XL + mmproj-F16, ~4 GB). One model
# serves the live captioner, the deep worker and the ask LLM — see CLAUDE.md invariant 1.
#
# Search order, so an existing copy is never re-downloaded:
#   1. $SPARK_MODEL_DIR                       (explicit override)
#   2. the HuggingFace cache                  (where it already is on this box)
#   3. ~/.cache/spark-vision/models           (where we put it if we fetch it)
#
# Prints the resolved directory on stdout. Everything else goes to stderr so callers
# can do:  MODEL_DIR="$(scripts/fetch_models.sh)"
set -euo pipefail

REPO="${SPARK_MODEL_REPO:-unsloth/gemma-4-E2B-it-GGUF}"
WEIGHTS="${SPARK_MODEL_WEIGHTS:-gemma-4-E2B-it-UD-Q4_K_XL.gguf}"
MMPROJ="${SPARK_MODEL_MMPROJ:-mmproj-F16.gguf}"
FALLBACK_DIR="${SPARK_MODEL_DIR:-$HOME/.cache/spark-vision/models}"
MIN_BYTES=$((500 * 1024 * 1024))   # a real weights file is GBs; anything less is a stub

log() { printf '  %s\n' "$*" >&2; }

have_both() {
    [ -f "$1/$WEIGHTS" ] && [ -f "$1/$MMPROJ" ] &&
    [ "$(stat -Lc%s "$1/$WEIGHTS" 2>/dev/null || echo 0)" -ge "$MIN_BYTES" ]
}

# 1 + 2 — already on disk?
for dir in "${SPARK_MODEL_DIR:-}" \
           $(ls -d "$HOME"/.cache/huggingface/hub/models--"${REPO/\//--}"/snapshots/*/ 2>/dev/null) \
           "$FALLBACK_DIR"; do
    [ -n "$dir" ] || continue
    if have_both "$dir"; then
        log "model already present: $dir"
        printf '%s\n' "${dir%/}"
        exit 0
    fi
done

# 3 — fetch. Public repo, so no token and no NGC account is involved.
log "model not found locally — downloading ~4 GB from huggingface.co/$REPO"
log "(one time; set SPARK_MODEL_DIR to reuse an existing copy instead)"
mkdir -p "$FALLBACK_DIR"
for f in "$WEIGHTS" "$MMPROJ"; do
    if [ -f "$FALLBACK_DIR/$f" ] && [ "$(stat -Lc%s "$FALLBACK_DIR/$f")" -ge "$MIN_BYTES" ]; then
        log "  have $f"; continue
    fi
    log "  fetching $f"
    # -C - resumes a partial file; --fail turns an HTML error page into a non-zero exit
    # rather than a "model" that is really a 404 page.
    curl -fL -C - --retry 3 --retry-delay 2 \
         -o "$FALLBACK_DIR/$f" \
         "https://huggingface.co/$REPO/resolve/main/$f" >&2
done

have_both "$FALLBACK_DIR" || { log "download finished but files look wrong"; exit 1; }
log "model ready: $FALLBACK_DIR"
printf '%s\n' "$FALLBACK_DIR"
