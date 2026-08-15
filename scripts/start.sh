#!/usr/bin/env bash
# Bring the whole system up on a DGX Spark, in dependency order, and wait for each piece
# to actually answer before starting the next.
#
#   model server  ->  recorder  ->  ingest  ->  agent + UI
#
# Anything missing that CAN be fetched safely is fetched (the ~4 GB model). Anything that
# needs a human — a package install, a camera, LM Studio — is reported with the exact
# command to fix it, and we stop rather than limp along in a state that looks fine and
# is not.
#
#   ./scripts/start.sh              # everything
#   ./scripts/start.sh --no-record  # reuse whatever is already in data/archive
#   ./scripts/stop.sh               # shut down cleanly (SIGTERM — see below)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

RUN_DIR="${SPARK_RUN_DIR:-$ROOT/.run}"
LOG_DIR="${SPARK_LOG_DIR:-$ROOT/.run/logs}"
mkdir -p "$RUN_DIR" "$LOG_DIR"

RECORD=1
[ "${1:-}" = "--no-record" ] && RECORD=0

PY="${PY:-python3}"
VLM_PORT="$($PY -c 'from shared.config import get; print(get("vlm.endpoint").rsplit(":",1)[1].split("/")[0])' 2>/dev/null || echo 8000)"
UI_PORT="$($PY -c 'from shared.config import get; print(get("agent.port"))' 2>/dev/null || echo 8080)"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
step()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()   { red "$*"; exit 1; }

# --------------------------------------------------------------------------------------
# 0. Preflight — everything a human must fix, reported together rather than one per run
# --------------------------------------------------------------------------------------
step "Preflight"
MISSING=0

$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || { red "  python >= 3.11 required (have $($PY -V 2>&1))"; MISSING=1; }
$PY -c 'import yaml' 2>/dev/null \
    || { red "  PyYAML missing            fix: pip3 install --user pyyaml"; MISSING=1; }

command -v ffmpeg >/dev/null \
    || { red "  ffmpeg missing            fix: sudo apt install ffmpeg"; MISSING=1; }
command -v ffprobe >/dev/null \
    || { red "  ffprobe missing           fix: sudo apt install ffmpeg"; MISSING=1; }

SOURCE="$($PY -c 'from shared.config import get; print(get("recorder.source") or "")' 2>/dev/null || echo "")"
if [ "$RECORD" = 1 ]; then
    case "$SOURCE" in
        /dev/video*)
            # Absent and unreadable are different problems with different fixes, so
            # report exactly one of them rather than both for the same device.
            if [ ! -e "$SOURCE" ]; then
                red "  camera $SOURCE not present   fix: plug the webcam in, or run --no-record"; MISSING=1
            elif [ ! -r "$SOURCE" ]; then
                red "  camera $SOURCE not readable  fix: sudo usermod -aG video \$USER, then re-login"; MISSING=1
            else
                green "  camera $SOURCE"
            fi
            ;;
        "") red "  recorder.source is unset  fix: set it in config/settings.yaml"; MISSING=1 ;;
    esac
fi

# nvidia-smi is informational: the model runs on CPU too, just far slower.
if command -v nvidia-smi >/dev/null; then
    CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    [ "$CAP" = "12.1" ] && green "  GPU sm_121 (GB10)" \
                        || printf '  GPU compute_cap %s (expected 12.1 on a DGX Spark)\n' "${CAP:-unknown}"
else
    printf '  nvidia-smi not found — the model will fall back to CPU and be very slow\n'
fi

[ "$MISSING" = 0 ] || die "
Preflight failed. Fix the lines above and re-run. Nothing was started."
green "  preflight ok"

# --------------------------------------------------------------------------------------
# 1. Model server — fetches the ~4 GB model on first run
# --------------------------------------------------------------------------------------
step "Model server (port $VLM_PORT)"
if curl -sf -o /dev/null "http://127.0.0.1:$VLM_PORT/v1/models" 2>/dev/null; then
    green "  already serving — reusing it"
else
    nohup "$HERE/serve_models.sh" >"$LOG_DIR/model.log" 2>&1 &
    echo $! >"$RUN_DIR/model.pid"
    printf '  loading'
    for _ in $(seq 1 120); do
        curl -sf -o /dev/null "http://127.0.0.1:$VLM_PORT/v1/models" 2>/dev/null && break
        kill -0 "$(cat "$RUN_DIR/model.pid")" 2>/dev/null || { echo; tail -20 "$LOG_DIR/model.log"; die "  model server exited — see $LOG_DIR/model.log"; }
        printf '.'; sleep 2
    done
    echo
    curl -sf -o /dev/null "http://127.0.0.1:$VLM_PORT/v1/models" || die "  model server did not come up — see $LOG_DIR/model.log"
    green "  serving"
fi

# --------------------------------------------------------------------------------------
# 2. Recorder — writes data/archive continuously, independent of any AI (SPEC §2.1)
# --------------------------------------------------------------------------------------
if [ "$RECORD" = 1 ]; then
    step "Recorder ($SOURCE)"
    if pgrep -f "services\.recorder" >/dev/null; then
        green "  already recording"
    else
        nohup $PY -m services.recorder >"$LOG_DIR/recorder.log" 2>&1 &
        echo $! >"$RUN_DIR/recorder.pid"
        sleep 3
        kill -0 "$(cat "$RUN_DIR/recorder.pid")" 2>/dev/null \
            || { tail -20 "$LOG_DIR/recorder.log"; die "  recorder exited — see $LOG_DIR/recorder.log"; }
        green "  recording to data/archive"
    fi
else
    step "Recorder"; printf '  skipped (--no-record); using whatever is in data/archive\n'
fi

# --------------------------------------------------------------------------------------
# 3. Ingest — gate + caption + index (SPEC §2.2-§2.4)
# --------------------------------------------------------------------------------------
step "Ingest (gate + caption)"
if pgrep -f "services\.ingest" >/dev/null; then
    green "  already running"
else
    nohup $PY -m services.ingest --follow >"$LOG_DIR/ingest.log" 2>&1 &
    echo $! >"$RUN_DIR/ingest.pid"
    sleep 3
    kill -0 "$(cat "$RUN_DIR/ingest.pid")" 2>/dev/null \
        || { tail -20 "$LOG_DIR/ingest.log"; die "  ingest exited — see $LOG_DIR/ingest.log"; }
    green "  following the archive"
fi

# --------------------------------------------------------------------------------------
# 4. Agent + UI
# --------------------------------------------------------------------------------------
step "Ask agent + UI (port $UI_PORT)"
if curl -sf -o /dev/null "http://127.0.0.1:$UI_PORT/api/config" 2>/dev/null; then
    green "  already serving"
else
    nohup $PY -m services.agent >"$LOG_DIR/agent.log" 2>&1 &
    echo $! >"$RUN_DIR/agent.pid"
    for _ in $(seq 1 40); do
        curl -sf -o /dev/null "http://127.0.0.1:$UI_PORT/api/config" 2>/dev/null && break
        sleep 1
    done
    curl -sf -o /dev/null "http://127.0.0.1:$UI_PORT/api/config" \
        || { tail -20 "$LOG_DIR/agent.log"; die "  agent did not come up — see $LOG_DIR/agent.log"; }
    green "  serving"
fi

cat <<EOF

$(green "Running.")

  UI            http://127.0.0.1:$UI_PORT/?mode=live
                (?mode=live matters — the page defaults to mock fixtures)

  logs          $LOG_DIR/{model,recorder,ingest,agent}.log
  stop          ./scripts/stop.sh

Captions take a few window-strides to appear; until then the index is empty and
questions will honestly answer "nothing indexed covers that".
EOF
