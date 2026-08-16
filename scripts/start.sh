#!/usr/bin/env bash
# Bring the whole system up on a DGX Spark, in dependency order, and wait for each piece
# to actually answer before starting the next.
#
#   model servers (checked, not started)  ->  recorder  ->  ingest  ->  agent + UI
#
# Nothing here starts a model. The two Nemotron servers are this box's own containers
# (DEPLOY_GN100.md §2) and a second engine OOMs the box, so a missing one is reported
# with the command that fixes it. Same for anything else a human must resolve — a
# package, a camera. We stop rather than limp along in a state that looks fine and is
# not.
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
VLM_PORT="$($PY -c 'from shared.config import get; print(get("vlm.endpoint").rsplit(":",1)[1].split("/")[0])' 2>/dev/null || echo 8082)"
LLM_PORT="$($PY -c 'from shared.config import get; print(get("agent.endpoint").rsplit(":",1)[1].split("/")[0])' 2>/dev/null || echo 8000)"
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
# 1. Model servers — checked, never started
# --------------------------------------------------------------------------------------
# This script used to fall back to downloading gemma-4-E2B-it (~4 GB) and launching its
# own llama-server whenever the VLM port did not answer. On gn100-2f74 that fallback was
# a trap in three ways, and it would only ever spring at the worst moment — while the
# real model server was restarting or wedged:
#
#   1. It starts a SECOND model process. 121 GB is shared between CPU and GPU, the two
#      Nemotron containers already hold ~80 GB of it, and a third engine is what OOMs
#      the box. CLAUDE.md invariant 1 exists because of exactly this.
#   2. It serves the WRONG MODEL. Every measurement in DEPLOY_GN100.md is Nemotron; a
#      demo silently answering from gemma would look like it worked.
#   3. It downloads 4 GB, on a box whose demo is rehearsed with the network off.
#
# So: report and stop. Both servers belong to this machine's own provisioning and are
# started by systemd and docker, never by us.
step "Model servers (VLM :$VLM_PORT, ask LLM :$LLM_PORT)"
MODELS_DOWN=0
curl -sf -o /dev/null "http://127.0.0.1:$VLM_PORT/v1/models" 2>/dev/null \
    || { red "  VLM not answering on :$VLM_PORT      fix: sudo systemctl start gn100-vlm"; MODELS_DOWN=1; }
curl -sf -o /dev/null "http://127.0.0.1:$LLM_PORT/v1/models" 2>/dev/null \
    || { red "  ask LLM not answering on :$LLM_PORT   fix: docker start nemoclaw-vllm"; MODELS_DOWN=1; }
if [ "$MODELS_DOWN" = 1 ]; then
    die "
The model servers are this box's own containers; this script does not start them and
must not start one of its own. Bring the missing one up, WARM IT (the VL's first request
after a restart takes ~26 s), then re-run.

Order matters: the VL container joins nemoclaw-vllm's network namespace, so touching
nemoclaw-vllm kills the VL's networking. Always:

    sudo systemctl stop gn100-vlm  ->  fix nemoclaw-vllm  ->  wait for :$LLM_PORT
                                   ->  sudo systemctl start gn100-vlm

Nothing was started."
fi
green "  both serving — this script starts no model process (invariant 1)"

# --------------------------------------------------------------------------------------
# 2. AlertBridge — the credential notify_discord needs, which this repo does not hold
# --------------------------------------------------------------------------------------
# `notify_discord` posts through AlertBridge, a separate service that owns the Discord
# webhook. Nothing here or in git holds a webhook URL or a token: alertbridge.py reads
# ALERTBRIDGE_SERVICE_TOKEN from the process environment and nowhere else, deliberately.
# This block's whole job is to make sure the environment HAS it before the agent is
# launched — an exported token is inherited by the agent, and by M5 running inside it.
#
# Getting this wrong is quiet in both directions, which is why it is checked here with
# everything else a human must fix rather than discovered mid-demo:
#
#   * At task-creation time the form rejects notify_discord outright.
#   * At fire time a send failure writes NO action-log row (by design — the event must
#     stay retryable), so a missing token looks exactly like nothing ever matching.
#
# Precedence is deliberate: an already-exported token wins, so a rotated credential or a
# different tailnet needs no edit here.
step "AlertBridge (notify_discord)"
AB_URL="$($PY -c 'from shared.config import get; print(get("mcp.discord.base_url"))' 2>/dev/null || echo "")"
AB_SOURCE=""
if [ -n "${ALERTBRIDGE_SERVICE_TOKEN:-}" ]; then
    AB_SOURCE="the environment"
else
    for AB_ENV in "${SPARK_ALERTBRIDGE_ENV:-}" "$ROOT/.env.local" /opt/alertbridge/.env; do
        [ -n "$AB_ENV" ] && [ -r "$AB_ENV" ] || continue
        # Only this one key is read. Anchored on '=' so the audit and HMAC tokens
        # sitting next to it in the same file cannot be picked up by mistake.
        AB_TOKEN="$(sed -n 's/^ALERTBRIDGE_SERVICE_TOKEN=//p' "$AB_ENV" | head -1 \
                    | sed -e 's/^["'"'"']//' -e 's/["'"'"']$//')"
        if [ -n "$AB_TOKEN" ]; then
            export ALERTBRIDGE_SERVICE_TOKEN="$AB_TOKEN"
            AB_SOURCE="$AB_ENV"
            unset AB_TOKEN
            break
        fi
    done
fi

# Does anything actually REQUIRE it? A seed task that notifies Discord with no token is
# the bad case: build_monitor() refuses, the agent catches that and comes up with M5
# disabled entirely, and every OTHER standing task stops being evaluated too — announced
# by one line in agent.log. Fail here instead, where the fix is on screen.
SEED_NEEDS_DISCORD="$($PY - <<'PY' 2>/dev/null || echo 0
import yaml
from shared import config

try:
    doc = yaml.safe_load(config.repo_path("monitor.tasks_file").read_text()) or {}
except Exception:
    print(0)
else:
    tasks = doc.get("tasks") or []
    print(int(any(
        (t or {}).get("action") == "notify_discord" and (t or {}).get("enabled", True)
        for t in tasks
    )))
PY
)"

if [ -n "$AB_SOURCE" ]; then
    green "  token loaded from $AB_SOURCE"
    # Unauthenticated liveness probe — sends nothing, posts nothing.
    curl -sf -o /dev/null "$AB_URL/health/live" 2>/dev/null \
        && green "  $AB_URL reachable" \
        || red "  $AB_URL not answering — notify_discord will fail at fire time"
elif [ "$SEED_NEEDS_DISCORD" = 1 ]; then
    die "
A task in config/tasks.yaml uses notify_discord and no ALERTBRIDGE_SERVICE_TOKEN was
found. Starting anyway would bring the agent up with M5 disabled entirely — every
standing task silently unevaluated, not just that one.

    export ALERTBRIDGE_SERVICE_TOKEN=...        # or put it in $ROOT/.env.local
    curl -fsS $AB_URL/health/live               # check the service first

Nothing was started."
else
    printf '  no token found — notify_discord tasks will be refused at creation\n'
    printf '  fix: export ALERTBRIDGE_SERVICE_TOKEN=... or put it in %s\n' "$ROOT/.env.local"
    printf '  every other action (save_clip, raise_alert, file_ticket) is unaffected\n'
fi

# --------------------------------------------------------------------------------------
# 3. Recorder — writes data/archive continuously, independent of any AI (SPEC §2.1)
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
# 4. Ingest — gate + caption + index (SPEC §2.2-§2.4)
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
# 5. Agent + UI
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

  UI            http://127.0.0.1:$UI_PORT/
                (detects M3 and runs live; add ?mode=mock for the fixtures)

  logs          $LOG_DIR/{model,recorder,ingest,agent}.log
  stop          ./scripts/stop.sh

Captions take a few window-strides to appear; until then the index is empty and
questions will honestly answer "nothing indexed covers that".
EOF
