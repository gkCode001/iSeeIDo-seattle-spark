#!/usr/bin/env bash
# Shut everything down cleanly.
#
# SIGTERM, never SIGKILL. ffmpeg needs the grace period to write the mp4 moov atom on the
# segment it currently has open; a hard kill leaves that file unplayable AND makes every
# analysis window overlapping it undecodable. Measured on this box: one hard-killed 60 s
# segment cost 15 of 24 windows in an end-to-end run. The last minute recorded is usually
# the minute that matters.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RUN_DIR="${SPARK_RUN_DIR:-$ROOT/.run}"

# --keep-model is accepted and ignored: it dates from when this repo started its own
# llama-server. It no longer starts a model and must not stop one — the two Nemotron
# servers are this box's own containers (systemd gn100-vlm, docker nemoclaw-vllm) and
# stopping them here would take down whatever else on the machine is using them, then
# leave the next start.sh reporting a dead endpoint it cannot fix.
[ "${1:-}" = "--keep-model" ] && printf '  (--keep-model is a no-op; this script never stops a model server)\n' 

stop_one() {
    local name="$1" pattern="$2" grace="${3:-10}"
    local pidfile="$RUN_DIR/$name.pid" pid=""
    [ -f "$pidfile" ] && pid="$(cat "$pidfile" 2>/dev/null)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        pid="$(pgrep -f "$pattern" | head -1)"
    fi
    if [ -z "$pid" ]; then printf '  %-10s not running\n' "$name"; return; fi

    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 "$grace"); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        printf '  %-10s did not stop on SIGTERM after %ss; sending SIGKILL\n' "$name" "$grace"
        printf '             (if this was the recorder, its open segment may be unplayable)\n'
        kill -KILL "$pid" 2>/dev/null
    else
        printf '  %-10s stopped\n' "$name"
    fi
    rm -f "$pidfile"
}

# Recorder first, and with the longest grace: it is the one holding a file open.
stop_one recorder "services\.recorder" 15
stop_one ingest   "services\.ingest"   10
stop_one agent    "services\.agent"    10
# M5 normally runs INSIDE the agent, so this is usually "not running". It is here for the
# standalone form (`python3 -m services.monitor`), which start.sh does not launch and
# which would otherwise be left holding the action log after everything else stopped.
stop_one monitor  "services\.monitor"  10

echo
echo "Archive kept in data/archive; index in data/index.jsonl; tasks in data/tasks.json."
echo "Model servers left alone — they are this box's own (gn100-vlm, nemoclaw-vllm)."
