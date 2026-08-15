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

KEEP_MODEL=0
[ "${1:-}" = "--keep-model" ] && KEEP_MODEL=1

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
if [ "$KEEP_MODEL" = 1 ]; then
    printf '  %-10s left running (--keep-model)\n' "model"
else
    stop_one model "llama-server" 10
fi

echo
echo "Archive kept in data/archive; index in data/index.jsonl."
