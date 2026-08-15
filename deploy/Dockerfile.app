# Image for the Python app services: recorder, ingest, agent, worker, monitor.
#
# NEVER BUILT. Nothing in this repo has run a container as of 2026-08-15 — the user is
# not in the `docker` group. Treat every line below as a proposal that has not compiled.
#
# Design notes:
#
# * No source is COPYed in. docker-compose.yml bind-mounts shared/, services/, scripts/
#   and config/ instead, so a code edit during a 40-hour build does not cost a rebuild.
#   Only the interpreter and its dependencies are baked.
#
# * Dependencies are exactly what pyproject.toml declares as required — pyyaml and
#   requests. The `serve` (fastapi, uvicorn, websockets) and `index` (pymilvus, numpy)
#   extras are marked "pending approval" in pyproject.toml and are NOT installed here.
#   CLAUDE.md: do not add dependencies without asking, and verify an aarch64 build exists
#   first. When M3 needs uvicorn, that is a decision for pyproject.toml's owner; this
#   file follows it rather than leading it.
#
# * The tests are stdlib `unittest` on purpose, so they need none of the above.

# python:3.11-slim-bookworm publishes a linux/arm64 manifest. That is the one image in
# this build that is not in doubt — but it is still UNPULLED here, so confirm with
#     docker manifest inspect python:3.11-slim-bookworm | grep -i arm64
FROM python:3.11-slim-bookworm

# ffmpeg is required by the recorder (SPEC §2.1) and by services/mcp/clips.py for
# evidence-clip cuts.
#
# CAVEAT, and it matters: this installs Debian bookworm's ffmpeg (5.1.x), NOT the
# ffmpeg 6.1.1 that was actually verified on the host — the one confirmed to have cuda
# hwaccel, h264_cuvid/av1_cuvid, working h264_nvenc on this GB10, and v4l2 capture
# (CLAUDE.md machine state). The container build is UNVERIFIED for all four. nvenc and
# nvdec additionally require the nvidia runtime to inject libnvidia-encode.so.1 /
# libnvcuvid.so.1, which is what NVIDIA_DRIVER_CAPABILITIES=...,video in the compose file
# is for. Until this has been run, the tested recorder path is the host one:
#     python3 -m services.recorder
# Re-verify inside the container with:
#     ffmpeg -hide_banner -hwaccels
#     ffmpeg -hide_banner -encoders | grep nvenc
# and if the Debian build falls short, switch this FROM to an NVIDIA CUDA base image with
# a CUDA-enabled ffmpeg — noting that such a base must itself be checked for sm_121.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ingest.overlay.fontfile in config/settings.yaml points at
# /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf — supplied by fonts-dejavu-core above.
# The burned-in wall-clock overlay is what the VLM reads for temporal localization
# (CLAUDE.md invariant 8); a missing font makes ffmpeg's drawtext fail, so fail the
# BUILD instead of discovering it at 3am.
RUN test -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

# Pinned to the versions already present in the host's system python, so the container
# and the host run the same code paths.
RUN pip install --no-cache-dir "pyyaml>=6.0" "requests>=2.31"

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Runs as root. The recorder needs /dev/video0, which on the host is granted by a per-user
# ACL for uid 1000 rather than the `video` group (CLAUDE.md), so a non-root container user
# would need group_add or its own ACL entry. Root is the shippable answer for a
# single-box demo; note it here rather than pretend otherwise.

# No ENTRYPOINT: docker-compose.yml gives each service its own
# `command: ["python3", "-m", "services.<name>"]`, which keeps the compose file the one
# place the process list is written down.
CMD ["python3", "-c", "import sys; sys.exit('set an explicit command: — see docker-compose.yml')"]
