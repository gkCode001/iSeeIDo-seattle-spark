# CLAUDE.md

Local video intelligence on a single DGX Spark. One camera, two surfaces over one
shared index: an **ask** agent (question → instant answer, or escalate to a worker
that re-watches the footage) and a **watch** monitor (standing tasks → MCP actions).

Full design in `SPEC.md`. Read §2–§3 before touching ingest or the schema, and §11
before touching the UI.

---

## Hard invariants

Violating any of these breaks the system in ways that are slow to diagnose. If a task
seems to require breaking one, stop and ask.

1. **One VLM process, ever.** Memory is unified — 128 GB shared between CPU and GPU,
   no VRAM to spill into. A second model instance OOMs the box. All VLM access goes
   through `shared/vlm_client.py` and `shared/queue.py`. Never call the inference
   endpoint directly from a service.

2. **Every chunk carries absolute wall-clock time.** `t_start`/`t_end` in UTC, plus
   `segment` and `pts_offset`. PTS restarts at zero every segment file — it is
   meaningless on its own. This tuple is the only join between a text hit and the
   pixels; without it the deep worker cannot exist.

3. **Fetching video takes a time range, never a filename.** An event can span two
   segment files. Always go through `shared/timecode.py`.

4. **Never block a user turn on `deep_analyze`.** Answer provisionally, return a
   `job_id`, stream the refinement over WebSocket. **The refinement is appended, never
   substituted** — on both surfaces (§4.3, §6.3). Rendering it in place is the obvious
   implementation and it hides the escalation, which is the one thing the demo exists
   to show.

5. **MCP actions go through cooldown + time-range dedupe + the append-only log.**
   No exceptions, no direct calls to the action server. Actions cannot be un-fired.

6. **Live path: `enable_reasoning=false`, `max_tokens=80`.** Deep path:
   `enable_reasoning=true`, `max_tokens≈600`. Decode is ~95% of latency and is
   bandwidth-bound; output token count is the main dial. Do not "improve" captions by
   letting them get longer.

7. **The archive stays at native resolution.** Downscaling only ever applies to the
   live analysis path. The archive is what the worker re-reads.

8. **Burn the wall-clock overlay *after* any resize.** The VLM reads it for temporal
   localization. If it becomes illegible, localization fails silently.

9. **Ingest APIs take a list of chunks**, even though we always pass one. Batch
   dimension now = config change later; single-chunk signature = refactor.

10. **The UI ships no remote assets.** Vendor everything. The demo is rehearsed and run
    with the network off — a CDN link works on every laptop you test on and fails on
    stage.

---

## Platform gotchas

- **ARM64 + sm_121.** Not sm_100 (datacenter Blackwell), not sm_120 (RTX 50xx).
  Prebuilt wheels and containers frequently lack sm_121 support. Verify a package has
  an ARM64 build *before* adding it.
- **Do not add dependencies without asking.** Dependency availability is the most
  common way this box eats an afternoon.
- **Bandwidth, not FLOPs, is the constraint.** 273 GB/s shared. When something is
  slow, the question is how many bytes are being read per token, not how much compute
  is idle.
- **Batching does not help us.** One camera = one request in flight. Optimizations
  should reduce work per chunk, not parallelize.

### Machine state — audited 2026-08-15

The dev box *is* the target hardware, not a stand-in. Confirmed: NVIDIA GB10,
`compute_cap 12.1` (**sm_121**), aarch64, 127.6 GB unified, 3.3 TB free on `/`, driver
580.173.02, CUDA 13.0 (nvcc 13.0.88), Ubuntu 24.04.4. A working `torch 2.11.0+cu130`
aarch64 build already exists in a local venv — CUDA 13 PyTorch on this arch is less of
a problem than it looks.

Not yet set up, in the order they bite:

| Gap | Fix | Gates |
|---|---|---|
| User not in `docker` group (daemon is running) | `sudo usermod -aG docker $USER`, re-login | everything containerized |
| No `/etc/docker/daemon.json` — nvidia runtime likely unregistered (toolkit 1.19.1 *is* installed) | `sudo nvidia-ctk runtime configure --runtime=docker` | GPU inside containers |
| ~~No NGC credentials~~ — **no longer blocking** | none | D1/D3 resolved without NGC; see "Models" below |
| ~~`ffmpeg` absent~~ — **installed 2026-08-15**, 6.1.1 | none | — |
| DeepStream absent, no NVIDIA GStreamer plugins | verify sm_121 support before committing; a TensorRT or frame-diff gate preserves the architecture | the detector gate (§2.3) |

HuggingFace and PyPI are reachable, so Cosmos weights may be pullable outside NGC — the
NIM containers are not.

**ffmpeg turned out better than the package metadata suggested.** The stock apt 6.1.1
build has `cuda` hwaccel (verified initialising on this box), `h264_cuvid`/`av1_cuvid`
NVDEC decoders, `h264_nvenc` (verified encoding on GB10), and v4l2 capture. No CUDA
rebuild is needed for §2.4.

### Models — D1 and D3, resolved 2026-08-15

**One model, one process, no NGC, no docker.** `gemma-4-E2B-it` (Q4_K_XL + mmproj-F16,
~4 GB) serves the live captioner, the deep worker AND the ask LLM. It was already in the
HuggingFace cache. Start it with `make serve` (`scripts/serve_models.sh`).

Served by the **CUDA-13 ARM64 `llama-server` bundled inside LM Studio**
(`~/.lmstudio/extensions/backends/llama.cpp-linux-arm64-nvidia-cuda13-*/`). That prebuilt
sm_121 binary is why none of the NGC/docker path is needed.

Measured on this box, real 1080p webcam footage:

| | |
|---|---|
| Caption, 5 frames | **2.55 s cold, 1.17 s warm** (budget 4 s, SPEC target ~2 s) |
| Decode floor | ~31 tok/s |
| Gate skip rate | **78%** of judged windows |
| Deep analysis | **8.6 s** end to end (target 20–60 s) |
| Reads the burned clock? | **Yes** — captions cite `2026-08-15 09:21:13 UTC` unprompted |

**`--reasoning off` is load-bearing, not a preference.** This is a thinking model: left
on, it spends the entire 80-token live budget inside `reasoning_content` and returns an
**empty caption** with `finish_reason=length`. That is invariant 6 as a server flag —
`enable_reasoning: false` in settings.yaml is NIM/Cosmos vocabulary that llama-server does
not speak, so the switch must be made at launch. `scripts/serve_models.sh` sets it.

Bigger local vision models (Qwen3.8-27B, Ornith-35B, Muse-Glimmer-30B, 15–30 GB) are
4–8× slower per token and cannot hold a 4 s stride. Keep them as deep-path fallbacks.

**Camera: a USB webcam is the demo source (D2 resolved).** Attached and verified
end-to-end 2026-08-15. `/dev/video0` is the capture node (`/dev/video1` exposes no
formats — metadata only). Access is via a per-user ACL, not the `video` group, so `id`
looking wrong is not a problem.

The camera offers **mjpeg only** — 1920x1080, 1080x1920, 3840x2160, 1728x3072 — plus an
HEVC mode this ffmpeg build cannot take over v4l2. **There is no raw format and no
1280x720**, so `input_format: mjpeg` is mandatory rather than optional, and a
plausible-looking resolution will be silently substituted. Re-run this after any camera
change:

```bash
ffmpeg -f v4l2 -list_formats all -i /dev/video0
```

Measured archive: h264 1920x1080@30 via `h264_nvenc`, ~2.1 Mbps → **~22 GB/day**, half
of SPEC §2.1's 43 GB/day budget. 3840x2160 is available if the deep worker ever wants
more detail to find.

**Stop the recorder with SIGTERM, never SIGKILL.** Verified: SIGTERM lets ffmpeg write
the mp4 moov atom and the final partial segment stays playable. A hard kill leaves the
open segment corrupt (`moov atom not found`) — the last minute of footage, which at hour
39 is the minute that matters.

The cost is larger than one file. A corrupt segment is a normal size on disk and passes
any "is it big enough" check, but **every analysis window overlapping it fails to
decode** — one hard-killed 60 s segment produced 15 undecodable windows out of 24 in an
end-to-end run. M1 counts these as `decode_failures` rather than skips, which is what
keeps the gate's skip rate honest; measure the rate over windows the gate actually
judged, never over all windows.

**Keyframe interval is correctness, not quality.** A stream-copied cut can only start on
a keyframe, so the GOP bounds how accurately *any* time-range fetch lands — evidence
clips and the deep worker's seek both. Measured here: an 8.3 s GOP served 8.07 s for a
2.0 s request, starting 6 s early; 1 s served 2.1 s. Hence
`recorder.device.keyframe_interval_seconds: 1.0` and `-g` on the capture path. Raising it
silently degrades `pts_offset` precision, which is invariant 2's whole point.

Beware local model runners: LM Studio and `unsloth studio` (port 8888) load models into
the same unified memory and will silently eat the budget in §7.1. Shut them down before
ingest — see invariant 1.

---

## Conventions

- Python 3.11+, `ruff` for lint/format, type hints on anything crossing a module
  boundary.
- All tunable numbers (windows, strides, thresholds, model names) live in
  `config/settings.yaml`. **No magic numbers in service code.**
- `shared/schema.py` is the single source of truth for `ChunkRecord`, `Task`,
  `ActionLog`. Change it there and nowhere else.
- Structured logging. Every VLM call logs model, profile, token counts, wall time —
  we cannot tune what we cannot see.
- Timestamps are UTC, ISO 8601, `Z` suffix. Local time only at the UI layer — one
  conversion helper, called at render, nowhere else.
- **The action log is the only history store.** M3 reads it via `read_action_log`; the
  Timeline pane renders the same rows. Never give the UI a parallel store — two sources
  drift, and the drift surfaces as the agent contradicting the screen.
- Chat history is append-only JSONL keyed by turn, and persists the *job*, not just the
  message text. Refinements land after the turn ends and must survive a reload.

## Testing

- `shared/timecode.py` needs real tests: boundary-spanning ranges, segment gaps,
  clock drift, DST-adjacent local conversions. This module is load-bearing.
- Mock the VLM in unit tests. Never call the real endpoint from `pytest` — it
  contends with ingest.
- Keep a 2-minute fixture clip in `tests/fixtures/` with a known event at a known
  timestamp, and assert retrieval returns the right range.

---

## Commands

```bash
make up            # docker compose: milvus, vllm, nim, services
make ingest        # run M1 against the configured source
make bench         # time a single caption — the number that governs everything
make test
make lint
```

---

## Working style here

- **Benchmark before optimizing.** The block-0 caption benchmark decides the model
  choice (D1 in SPEC §10). Don't guess at it.
- Prefer deleting work over parallelizing it. The detector gate skipping 80% of
  windows is worth more than any inference tuning.
- This is a 40-hour hackathon build. When a choice is between "correct" and
  "shippable", say so explicitly and let a human decide — but keep the invariants
  above regardless, because those are what break demos.
- Open decisions are in `SPEC.md` §10. If a task depends on one that's unresolved,
  flag it rather than picking silently.
