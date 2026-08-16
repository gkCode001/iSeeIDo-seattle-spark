# Two-Speed Vision Agent

A local video intelligence system on a single DGX Spark. One camera. Two surfaces over
one shared understanding of the footage:

- **Ask** — a human asks a question. It is answered instantly from the index, or
  escalated to a worker that re-watches the original footage.
- **Watch** — standing tasks run continuously. When footage matches one, an MCP action
  fires.

The design thesis: *ingest is lossy on purpose, and the system knows when its own summary
isn't good enough.* Everything runs on the box — no cloud, and the demo is rehearsed with
the network off.

Full design in [`SPEC.md`](SPEC.md). Working rules and hard invariants in
[`CLAUDE.md`](CLAUDE.md). Read both before changing anything.

---

## Where this actually is

**The whole chain runs on this box, on live camera footage.** There is no test suite: `tests/` was removed deliberately (see CLAUDE.md). Changes are verified by running the stack and reading `.run/logs/`.

| Module | State |
|---|---|
| `shared/` — schema, timecode, VLM client, priority queue | **Built.** `timecode.py` is load-bearing: boundary-spanning ranges, segment gaps, drift, DST. |
| `services/recorder/` — the ffmpeg segmenter (SPEC §2.1) | **Built.** Verified end-to-end against a USB webcam (2026-08-15) and an iPhone over RTSP (2026-08-16). |
| `services/index/` — M2, the index (SPEC §3) | **Built.** Runs today on the `memory` + `hashing` + `lexical` backends, i.e. with no Milvus and no NGC key. |
| `services/mcp/` — the action server and the three brakes (SPEC §6.4) | **Built.** |
| `ui/` — the single page, three panes (SPEC §11) | **Built.** Renders from fixtures in `ui/mock/`; assets vendored, no CDN. |
| `services/ingest/` — M1, gate + captioning (SPEC §2) | **Built.** Motion gate needs no new dependency: ffmpeg emits 32x32 grayscale thumbnails and the diff is pure Python. Measured **78% skip rate** on real webcam footage (SPEC targets ≥80%). |
| `services/worker/` — M4, the deep worker (SPEC §5) | **Built.** Cuts real evidence clips across segment boundaries. Confidence is a documented *coverage* heuristic, not certainty. |
| `services/agent/` — M3, the ask agent + server (SPEC §4) | **Built.** stdlib `http.server` + a hand-rolled RFC 6455 WebSocket, because fastapi is not installed and was not approved. |
| `services/monitor/` — M5, the standing-task funnel (SPEC §6) | **Built.** The brakes are the point: 180 overlapping chunks across 12 minutes of one event fire **exactly one** action. |
| `docker-compose.yml`, `deploy/` | **Written, never run.** See "The deployment layer has never executed" below. |

**The full chain runs on real models, on real footage.** camera → recorder → M1 → M2 →
M3 → M4 → M5, on video the recorder captured from the attached webcam. Measured on this
box:

| | Result | Budget |
|---|---|---|
| Caption, 5 frames | **2.55 s cold, 1.17 s warm** | 4 s cap, ~2 s target (SPEC §2.4) |
| Detector gate | **78–81% skip rate** | ≥80% (SPEC §2.3) |
| Deep analysis | **2–9 s** | 20–60 s (SPEC §5) |
| Escalation (D6) | grounded ✅ / escalated ✅ on the same footage | the design thesis |

The VLM demonstrably **reads the burned-in wall clock** — captions cite
`2026-08-15 09:21:13 UTC` unprompted, which is what invariant 8 exists to guarantee.

### What is not wired yet

**M5 standing tasks do not fire.** `services/monitor` has no `__main__.py` and its
worker-verifier cannot receive a verdict, so the Watch pane renders tasks and funnel
state but no action fires and no retraction happens. Fix before demoing that pane.

**`docker-compose.yml` has never been run.** It is not needed — see Prerequisites — but
it is also unverified.

### Backends are swappable, by design

`config/settings.yaml` runs `vlm.backend: vllm` and `agent.backend: nim` against the
local `llama-server`, with `index.store.backend: memory`, `index.embed.backend: hashing`
and `index.rerank.backend: lexical`. Each has a stub/in-memory sibling that needs no
model at all, which is how the pipeline was proven before a model was serving — set
`vlm.backend: stub` to get back there. Flip the index backends to `milvus` / `nim` if you
ever stand those containers up.

**`index.store.memory_path` must stay set** while the in-memory backend is in use: M1, M3
and M5 are separate processes, and with it null each gets its own empty corpus — ingest's
captions die with the ingest process and every question retrieves nothing.

---

## Prerequisites

A DGX Spark (GB10, sm_121, ARM64). Everything below runs on the host — **no Docker, no
NGC account, no NVIDIA container**.

| | Why | If missing |
|---|---|---|
| **Python ≥ 3.11** + PyYAML | everything | `pip3 install --user pyyaml` |
| **ffmpeg** (with ffprobe) | recording, frame sampling, clips | `sudo apt install ffmpeg` |
| **LM Studio** | ships the ARM64 + CUDA-13 `llama-server` this project serves its model with | [lmstudio.ai/download](https://lmstudio.ai/download) — install, open once, let it install its llama.cpp backend |
| **A USB webcam** on `/dev/video0` | the demo source (SPEC §10 D2) | plug it in, or run with `--no-record` against existing footage |
| ~4 GB disk for the model | `gemma-4-E2B-it` (SPEC §10 D1/D3) | **downloaded automatically on first run** |

**Why LM Studio?** Only for the `llama-server` binary it bundles — a prebuilt llama.cpp
for ARM64 + CUDA 13. Building that yourself, or getting an NGC key for the NIM
containers, are the alternatives; this is by far the shortest path on this hardware. If
you already have a `llama-server`, point at it instead and skip LM Studio entirely:

```bash
export SPARK_LLAMA_SERVER=/path/to/llama-server
```

The model itself comes from **HuggingFace, public, no token** — `scripts/fetch_models.sh`
fetches it only if it is not already on disk (it checks your HF cache first).

---

## Run it

```bash
./scripts/start.sh          # everything, in dependency order
```

Then open **<http://127.0.0.1:8080/>**. The page detects M3 and runs live; add
`?mode=mock` to rehearse against the fixtures in `ui/mock/` instead.

**<http://127.0.0.1:8080/browse.html?mode=live>** is the index browser: every analysis
window the system has written, newest first, paged — the corpus an Ask answer is drawn
from. Gate-skipped windows are listed too (captionless rows), because they are ~78% of it.
Filter by caption substring, tier, time range; `←`/`→` page. There is a link to it in the
console's top bar.

`start.sh` runs a preflight (reporting *every* missing prerequisite at once, with the fix
for each), downloads the model if needed, then starts **model server → recorder → ingest
→ agent**, waiting for each to actually answer before starting the next. Re-running it is
safe: anything already up is reused.

```bash
./scripts/start.sh --no-record   # skip the camera, use existing data/archive
./scripts/stop.sh                # clean shutdown
./scripts/stop.sh --keep-model   # ... but leave the model loaded
```

**Stop with `./scripts/stop.sh`, never `kill -9`.** It sends SIGTERM so ffmpeg can write
the moov atom on the segment it has open. A hard kill leaves that file unplayable *and*
makes every analysis window overlapping it undecodable — measured here, one hard-killed
60 s segment cost 15 of 24 windows.

Captions take a few window-strides to appear. Until the index has content, questions
honestly answer "nothing indexed covers that" — that is the groundedness gate working,
not a failure.

Logs land in `.run/logs/{model,recorder,ingest,agent}.log`.

### Running the pieces by hand

```bash
make serve                          # 1. the one model process (must be first)
python3 -m services.recorder        # 2. webcam -> data/archive, 60 s segments
python3 -m services.ingest --follow # 3. gate + caption + index
python3 -m services.agent           # 4. ask agent + UI on :8080
```

### Checks

```bash
make doctor    # this box vs CLAUDE.md's machine-state table
make bench     # SPEC §9 block 0 — time a single caption
make lint      # ruff
```

---

## Setup blockers

These affect the Docker path only — the host path in **Run it** needs none of them. `make doctor` prints each of these too; this is the same
list so you can act on it without running anything.

### 1. This user cannot talk to the Docker daemon

The daemon is running (Docker 29.2.1, Compose v5.0.2) but the user is not in the `docker`
group, so every `docker` command fails with a permission error.

```bash
sudo usermod -aG docker $USER
# then log out and back in, or for the current shell only:
newgrp docker
```

Gates: everything containerized.

### 2. The nvidia runtime is not registered with Docker

`/etc/docker/daemon.json` is absent. The NVIDIA container toolkit 1.19.1 *is* installed,
so this is one command — but until it is run, `runtime: nvidia` in `docker-compose.yml`
fails with "unknown or invalid runtime name".

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Gates: GPU access inside containers, and ffmpeg's `h264_nvenc` / NVDEC inside the app
image (they need the runtime to inject `libnvidia-encode.so.1` and `libnvcuvid.so.1`).

### ~~No NGC credentials~~ — no longer a blocker

`nvcr.io` still returns 401 and there is no `~/.ngc`, but nothing needs it. SPEC §10 D1
and D3 both resolved to a local model served by the ARM64 + CUDA-13 `llama-server`
bundled with LM Studio. An NGC key is only required if you want the NIM containers for
embed/rerank, which the lexical and hashing fallbacks already stand in for.

### 4. DeepStream is absent and its sm_121 support is unverified

No NVIDIA GStreamer plugins are installed. This does not block the build: the detector
gate (SPEC §2.3) ships with `ingest.gate.backend: motion`, which downscales to a 32x32
grayscale thumbnail via ffmpeg and diffs ~1 KB/frame in pure Python — no numpy, no
OpenCV, no new dependency. The architecture is preserved; a `deepstream` or `tensorrt`
backend can replace it later.

Verify sm_121 support before committing to DeepStream. SPEC §7.1's 5 GB DeepStream row
is left unallocated in `docker-compose.yml` so switching later does not require
re-cutting anyone else's memory budget.

### Already resolved

- **ffmpeg** — installed 2026-08-15, 6.1.1. Better than the package metadata suggested:
  it has `cuda` hwaccel, `h264_cuvid`/`av1_cuvid`, working `h264_nvenc` on this GB10, and
  v4l2 capture. No CUDA rebuild needed.
- **Camera** (SPEC §10 D2) — a USB webcam on `/dev/video0`, attached and verified
  end-to-end. Access is via a per-user ACL, not the `video` group, so `id` looking wrong
  is not a problem.

---

## The deployment layer has never executed

`docker-compose.yml` and `deploy/` were written against SPEC §1, §7.1 and §12 and
validated as far as this box allows, which is not very far. Stated plainly:

**What was verified**

- The compose file is valid YAML and parses cleanly (`yaml.safe_load`).
- Its anchor merges resolve to the intended per-service keys, no duplicate `<<` key
  silently dropping one.
- Every published port matches `config/settings.yaml` exactly: vLLM 8000, embed 8001,
  rerank 8002, agent LLM 8003, Milvus 19530, agent server 8080. No host-port collisions.
- Every service carries an explicit `mem_limit`, and the totals reconcile to SPEC §7.1's
  128 GB table.

**What was NOT verified — nothing here is evidence of anything**

- **No container has been started, and no image has been pulled or built.** Blockers 1
  and 3 above make both impossible right now.
- `docker compose config` has not been run, so Compose's own schema validation and its
  `${VAR:?}` interpolation have not passed judgement on this file.
- **No image tag is confirmed.** Not one. Every image is an environment variable, and
  `deploy/.env.example` marks what still needs checking for each. The two things to
  confirm per image are (a) an aarch64 manifest exists and (b) it actually supports
  **sm_121** — which is *not* sm_100 (datacenter Blackwell) and *not* sm_120 (RTX 50xx).
  A container built for either pulls cleanly, starts cleanly, then fails at the first
  kernel launch.
- The riskiest single line is `VLLM_IMAGE`: an ARM64 + sm_121 vLLM container is not a
  solved problem, and there is deliberately no default for it.
- Every healthcheck command is a guess at what the image contains.
- Whether `mem_limit` (a cgroup limit) actually catches CUDA allocations on this box's
  *unified* memory is unknown. It is treated as a backstop; the load-bearing cap is the
  application flag (`--gpu-memory-utilization`).
- `deploy/Dockerfile.app` installs Debian's ffmpeg (5.1.x), which is **not** the 6.1.1
  build verified on the host. Until someone runs it, the tested recorder path is the host
  one.

### Memory, which is the whole game

128 GB is unified — CPU and GPU share it and there is no VRAM to spill into. A second
model instance does not run slowly, it **OOMs the box**. So `docker-compose.yml` sets an
explicit limit on every service, each commented with the SPEC §7.1 row it came from:

| SPEC §7.1 row | GB | Service |
|---|---|---|
| VLM weights + VLM KV-cache | 18 + 34 | `vllm` — `mem_limit: 52g`, `--gpu-memory-utilization 0.40` |
| LLM weights + LLM KV | 19 + 8 | `llm` — `mem_limit: 27g` |
| Embed + rerank | 3 | `embed` 2g + `rerank` 1g |
| Milvus | 6 | `milvus` 5g + `etcd` 512m + `minio` 512m |
| DeepStream | 5 | reserved, unallocated (see blocker 4) |
| OS + headroom | 35 | app services take 14–17g; the rest is OS and page cache |

The VLM KV-cache is capped three ways, because left alone it takes the headroom the ask
LLM needs: `--gpu-memory-utilization 0.40` (the allocator ceiling, 52/128),
`--max-num-seqs 2 --max-model-len 16384` (an arithmetic cap on KV bytes), and
`mem_limit: 52g` (the cgroup backstop). A direct byte cap
(`--kv-cache-memory-bytes` / `--num-gpu-blocks-override`) is documented in
`deploy/.env.example` but not set, because its availability and syntax depend on a vLLM
version nobody here has seen.

Beware local model runners: LM Studio and `unsloth studio` (port 8888) load into the same
unified memory and will silently eat this budget. `make doctor` warns about them. Shut
them down before ingest.

### Compose profiles

`make up` starts only the app services, which is what today's stub configuration wants:

```bash
make up                                                    # recorder, ingest, agent, worker, monitor
docker compose --profile index up -d                       # + etcd, minio, milvus
docker compose --profile models up -d                      # + vllm, embed, rerank, llm  (needs NGC)
docker compose --profile index --profile models up -d      # the full SPEC §1 stack
```

App services use `network_mode: host` because `config/settings.yaml` hard-codes
`localhost` for every endpoint and that file is the single source of tunables — forking
it for containers would mean two copies of every number, which drift.

`data/` is a bind mount, never a named volume: `docker compose down -v` must not be able
to delete the archive the deep worker exists to re-read.

`services/index` and `services/mcp` are listed under a `standalone` profile and are
**off** by default. As built they are in-process libraries, not daemons — neither has a
`__main__`, and running the action server as a second process would mean a second writer
against the append-only `data/actions.jsonl`. See the comment above them in
`docker-compose.yml`.

Copy the env template before using any profile:

```bash
cp deploy/.env.example deploy/.env
$EDITOR deploy/.env
set -a; . deploy/.env; set +a           # Compose does not auto-load deploy/.env
```

No secret is stored in `docker-compose.yml`. `NGC_API_KEY` and `HF_TOKEN` are referenced
from the environment; `deploy/.env` is gitignored.

---

## Open decisions — SPEC §10

Two of these are wired into the deployment as fail-loud environment variables rather than
silently defaulted, because picking one here would hide the decision.

| # | Decision | State |
|---|---|---|
| D1 | Which Cosmos 3 variant on the live path? | **OPEN.** Decided by the block-0 caption benchmark (SPEC §9), not by taste — if a 5-frame caption takes >4 s, D1 is forced to a smaller variant. `vlm.model` is null; `VLM_MODEL` has no default and stops Compose if unset. |
| D2 | Live camera or pre-ingested recording? | **RESOLVED: USB webcam**, `/dev/video0` via v4l2. RTSP and file sources remain supported and tested. |
| D3 | Nemotron 3 Nano or 3.5 Lightning? | **OPEN.** Lightning is newer and desktop-targeted; confirm an ARM64/sm_121 container exists before betting the demo. `agent.model` is null; `LLM_IMAGE` and `LLM_SERVED_NAME` have no defaults. |
| D4 | Ship the `rollup` index tier? | **OPEN**, stretch goal. `index.rollup.enabled: false`. Skip if the 30 h block is at risk. |
| D5 | Who writes standing tasks? | **OPEN.** Proposed: build `register_task` as an endpoint and have the §11.3 form POST to it; binding M3 to it is then a tool schema, not a half-day bet made now. |
| D6 | The two demo questions | **OPEN.** Need one the index genuinely answers and one it genuinely can't, on the same footage. Choose before shooting, not after. |
| D7 | Funnel visualization in the Watch pane? | **OPEN.** ~45 min over a plain task list, and it is what makes the SPEC §6.4 brakes provable on stage rather than asserted. |
| D8 | What timezone does the burned-in overlay carry? | **OPEN and time-critical.** The overlay burns UTC while the UI renders local, so on stage that reads as two clocks in one card. **It is baked into the archive at capture time and cannot be changed afterwards — decide before shooting footage.** |

---

## Layout

```
config/settings.yaml     every tunable number in the system. No magic numbers in service code.
config/tasks.yaml        standing tasks for M5 (SPEC §6.1)
shared/                  schema, timecode, the one VLM client, the priority queue
services/                recorder, ingest (M1), index (M2), agent (M3), worker (M4),
                         monitor (M5), mcp (the action server)
ui/                      console (index.html: three panes + players) and the index
                         browser (browse.html). Vendored assets, no CDN.
scripts/doctor.py        `make doctor`
docker-compose.yml       the container topology — never run, see above
deploy/                  Dockerfile.app, .env.example
data/                    archive/, clips/, actions.jsonl, chats.jsonl — bind-mounted, never committed
```
