# Running on gn100-2f74

Status and runbook for **this GN100** — Gautam's own machine, physically at hand.
Everything here happens locally on the box (no SSH indirection); if you are Claude
Code running on gn100-2f74, this file is your ground truth. Facts below were
verified live on **2026-08-15**; the app was deployed and run end-to-end that day.

**State: DEPLOYED AND RUNNING.** The repo lives at `/root/gautam/spark-vision`,
the UI serves at `http://localhost:8080/`, and the full chain — gate → caption →
index → ask → escalate → deep re-watch — has executed against real inference on
this box's own model servers. What remains is the camera hookup (§5) and two
small chores (§6).

---

## 1. This machine

GB10, sm_121, aarch64, Ubuntu 24.04.4, driver 580.173.02, 121 GB unified memory,
3.3 TB free. Python 3.12.3 with PyYAML + requests (the app's only deps), ffmpeg
6.1.1 with working `h264_nvenc` (verified — it encoded the test archive), Docker
29.2.1, git, rsync. No NGC key, no LM Studio — neither is needed.

Memory note: the two model servers commit ~64 GB between them. The app itself
adds 1–2 GB. Never start another model process — CLAUDE.md invariant 1. The
unused extras were already stopped and disabled to free memory:
`gn100-nemotron.service` (a spare Nemotron 3 Nano 4B llama-server on :8081) and
acer01's `openclaw-gateway.service`. Re-enable either with `systemctl [--user]
enable --now ...` if ever wanted.

## 2. The models — serving locally, verified, and modified by us

Both are the box's own provisioning (local user `acer01` owns the weights), now
tuned for this app. **We changed their launch configs on 2026-08-15**:

| Slot | Model | Endpoint | Managed by |
|---|---|---|---|
| VLM — live captions + deep worker (D1) | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD` | `http://localhost:8082/v1` | systemd `gn100-vlm.service` (docker, NGC vLLM 0.21 image) |
| Ask LLM — M3 + monitor stage-2 (D3) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `http://localhost:8000/v1` | plain container `nemoclaw-vllm`, `restart: unless-stopped` |

**Changes we made to the VL unit** (`/etc/systemd/system/gn100-vlm.service`;
pre-change backup at `/root/gn100-vlm.service.bak`):
`--max-model-len` 8192 → **40960**, `--gpu-memory-utilization` 0.13 → **0.25**,
image limit 8 @1024×1024 → **16 @1920×1080**. This is what lets the deep worker
send 14 native-1080p frames (~32k prompt tokens) in one request.

**Changes we made to the Lightning container** (recreated with the original
flags plus): `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. The
parser choice is from evidence, not docs: the model's chat template emits
`<tool_call><function=name><parameter=key>…` — Qwen3-coder format. `hermes` was
tried first and does NOT parse it (the call leaks into the answer as raw text).

**Two operational trapdoors, learned the hard way:**

- **The VL container joins the Lightning container's network namespace**
  (`--network container:nemoclaw-vllm`). Touching `nemoclaw-vllm` therefore
  kills the VL's networking. Order is always: `systemctl stop gn100-vlm` →
  recreate/restart `nemoclaw-vllm` → wait for :8000 → `systemctl start gn100-vlm`.
- **Cold-start latency lies.** The VL's first caption after a restart takes
  ~26 s; warm it is 2–3 s. Send one throwaway request before believing any
  latency number, and after every model-server restart.

## 3. App configuration — already in the repo

`config/settings.yaml` in this repo is the deployed configuration; no per-box
edits are needed on a fresh pull. The load-bearing entries, and why:

- `vlm.endpoint: :8082`, `agent.endpoint: :8000`, model names as above.
- `agent.extra_body.chat_template_kwargs.enable_thinking: false` — **do not
  remove.** Lightning is a hybrid reasoner; without this it writes "Here's a
  thinking process:" into `content` (measured: 466 tokens of it) and the
  groundedness gate's YES/NO first-line parse breaks. The `/no_think` and
  "detailed thinking off" system prompts were tested and do nothing; this
  per-request kwarg is the only working switch. Plumbed via
  `services/agent/settings.py` → `services/agent/llm.py`.
- `vlm.profiles.deep.sample_fps: 0.4` — sized to the server's 16-image cap
  (max range 25 s + 2×5 s pad = 35 s ⇒ 14 frames), not to taste.
- `index.store.backend: memory` with `memory_path` set — M1/M3/M5 are separate
  processes sharing the corpus through that file. Null it and every question
  retrieves nothing.

## 4. Measured on this box, 2026-08-15

| | Result | Budget |
|---|---|---|
| Live caption, 5 frames @512px (warm) | **3.0–3.1 s** | 4 s cap |
| Deep prompt, 14 native 1080p frames (~32k tok) | **30 s** | 20–60 s |
| Reads the burned-in UTC clock | **Yes, unprompted** — "timestamp of 2026-08-15 23:00:57 UTC" | invariant 8 |
| Motion gate | skips `testsrc2` as "still" (0.014 < 0.02), fires on real motion (0.029) | working as designed |
| Groundedness gate through real Lightning | clean first-line "NO, …", escalates, returns job id | SPEC §4.2/§4.3 |
| Deep job on empty archive | honest refusal: "a fact about the recording, not a failure of the analysis" | correct |

## 5. Camera — the remaining physical step

No `/dev/video*` exists; the camera is an **iPhone over RTSP on WiFi** (an
iPhone cannot be a USB webcam on Linux):

1. iPhone on the **same network** as this box, plugged into power, RTSP-server
   app installed (e.g. "IP Camera Lite"): 1920×1080, H.264, keyframe/GOP
   interval 1 s if the app offers it (it bounds evidence-clip seek precision —
   see CLAUDE.md on keyframe interval).
2. Set `recorder.source` in `config/settings.yaml` to the app's URL, e.g.
   `rtsp://<iphone-ip>:8554/live`. `rtsp_transport: tcp` is already set;
   `copy_codec: true` will stream-copy the phone's H.264 into the archive.
3. `./scripts/stop.sh && ./scripts/start.sh` (without `--no-record` now).
4. Verify: `data/archive/` grows a new `cam01_*.mp4` every 60 s, and
   `.run/logs/ingest.log` shows `gated: false` chunks with captions when
   something moves in frame.

Until then, `./scripts/start.sh --no-record` runs against whatever is in
`data/archive/` — currently the generated test segments (1080p testsrc2 with a
burned UTC clock and, in the last one, a moving white square that trips the
gate).

## 6. Remaining chores

1. **M5 monitor is not in `start.sh`.** It has a runner; start it alongside the
   rest with `python3 -m services.monitor` when demoing the Watch pane.
2. **~10 tests assert the old single-model gemma config** (e.g.
   `test_the_resolved_model_is_read_from_config` asserts
   `agent.model == vlm.model`; `test_endpoint...` asserts the literal `:8000`
   VLM endpoint). They fail against this config for encoding the old decision,
   not for finding bugs. Update them to the two-model reality. Separately,
   `test_ask_grounded` (and siblings) fail at HEAD on every machine tried —
   pre-existing, not this box.
3. **D8 (overlay timezone) is still open and time-critical**: the clock is
   burned into the archive at capture and cannot be changed later. Decide
   before recording demo footage. The UI's `display_timezone` should match
   what the demo audience expects.

## 7. Runbook

```bash
cd /root/gautam/spark-vision
./scripts/start.sh              # models are external and already up; start.sh
                                #   detects the answering endpoint and skips its
                                #   own model step (verified behavior)
./scripts/stop.sh               # always this, never kill -9 — a hard-killed
                                #   segment poisons every window overlapping it
tail -f .run/logs/ingest.log    # gate decisions + captions, one JSON line each
tail -f .run/logs/agent.log     # ask turns, escalations, deep-job states
```

UI: `http://localhost:8080/` (Ask + Watch + Timeline), `browse.html` for the
raw index. From another machine on the LAN: `http://gn100-2f74:8080/`.

Model servers, when needed:

```bash
systemctl status gn100-vlm            # the VL — edit the unit file to change flags
docker ps                             # nemoclaw-vllm — recreate to change flags,
                                      #   in the §2 order, then WARM both
```
