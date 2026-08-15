# Opening prompt for Claude Code on gn100-2f74

Paste everything below the line as the first message to a `claude` session
started in `/root/gautam/spark-vision`. Context: the app is already deployed
and running there against the box's own model servers; this prompt hands the
session the camera bring-up and verification.

---

You are running on gn100-2f74, my own DGX Spark (GB10, sm_121, aarch64), inside
`/root/gautam/spark-vision`. This app — a two-speed vision agent — is ALREADY
deployed and partially running here. Your job is to bring it fully live with the
camera I just attached, not to set it up from scratch.

Before doing anything else:

1. Read `DEPLOY_GN100.md` end to end — it is the ground truth for this box:
   current state, the model servers, measured budgets, and the runbook.
2. Read `CLAUDE.md` — the hard invariants. Invariant 1 especially: NEVER start a
   model process. Both models are already served by this box's own containers
   (Nemotron Nano 12B v2 VL on :8082, Nemotron 3.5 Lightning on :8000) and the
   app is configured to use them.
3. Run `git pull` (origin is wired via a deploy key with write access), then
   check what is already running (`pgrep -af services`, curl `:8000`/`:8082`
   `/v1/models`, curl `:8080`) before starting or restarting anything.

Then do this, in order:

**A. CAMERA.** I attached a USB camera. Probe it:

    ffmpeg -f v4l2 -list_formats all -i /dev/video0

Camera quirks are per-device — do not assume the formats in the config
comments. Update `recorder.device` in `config/settings.yaml` (`video_size`,
`framerate`, `input_format`) to what this camera actually offers. Prefer
1920x1080. If there's no `/dev/video0`, stop and tell me what `ls /dev/video*`
shows.

**B. RESTART with the recorder:**

    ./scripts/stop.sh && ./scripts/start.sh

(No `--no-record` this time. Never `kill -9` anything — a hard-killed ffmpeg
corrupts the open segment and poisons every analysis window overlapping it.)

**C. VERIFY the chain**, in this order, and show me the evidence:

- `data/archive/` grows a new `cam01_*.mp4` every ~60 s
- `.run/logs/ingest.log` shows gate decisions; wave at the camera and confirm
  a chunk with `"gated": false` and a non-empty caption appears
- the caption cites the burned-in UTC clock
- POST a question to `http://localhost:8080/api/ask` about what just happened
  and confirm a grounded answer, or a provisional answer + deep job if it
  escalates. Remember the VL's first request after any restart takes ~26 s
  cold — send a warm-up request before judging latency.

**D. START THE MONITOR (M5)**, which `start.sh` does not launch:

    nohup python3 -m services.monitor >> .run/logs/monitor.log 2>&1 &

and confirm it matches chunks against `config/tasks.yaml`.

**E. COMMIT AND PUSH** whatever you changed (config edits, fixes) with clear
messages. I pull the same repo from my laptop.

Guardrails — things that look wrong but are load-bearing; do not "fix" them:

- `agent.extra_body.chat_template_kwargs.enable_thinking: false` — without it
  Lightning writes its reasoning into `content` and breaks the groundedness
  gate.
- `vlm.profiles.deep.sample_fps: 0.4` — sized to the VL server's 16-image cap.
- `index.store.memory_path` must stay set — three processes share the corpus
  through it.
- The VL container joins `nemoclaw-vllm`'s network namespace: if you ever must
  touch the model servers, the order is `systemctl stop gn100-vlm` → deal with
  `nemoclaw-vllm` → wait for `:8000` → `systemctl start gn100-vlm` → warm both.
- Do not add Python dependencies; the app is stdlib + PyYAML + requests by
  design, and this is an ARM64/sm_121 box where wheels are often missing.

Known loose ends you may hit but should not rabbit-hole on: ~10 tests assert
the old single-model gemma config and fail against this one (update them only
if asked); D8 (the timezone burned into the overlay vs the UI's display
timezone) is an open decision — flag it before I record demo footage, don't
decide it yourself.

Report back with: camera formats found, config changes made, evidence from
step C, and anything that didn't match `DEPLOY_GN100.md`'s description.
