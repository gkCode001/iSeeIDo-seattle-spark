# SPEC — Two-Speed Vision Agent

A local video intelligence system on a single DGX Spark. One camera. Two surfaces
over one shared understanding of the footage:

- **Ask** — a human asks a question. Answered instantly from the index, or escalated
  to a worker that re-watches the original footage.
- **Watch** — standing tasks run continuously. When footage matches one, an MCP
  action fires.

The design thesis: *ingest is lossy on purpose, and the system knows when its own
summary isn't good enough.*

---

## 0. Hardware and constraints

| | |
|---|---|
| Box | 1 × DGX Spark, GB10 Grace Blackwell |
| Memory | 128 GB unified LPDDR5X, **273 GB/s** shared CPU+GPU |
| Compute capability | **sm_121** (not sm_100, not sm_120) |
| Arch | ARM64 |
| Cameras | 1 |

Two facts drive nearly every decision below:

1. **There is no separate VRAM.** A process that loads a second copy of a model does
   not run slowly, it OOMs.
2. **Decode is bandwidth-bound.** An 8B bf16 model reads ~16 GB per output token
   → ~17 tok/s single-stream, ceiling. Prefill is fast (thousands of tok/s);
   generation is not. **Output token count is the primary latency dial.**

Corollary: with one camera there is never more than one request in flight, so
batching does not help the live path. Real-time comes from doing *less work per
chunk*, not more work in parallel.

---

## 1. Architecture

```
  camera
    │
    ├─► recorder ──────────────► segment files on disk (archive, full res)
    │                                    ▲
    ▼                                    │ fetch(t_start, t_end)
  M1 ingest                              │
   detector gate ─► VLM caption          │
    │                                    │
    ▼                                    │
  M2 index (Milvus)                      │
    │            │                       │
    │            └──────────┐            │
    ▼                       ▼            │
  M3 ask agent          M5 monitor       │
    │  │                   │             │
    │  └──► M4 deep worker ◄─────────────┘
    │            │          (shared, priority queue)
    ▼            ▼
  user      MCP actions
```

Five modules, two surfaces, **one VLM process**.

---

## 2. M1 — Ingest

### 2.1 Recording (separate from analysis)

The recorder writes the stream to disk continuously, independent of any AI.
Fixed-length segments named by start time:

```
cam01_20260814_211000.mp4    # 21:10:00 → 21:11:00
cam01_20260814_211100.mp4    # 21:11:00 → 21:12:00
```

- 60 s per file. Small enough that a 30 s fetch touches ≤2 files; large enough to
  avoid managing 86,400 files/day.
- `ffmpeg -f segment -strftime 1` is sufficient. VSS/VIOS is optional.
- **Record at full resolution. Never downscale the archive.** It is the deep
  worker's entire reason for existing.
- Budget: 1080p @ 4 Mbps ≈ **1.8 GB/hour, 43 GB/day**.

**Keyframe interval is load-bearing.** A stream-copied clip can only *begin* on a
keyframe, so the GOP is the floor on how accurately any fetch can hit a requested time
range — for evidence clips and for the deep worker's seek alike. Measured on this box:
an 8.3 s GOP turned a 2.0 s request into **8.07 s of footage starting 6 s early**, while
a 1 s GOP returned 2.1 s. The webcam path sets `-g` from
`recorder.device.keyframe_interval_seconds` (1 s). §3.1's `pts_offset` is only as
precise as this number, so it is not a quality knob and must not be raised to save disk.

The webcam path is the one place the archive is **encoded** rather than stream-copied —
v4l2 emits rawvideo or mjpeg and neither muxes into mp4 as anything a player will open.
It encodes at capture resolution with no scale filter, which is what invariant 7
protects; the rule is never to *downscale* the archive, not never to touch a codec.

### 2.2 Analysis windows

Windows are **time ranges pointing into the segment files**. No video is copied or cut.

| Parameter | Value | Why |
|---|---|---|
| Window | **5 s** | Latency floor is "wait for window to close". 5 frames still shows motion; below ~4 frames the VLM can't tell "reversing toward" from "parked near". |
| Stride | 4 s (1 s overlap) | An event on a boundary would otherwise be halved and described badly in both. |
| Sample rate | 1 fps → 5 frames | |
| Resolution (live) | ~512 px short side | Memory dial, not a speed dial — see §2.5. |
| Resolution (deep) | native | |

### 2.3 The detector gate

**This is the main reason real-time is achievable.** On a fixed camera most windows
are an empty scene. DeepStream's detector runs on every window; the VLM only runs
when something is there.

- No motion / no new objects → write a null record, skip inference entirely.
- Target: **≥80% of windows skipped**. Effective VLM load drops accordingly, and the
  VLM is idle and instantly available when something does happen.
- Log the skip rate. If it's below 60% the gate is mistuned and real-time is gone.

### 2.4 Captioning

Per surviving window:

1. Decode on NVDEC (frames stay in GPU memory).
2. Sample to 5 frames, resize.
3. **Burn wall-clock into the bottom of each frame, after the resize.** Cosmos Reason
   models are trained to read burned-in timestamps for temporal localization — this
   is what lets the model cite times back to us. If the overlay is illegible after
   downscaling, localization silently breaks.
4. Frames + caption prompt → VLM.
5. Caption → embedding.

Hard settings on the live path:

| Setting | Value | Why |
|---|---|---|
| `enable_reasoning` | **false** | A CoT caption is 1000+ tokens → 60 s+/chunk. Non-negotiable. |
| `max_tokens` | **320** (was 80) | Decode is ~95% of cost and this is the dial — but measured on this model, 80 bought less than it looked: 97% of captions spent a quarter of their words describing the burned-in clock, and raising the cap alone changed nothing (the model stops when the prompt is satisfied). A caption with real detail costs 2.51 s; the gate's ~78% skip makes that 0.55 s per 4 s stride. See CLAUDE.md invariant 6. |
| Model | smallest viable Cosmos 3 variant | ~4× decode speed vs 8B. See open decision D1. |
| Precision | bf16 | Quantized Cosmos Reason variants are documented to hallucinate. |
| Batch | 1 in practice | **But the API must take a list.** See §8. |

Expected: **~2 s per captioned chunk.** Benchmark this first (§9, block 0).

### 2.5 Resolution — what it does and doesn't do

Resolution and fps affect **prefill only**, which is ~5% of chunk cost. Halving
resolution saves ~0.3 s. It is worth doing for **KV-cache pressure**, not speed.
Do not lower it on the deep path — that path exists to read fine detail.

---

## 3. M2 — Index

### 3.1 The chunk record

**Agree on this before writing any ingest code.** It is the only join between a text
hit and the pixels it came from. If it's wrong, M4 cannot exist.

```json
{
  "chunk_id":   "cam01_20260814T211107_211112",
  "camera_id":  "cam01",
  "t_start":    "2026-08-14T21:11:07Z",
  "t_end":      "2026-08-14T21:11:12Z",
  "segment":    "cam01_20260814_211100.mp4",
  "pts_offset": 7.00,
  "tier":       "live",
  "gated":      false,
  "caption":    "A white panel van reverses toward the loading door...",
  "embedding":  [768 floats]
}
```

Rules that fall out of it:

- **Never trust PTS alone.** Recorders restart PTS at zero every file; users think in
  wall clock. Store both. `pts_offset = t_start − segment_start`, and segment_start
  comes from the filename — which is the whole reason for naming files that way.
- **Always stitch.** An event at 21:11:58 running 12 s spans two files. The fetch
  function takes a *time range*, never a filename.
- `camera_id` is constant today. Keep the field anyway — one line now versus
  reindexing everything later.

### 3.2 Storage

Milvus holds the vector plus the full payload above. No separate database for the
hackathon.

Size contrast, which is the point: 768 floats ≈ 3 KB; at a 4 s stride with an 80%
gate skip, ~4,300 captioned chunks/day ≈ **13 MB/day of index against 43 GB/day of
video**. The index is free. That is what makes re-analysis affordable — keep
everything, pay to look closely only when asked.

### 3.3 Two-tier index

| Tier | Window | Purpose |
|---|---|---|
| `live` | 5 s | Alert path. Fresh, thin, low latency. |
| `rollup` | 60 s | Search path. A background job merges 12 live chunks and re-captions the merged window. |

Rationale: short windows keep latency low but dilute retrieval — neighbouring
embeddings look alike and the reranker has nothing to discriminate on. The rollup
restores context. It runs on idle capacity and is not on any critical path.
*Ship `live` first; `rollup` is a stretch goal.*

### 3.4 Retrieval

```
question → embed (same model as ingest) → Milvus ANN, k=20
        → rerank (cross-encoder) → top 5
```

| Component | Model |
|---|---|
| Embed | `llama-3.2-nemoretriever-300m-embed-v1`, 768 dims (Matryoshka-truncated) |
| Rerank | `llama-3.2-nv-rerankqa-1b-v2` |
| Store | Milvus |

The top 5 carry their time ranges. That is what M3 hands to M4.

---

## 4. M3 — Ask agent

**Model:** Nemotron 3 Nano (NVFP4) served via NIM, OpenAI-compatible endpoint.

### 4.1 Tools

| Tool | Purpose |
|---|---|
| `search_index(query, t_from?, t_to?)` | M2 retrieval |
| `request_deep_analysis(t_start, t_end, question)` | dispatch to M4, returns `job_id` immediately |
| `read_action_log(t_from, t_to)` | answer "why did you alert at 21:11?" |
| MCP actions | `save_clip`, `raise_alert`, `file_ticket` |

### 4.2 Escalation

Retrieval distance is **not** a confidence signal. ANN always returns a top-k with a
plausible score even when the answer was never indexed; it measures the closest thing
it has, not whether the answer is present.

Two mechanisms, both on:

1. **Groundedness gate.** Give the reranked chunks to the agent and ask, before
   answering: *can this question be answered from this context alone, yes/no?* One
   extra call, catches most cases.
2. **Tool choice.** `request_deep_analysis` is described so the model reaches for it
   on fine visual detail. This makes the decision legible — **print it in the UI**.

### 4.3 Never block

Deep analysis is tens of seconds. The user turn must never await it.

```
answer provisionally  →  return job_id  →  stream refined answer over WebSocket
```

Backstops: one deep job in flight; dedupe identical ranges (an impatient user
clicking twice must not queue the work twice); 90 s timeout, stated to the user.

---

## 5. M4 — Deep worker

Headless. One entry point, shared by M3 and M5:

```python
deep_analyze(t_start, t_end, question) -> {answer, evidence_clip, confidence}
```

1. Resolve the time range to segment file(s), stitching across boundaries.
2. Re-decode at **4 fps, native resolution**.
3. Same VLM process, deep request profile:

| Setting | Value |
|---|---|
| `enable_reasoning` | **true** (returns `reasoning_description` per chunk) |
| `max_tokens` | ~600 |
| Sample | 4 fps |

**Latency warning.** 4 fps over 30 s is ~120 frames. Prefill is fine; the CoT is not.
At ~17 tok/s, a 2000-token trace is over two minutes. Budget the trace to ~600 tokens,
or run the worker on the smaller Cosmos variant and accept weaker reasoning. Realistic
target: **20–60 s.**

---

## 6. M5 — Standing-task monitor

Long-running, push-triggered. Subscribes to every chunk M1 emits. **The only module
that changes the outside world unprompted**, so it is the one that needs brakes.

### 6.1 Task definition

```json
{
  "task_id":  "fire-door-blocked",
  "describe": "a vehicle stopped in front of the fire door",
  "window":   120,
  "action":   "raise_alert",
  "cooldown": 300,
  "active":   "18:00-06:00"
}
```

### 6.2 Three-stage funnel

| Stage | Runs on | Cost | Does |
|---|---|---|---|
| 1. Embedding match | every chunk | free | Task descriptions embedded once at registration; cosine against each new caption. Deliberately loose — over-trigger here. |
| 2. LLM confirm | candidates | ~1 s | Nemotron reads caption + task, says match/no. Also holds the sustain window (`window` seconds of consecutive matches before promoting). |
| 3. Worker verify | promoted | 20–60 s | `deep_analyze` on the matched range. Captions are lossy; don't file a ticket on a 1 fps guess. |

### 6.3 Acting — non-blocking

Stage 3 is **not** a blocking precondition. Real-time requires firing on stage-2
confidence, then attaching the verified verdict and clip when the worker finishes —
the same provisional-then-refined shape M3 uses. One pattern, both surfaces.

Split by *action severity*, not by task:

- Low stakes (`save_clip`) → fire on stage 2, no verification.
- Reaches a human (`raise_alert`, `file_ticket`) → fire provisionally, mark
  `unverified`, update on stage 3. Retract if stage 3 disagrees.

### 6.4 Three brakes — non-negotiable

1. **Cooldown** per task (default 300 s). One event → one alert.
2. **Dedupe by overlapping time range.** Consecutive chunks share 1 s and will
   double-report the same moment.
3. **Append-only action log**, readable by M3, with the clip attached. "Why did you
   alert at 21:11?" must be an answerable question.

The demo failure mode here is not missing an event. It is firing thirty alerts for one.

---

## 7. Shared VLM and queueing

**One VLM process. Two request profiles.** A worker that instantiates its own model
OOMs; lazily loading one per job costs minutes of cold start.

Priority order:

```
1. interactive (M3/M4 on a user's behalf)
2. background verification (M5 stage 3)
3. ingest captioning (M1)
```

Ingest may be **paused, never starved** — M5 has nothing to match against if captions
stop arriving. Cap any pause at a few seconds and let it catch up.

### 7.1 Memory budget (128 GB)

| Component | GB |
|---|---|
| VLM weights | 18 |
| VLM KV-cache | 34 |
| LLM weights | 19 |
| LLM KV | 8 |
| Embed + rerank | 3 |
| Milvus | 6 |
| DeepStream | 5 |
| OS + headroom | 35 |

Set `VLLM_GPU_MEMORY_UTILIZATION` conservatively. **Cap the VLM KV-cache explicitly** —
left alone it takes the headroom the LLM needs. Keep the Spark cache-cleaner running.

---

## 8. Latency budget (event → alert)

| Stage | Time |
|---|---|
| Window closes | 0–5 s (inherent) |
| Detector gate | ~0.1 s |
| VLM caption (small model, 80 tok) | ~2 s |
| Embed + task match | ~0.2 s |
| LLM confirm | ~1 s |
| **Alert fires** | **~4–9 s** |
| Deep verification attaches | +20–60 s |

Ask path: **~2–3 s** provisional, **+20–60 s** refined.

**Scaling note (for the writeup).** The bottleneck is single-stream bandwidth, and the
reason we're single-stream is that we have one camera. The same code with 40 cameras
would batch well on this box. The workload shape is the limit, not the silicon —
which is why the ingest API takes a list even though we always pass one.

---

## 9. Build order

Boring video end-to-end first. The interesting parts are worthless if the plumbing
isn't proven.

| Block | Ship | Done when |
|---|---|---|
| 0–2 h | **Benchmark one chunk** | `time` a single 5-frame caption. If >4 s, D1 is forced. |
| 2–6 h | Recorder + M1 with detector gate | Segment files on disk; gate skip-rate logged. |
| 6–12 h | M2 with the real schema | A text query returns a chunk with correct wall-clock times. |
| 12–18 h | M4, called by hand | Given a time range it re-watches and beats the caption. |
| 18–26 h | M3 + escalation gate | Agent decides alone. Refined answer streams in without blocking. |
| 26–30 h | MCP actions + UI (§11) | Escalation visible on screen. Clip saves. Vendor UI assets *now* — the 36 h rehearsal is network-off. |
| 30–36 h | M5, one task | A staged event fires **exactly one** verified alert. Not thirty. |
| 36–40 h | Rehearse twice | Both questions + the standing task land cold, network off. |

---

## 10. Open decisions

| # | Decision | Notes | Owner |
|---|---|---|---|
| D1 | Which VLM on the live path? | **RESOLVED 2026-08-15 by the block-0 benchmark.** Not Cosmos — no NGC key, and nothing Cosmos-shaped was reachable. `gemma-4-E2B-it` Q4_K_XL + mmproj-F16, ~4 GB resident, already in the HF cache. Measured: **2.55 s cold, 1.17 s warm per 5-frame caption** against a 4 s budget, and it demonstrably **reads the burned-in wall clock** (invariant 8) — captions cite `2026-08-15 09:21:13 UTC` unprompted. The 15–30 GB vision models on this box are 4–8× slower per token and cannot hold a 4 s stride. | |
| D2 | Live camera or pre-ingested recording? | **RESOLVED: USB webcam** (`/dev/video0`, v4l2). Live, no network, on-device, and an event can be staged in front of it on cue. RTSP and file sources remain supported and tested — a webcam failing on the day must not also be a code change. Note the webcam path **encodes** (`h264_nvenc`, verified on this GB10) because raw/mjpeg cannot be stream-copied into mp4; that is not a downscale, so invariant 7 holds. | |
| D3 | Nemotron 3 Nano or 3.5 Lightning? | **RESOLVED 2026-08-15: neither.** No NGC key, and no ARM64/sm_121 container could be verified for either — exactly the risk this decision existed to check. The live VLM is multimodal and answers text-only prompts, so M3 and M5 stage-2 share the **same model in the same process** as the VLM. Stricter than §7's two-process design, not looser: invariant 1 exists because a second model instance OOMs the box, and one process makes that impossible. The priority queue already arbitrates contention. | |
| D4 | Ship the `rollup` tier? | Stretch goal. Skip if block 30 h is at risk. | |
| D5 | Who writes standing tasks? | Hardcoded YAML is 2 h. Letting M3 register one from conversation ("alert me if that happens again") is a better story, half a day. **Proposed:** build `register_task` as an endpoint and have the §11.3 form POST to it — binding M3 is then a tool schema (~1 h at hour 34), not a half-day bet made now. | |
| D6 | The two demo questions | Need one the index genuinely answers and one it genuinely can't, **on the same footage**. Choose before shooting, not after. **Verified working end to end 2026-08-15**: a scene question returns `grounded=true, escalated=false`; a fine-visual-detail question returns `grounded=false, escalated=true` with the gate's reason printable — on the same webcam footage, with the real model. The exact wording is still to be chosen. | |
| D7 | Funnel visualization in the Watch pane? | ~45 min over a plain task list. It is what makes the §6.4 brakes *provable* on stage rather than asserted. Recommend yes; cut first if block 30 h is at risk. | |
| D8 | What timezone does the burned-in overlay carry? | `ingest.overlay.format` burns **UTC**, so the deep path cites UTC times *inside* the answer text while the UI chrome renders local (§11.5). On stage that reads as two different clocks in one card. Either burn the overlay in the display timezone, or have M3 convert cited times before returning text. The UI renders answer text verbatim and performs no inverse conversion — one conversion, one direction. **Not urgent and not destructive:** the overlay is drawn onto *sampled frames* in the analysis paths (§2.4 step 3, and M4's re-decode), never into the archive — the recorder emits no `drawtext`. So this is a config change, not a re-shoot. | |

---

## 11. UI

The UI is the demo's evidence, not chrome. The system's claim — *it knows when its own
summary isn't good enough* — is invisible unless the screen shows it. §4.2 says so
directly: **print the escalation decision in the UI.** Block 26–30 h is done when that
is true.

Budget is 4 h, shared with MCP actions. Every choice below is sized to that.

### 11.1 Shape

Single page served by the M3 FastAPI process — the §4.3 WebSocket lives there anyway.
Three panes over one shared archive player.

| Pane | Surface | Reads |
|---|---|---|
| **Ask** | chat | M3 turns + WS refinements |
| **Watch** | standing tasks | M5 funnel state + task registry |
| **Timeline** | history | the action log (§6.4) |

Plain JS is sufficient; the interactive surface is two lists and a log. If you prefer a
framework, vendor it and budget the build step.

**All assets vendored locally — no CDN, no remote fonts.** §9 rehearses with the network
off. You want to discover a missing asset at hour 27, not hour 39.

### 11.2 Ask pane — never overwrite the provisional answer

§4.3 gives the shape: provisional → `job_id` → refinement over WebSocket. Rendering the
refinement *in place* is the obvious implementation and it destroys the demo — the
audience sees a chatbot that took 40 seconds. Stack them. The delta is the point.

```
┌─ You ──────────────────────────────────────────────┐
│ Was the van's rear door open when it backed up?    │
└────────────────────────────────────────────────────┘

  ⌕ searched index · 20 → 5 chunks · 21:11:07–21:11:52

┌─ Provisional ─────────────────── 2.1 s ────────────┐
│ A white panel van reversed toward the loading      │
│ door around 21:11. The caption doesn't record      │
│ whether its rear door was open.                    │
│                                                    │
│ ⚠ NOT ANSWERABLE FROM INDEX → escalated            │
│   job 7f3a · re-watching 21:11:07–21:11:52         │
│   native res, 4 fps · ⏱ 12s / 90s timeout          │
└────────────────────────────────────────────────────┘

┌─ Refined ─────────────────────── 34.8 s ───────────┐
│ Yes. The rear doors are open from 21:11:19         │
│ onward — visible as the van turns. Two figures     │
│ approach from the right at 21:11:41.               │
│                          [▶ clip]  [reasoning ▾]   │
└────────────────────────────────────────────────────┘
```

| Element | Why |
|---|---|
| Groundedness badge — `answered from index` / `escalated` | §4.2's gate returns a literal yes/no. The most important pixel in the build. |
| Elapsed timer against the 90 s timeout | §4.3 requires the timeout be stated to the user. Also covers the dead air. |
| Cited time ranges, clickable | The chunk carries `segment` + `pts_offset`; a click scrubs the player. Makes the §3.1 join tangible. |
| Reasoning trace, collapsed | The deep path returns `reasoning_description` (~600 tok). Proof, not prose. |
| Dedupe notice — "already running — job 7f3a" | §4.3 dedupes identical ranges. A silent no-op reads as a bug in rehearsal. |

### 11.3 Watch pane — render the funnel

§6.2's three stages are invisible by default, and §6.4 warns the demo failure mode is
firing thirty alerts for one. One fix covers both: show funnel state per task, with the
cooldown timer.

```
┌─ fire-door-blocked ─────────────────── ACTIVE ─────┐
│ "a vehicle stopped in front of the fire door"      │
│ 18:00–06:00 · window 120s · cooldown 300s          │
│ → raise_alert                                      │
│                                                    │
│ ① embed match  ●●●○○  0.71  (loose gate)           │
│ ② llm confirm  ✓ match · sustain ████████░░ 96/120s│
│ ③ verify       —                                   │
│                                                    │
│ last fired 21:11:14 · 🔒 cooling 247s remain       │
└────────────────────────────────────────────────────┘
```

The visible cooldown is how the brake gets *proved* rather than asserted: the event keeps
matching for four more minutes, the panel visibly holds, nothing fires.

**Defining tasks** — a form with the six §6.1 fields, POSTing to a `register_task`
endpoint. Build the endpoint first and bind M3 to it later: conversational registration
("alert me if that happens again") then costs a tool schema, not new plumbing. See D5.
`config/tasks.yaml` stays the cold-start seed so a fresh boot has tasks without clicking.

### 11.4 History — two logs, different durability

**The action log is authoritative.** Append-only (§6.4), and M3 already reads it through
`read_action_log` (§4.1). So the Timeline pane renders *exactly* what the agent
introspects — one source, no drift. "Why did you alert at 21:11?" is answerable in chat
and browsable on screen from the same rows.

Retraction (§6.3) cannot mutate an append-only log. A retracted alert renders as the
original struck through, with the retraction linked beneath it:

```
21:11:14  ⚠ raise_alert  fire-door-blocked      unverified
          └ 21:11:52  ✓ verified · clip attached
21:34:02  ⚠ raise_alert  fire-door-blocked      unverified
          └ 21:34:39  ✗ RETRACTED — worker found no vehicle
21:40:11  ⧉ save_clip    loading-bay-activity   (no verify)
```

Show retractions on stage. A visible retraction is the thesis, restated on the Watch
surface.

**Chat history** is the softer one, and a gap in §3.2: "no separate database" is right,
but Milvus is a vector store and conversations are not vectors. Append JSONL, same shape
as the action log.

It matters for one reason — **refinements arrive after the turn ends.** Persist the *job*
keyed to its turn, not just the message text, or a page reload loses the 34-second
answer.

### 11.5 Time

UTC everywhere beneath; conversion in exactly one helper, at render. All three panes show
local time, every payload underneath stays `Z`-suffixed.

---

## 12. Repo layout

```
├── CLAUDE.md
├── SPEC.md
├── docker-compose.yml
├── config/
│   ├── settings.yaml        # windows, strides, thresholds, model names
│   └── tasks.yaml           # standing tasks (M5)
├── shared/
│   ├── schema.py            # ChunkRecord, Task, ActionLog — single source of truth
│   ├── timecode.py          # wall clock ↔ (segment, pts_offset). Critical, test hard.
│   ├── vlm_client.py        # the ONE client; live + deep profiles
│   └── queue.py             # priority queue in front of the VLM
├── services/
│   ├── recorder/            # ffmpeg segmenter
│   ├── ingest/              # M1
│   ├── index/               # M2
│   ├── agent/               # M3
│   ├── worker/              # M4
│   ├── monitor/             # M5
│   └── mcp/                 # action server
├── ui/                      # single page, three panes — §11. Assets vendored, no CDN.
├── data/
│   ├── actions.jsonl        # append-only action log (§6.4) — M3 and Timeline read this
│   └── chats.jsonl          # chat turns + their job refs (§11.4)
└── tests/
```
