# Architecture

This document explains the whole system in plain language, first from 10,000 feet,
then close to the ground. The authoritative design is [`SPEC.md`](SPEC.md); the hard
rules are in [`CLAUDE.md`](CLAUDE.md).

## The idea in one paragraph

One camera records everything to disk, all the time. A cheap AI pass watches the live
feed and writes a short text note ("a white van reverses toward the loading door")
about every few seconds of footage — but only when something is actually happening.
Those notes go into a searchable index. When a human asks a question, the system
answers instantly from the notes; when the notes aren't good enough, it goes back and
**re-watches the original footage** carefully. Standing watch-tasks ("alert me if a
vehicle blocks the fire door") run against the same notes and fire real actions — up
to and including a Discord message. Recorded video — a demo clip, an open dataset —
can be dropped into a folder and imported onto the same timeline, where the identical
pipeline picks it up. Everything runs on one machine, offline — no cloud.

That's why it's called **two-speed**: a fast, cheap, always-on pass, and a slow,
careful, on-demand pass. The system knows when its own quick summary isn't good
enough, and escalates itself.

---

## High level — what talks to what

```mermaid
flowchart TB
    CAM(["📷 USB webcam"])
    DROP(["📂 Recorded video\ndropped in data/inbox/\n(demo clips, open datasets)"])

    subgraph ALWAYS["Always running (the fast path)"]
        REC["Recorder\nsaves everything to disk,\n60-second video files"]
        IMP["Importer\nslices a file onto the real\ntimeline, as its own camera"]
        ING["M1 · Ingest\nmotion gate: skips quiet footage\nVLM writes a short caption for the rest"]
        IDX[("M2 · Index\nsearchable text notes,\neach pinned to an exact\nwall-clock time range")]
    end

    ARCH[("Video archive\nfull resolution,\nnever downscaled")]

    subgraph SURFACES["Two surfaces, same index"]
        ASK["M3 · Ask agent\na human asks a question"]
        MON["M5 · Monitor\nstanding tasks watch\nevery new caption"]
    end

    DEEP["M4 · Deep worker (the slow path)\nre-watches the original footage\ncarefully, cuts an evidence clip"]
    MCP["MCP actions\nsave_clip · raise_alert · file_ticket · notify_discord\n(with brakes: cooldown, dedupe, action log)"]
    AB(["📨 AlertBridge → Discord\n(the box's service, not this repo's)"])
    USER(["🧑 Person at the console"])

    CAM --> REC --> ARCH
    CAM --> ING --> IDX
    DROP --> IMP --> ARCH
    IMP --> ING
    IDX --> ASK
    IDX --> MON
    USER <--> ASK
    ASK -. "notes not good enough?\nescalate" .-> DEEP
    MON -. "verify before it counts" .-> DEEP
    DEEP -->|"fetch by time range"| ARCH
    MON --> MCP
    MCP -->|"notify_discord"| AB
```

**How to read it:** solid arrows are the everyday flow; dashed arrows are the
escalation — the moment the system decides its quick notes aren't trustworthy enough
and pays for a careful second look. Both surfaces (a human asking, a robot watching)
escalate to the **same** deep worker, and the deep worker's whole job is to turn a
*time range* back into the original pixels.

Three design rules make the picture work:

1. **The archive is sacred.** Recording is dumb and independent — no AI in that path,
   full resolution, so there is always ground truth to go back to.
2. **The index is nearly free.** Text notes cost ~13 MB/day against ~22–43 GB/day of
   video, so the system keeps everything and pays to look closely only when asked.
3. **Never block, never spam.** Answers arrive instantly and get *refined* later
   (never replaced), and actions pass through brakes so one event fires one alert,
   not thirty.

---

## Low level — how each piece actually works

```mermaid
flowchart TB
    subgraph CAPTURE["Capture"]
        CAM(["/dev/video0 · mjpeg 1080p\n(DJI Osmo Pocket 4, webcam mode)"])
        REC["services/recorder\nffmpeg → h264_nvenc, 1 s keyframes\ncam01_YYYYMMDD_HHMMSS.mp4"]
        ARCH[("data/archive/\n60 s segments, full res")]
        CAM --> REC --> ARCH
    end

    subgraph IMPORT["Importer · services/importer"]
        INBOX(["data/inbox/ — the drop folder"])
        IMP["probe → slice with -c copy\ncut points read back, never assumed\nplaced ending NOW, own id (clip01)\nthen M1 walks the imported range"]
        INBOX --> IMP
    end

    subgraph M1["M1 · services/ingest"]
        WIN["Windowing\n5 s windows, 4 s stride\n(1 s overlap so nothing\nfalls on a boundary)"]
        GATE{"Motion gate\nffmpeg 32×32 grayscale\nthumbnails, pure-Python diff"}
        NULLREC["Write a null record\n(~78% of windows —\nthis is why real-time works)"]
        CAP["Caption\n5 frames @ 1 fps, ~512 px,\nwall-clock burned into each frame\nVLM: no reasoning, 80 tokens max"]
        WIN --> GATE
        GATE -- "no motion" --> NULLREC
        GATE -- "something moved" --> CAP
    end

    subgraph M2["M2 · services/index"]
        CHUNK["Chunk record\nchunk_id · t_start/t_end (UTC)\nsegment · pts_offset · caption · embedding"]
        STORE[("Store\nmemory → data/index.jsonl\n(Milvus optional)")]
        RETR["Retrieval\nembed → ANN top-20 → rerank → top-5\nrecency breaks ties the reranker cannot\n(hashing + lexical backends today,\nNIM models optional)"]
        CHUNK --> STORE --> RETR
    end

    subgraph VLM["The box's model servers — this repo starts none"]
        Q["shared/queue.py · priority queue\n1. interactive (a human is waiting)\n2. M5 verification\n3. ingest captions (paused, never starved)"]
        LS["Nemotron Nano 12B v2 VL · :8082\nlive profile: no reasoning, 80 tok\ndeep profile: reasoning, ~600 tok"]
        LLM["Nemotron 3.5 Lightning · :8000\nM3's ask LLM + M5's stage-2 confirmer\n(topbar can rebind M3 to LM Studio)"]
        Q --> LS
    end

    subgraph M3["M3 · services/agent — Ask"]
        SRV["stdlib http.server :8080\n+ hand-rolled RFC 6455 WebSocket"]
        TOOLS["Tools\nsearch_index · request_deep_analysis\nread_action_log"]
        GROUND{"Groundedness gate\ncan the retrieved notes\nanswer this? yes/no"}
        PROV["Answer now (~2–3 s)\nreturn job_id, stream the\nrefinement over WebSocket —\nappended, never substituted"]
        SRV --> TOOLS --> GROUND
        GROUND -- "yes" --> PROV
        GROUND -- "no → escalate" --> DEEP
    end

    subgraph M4["M4 · services/worker — Deep"]
        DEEP["deep_analyze(t_start, t_end, question)\nresolve time range → segment files\n(stitch across boundaries, shared/timecode.py)\nre-decode 4 fps at native resolution\n→ answer + evidence clip + confidence\nmeasured 2–9 s"]
    end

    subgraph M5["M5 · services/monitor — Watch"]
        TASKS["config/tasks.yaml\ndescribe · window · action\ncooldown · active hours"]
        F1["Stage 1 · embedding match\nevery caption, ~free,\ndeliberately loose"]
        F2["Stage 2 · LLM confirm\n~1 s + sustain window"]
        F3["Stage 3 · worker verify\nfire provisionally first,\nattach verdict — retract if wrong"]
        TASKS --> F1 --> F2 --> F3
    end

    subgraph ACT["services/mcp — Actions"]
        BRAKES["Three brakes\ncooldown per task\ntime-range dedupe\nappend-only log"]
        LOG[("data/actions.jsonl\nthe only history store —\n'why did you alert at 21:11?'\nis answerable")]
        CLIPS[("data/clips/")]
        AB["AlertBridge :8081\nthe box's Discord relay —\nowns the webhook, we hold no secret\n202 = accepted, never 'delivered'"]
        BRAKES --> LOG
        BRAKES --> CLIPS
        BRAKES -->|"notify_discord,\nonly after the brakes say yes"| AB
    end

    subgraph UI["ui/ — vendored assets, works offline"]
        PANES["index.html — three panes\nAsk · Watch · Timeline"]
        BROWSE["browse.html\nevery indexed window, paged"]
    end

    ARCH -->|"decode windows"| WIN
    IMP --> ARCH
    IMP -->|"walk the imported range"| WIN
    CAP -->|"VLM call"| Q
    CAP --> CHUNK
    NULLREC --> STORE
    RETR --> TOOLS
    F1 -.->|"subscribes to\nevery new chunk"| STORE
    F2 -->|"yes/no verdict"| LLM
    F3 --> DEEP
    DEEP -->|"fetch(t_start, t_end)\nnever a filename"| ARCH
    DEEP -->|"VLM calls,\ndeep profile"| Q
    DEEP -->|"refined answer"| PROV
    F3 --> BRAKES
    TOOLS -->|"read_action_log"| LOG
    PANES <--> SRV
    BROWSE --> SRV
```

### The one non-obvious join: time

Every chunk record carries `t_start`/`t_end` in UTC **plus** the segment filename and
a `pts_offset`. That tuple is the only link between a text note and the pixels it
describes. Video files restart their internal clock at zero every segment, so a
timestamp alone is meaningless — and an event can span two files, which is why
*everything* that touches video takes a **time range, never a filename**, and goes
through `shared/timecode.py` to stitch across boundaries. The wall clock is even
burned into the frames the VLM sees, so the model itself can cite real times back.

### The one hard resource rule: this repo starts no model

The box has 128 GB of *unified* memory — CPU and GPU share it, and an extra model
instance doesn't run slowly, it crashes the machine. So the two model servers are the
machine's own (the VLM on :8082, Nemotron 3.5 Lightning on :8000), `start.sh` checks
them and stops with the fix if either is down, and nothing in this repo ever spawns an
engine. The VLM serves the live captioner and the deep worker with two request profiles
(fast/no reasoning vs. slow/reasoning) and a priority queue in front: a waiting human
beats background verification beats ingest, and ingest may be paused but never starved.
Lightning answers the ask surface and speaks M5's stage-2 yes/no verdicts — with
thinking disabled there, because a confirmer with an 8-token budget that spends it
reasoning fails closed on every chunk.

### The end-to-end speeds, measured on the real box

| Path | Latency |
|---|---|
| Event → caption in the index | ~4–9 s |
| Question → provisional answer | ~2–3 s |
| Escalation → deep, verified answer | +2–9 s (budget 20–60 s) |
| Quiet footage skipped by the gate | 78–81% |

### What runs where

| Process | Port | Started by |
|---|---|---|
| Nemotron Nano 12B v2 VL | 8082 | the box: `systemctl start gn100-vlm` — never this repo |
| Nemotron 3.5 Lightning (ask LLM) | 8000 | the box: `docker start nemoclaw-vllm` — never this repo |
| AlertBridge (Discord relay) | 8081 | the box: `/opt/alertbridge` — never this repo |
| `services/recorder` | — | `scripts/start.sh` |
| `services/ingest --follow` | — | `scripts/start.sh` |
| `services/agent` (API + UI, M5 in-process) | 8080 | `scripts/start.sh` |
| `services/importer` | — | by hand, when a file lands in `data/inbox/` |

`services/index` and `services/mcp` are in-process libraries, not daemons. M5 runs
inside M3's process by default, so a task registered through the Watch pane
(`POST /api/register_task`) is the same object the funnel evaluates; `python3 -m
services.monitor` runs it standalone instead, with tasks seeded from
`config/tasks.yaml`. Either way an action reaches the world only through
`services/mcp`, so the brakes apply.
