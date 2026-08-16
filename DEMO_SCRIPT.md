# Demo video script — Two-Speed Vision Agent

**Target: 4:30.** Hard floor 3:00, hard ceiling 5:00. VO is ~700 words at ~155 wpm.
Numbers below are the measured ones (`DEPLOY_GN100.md` §4, README). Do not round them up.

---

## 0:00 — 0:25 · Cold open

**VISUAL** — Three short clips, hard cuts, no music yet. Timestamps burned in the corner.
An elderly person collapsing in a hallway. A toddler going over the edge of a pool. A man
walking into a shop with a gun. Cut each one a beat *before* the impact. Then black.

**VO**

> There are a billion cameras in the world. Every one of these was recorded.
>
> Every one of these was watched — afterwards. By a person, scrubbing back through the
> footage, deciding what had already happened.
>
> The camera was never the problem. Nobody was watching it.

---

## 0:25 — 0:50 · Why nobody is watching

**VISUAL** — Simple animated line: 1 camera → 2.6 million frames a day. Then a second line
underneath: "→ cloud → $$$ / bandwidth / your living room, uploaded."

**VO**

> One camera is two and a half million frames a day. You cannot send that to a cloud model.
> Not for the cost, not for the bandwidth, and not if the camera is pointed at your
> hallway, your classroom, or your kid.
>
> So you compress. You keep a summary. And the moment you do, you've thrown away the exact
> detail somebody is going to ask you for.

---

## 0:50 — 1:15 · The idea

**VISUAL** — The DGX Spark on the desk, camera plugged into it, single cable. Then the
`ARCHITECTURE.md` diagram animating in: camera → recorder → gate → caption → index, with
two arms off the index labelled **ASK** and **WATCH**.

**VO**

> This is a DGX Spark. One box, on the desk, no network. Everything you're about to see
> runs on it.
>
> It watches at two speeds. Fast: a cheap pass over every window of footage, all day, so
> there's always an answer. Slow: when the fast pass isn't good enough, it goes back to the
> original video and *re-watches* it.
>
> The whole thesis is that second part. The summary is lossy on purpose — and the system
> knows when its own summary isn't good enough.

---

## 1:15 — 2:05 · Demo 1 — WATCH (the standing task fires)

**VISUAL** — Screen capture of the console, Watch pane. Show the standing task already
registered: *"a person lying on the floor not moving"* → `notify_discord`. Then cut to the
live camera view: someone in frame clutches their chest and goes down. Split-screen the
ingest log scrolling captions. Then the Discord notification landing on a phone.

**VO**

> A standing task, in plain English: tell me if a person is lying on the floor, not moving.
>
> No training. No labels. No detector class. It's a sentence.
>
> [beat — the fall]
>
> The captioner sees it. A second model confirms it's real, not a false alarm. And the
> action fires — off the box, to a phone.
>
> Here's the part that matters for anything you'd actually deploy: that event lasted twelve
> minutes and produced a hundred and eighty matching windows. It sent **one** message. One
> event, one alert — a cooldown, a time-range dedupe, and an append-only log that can't be
> rewritten. Because an alert cannot be un-sent.

---

## 2:05 — 2:45 · Demo 2 — ASK (instant, and grounded)

**VISUAL** — Ask pane. Type: *"was anyone in the room in the last ten minutes?"* Answer
comes back fast. Cursor hovers the citation chips — click one, the actual video segment
plays back at that timestamp.

**VO**

> The same index answers questions.
>
> That came out of the index — no video was decoded to answer it. And every claim is a
> link. Click it and you're looking at the frames that claim came from, at the second it
> happened, in the original recording.
>
> No citation, no answer. If nothing in the index covers your question, it says so.

---

## 2:45 — 3:35 · Demo 3 — the escalation (the actual point)

**VISUAL** — Ask pane again. Type a question the summary genuinely can't hold — e.g.
*"what colour was the bag the person left behind?"* First response appears immediately,
hedged, with a **job ID** badge. Timeline shows the deep job running. Then — without the
page reloading, and without the first answer disappearing — a second block **appends**
below it, with an evidence clip.

**VO**

> Now the interesting one. Something the summary never recorded.
>
> Watch what it does *not* do. It doesn't stall, and it doesn't guess. It answers with what
> it has, admits that isn't enough, and hands back a job.
>
> Behind that, a worker is pulling the original footage — full resolution, the frames the
> fast pass downscaled away — and re-watching it properly.
>
> [beat — refinement appends]
>
> And the refinement is **added**, not swapped in. You can see the system change its mind.
> That's not a UI detail. A system that silently overwrites its first answer is a system
> you can never audit.

---

## 3:35 — 4:05 · What it costs

**VISUAL** — Clean numbers on screen, one at a time, over B-roll of the box idling.

**VO**

> Measured on this box, on live 1080p:
>
> Three seconds a caption. A motion gate that throws away eighty percent of windows before
> a model ever sees them — that gate is the only reason this is real-time. Half a minute
> for a deep re-watch, off the critical path, so nothing ever blocks on it. Twenty-two
> gigabytes of archive a day.
>
> Two NVIDIA Nemotron models on one Spark — Nano 12B Vision for the eyes, 3.5 Lightning for
> the reasoning. A hundred and twenty-one gigabytes of unified memory, and no second copy
> of anything.
>
> And the network is off. It's been off this entire video.

---

## 4:05 — 4:35 · Close

**VISUAL** — Return to the three opening clips, same frames — but now each one has a
caption card and a fired action overlaid, timestamped. Hold on the last. Fade to the
project name.

**VO**

> The footage always existed. What was missing was something watching it that could tell
> you *why* it mattered, at the moment it mattered — and could go back and check itself
> when it wasn't sure.
>
> One camera. One box. No cloud.
>
> That's the whole thing.

---

# Production notes

### Before you roll
- **Warm both models.** The first request after a restart takes ~26 s. Send a throwaway
  caption and a throwaway ask before recording anything, or the demo looks broken.
- `./scripts/start.sh`, confirm `.run/logs/ingest.log` is writing captions, then leave it
  running for a few strides so the index isn't empty.
- Decide **D8** — the overlay burns UTC and the UI renders local. Two clocks on screen in
  one shot is the single most confusing thing a judge can see. Fix it before shooting,
  because it's baked in at capture and cannot be changed after.
- Have `browse.html` open in a second tab. If a judge asks "is that real?", that page is
  the answer — every window the system ever wrote, skipped ones included.

### Shot list
1. Three stock/open clips for the cold open (also: run them through
   `python3 -m services.importer --inbox` beforehand — then the opening footage and the
   demo footage are literally the same pipeline, and you can say so).
2. Hero shot of the Spark + camera, one cable.
3. Screen capture, Watch pane → fall → Discord on a phone. Get the phone in frame.
4. Screen capture, Ask grounded → click a citation → video plays.
5. Screen capture, Ask escalated → job badge → refinement appends. **Do not cut this one.**
6. B-roll of the box.

### If you're over 5:00, cut in this order
1. The cost numbers (3:35) down to two lines: three seconds a caption, eighty percent
   skipped.
2. The "why nobody is watching" beat (0:25) — fold it into one sentence in the cold open.
3. Never cut Demo 3. It is the only thing in the video nobody else at the hackathon has.

### Lines worth memorising
- *"The camera was never the problem. Nobody was watching it."*
- *"An alert cannot be un-sent."*
- *"You can see the system change its mind."*
- *"The network has been off this entire video."*
