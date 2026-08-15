/* The shared archive player (SPEC §11.1: "three panes over one shared archive player").
 *
 * CLAUDE.md invariant 3: fetching video takes a TIME RANGE, never a filename. Every
 * entry point here is scrubTo(t_start, t_end); there is deliberately no scrubToFile().
 * The segment list under the transport is *output* — what the range resolved to — and
 * when a range straddles a minute boundary the panel says so, because that stitching is
 * the thing most likely to be quietly broken and never noticed.
 *
 * In mock mode the picture is a drawn placeholder, clearly marked, with the wall-clock
 * overlay burned in the way ingest burns it (after the resize, in UTC — invariant 8).
 * In live mode the same range becomes GET /api/video?t_from=..&t_to=.. on a <video>.
 */
window.SPARK = window.SPARK || {};

SPARK.player = (function () {
  "use strict";

  var els = {};
  var cfg = {};
  var chunks = [];
  var state = {
    tStart: null, // ISO Z
    tEnd: null, // ISO Z
    posMs: 0, // ms from range start
    playing: false,
    source: "",
  };
  var raf = null;
  var lastTick = 0;

  function init(root, config, chunkRecords) {
    cfg = config || {};
    chunks = chunkRecords || [];
    els.root = root;
    els.range = root.querySelector("[data-player-range]");
    els.source = root.querySelector("[data-player-source]");
    els.canvas = root.querySelector("[data-player-canvas]");
    els.video = root.querySelector("[data-player-video]");
    els.clock = root.querySelector("[data-player-clock]");
    els.scrub = root.querySelector("[data-player-scrub]");
    els.play = root.querySelector("[data-player-play]");
    els.segments = root.querySelector("[data-player-segments]");
    els.fetchline = root.querySelector("[data-player-fetch]");

    els.play.addEventListener("click", toggle);
    els.scrub.addEventListener("input", function () {
      state.posMs = (parseFloat(els.scrub.value) / 1000) * durationMs();
      // Seek the element too, or the slider moves the digits while the picture sits
      // still — the same class of bug as play() not starting it.
      if (usingVideo() && isFinite(els.video.duration)) {
        els.video.currentTime = Math.min(state.posMs / 1000, els.video.duration);
      }
      draw();
    });
    window.addEventListener("resize", draw);
    draw();
  }

  function setChunks(records) {
    chunks = records || [];
  }

  function durationMs() {
    if (!state.tStart || !state.tEnd) return 0;
    return SPARK.time.epochMs(state.tEnd) - SPARK.time.epochMs(state.tStart);
  }

  /** THE entry point. Everything that cites a moment calls this with a time range. */
  function scrubTo(tStart, tEnd, opts) {
    var o = opts || {};
    state.tStart = tStart;
    state.tEnd = tEnd;
    state.posMs = 0;
    state.source = o.source || "";
    resolveSegments(tStart, tEnd);
    if (!SPARK.data.isMock()) {
      // Live: the server stitches. The URL carries a range; it never carries a filename.
      els.video.src =
        SPARK.data.endpoints.video +
        "?t_from=" + encodeURIComponent(tStart) +
        "&t_to=" + encodeURIComponent(tEnd);
      els.video.hidden = false;
      els.canvas.hidden = true;
    }
    play();
    els.root.classList.add("player--flash");
    setTimeout(function () {
      els.root.classList.remove("player--flash");
    }, 400);
    draw();
  }

  /** Which segment files does this range touch? Answered from the chunk records, which
   *  carry `segment` + `pts_offset` per SPEC §3.1 — the UI does not re-derive filenames
   *  from wall clock. That derivation belongs to shared/timecode.py, server side. */
  function resolveSegments(tStart, tEnd) {
    var a = SPARK.time.epochMs(tStart);
    var b = SPARK.time.epochMs(tEnd);
    var hits = chunks.filter(function (c) {
      return SPARK.time.epochMs(c.t_end) > a && SPARK.time.epochMs(c.t_start) < b;
    });
    // Sort in UTC. Never sort on a converted local string: inside a fall-back hour two
    // distinct instants compare equal as local wall clock (PEP 495), and an hour of
    // history silently collapses.
    hits.sort(function (x, y) {
      return SPARK.time.epochMs(x.t_start) - SPARK.time.epochMs(y.t_start);
    });

    var segs = [];
    hits.forEach(function (c) {
      if (segs.indexOf(c.segment) === -1) segs.push(c.segment);
    });

    els.segments.innerHTML = "";
    if (!segs.length) {
      var none = document.createElement("span");
      none.className = "muted";
      none.textContent = "no indexed chunk in range — segments resolve server-side via shared/timecode.py";
      els.segments.appendChild(none);
    } else {
      segs.forEach(function (s, i) {
        var chip = document.createElement("span");
        chip.className = "seg";
        var first = hits.filter(function (c) {
          return c.segment === s;
        })[0];
        chip.textContent = s;
        chip.title = "pts_offset " + first.pts_offset.toFixed(2) + "s into this file";
        els.segments.appendChild(chip);
        if (i < segs.length - 1) {
          var join = document.createElement("span");
          join.className = "seg-join";
          join.textContent = "+";
          els.segments.appendChild(join);
        }
      });
      if (segs.length > 1) {
        var warn = document.createElement("span");
        warn.className = "seg-span";
        warn.textContent = "spans " + segs.length + " files · stitched";
        els.segments.appendChild(warn);
      }
    }

    els.fetchline.textContent =
      "fetch(t_start=" + tStart + ", t_end=" + tEnd + ")";
  }

  /** True when a real <video> is on screen rather than the mock-mode canvas.
   *
   *  The two surfaces need opposite treatment and only one is ever live: the canvas is
   *  driven frame-by-frame from `state.posMs`, while the <video> drives ITSELF and the
   *  playhead has to follow it. Forgetting the second half is what left live mode
   *  showing a single frame — the element had a src and was simply never started. */
  function usingVideo() {
    return !els.video.hidden && !!els.video.src;
  }

  function play() {
    if (!state.tStart) return;
    state.playing = true;
    els.play.textContent = "❚❚";
    els.play.setAttribute("aria-label", "pause");
    lastTick = performance.now();
    if (usingVideo()) {
      // Muted + playsinline are set in the markup precisely so this is allowed to
      // autoplay. A rejected promise is not fatal — the user can press play — but it
      // must not go unreported, because "nothing happens" is indistinguishable from
      // the bug this replaces.
      var started = els.video.play();
      if (started && typeof started.catch === "function") {
        started.catch(function (err) {
          state.playing = false;
          els.play.textContent = "▶";
          if (els.fetchline) els.fetchline.textContent = "autoplay blocked — press ▶ (" + err.name + ")";
        });
      }
    }
    if (!raf) raf = requestAnimationFrame(tick);
  }

  function pause() {
    state.playing = false;
    els.play.textContent = "▶";
    els.play.setAttribute("aria-label", "play");
    if (usingVideo()) els.video.pause();
  }

  function toggle() {
    if (state.playing) pause();
    else play();
  }

  function tick(now) {
    raf = null;
    if (state.playing) {
      if (usingVideo()) {
        // The element owns the clock here. Deriving the playhead from currentTime keeps
        // the burned-in wall clock, the segment strip and the digits agreeing with the
        // pixels actually on screen — a separately-ticking playhead would drift away
        // from the frame it claims to describe.
        state.posMs = Math.max(0, els.video.currentTime * 1000);
        if (els.video.ended) pause();
      } else {
        // Clamp both ends. requestAnimationFrame's timestamp and performance.now() can
        // disagree on the first frame (headless/virtual-time engines especially), and a
        // negative delta would park the playhead a second BEFORE the range it claims to
        // be inside — which reads as the wall-clock join being off by one.
        state.posMs = Math.max(0, state.posMs + Math.max(0, now - lastTick));
        if (state.posMs >= durationMs()) {
          state.posMs = durationMs();
          pause();
        }
      }
    }
    lastTick = now;
    draw();
    if (state.playing) raf = requestAnimationFrame(tick);
  }

  function playheadIso() {
    if (!state.tStart) return null;
    return new Date(SPARK.time.epochMs(state.tStart) + state.posMs).toISOString();
  }

  function draw() {
    if (!els.root) return;
    if (!state.tStart) {
      els.range.textContent = "no range loaded";
      els.clock.textContent = "--:--:--";
      drawPlaceholder("click any cited time range");
      return;
    }
    var dur = durationMs() / 1000;
    els.range.textContent =
      SPARK.time.range(state.tStart, state.tEnd) + "  ·  " + dur.toFixed(1) + "s";
    els.range.title = state.tStart + " → " + state.tEnd + " (UTC, as stored)";
    els.source.textContent = state.source || "";
    var pos = durationMs() ? state.posMs / durationMs() : 0;
    els.scrub.value = String(Math.round(pos * 1000));
    var head = playheadIso();
    els.clock.textContent = SPARK.time.fmt(head);
    els.clock.title = head + " (UTC, as stored)";
    if (!els.canvas.hidden) drawFrame(head, pos);
  }

  // -----------------------------------------------------------------------------------
  // Placeholder picture. Marked MOCK on every frame — nobody should mistake this for
  // footage. What it does carry faithfully is the burned-in wall clock: ingest writes it
  // AFTER the resize and the VLM reads it back for temporal localization (invariant 8),
  // so if it is ever illegible here it would be illegible there too.
  // -----------------------------------------------------------------------------------
  function fitCanvas() {
    var c = els.canvas;
    var w = c.clientWidth || 480;
    var h = Math.round((w * 9) / 16);
    var dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    c.style.height = h + "px";
    var ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  function drawPlaceholder(message) {
    var f = fitCanvas();
    var ctx = f.ctx;
    ctx.clearRect(0, 0, f.w, f.h);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, f.w, f.h);
    ctx.fillStyle = "#4b5563";
    ctx.font = "13px " + monoStack();
    ctx.textAlign = "center";
    ctx.fillText(message, f.w / 2, f.h / 2);
    ctx.textAlign = "left";
  }

  function monoStack() {
    return 'ui-monospace, "DejaVu Sans Mono", "Liberation Mono", Menlo, Consolas, monospace';
  }

  function drawFrame(headIso, pos) {
    var f = fitCanvas();
    var ctx = f.ctx;
    var w = f.w;
    var h = f.h;

    // ground / wall
    ctx.fillStyle = "#11161d";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#161d26";
    ctx.fillRect(0, h * 0.55, w, h * 0.45);

    // fire door + keep-clear hatching
    var doorX = w * 0.62;
    ctx.fillStyle = "#1d2733";
    ctx.fillRect(doorX, h * 0.18, w * 0.26, h * 0.42);
    ctx.strokeStyle = "#2b3644";
    ctx.strokeRect(doorX, h * 0.18, w * 0.26, h * 0.42);
    ctx.strokeStyle = "rgba(224,168,64,0.35)";
    ctx.lineWidth = 2;
    for (var i = 0; i < 8; i++) {
      var x0 = doorX + (i * w * 0.26) / 8;
      ctx.beginPath();
      ctx.moveTo(x0, h * 0.62);
      ctx.lineTo(x0 + w * 0.04, h * 0.86);
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    // the van, reversing toward the door across the range
    var vanW = w * 0.3;
    var vanH = h * 0.26;
    var vanX = w * 0.06 + pos * (w * 0.42);
    var vanY = h * 0.5;
    ctx.fillStyle = "#c9d3e0";
    ctx.fillRect(vanX, vanY, vanW, vanH);
    ctx.fillStyle = "#8e9bb0";
    ctx.fillRect(vanX + vanW * 0.72, vanY + vanH * 0.15, vanW * 0.26, vanH * 0.4);
    ctx.fillStyle = "#0b0e13";
    ctx.beginPath();
    ctx.arc(vanX + vanW * 0.22, vanY + vanH, h * 0.035, 0, Math.PI * 2);
    ctx.arc(vanX + vanW * 0.8, vanY + vanH, h * 0.035, 0, Math.PI * 2);
    ctx.fill();

    // rear doors swing open partway through — the detail the caption could not record
    if (pos > 0.26) {
      ctx.strokeStyle = "#e8eef7";
      ctx.lineWidth = 3;
      var swing = Math.min(1, (pos - 0.26) / 0.15);
      ctx.beginPath();
      ctx.moveTo(vanX, vanY);
      ctx.lineTo(vanX - vanW * 0.22 * swing, vanY - vanH * 0.18 * swing);
      ctx.moveTo(vanX, vanY + vanH);
      ctx.lineTo(vanX - vanW * 0.22 * swing, vanY + vanH + vanH * 0.18 * swing);
      ctx.stroke();
      ctx.lineWidth = 1;
    }

    // two figures from the right, late in the range
    if (pos > 0.72) {
      var fp = (pos - 0.72) / 0.28;
      ctx.fillStyle = "#7f8ea3";
      [0, 1].forEach(function (n) {
        var fx = w * (0.95 - fp * 0.22) + n * w * 0.04;
        ctx.fillRect(fx, h * 0.58, w * 0.014, h * 0.16);
        ctx.beginPath();
        ctx.arc(fx + w * 0.007, h * 0.565, w * 0.011, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // MOCK watermark — honesty about what this picture is
    ctx.fillStyle = "rgba(148,163,184,0.5)";
    ctx.font = "11px " + monoStack();
    ctx.fillText("MOCK FRAME · no archive attached", 10, 18);

    // burned-in wall clock, bottom, UTC per ingest.overlay.format
    var overlayFmt = SPARK.data.cfg(cfg, "ingest.overlay.format", "%Y-%m-%d %H:%M:%S UTC");
    var minH = SPARK.data.cfg(cfg, "ingest.overlay.min_height_px", 16);
    var stamp = SPARK.time.fmtUtc(headIso, overlayFmt);
    var fontPx = Math.max(minH - 4, Math.round(h * 0.055));
    ctx.font = fontPx + "px " + monoStack();
    var tw = ctx.measureText(stamp).width;
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(w / 2 - tw / 2 - 8, h - fontPx - 14, tw + 16, fontPx + 10);
    ctx.fillStyle = "#f4f7fb";
    ctx.fillText(stamp, w / 2 - tw / 2, h - 12);
  }

  return {
    init: init,
    setChunks: setChunks,
    scrubTo: scrubTo,
    play: play,
    pause: pause,
  };
})();
