/* The index browser — ui/browse.html.
 *
 * One row per analysis window, newest first, paged server-side. The console answers
 * "what happened?"; this answers "what is in the index that the answer came from?".
 *
 * Three rules this file inherits from the console and does not get to relax:
 *   - UTC underneath, converted at render, only through SPARK.time (SPEC §11.5).
 *   - Footage is fetched by TIME RANGE, never by filename (invariant 3). The ▶ button
 *     links to /api/video?t_from&t_to; `segment` is displayed as provenance only.
 *   - Every tunable comes from config or from the server payload. The page-size choices
 *     below are bounded by agent.browse.max_page_size, not by a number typed here.
 *
 * Paging is server-side on purpose. The corpus is ~17k windows/day at a 5 s stride;
 * fetching it all to slice it in the browser works on a fixture and dies on day two.
 */
window.SPARK = window.SPARK || {};

SPARK.browse = (function () {
  "use strict";

  var cfg = {};
  var els = {};

  // The request. Everything about what is on screen lives here and in the URL.
  var state = {
    offset: 0,
    limit: null, // filled from config once it loads
    q: "",
    tier: "",
    gated: "", // "" = all | "false" = captioned | "true" = gate-skipped
    rangeSeconds: 0, // 0 = all time
    newestFirst: true,
  };

  var last = { total: 0, pages: 0, page: 0, previewChars: null };
  var inflight = 0;

  // ---------------------------------------------------------------------------------
  // boot
  // ---------------------------------------------------------------------------------
  function init() {
    els = {
      banner: document.querySelector("[data-banner]"),
      q: document.querySelector("[data-filter-q]"),
      gated: document.querySelector("[data-filter-gated]"),
      tier: document.querySelector("[data-filter-tier]"),
      range: document.querySelector("[data-filter-range]"),
      order: document.querySelector("[data-filter-order]"),
      limit: document.querySelector("[data-filter-limit]"),
      form: document.querySelector("[data-filters]"),
      reset: document.querySelector("[data-filter-reset]"),
      body: document.querySelector("[data-rows-body]"),
      empty: document.querySelector("[data-rows-empty]"),
      scroll: document.querySelector("[data-rows-scroll]"),
      summary: document.querySelector("[data-rows-summary]"),
      pagerState: document.querySelector("[data-pager-state]"),
      first: document.querySelector("[data-page-first]"),
      prev: document.querySelector("[data-page-prev]"),
      next: document.querySelector("[data-page-next]"),
      last: document.querySelector("[data-page-last]"),
      corpus: {
        total: document.querySelector("[data-corpus-total]"),
        captioned: document.querySelector("[data-corpus-captioned]"),
        gated: document.querySelector("[data-corpus-gated]"),
        skiprate: document.querySelector("[data-corpus-skiprate]"),
        health: document.querySelector("[data-corpus-health]"),
      },
    };

    return SPARK.data
      .loadConfig()
      .then(function (loaded) {
        cfg = loaded || {};
        SPARK.time.configure(cfg);
        header();
        pageSizes();
        readUrl();
        writeControls();
        wire();
        return load();
      })
      .catch(fail);
  }

  function header() {
    var mode = document.querySelector("[data-mode-pill]");
    if (SPARK.data.isMock()) {
      mode.textContent = "MOCK FIXTURES";
      mode.className = "pill pill--mock";
      mode.title =
        "Paging ui/mock/chunks.json in the browser. The real index is served by M3 at " +
        "/api/index — flip MODE in ui/static/data.js, or add ?mode=live.";
    } else {
      mode.textContent = "LIVE · M3";
      mode.className = "pill pill--live";
    }

    var tz = document.querySelector("[data-tz-pill]");
    tz.textContent = SPARK.time.tzLabel();
    tz.title =
      "SPEC §11.5: UTC everywhere underneath, converted once at render. " +
      "Hover any timestamp for the Z-suffixed value it came from.";

    document.querySelector("[data-camera-id]").textContent = SPARK.data.cfg(cfg, "camera.id", "cam01");

    var clock = document.querySelector("[data-clock-pill]");
    setInterval(function () {
      clock.textContent = SPARK.time.fmt(new Date().toISOString());
    }, 500);
  }

  /** Page-size options, derived from config rather than hardcoded in the markup. */
  function pageSizes() {
    var dflt = SPARK.data.cfg(cfg, "agent.browse.page_size", 25);
    var max = SPARK.data.cfg(cfg, "agent.browse.max_page_size", 200);
    var sizes = [dflt, dflt * 2, dflt * 4, max].filter(function (n, i, arr) {
      return n > 0 && n <= max && arr.indexOf(n) === i;
    });
    sizes.sort(function (a, b) {
      return a - b;
    });
    sizes.forEach(function (n) {
      var opt = document.createElement("option");
      opt.value = String(n);
      opt.textContent = n + " rows";
      els.limit.appendChild(opt);
    });
    state.limit = dflt;
  }

  // ---------------------------------------------------------------------------------
  // URL <-> state. A page of the index should be linkable and survive a reload — the
  // whole point of a browser is being able to say "look at this row".
  // ---------------------------------------------------------------------------------
  function readUrl() {
    var p = new URLSearchParams(window.location.search);
    state.q = p.get("q") || "";
    state.tier = p.get("tier") || "";
    state.gated = p.get("gated") || "";
    state.rangeSeconds = parseInt(p.get("range") || "0", 10) || 0;
    state.newestFirst = p.get("order") !== "oldest";
    var limit = parseInt(p.get("limit") || "0", 10);
    if (limit > 0) state.limit = limit;
    var page = parseInt(p.get("page") || "1", 10);
    state.offset = page > 1 ? (page - 1) * state.limit : 0;
  }

  function writeUrl() {
    var p = new URLSearchParams();
    if (state.q) p.set("q", state.q);
    if (state.tier) p.set("tier", state.tier);
    if (state.gated) p.set("gated", state.gated);
    if (state.rangeSeconds) p.set("range", String(state.rangeSeconds));
    if (!state.newestFirst) p.set("order", "oldest");
    if (state.limit) p.set("limit", String(state.limit));
    var page = Math.floor(state.offset / state.limit) + 1;
    if (page > 1) p.set("page", String(page));
    // The mode override is the one param that is not ours; carry it through so
    // ?mode=live survives paging.
    var mode = new URLSearchParams(window.location.search).get("mode");
    if (mode) p.set("mode", mode);
    var qs = p.toString();
    window.history.replaceState(null, "", qs ? "?" + qs : window.location.pathname);
  }

  function writeControls() {
    els.q.value = state.q;
    els.tier.value = state.tier;
    els.gated.value = state.gated;
    els.range.value = state.rangeSeconds ? String(state.rangeSeconds) : "";
    els.order.value = state.newestFirst ? "newest" : "oldest";
    els.limit.value = String(state.limit);
  }

  function wire() {
    els.form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      applyFilters();
    });
    // The selects are a single click each; making them wait for "apply" is a click the
    // user should not have to find. The text box keeps the explicit submit.
    [els.gated, els.tier, els.range, els.order, els.limit].forEach(function (el) {
      el.addEventListener("change", applyFilters);
    });
    els.reset.addEventListener("click", function () {
      state.q = "";
      state.tier = "";
      state.gated = "";
      state.rangeSeconds = 0;
      state.newestFirst = true;
      state.offset = 0;
      writeControls();
      load();
    });

    els.first.addEventListener("click", function () {
      goto(0);
    });
    els.prev.addEventListener("click", function () {
      goto(state.offset - state.limit);
    });
    els.next.addEventListener("click", function () {
      goto(state.offset + state.limit);
    });
    els.last.addEventListener("click", function () {
      goto((Math.max(1, last.pages) - 1) * state.limit);
    });

    document.addEventListener("keydown", function (ev) {
      var typing = ev.target && /^(INPUT|TEXTAREA|SELECT)$/.test(ev.target.tagName);
      if (typing || ev.metaKey || ev.ctrlKey || ev.altKey) return;
      if (ev.key === "ArrowLeft") goto(state.offset - state.limit);
      else if (ev.key === "ArrowRight") goto(state.offset + state.limit);
    });
  }

  function applyFilters() {
    state.q = els.q.value.trim();
    state.tier = els.tier.value;
    state.gated = els.gated.value;
    state.rangeSeconds = parseInt(els.range.value || "0", 10) || 0;
    state.newestFirst = els.order.value !== "oldest";
    state.limit = parseInt(els.limit.value, 10) || state.limit;
    state.offset = 0; // a new filter invalidates the page number, always
    load();
  }

  function goto(offset) {
    var maxOffset = Math.max(0, (Math.max(1, last.pages) - 1) * state.limit);
    var next = Math.min(Math.max(0, offset), maxOffset);
    if (next === state.offset) return;
    state.offset = next;
    load();
  }

  // ---------------------------------------------------------------------------------
  // fetch + render
  // ---------------------------------------------------------------------------------
  function load() {
    writeUrl();
    var token = ++inflight;
    els.summary.textContent = "loading…";
    var params = {
      offset: state.offset,
      limit: state.limit,
      q: state.q,
      tier: state.tier,
      gated: state.gated,
      newest_first: state.newestFirst,
    };
    if (state.rangeSeconds) {
      // Computed from the epoch, not from a local wall clock: this is arithmetic on an
      // instant, not a timezone conversion, so it stays out of time.js's territory.
      params.t_from = new Date(Date.now() - state.rangeSeconds * 1000).toISOString();
    }
    return SPARK.data
      .loadIndexPage(params)
      .then(function (resp) {
        if (token !== inflight) return; // a newer request already won
        render(resp);
      })
      .catch(fail);
  }

  function render(resp) {
    last.total = resp.total || 0;
    last.pages = resp.pages || 0;
    last.page = resp.page || 0;
    if (resp.caption_preview_chars) last.previewChars = resp.caption_preview_chars;

    corpus(resp.stats || {});

    var rows = resp.chunks || [];
    // An offset past the end — a stale link, or rows that aged out from under a filter.
    // Land on the last real page instead of showing an empty one with a live total.
    if (!rows.length && last.total > 0 && state.offset > 0) {
      var clamped = Math.max(0, (Math.max(1, last.pages) - 1) * state.limit);
      // Only retry if that is somewhere new — otherwise render the empty page and let
      // the reader see it, rather than reloading forever against a moving corpus.
      if (clamped !== state.offset) {
        state.offset = clamped;
        load();
        return;
      }
    }
    els.body.textContent = "";
    rows.forEach(function (chunk) {
      els.body.appendChild(row(chunk));
    });

    els.empty.hidden = rows.length > 0;
    if (!rows.length) {
      els.empty.textContent = last.total
        ? "No rows on this page — the corpus moved under the filter. Try page 1."
        : "Nothing in the index matches these filters." +
          (SPARK.data.isMock() ? " (Mock fixtures hold a dozen rows.)" : "");
    }

    var shownFrom = last.total ? state.offset + 1 : 0;
    var shownTo = state.offset + rows.length;
    els.summary.textContent = last.total
      ? shownFrom + "–" + shownTo + " of " + last.total
      : "0 windows";
    els.pagerState.textContent = last.total
      ? "page " + last.page + " of " + last.pages
      : "no pages";

    var atStart = state.offset <= 0;
    var atEnd = state.offset + state.limit >= last.total;
    els.first.disabled = atStart;
    els.prev.disabled = atStart;
    els.next.disabled = atEnd;
    els.last.disabled = atEnd;

    els.scroll.scrollTop = 0;
  }

  function corpus(stats) {
    els.corpus.total.textContent = num(stats.total);
    els.corpus.captioned.textContent = num(stats.captioned);
    els.corpus.gated.textContent = num(stats.gated);
    var rate = typeof stats.skip_rate === "number" ? Math.round(stats.skip_rate * 100) : null;
    els.corpus.skiprate.textContent = rate === null ? "–" : rate + "%";

    var health = stats.gate_health || "";
    els.corpus.health.textContent = health ? "skip rate · " + health : "skip rate";
    els.corpus.health.className = "corpus-label" + (health === "low" ? " corpus-label--warn" : "");
    els.corpus.health.title =
      "SPEC §2.3: the fraction of windows the detector gate skipped before inference. " +
      "Measured against ingest.gate.warn_skip_rate — 'low' means the gate is mistuned " +
      "and the live path is doing work it does not need to.";
  }

  function num(value) {
    return typeof value === "number" ? String(value) : "–";
  }

  // ---------------------------------------------------------------------------------
  // one row
  // ---------------------------------------------------------------------------------
  function row(chunk) {
    var tr = document.createElement("tr");
    tr.className = chunk.gated ? "row row--gated" : "row";

    tr.appendChild(timeCell(chunk));
    tr.appendChild(stateCell(chunk));
    tr.appendChild(captionCell(chunk));
    tr.appendChild(sourceCell(chunk));
    return tr;
  }

  function timeCell(chunk) {
    var td = document.createElement("td");
    td.className = "col-time";

    var clock = document.createElement("div");
    clock.className = "row-clock";
    clock.textContent = SPARK.time.range(chunk.t_start, chunk.t_end);
    // The stored value, verbatim, beside the rendered one — SPEC §11.5 auditable rather
    // than asserted.
    clock.title = SPARK.time.utc(chunk.t_start) + " → " + SPARK.time.utc(chunk.t_end) + " (as stored)";
    td.appendChild(clock);

    var date = document.createElement("div");
    date.className = "row-date muted";
    var seconds = SPARK.time.durationSeconds(chunk.t_start, chunk.t_end);
    date.textContent = SPARK.time.fmt(chunk.t_start, "%Y-%m-%d") + " · " + SPARK.time.secs(seconds);
    td.appendChild(date);

    return td;
  }

  function stateCell(chunk) {
    var td = document.createElement("td");
    td.className = "col-state";

    var tag = document.createElement("span");
    if (chunk.gated) {
      tag.className = "tag tag--gated";
      tag.textContent = "skipped";
      tag.title =
        "The detector gate found no motion in this window, so it never reached the VLM " +
        "(SPEC §2.3). The row is kept: a gap in the record stream is indistinguishable " +
        "from a crashed ingest.";
    } else {
      tag.className = "tag tag--" + (chunk.tier === "rollup" ? "rollup" : "live");
      tag.textContent = chunk.tier || "live";
      tag.title =
        chunk.tier === "rollup"
          ? "A merged window (SPEC §3.3) — the search tier."
          : "A live-path caption: enable_reasoning=false, ~80 tokens (invariant 6).";
    }
    td.appendChild(tag);
    return td;
  }

  function captionCell(chunk) {
    var td = document.createElement("td");
    td.className = "col-caption";

    if (!chunk.caption) {
      var none = document.createElement("span");
      none.className = "muted caption-none";
      none.textContent = "— no caption (gate skipped this window before inference)";
      td.appendChild(none);
      return td;
    }

    var limit = last.previewChars || SPARK.data.cfg(cfg, "agent.browse.caption_preview_chars", 240);
    var text = document.createElement("span");
    text.className = "caption";
    var long = chunk.caption.length > limit;
    text.textContent = long ? chunk.caption.slice(0, limit).trim() + "…" : chunk.caption;
    td.appendChild(text);

    if (long) {
      var more = document.createElement("button");
      more.type = "button";
      more.className = "chip chip--quiet caption-more";
      more.textContent = "more";
      more.addEventListener("click", function () {
        var expanded = text.textContent === chunk.caption;
        text.textContent = expanded ? chunk.caption.slice(0, limit).trim() + "…" : chunk.caption;
        more.textContent = expanded ? "more" : "less";
      });
      td.appendChild(more);
    }

    if (state.q) highlight(text, state.q);
    return td;
  }

  /** Mark the searched substring inside an already-rendered caption. Operates on the
   *  text node, never on a built HTML string — a caption is model output and goes into
   *  the DOM as text, always. */
  function highlight(node, needle) {
    var haystack = node.textContent;
    var at = haystack.toLowerCase().indexOf(needle.toLowerCase());
    if (at === -1) return;
    var mark = document.createElement("mark");
    mark.textContent = haystack.slice(at, at + needle.length);
    node.textContent = "";
    node.appendChild(document.createTextNode(haystack.slice(0, at)));
    node.appendChild(mark);
    node.appendChild(document.createTextNode(haystack.slice(at + needle.length)));
  }

  function sourceCell(chunk) {
    var td = document.createElement("td");
    td.className = "col-source";

    // Footage by RANGE, never by filename (invariant 3). The segment below is shown as
    // provenance — it is what pts_offset is an offset into — and is deliberately not a
    // link, because a link to a file is the exact mistake the invariant forbids.
    if (!SPARK.data.isMock()) {
      var play = document.createElement("a");
      play.className = "chip chip--action";
      play.textContent = "▶ range";
      play.target = "_blank";
      play.rel = "noopener";
      play.href =
        SPARK.data.endpoints.video +
        "?t_from=" +
        encodeURIComponent(chunk.t_start) +
        "&t_to=" +
        encodeURIComponent(chunk.t_end);
      play.title = "Fetch this window's footage by time range — it may span two segment files.";
      td.appendChild(play);
    }

    var seg = document.createElement("div");
    seg.className = "row-segment muted";
    seg.textContent = chunk.segment + " @ " + Number(chunk.pts_offset).toFixed(1) + "s";
    seg.title =
      "PTS restarts at zero in every segment file, so this offset is meaningless " +
      "without the wall-clock range beside it (invariant 2).";
    td.appendChild(seg);

    var id = document.createElement("div");
    id.className = "row-id muted";
    id.textContent = chunk.chunk_id;
    id.title = "chunk_id — click to copy";
    id.addEventListener("click", function () {
      copy(chunk.chunk_id, id);
    });
    td.appendChild(id);

    return td;
  }

  function copy(text, el) {
    var done = function () {
      var was = el.textContent;
      el.textContent = "copied";
      setTimeout(function () {
        el.textContent = was;
      }, 900);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {
        /* a denied clipboard permission is not worth a banner */
      });
    }
  }

  // ---------------------------------------------------------------------------------
  function fail(err) {
    els.banner.hidden = false;
    els.banner.className = "banner banner--bad";
    els.banner.textContent = err && err.message ? err.message : String(err);
    els.summary.textContent = "failed";
    console.error(err);
  }

  return { init: init };
})();

SPARK.browse.init();
