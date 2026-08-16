/* The delete-old-footage control (topbar button + confirmation panel).
 *
 * Every other pane on this page reads. This one destroys, and nothing puts it back: the
 * archive is the only thing the deep worker can re-read (CLAUDE.md invariant 7), so an
 * hour swept from disk can never be re-analysed — only remembered through whatever
 * caption outlived it, if the caption outlived it, which here it deliberately does not.
 *
 * Hence the shape: click asks the server what WOULD go, the panel prints the real
 * counts, and only a second, separate click on a differently-labelled button deletes.
 * A window.confirm() would have been fewer lines and would have shown the operator a
 * sentence instead of the file count, the caption count and the gigabytes.
 */
window.SPARK = window.SPARK || {};

SPARK.retention = (function () {
  "use strict";

  var refs = null;
  var cfg = null;
  var plan = null;
  var busy = false;

  function init(cfgIn) {
    cfg = cfgIn;
    refs = {
      button: document.querySelector("[data-purge]"),
      panel: document.querySelector("[data-purge-panel]"),
      summary: document.querySelector("[data-purge-summary]"),
      note: document.querySelector("[data-purge-note]"),
      confirm: document.querySelector("[data-purge-confirm]"),
      cancel: document.querySelector("[data-purge-cancel]"),
      reload: document.querySelector("[data-purge-reload]"),
    };
    if (!refs.button) return Promise.resolve();

    refs.button.textContent = "delete > " + ageLabel(maxAge());

    // Mock mode has no archive to delete and must not pretend otherwise. Every other
    // pane degrades to fixtures; a destructive control that degrades to a scripted
    // success is the one kind of mock that could cost someone real footage.
    if (SPARK.data.isMock()) {
      refs.button.disabled = true;
      refs.button.title =
        "live mode only — this page is rendering ui/mock/*.json and there is no archive behind it";
      return Promise.resolve();
    }

    refs.button.addEventListener("click", open);
    refs.cancel.addEventListener("click", close);
    refs.confirm.addEventListener("click", run);
    refs.reload.addEventListener("click", function () {
      window.location.reload();
    });
    return Promise.resolve();
  }

  function maxAge() {
    return Number(SPARK.data.cfg(cfg, "retention.max_age_seconds", 10800));
  }

  /** "3 h" / "45 min" / "90 s" — the age as an operator would say it. */
  function ageLabel(seconds) {
    var s = Number(seconds) || 0;
    if (s >= 3600) return round1(s / 3600) + " h";
    if (s >= 60) return round1(s / 60) + " min";
    return round1(s) + " s";
  }

  function round1(n) {
    return String(Math.round(n * 10) / 10);
  }

  function bytesLabel(n) {
    var b = Number(n) || 0;
    if (b >= 1e9) return (b / 1e9).toFixed(1) + " GB";
    if (b >= 1e6) return Math.round(b / 1e6) + " MB";
    if (b >= 1e3) return Math.round(b / 1e3) + " kB";
    return b + " B";
  }

  function plural(n, one, many) {
    return n + " " + (n === 1 ? one : many);
  }

  // -----------------------------------------------------------------------------------
  // open — ask the server what would go, and say so
  // -----------------------------------------------------------------------------------
  function open() {
    if (busy) return;
    show();
    refs.reload.hidden = true;
    refs.confirm.hidden = false;
    refs.confirm.disabled = true;
    refs.cancel.disabled = false;
    refs.summary.textContent = "checking what is older than " + ageLabel(maxAge()) + "…";
    refs.note.textContent = "";

    SPARK.data
      .retentionPlan()
      .then(function (p) {
        plan = p;
        render(p);
      })
      .catch(function (err) {
        plan = null;
        refs.summary.textContent = "could not read the archive: " + err.message;
        refs.confirm.disabled = true;
      });
  }

  function render(p) {
    var files = p.segment_count || 0;
    var chunks = p.chunk_count || 0;

    if (p.empty) {
      refs.summary.textContent =
        "Nothing is older than " + ageLabel(p.older_than_seconds) + " — nothing to delete.";
      refs.confirm.disabled = true;
    } else {
      refs.summary.textContent =
        "Delete " +
        plural(files, "segment file", "segment files") +
        " (" +
        bytesLabel(p.bytes_to_free) +
        ") and " +
        plural(chunks, "caption", "captions") +
        " ending before " +
        SPARK.time.fmt(p.cutoff) +
        "? This cannot be undone.";
      refs.confirm.disabled = false;
      // Label the button with whatever is actually going. "delete 0 B" is the wrong
      // thing to put on a button that is about to remove a thousand captions — which is
      // exactly the case when the archive is already gone and only the index remains.
      refs.confirm.textContent =
        "delete " + (files ? bytesLabel(p.bytes_to_free) : plural(chunks, "caption", "captions"));
    }

    // The three things a reader will otherwise assume went and did not.
    var notes = [];
    var kept = p.kept_live_segments || [];
    if (kept.length) {
      notes.push(
        "keeping " +
          kept.join(", ") +
          " — the recorder is still writing, and unlinking an open segment leaves it unplayable"
      );
    }
    if (p.archive_missing) {
      notes.push("no archive directory on disk; only index rows would go");
    }
    notes.push("evidence clips and the action log are not touched");
    refs.note.textContent = notes.join(" · ");
  }

  // -----------------------------------------------------------------------------------
  // run — the irreversible half
  // -----------------------------------------------------------------------------------
  function run() {
    if (busy || !plan || plan.empty) return;
    busy = true;
    refs.confirm.disabled = true;
    refs.cancel.disabled = true;
    refs.summary.textContent = "deleting…";

    SPARK.data
      .applyRetention(plan.older_than_seconds)
      .then(function (result) {
        busy = false;
        refs.confirm.hidden = true;
        refs.cancel.disabled = false;
        refs.cancel.textContent = "close";
        refs.summary.textContent =
          "Deleted " +
          plural(result.segments_deleted, "segment file", "segment files") +
          " (" +
          bytesLabel(result.bytes_freed) +
          ") and " +
          plural(result.chunks_deleted, "caption", "captions") +
          ".";
        // Every pane on this page is now holding chunks, a player range and an index
        // page that describe footage which no longer exists. Offering a reload beats
        // refreshing them piecemeal and beats silently leaving stale rows on screen —
        // and it is offered rather than taken, because a page that navigates itself
        // mid-demo is its own kind of surprise.
        refs.note.textContent = (result.errors || []).length
          ? "could not delete: " + result.errors.join("; ")
          : "the panes above are still showing the deleted range until you reload";
        refs.reload.hidden = false;
      })
      .catch(function (err) {
        busy = false;
        refs.cancel.disabled = false;
        refs.confirm.disabled = false;
        refs.summary.textContent = "delete failed: " + err.message;
      });
  }

  function show() {
    refs.panel.hidden = false;
  }

  function close() {
    if (busy) return;
    refs.panel.hidden = true;
    refs.confirm.hidden = false;
    refs.cancel.textContent = "cancel";
    refs.reload.hidden = true;
    plan = null;
  }

  return { init: init };
})();
