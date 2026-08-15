/* TIMELINE pane — SPEC §11.4, rendering the append-only action log (§6.4).
 *
 * This pane renders exactly the rows M3 introspects through read_action_log. One source,
 * no drift — a parallel store would eventually have the agent contradicting the screen.
 *
 * Rows are never mutated. Verification and retraction arrive as SEPARATE entries
 * carrying parent_id, and this file folds them for display only: the original stays
 * visible, struck through when it was retracted, with the amendment linked beneath it.
 * Showing the retraction is the point — a visible retraction is the thesis restated.
 *
 * Ordering rule, learned the hard way from shared/timecode.py's DST tests: SORT AND
 * COMPARE IN UTC. Two distinct instants inside a fall-back hour compare EQUAL as local
 * wall clock (PEP 495), so ordering on a converted string silently collapses an hour of
 * history once a year. Local strings here are output only, including the day headings.
 */
window.SPARK = window.SPARK || {};

SPARK.timeline = (function () {
  "use strict";

  var els = {};
  var cfg = {};

  var GLYPH = {
    raise_alert: "⚠",
    file_ticket: "🎫",
    save_clip: "⧉",
  };

  function init(root, config) {
    cfg = config || {};
    els.root = root;
    els.list = root.querySelector("[data-timeline-list]");
    els.summary = root.querySelector("[data-timeline-summary]");
    return refresh();
  }

  function refresh() {
    return SPARK.data
      .loadActions()
      .then(render)
      .catch(function (err) {
        els.list.innerHTML = "";
        var e = div("form-msg form-msg--bad");
        e.textContent = err.message;
        els.list.appendChild(e);
      });
  }

  function render(entries) {
    els.list.innerHTML = "";

    var byId = {};
    entries.forEach(function (e) {
      byId[e.entry_id] = e;
    });

    var roots = [];
    var children = {};
    entries.forEach(function (e) {
      if (e.parent_id && byId[e.parent_id]) {
        (children[e.parent_id] = children[e.parent_id] || []).push(e);
      } else {
        // An amendment whose parent is outside the loaded range is still shown; hiding
        // it would be the one thing an append-only log must never do.
        roots.push(e);
      }
    });

    var byUtc = function (a, b) {
      return SPARK.time.epochMs(a.ts) - SPARK.time.epochMs(b.ts);
    };
    roots.sort(byUtc);
    Object.keys(children).forEach(function (k) {
      children[k].sort(byUtc);
    });

    var retracted = 0;
    var day = null;
    roots.forEach(function (entry) {
      var kids = children[entry.entry_id] || [];
      var final = kids.length ? kids[kids.length - 1] : entry;
      if (final.status === "retracted") retracted += 1;

      // Heading label is local (a human's day); the grouping still walks UTC order.
      var label = SPARK.time.fmt(entry.ts, "%Y-%m-%d");
      if (label !== day) {
        day = label;
        var h = div("day-head");
        h.textContent = label;
        h.title = "day boundary in " + SPARK.time.tzLabel();
        els.list.appendChild(h);
      }

      els.list.appendChild(entryNode(entry, kids, final));
    });

    els.summary.textContent =
      entries.length + " entries · " + roots.length + " actions · " + retracted + " retracted";
    els.summary.classList.toggle("has-retraction", retracted > 0);
  }

  function entryNode(entry, kids, final) {
    var node = div("entry");
    node.dataset.entryId = entry.entry_id;
    if (final.status === "retracted") node.classList.add("entry--retracted");
    else if (final.status === "verified") node.classList.add("entry--verified");

    var head = div("entry-head");
    head.appendChild(timeChip(entry.ts, entry.t_start, entry.t_end, "action " + entry.entry_id));
    head.appendChild(span("kind", (GLYPH[entry.action] || "•") + " " + entry.action));
    head.appendChild(
      span("entry-task", entry.task_id ? entry.task_id : "via chat (M3)")
    );
    head.appendChild(statusChip(entry.status, entry.action));
    node.appendChild(head);

    if (entry.reason) {
      var reason = document.createElement("p");
      reason.className = "entry-reason";
      reason.textContent = entry.reason;
      node.appendChild(reason);
    }

    var footer = div("entry-foot");
    footer.appendChild(
      rangeChip(entry.t_start, entry.t_end, "footage this action is about")
    );
    if (entry.clip_path) footer.appendChild(clipChip(entry));
    if (entry.job_id) footer.appendChild(span("muted", "job " + entry.job_id));
    node.appendChild(footer);

    kids.forEach(function (kid) {
      node.appendChild(amendmentNode(kid));
    });

    return node;
  }

  function amendmentNode(kid) {
    var node = div("amend amend--" + kid.status);
    var head = div("amend-head");
    head.appendChild(span("amend-elbow", "└"));
    head.appendChild(timeChip(kid.ts, kid.t_start, kid.t_end, "amendment " + kid.entry_id));
    if (kid.status === "retracted") {
      head.appendChild(span("amend-verdict amend-verdict--bad", "✗ RETRACTED"));
    } else if (kid.status === "verified") {
      head.appendChild(span("amend-verdict amend-verdict--ok", "✓ verified"));
    } else {
      head.appendChild(span("amend-verdict", kid.status));
    }
    if (kid.clip_path) head.appendChild(clipChip(kid));
    node.appendChild(head);

    if (kid.reason) {
      var reason = document.createElement("p");
      reason.className = "amend-reason";
      reason.textContent = kid.reason;
      node.appendChild(reason);
    }
    var note = div("amend-note");
    note.textContent =
      "separate append-only row " + kid.entry_id + " → parent " + kid.parent_id +
      " · the original above was not modified";
    node.appendChild(note);
    return node;
  }

  function statusChip(status, action) {
    if (action === "save_clip" && status === "unverified") {
      var s = span("status status--novrfy", "(no verify)");
      s.title = "low stakes: fires on stage-2 confidence, verification is not attempted (SPEC §6.3)";
      return s;
    }
    var chip = span("status status--" + status, status);
    if (status === "unverified") {
      chip.title = "fired provisionally on stage-2 confidence; stage 3 amends it (SPEC §6.3)";
    }
    return chip;
  }

  function timeChip(ts, tStart, tEnd, tooltip) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "chip chip--ts";
    b.textContent = SPARK.time.fmt(ts);
    b.title =
      (tooltip ? tooltip + "\n" : "") +
      "row appended " + ts + " (UTC, as stored)\nfootage " + tStart + " → " + tEnd;
    b.addEventListener("click", function () {
      SPARK.player.scrubTo(tStart, tEnd, { source: tooltip || "action log" });
    });
    return b;
  }

  function rangeChip(tStart, tEnd, tooltip) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "chip chip--cite";
    b.textContent = SPARK.time.range(tStart, tEnd);
    b.title = (tooltip ? tooltip + "\n" : "") + tStart + " → " + tEnd + " (UTC, as stored)";
    b.addEventListener("click", function () {
      SPARK.player.scrubTo(tStart, tEnd, { source: tooltip || "action log" });
    });
    return b;
  }

  function clipChip(entry) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "chip chip--action";
    b.textContent = "▶ clip";
    b.title = entry.clip_path + "\nopened by time range, never by filename (invariant 3)";
    b.addEventListener("click", function () {
      SPARK.player.scrubTo(entry.t_start, entry.t_end, { source: "clip · " + entry.entry_id });
    });
    return b;
  }

  function div(cls) {
    var d = document.createElement("div");
    d.className = cls;
    return d;
  }
  function span(cls, text) {
    var s = document.createElement("span");
    if (cls) s.className = cls;
    s.textContent = text;
    return s;
  }

  return { init: init, refresh: refresh };
})();
