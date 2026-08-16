/* WATCH pane — SPEC §11.3, the three-stage funnel (§6.2) and the three brakes (§6.4).
 *
 * The reason this pane exists in this shape: §6.2's stages are invisible by default and
 * the demo failure mode is not a missed event, it is thirty alerts for one. A cooldown
 * you can watch counting down while stages 1 and 2 keep matching *proves* the brake.
 * A task list would only assert it.
 *
 * Everything that counts is derived from an absolute timestamp on every frame:
 *     cooldown remaining = cooldown_seconds - (now - last_fired_ts)
 *     sustain elapsed    = now - stage2.since
 * Never a decrementing counter. A paused tab, a slow poll or a dropped frame must not
 * be able to make a brake look shorter than it is.
 */
window.SPARK = window.SPARK || {};

SPARK.watch = (function () {
  "use strict";

  var cfg = {};
  var els = {};
  var tasks = [];
  var monitor = { tasks: [] };
  var cards = {}; // task_id -> {node, refs}
  /* The task_id whose inline editor is open, or null. The pane re-renders on a 1 s poll
   * and render() rebuilds every card from scratch, so without this the editor is
   * destroyed a second after it opens — the form vanishes mid-keystroke and the card
   * snaps back to its read-only state. Suspending the rebuild is the right trade: the
   * funnel readout goes stale for the seconds someone is typing, and it refreshes the
   * moment they save or cancel. Losing what they typed is not recoverable; a stale
   * sustain bar is. */
  var editingTaskId = null;
  var pollTimer = null;

  function init(root, config) {
    cfg = config || {};
    els.root = root;
    els.list = root.querySelector("[data-watch-list]");
    els.form = root.querySelector("[data-watch-form]");
    els.formMsg = root.querySelector("[data-watch-formmsg]");
    els.details = root.querySelector("details.task-form");
    els.summary = els.details ? els.details.querySelector("summary") : null;
    els.endpoint = root.querySelector("[data-watch-endpoint]");
    els.stamp = root.querySelector("[data-watch-stamp]");

    els.endpoint.textContent = "POST " + SPARK.data.endpoints.registerTask;
    prefillForm();
    els.form.addEventListener("submit", onRegister);

    // Closing the form must not depend on finding the <summary> again: on a laptop the
    // open form is taller than the pane, so its foot is clipped and the heading is easy
    // to lose. Reset on close so a half-typed task does not reappear later looking real.
    var cancel = root.querySelector("[data-watch-cancel]");
    if (cancel && els.details) {
      cancel.addEventListener("click", function () {
        closeForm();
      });
    }

    // Reopening is the one unambiguous "I am done reading that" signal, so it is what
    // clears the last registration's receipt — no timer, nothing that expires while the
    // user is still looking at it.
    if (els.details) {
      els.details.addEventListener("toggle", function () {
        if (els.details.open) resetSummary();
      });
    }

    var pollMs = SPARK.data.cfg(cfg, "ui.poll_interval_ms", 1000);
    setInterval(tickLive, 250);

    return refresh().then(function () {
      pollTimer = setInterval(refresh, pollMs);
    });
  }

  function refresh() {
    return Promise.all([SPARK.data.loadTasks(), SPARK.data.loadMonitorState()])
      .then(function (r) {
        tasks = r[0];
        monitor = r[1];
        render();
      })
      .catch(function (err) {
        console.error("[watch] " + err.message);
      });
  }

  function monitorRow(taskId) {
    var hit = (monitor.tasks || []).filter(function (t) {
      return t.task_id === taskId;
    })[0];
    if (hit) return hit;
    var task = tasks.filter(function (t) {
      return t.task_id === taskId;
    })[0];
    return task ? SPARK.data.syntheticMonitorRow(task) : null;
  }

  // -----------------------------------------------------------------------------------
  // Render — full rebuild on new server state only. The per-frame tick below touches
  // text and bar widths, so a countdown never blows away a card mid-read.
  // -----------------------------------------------------------------------------------
  function render() {
    // Never rebuild the list under an open editor — see `editingTaskId`.
    if (editingTaskId) return;
    els.list.innerHTML = "";
    cards = {};
    if (monitor.generated_at) {
      els.stamp.textContent = "funnel state " + SPARK.time.fmt(monitor.generated_at);
      els.stamp.title =
        monitor.generated_at +
        " (UTC, as stored)" +
        (monitor._rebased ? "\nmock fixture rebased to now so the brakes visibly run" : "");
    }
    tasks.forEach(function (task) {
      els.list.appendChild(taskCard(task, monitorRow(task.task_id)));
    });
    tickLive();
  }

  /* The inline editor for one task. Returns {el, reset}; the caller owns placement.
   *
   * Only fields the server accepts are offered: PATCH /api/tasks/<id> rejects unknown
   * keys outright rather than ignoring them, and task_id is not among them. A field here
   * the server would refuse is a form that lies. */
  function buildEditor(task, onClose) {
    var el = document.createElement("form");
    el.className = "task-editor";
    el.hidden = true;

    function field(parent, label, hint) {
      var wrap = document.createElement("label");
      var head = document.createElement("span");
      head.textContent = label;
      if (hint) {
        var h = document.createElement("span");
        h.className = "muted";
        h.textContent = " \u2014 " + hint;
        head.appendChild(h);
      }
      wrap.appendChild(head);
      parent.appendChild(wrap);
      return wrap;
    }

    var describe = document.createElement("input");
    describe.autocomplete = "off";
    field(el, "describe", "paid on every caption, twice; re-embedded on save").appendChild(describe);

    var row1 = div("form-row");
    var win = document.createElement("input");
    win.type = "number";
    win.min = "1";
    field(row1, "window (s)").appendChild(win);
    var cool = document.createElement("input");
    cool.type = "number";
    cool.min = "1";
    field(row1, "cooldown (s)").appendChild(cool);
    el.appendChild(row1);

    var row2 = div("form-row");
    var action = document.createElement("select");
    ["save_clip", "raise_alert", "file_ticket", "notify_discord"].forEach(function (v) {
      var o = document.createElement("option");
      o.value = v;
      o.textContent = v;
      action.appendChild(o);
    });
    field(row2, "action").appendChild(action);
    var active = document.createElement("input");
    active.autocomplete = "off";
    field(row2, "active (local)").appendChild(active);
    el.appendChild(row2);

    var foot = div("form-foot");
    var save = document.createElement("button");
    save.type = "submit";
    save.className = "btn";
    save.textContent = "save";
    var cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "btn btn--quiet";
    cancel.textContent = "cancel";
    var msg = span("form-msg", "");
    foot.appendChild(save);
    foot.appendChild(cancel);
    foot.appendChild(msg);
    el.appendChild(foot);

    function reset() {
      describe.value = task.describe;
      win.value = task.window;
      cool.value = task.cooldown;
      action.value = task.action;
      active.value = task.active;
      msg.textContent = "";
      msg.className = "form-msg";
    }

    cancel.addEventListener("click", onClose);

    el.addEventListener("submit", function (ev) {
      ev.preventDefault();
      /* Send only what CHANGED. Restating every field would re-embed `describe` on a
       * window-only edit, and the embedding is what stage 1 matches on — an unrelated
       * edit quietly moving the match is not something anyone connects back to it. */
      var changes = {};
      if (describe.value.trim() !== task.describe) changes.describe = describe.value.trim();
      if (parseInt(win.value, 10) !== task.window) changes.window = parseInt(win.value, 10);
      if (parseInt(cool.value, 10) !== task.cooldown) changes.cooldown = parseInt(cool.value, 10);
      if (action.value !== task.action) changes.action = action.value;
      if (active.value.trim() !== task.active) changes.active = active.value.trim();

      function bad(text) {
        msg.textContent = text;
        msg.className = "form-msg bad";
      }
      if (!Object.keys(changes).length) {
        msg.textContent = "nothing changed";
        msg.className = "form-msg";
        return;
      }
      if (changes.describe !== undefined && changes.describe.length < 8)
        return bad("describe needs real words; it is embedded once and matched forever");
      if (changes.window !== undefined && !(changes.window > 0))
        return bad("window must be a positive number of seconds");
      if (changes.cooldown !== undefined && !(changes.cooldown > 0))
        return bad("cooldown must be positive \u2014 0 is how you get thirty alerts for one event");

      save.disabled = true;
      msg.className = "form-msg";
      msg.textContent = "PATCH \u2026";
      SPARK.data
        .patchTask(task.task_id, changes)
        .then(function () {
          save.disabled = false;
          onClose();
          return refresh();
        })
        .catch(function (err) {
          save.disabled = false;
          bad(err && err.message ? err.message : String(err));
        });
    });

    return { el: el, reset: reset };
  }

  function taskCard(task, row) {
    var refs = {};
    var card = document.createElement("section");
    card.className = "card task";
    card.dataset.taskId = task.task_id;

    var title = span("card-title", task.task_id);
    card.appendChild(title);
    refs.badge = span("card-badge", "");
    card.appendChild(refs.badge);

    // Delete. Confirmed first because a standing task cannot be un-deleted from here,
    // and because the seeded ones come back on restart while a registered one does not
    // — the dialog says which, so nobody discovers the difference by surprise.
    var del = document.createElement("button");
    del.type = "button";
    del.className = "task-delete";
    del.title = "delete this standing task";
    del.setAttribute("aria-label", "delete task " + task.task_id);
    del.textContent = "\u00d7";
    del.addEventListener("click", function () {
      var ok = window.confirm(
        "Delete standing task \u201c" + task.task_id + "\u201d?\n\n" +
        "It stops being evaluated immediately. Anything it already fired stays on the " +
        "Timeline \u2014 the action log is append-only and is never rewritten."
      );
      if (!ok) return;
      del.disabled = true;
      SPARK.data
        .deleteTask(task.task_id)
        .then(function () {
          delete cards[task.task_id];
          if (card.parentNode) card.parentNode.removeChild(card);
        })
        .catch(function (err) {
          del.disabled = false;
          refs.badge.textContent = "delete failed";
          refs.badge.title = String(err && err.message ? err.message : err);
        });
    });
    card.appendChild(del);

    /* Edit. Deliberately NOT a second "define a task" form: task_id is the cooldown and
     * dedupe key (SPEC §6.4) and the server refuses to change it, so a full form would
     * offer a field that cannot be edited. This edits exactly the five fields PATCH
     * accepts, in place on the card, next to the funnel readout that shows what the edit
     * did.
     *
     * `describe` is the one that matters most and the one people get wrong. Every word
     * of it is paid on EVERY captioned window, twice: prefill in the caption's watchlist
     * checklist, and decode in the verdict the model writes back. It is also embedded
     * once for stage 1, so saving re-embeds and changes what matches from the next chunk
     * on. The hint under the field says so, because that cost is otherwise invisible. */
    var edit = document.createElement("button");
    edit.type = "button";
    edit.className = "task-edit";
    edit.title = "edit this standing task";
    edit.setAttribute("aria-label", "edit task " + task.task_id);
    edit.textContent = "✎";
    card.appendChild(edit);

    var desc = document.createElement("p");
    desc.className = "task-describe";
    desc.textContent = "“" + task.describe + "”";
    card.appendChild(desc);

    var editor = buildEditor(task, function () {
      editor.el.hidden = true;
      desc.hidden = false;
      edit.disabled = false;
      // Let the poll resume BEFORE any refresh() the caller runs, or the refresh it
      // triggers on save would return early and the card would keep the old values.
      if (editingTaskId === task.task_id) editingTaskId = null;
    });
    card.appendChild(editor.el);
    edit.addEventListener("click", function () {
      editor.reset();
      editor.el.hidden = false;
      desc.hidden = true;
      edit.disabled = true;
      editingTaskId = task.task_id;
    });

    var meta = div("task-meta");
    meta.appendChild(span("", activeLabel(task.active)));
    meta.appendChild(span("sep", "·"));
    meta.appendChild(span("", "window " + task.window + "s"));
    meta.appendChild(span("sep", "·"));
    meta.appendChild(span("", "cooldown " + task.cooldown + "s"));
    card.appendChild(meta);

    var act = div("task-action");
    act.textContent = "→ " + task.action;
    act.classList.add(task.action === "save_clip" ? "sev-low" : "sev-human");
    if (task.action !== "save_clip") {
      act.title = "reaches a human: fires provisionally as unverified, amended on stage 3 (SPEC §6.3)";
    } else {
      act.title = "low stakes: fires on stage-2 confidence, no verification (SPEC §6.3)";
    }
    card.appendChild(act);

    // ── stage 1 ─────────────────────────────────────────────────────────────────────
    var s1 = stageRow("①", "embed match");
    var score = row && row.stage1 ? row.stage1.score : 0;
    var thr =
      row && row.stage1 && row.stage1.threshold !== null && row.stage1.threshold !== undefined
        ? row.stage1.threshold
        : SPARK.data.cfg(cfg, "monitor.stage1_cosine_threshold", null);
    s1.value.appendChild(span("dots", dots(score)));
    s1.value.appendChild(span("num", Number(score).toFixed(2)));
    s1.value.appendChild(
      span("muted", thr !== null ? "(loose gate ≥ " + thr + ")" : "(loose gate)")
    );
    s1.row.classList.add(row && row.stage1 && row.stage1.matched ? "stage--hit" : "stage--idle");
    card.appendChild(s1.row);

    // ── stage 2 ─────────────────────────────────────────────────────────────────────
    var s2 = stageRow("②", "llm confirm");
    var st2 = (row && row.stage2) || {};
    if (st2.verdict === "match") {
      s2.value.appendChild(span("verdict verdict--yes", "✓ match"));
      s2.value.appendChild(span("sep", "· sustain"));
      refs.sustainBar = bar();
      s2.value.appendChild(refs.sustainBar.wrap);
      refs.sustainText = span("num", "");
      s2.value.appendChild(refs.sustainText);
      refs.sustainSince = st2.since;
      refs.sustainWindow = st2.sustain_window_s || task.window;
      s2.row.classList.add("stage--hit");
    } else if (st2.verdict === "no_match") {
      s2.value.appendChild(span("verdict verdict--no", "✗ no match"));
      s2.row.classList.add("stage--idle");
    } else {
      s2.value.appendChild(span("muted", "—"));
      s2.row.classList.add("stage--idle");
    }
    card.appendChild(s2.row);

    // ── stage 3 ─────────────────────────────────────────────────────────────────────
    var s3 = stageRow("③", "verify");
    var st3 = (row && row.stage3) || {};
    if (st3.state === "running" || st3.state === "queued") {
      s3.value.appendChild(span("verdict verdict--run", "◐ " + st3.state));
      s3.value.appendChild(span("muted", "job " + (st3.job_id || "?") + " · 20–60 s"));
      s3.row.classList.add("stage--hit");
    } else if (st3.verdict === "verified") {
      s3.value.appendChild(span("verdict verdict--yes", "✓ verified"));
      s3.value.appendChild(span("muted", "job " + (st3.job_id || "?")));
      s3.row.classList.add("stage--hit");
    } else if (st3.verdict === "retracted") {
      s3.value.appendChild(span("verdict verdict--no", "✗ retracted"));
      s3.value.appendChild(span("muted", "job " + (st3.job_id || "?")));
      s3.row.classList.add("stage--bad");
    } else {
      s3.value.appendChild(span("muted", "—"));
      s3.row.classList.add("stage--idle");
      if (task.action === "save_clip") {
        s3.value.appendChild(span("muted", "(no verify — low stakes)"));
      }
    }
    card.appendChild(s3.row);

    // ── the brake ───────────────────────────────────────────────────────────────────
    var brake = div("brake");
    refs.brakeText = span("brake-text", "");
    brake.appendChild(refs.brakeText);
    refs.brakeBar = bar("brake-bar");
    brake.appendChild(refs.brakeBar.wrap);
    card.appendChild(brake);

    if (row && row.match_range) {
      var range = div("task-range");
      range.appendChild(span("muted", "matched range"));
      var b = document.createElement("button");
      b.type = "button";
      b.className = "chip chip--cite";
      b.textContent = SPARK.time.range(row.match_range.t_start, row.match_range.t_end);
      b.title = row.match_range.t_start + " → " + row.match_range.t_end + " (UTC, as stored)";
      b.addEventListener("click", function () {
        SPARK.player.scrubTo(row.match_range.t_start, row.match_range.t_end, {
          source: "task " + task.task_id,
        });
      });
      range.appendChild(b);
      card.appendChild(range);
    }

    refs.task = task;
    refs.row = row;
    cards[task.task_id] = { node: card, refs: refs };
    return card;
  }

  // -----------------------------------------------------------------------------------
  // The live tick — cooldown and sustain, recomputed from absolute time.
  // -----------------------------------------------------------------------------------
  function tickLive() {
    var now = Date.now();
    Object.keys(cards).forEach(function (taskId) {
      var c = cards[taskId];
      var refs = c.refs;
      var task = refs.task;
      var row = refs.row || {};

      var cooldownS = row.cooldown_seconds !== undefined && row.cooldown_seconds !== null
        ? row.cooldown_seconds
        : task.cooldown;
      var remaining = 0;
      if (row.last_fired_ts) {
        remaining = cooldownS - (now - SPARK.time.epochMs(row.last_fired_ts)) / 1000;
      }
      var cooling = remaining > 0;

      if (refs.brakeText) {
        if (row.last_fired_ts) {
          refs.brakeText.textContent =
            "last fired " + SPARK.time.fmt(row.last_fired_ts) +
            (cooling ? " · 🔒 cooling " + SPARK.time.countdown(remaining) + " remain" : " · 🔓 cooldown clear");
          refs.brakeText.title = row.last_fired_ts + " (UTC, as stored)";
        } else {
          refs.brakeText.textContent = "never fired · 🔓 cooldown clear";
          refs.brakeText.title = "";
        }
        refs.brakeText.classList.toggle("brake-text--cooling", cooling);
      }
      if (refs.brakeBar) {
        refs.brakeBar.set(cooling ? remaining / cooldownS : 0);
        refs.brakeBar.wrap.classList.toggle("bar--cooling", cooling);
      }

      if (refs.sustainBar && refs.sustainSince) {
        var elapsed = (now - SPARK.time.epochMs(refs.sustainSince)) / 1000;
        var win = refs.sustainWindow || task.window;
        var frac = Math.max(0, Math.min(1, elapsed / win));
        refs.sustainBar.set(frac);
        refs.sustainText.textContent = Math.floor(Math.min(elapsed, win)) + "/" + win + "s";
        refs.sustainBar.wrap.classList.toggle("bar--full", frac >= 1);
      }

      if (refs.badge) {
        var state = badgeState(task, row, cooling);
        refs.badge.textContent = state.label;
        refs.badge.className = "card-badge " + state.cls;
        refs.badge.title = state.hint;
        c.node.classList.toggle("task--cooling", cooling);
      }
    });
  }

  function badgeState(task, row, cooling) {
    if (task.enabled === false)
      return { label: "DISABLED", cls: "badge--off", hint: "task registered but not evaluated" };
    if (row.in_active_window === false)
      return {
        label: "OUT OF WINDOW",
        cls: "badge--off",
        hint: "outside " + task.active + " (local wall clock, may wrap midnight)",
      };
    if (cooling)
      return {
        label: "COOLING",
        cls: "badge--cool",
        hint: "brake 1 of 3: one event, one alert (SPEC §6.4). Stages keep matching; nothing fires.",
      };
    if (row.stage2 && row.stage2.verdict === "match")
      return { label: "MATCHING", cls: "badge--hot", hint: "stage 2 confirmed; holding the sustain window" };
    return { label: "ACTIVE", cls: "badge--on", hint: "in window, armed, nothing matching" };
  }

  // -----------------------------------------------------------------------------------
  // register_task — SPEC §11.3 / §10 D5. The six §6.1 fields, POSTed.
  // -----------------------------------------------------------------------------------
  function prefillForm() {
    var f = els.form;
    f.elements.window.value = SPARK.data.cfg(cfg, "monitor.stage2_sustain_default", 120);
    f.elements.cooldown.value = SPARK.data.cfg(cfg, "monitor.default_cooldown_seconds", 300);
    f.elements.active.value = "00:00-24:00";
  }

  /* task_id is the cooldown and dedupe key (SPEC §6.4), so it has to be a stable slug —
   * but that is our constraint, not something a person typing "Loading dock blocked"
   * should be made to satisfy by hand. Normalise instead of rejecting, and write the
   * result back into the field before submitting: the key is what the Timeline and the
   * action log will show forever, so the user must see the one they are actually getting.
   * Never transform it silently. */
  function slugify(raw) {
    return String(raw)
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "") // strip accents rather than eat the letter
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 49)
      .replace(/-+$/, ""); // the slice may have landed mid-separator
  }

  function onRegister(ev) {
    ev.preventDefault();
    var f = els.form;

    var typedId = f.elements.task_id.value.trim();
    var slugId = slugify(typedId);
    f.elements.task_id.value = slugId;

    var payload = {
      task_id: slugId,
      describe: f.elements.describe.value.trim(),
      window: parseInt(f.elements.window.value, 10),
      action: f.elements.action.value,
      cooldown: parseInt(f.elements.cooldown.value, 10),
      active: f.elements.active.value.trim(),
      enabled: true,
    };

    var problem = validate(payload);
    if (problem) return formMessage(problem, "bad");

    formMessage("POST " + SPARK.data.endpoints.registerTask + " …", "");
    SPARK.data
      .registerTask(payload)
      .then(function (task) {
        var receipt =
          "✓ registered " + task.task_id + " → " + task.action +
          (task.task_id !== typedId ? " (slugified from “" + typedId + "”)" : "") +
          (SPARK.data.isMock() ? " (mock: held in this page only)" : "");
        // The form is done; the new card in the list is the real confirmation. Collapse
        // it so the pane goes back to the funnel — but the receipt names the slug we
        // chose, so it moves to the <summary>, which is what stays on screen when the
        // <details> closes. Leaving it in the closed body would hide the rename.
        closeForm();
        summaryMessage(receipt);
        return refresh();
      })
      .catch(function (err) {
        formMessage(err.message, "bad");
      });
  }

  function validate(p) {
    // Anything typeable slugifies; only "nothing usable was typed" can reach this.
    if (!/^[a-z0-9][a-z0-9-]{0,48}$/.test(p.task_id))
      return "task_id needs at least one letter or digit — it is the cooldown and dedupe key";
    if (p.describe.length < 8) return "describe needs real words; it is embedded once and matched forever";
    if (!(p.window > 0)) return "window must be a positive number of seconds";
    if (!(p.cooldown > 0))
      return "cooldown must be positive — 0 is how you get thirty alerts for one event (SPEC §6.4)";
    if (!/^([01]\d|2[0-4]):[0-5]\d-([01]\d|2[0-4]):[0-5]\d$/.test(p.active))
      return "active must look like 18:00-06:00 (local wall clock, may wrap midnight)";
    return null;
  }

  var SUMMARY_IDLE = "+ define a standing task";

  /** Collapse and empty the form, so a half-typed task never reappears looking real. */
  function closeForm() {
    if (els.details) els.details.open = false;
    els.form.reset();
    prefillForm();
    formMessage("", "");
  }

  function summaryMessage(text) {
    if (!els.summary) return;
    els.summary.textContent = text;
    els.summary.classList.add("summary--ok");
  }

  function resetSummary() {
    if (!els.summary) return;
    els.summary.textContent = SUMMARY_IDLE;
    els.summary.classList.remove("summary--ok");
    formMessage("", "");
  }

  function formMessage(text, kind) {
    els.formMsg.textContent = text;
    els.formMsg.className = "form-msg " + (kind ? "form-msg--" + kind : "");
  }

  // -----------------------------------------------------------------------------------
  // small builders
  // -----------------------------------------------------------------------------------
  function stageRow(num, label) {
    var row = div("stage");
    row.appendChild(span("stage-num", num));
    row.appendChild(span("stage-label", label));
    var value = span("stage-value", "");
    row.appendChild(value);
    return { row: row, value: value };
  }

  function dots(score) {
    var filled = Math.max(0, Math.min(5, Math.round(Number(score || 0) * 5)));
    return "●●●●●".slice(0, filled) + "○○○○○".slice(0, 5 - filled);
  }

  function bar(extraClass) {
    var wrap = div("bar" + (extraClass ? " " + extraClass : ""));
    var fill = div("bar-fill");
    wrap.appendChild(fill);
    return {
      wrap: wrap,
      set: function (frac) {
        fill.style.width = Math.max(0, Math.min(1, frac)) * 100 + "%";
      },
    };
  }

  function activeLabel(active) {
    return String(active).replace("-", "–");
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
