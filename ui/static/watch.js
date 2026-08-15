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
  var pollTimer = null;

  function init(root, config) {
    cfg = config || {};
    els.root = root;
    els.list = root.querySelector("[data-watch-list]");
    els.form = root.querySelector("[data-watch-form]");
    els.formMsg = root.querySelector("[data-watch-formmsg]");
    els.endpoint = root.querySelector("[data-watch-endpoint]");
    els.stamp = root.querySelector("[data-watch-stamp]");

    els.endpoint.textContent = "POST " + SPARK.data.endpoints.registerTask;
    prefillForm();
    els.form.addEventListener("submit", onRegister);

    // Closing the form must not depend on finding the <summary> again: on a laptop the
    // open form is taller than the pane, so its foot is clipped and the heading is easy
    // to lose. Reset on close so a half-typed task does not reappear later looking real.
    var cancel = root.querySelector("[data-watch-cancel]");
    var details = root.querySelector("details.task-form");
    if (cancel && details) {
      cancel.addEventListener("click", function () {
        details.open = false;
        els.form.reset();
        prefillForm();
        formMessage("", "");
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

    var desc = document.createElement("p");
    desc.className = "task-describe";
    desc.textContent = "“" + task.describe + "”";
    card.appendChild(desc);

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

  function onRegister(ev) {
    ev.preventDefault();
    var f = els.form;
    var payload = {
      task_id: f.elements.task_id.value.trim(),
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
        formMessage(
          "registered " + task.task_id + " → " + task.action +
            (SPARK.data.isMock() ? " (mock: held in this page only)" : ""),
          "ok"
        );
        f.reset();
        prefillForm();
        return refresh();
      })
      .catch(function (err) {
        formMessage(err.message, "bad");
      });
  }

  function validate(p) {
    if (!/^[a-z0-9][a-z0-9-]{1,48}$/.test(p.task_id))
      return "task_id must be a lowercase slug — it is the cooldown and dedupe key";
    if (p.describe.length < 8) return "describe needs real words; it is embedded once and matched forever";
    if (!(p.window > 0)) return "window must be a positive number of seconds";
    if (!(p.cooldown > 0))
      return "cooldown must be positive — 0 is how you get thirty alerts for one event (SPEC §6.4)";
    if (!/^([01]\d|2[0-4]):[0-5]\d-([01]\d|2[0-4]):[0-5]\d$/.test(p.active))
      return "active must look like 18:00-06:00 (local wall clock, may wrap midnight)";
    return null;
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
