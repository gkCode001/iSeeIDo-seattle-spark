/* ASK pane — SPEC §11.2.
 *
 * ONE RULE ABOVE ALL OTHERS (CLAUDE.md invariant 4, SPEC §4.3 / §11.2):
 *
 *     THE REFINED ANSWER IS APPENDED. THE PROVISIONAL IS NEVER SUBSTITUTED.
 *
 * Rendering the refinement in place is the obvious implementation and it destroys the
 * demo: the audience sees a chatbot that took 40 seconds instead of a system that knew
 * its own summary was not good enough. So the provisional card is written once and then
 * treated as immutable — appendRefinement() creates a *new* sibling, and
 * assertProvisionalIntact() re-checks the frozen text afterwards and screams in the UI
 * if anything ever mutates it. That check is cheap insurance against a future edit
 * quietly turning this back into a normal chatbot.
 *
 * The five required elements from the §11.2 table, and where they live:
 *   groundedness badge      -> groundBadge()      (the most important pixel in the build)
 *   elapsed vs 90 s timeout -> the timer registry (absolute-time driven, never a counter)
 *   clickable cited ranges  -> citeChip()         (scrubs the shared player)
 *   reasoning trace         -> <details> in appendRefinement(), collapsed
 *   dedupe notice           -> dedupeNotice()     (a silent no-op reads as a bug)
 */
window.SPARK = window.SPARK || {};

SPARK.ask = (function () {
  "use strict";

  var cfg = {};
  var chunkById = {};
  var els = {};
  var timeoutSeconds = 90;
  var annK = null;
  var rerankN = null;

  // job_id -> {requestedAtMs, turnIds: [], done: bool, timerEls: []}
  var jobs = {};

  function init(root, config, chunks) {
    cfg = config || {};
    els.root = root;
    els.log = root.querySelector("[data-ask-log]");
    els.form = root.querySelector("[data-ask-form]");
    els.input = root.querySelector("[data-ask-input]");
    els.demos = root.querySelector("[data-ask-demos]");

    (chunks || []).forEach(function (c) {
      chunkById[c.chunk_id] = c;
    });

    timeoutSeconds = SPARK.data.cfg(cfg, "agent.deep.timeout_seconds", 90);
    annK = SPARK.data.cfg(cfg, "index.search.ann_k", null);
    rerankN = SPARK.data.cfg(cfg, "index.search.rerank_top_n", null);

    els.form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var q = els.input.value.trim();
      if (!q) return;
      els.input.value = "";
      submit(q);
    });
    els.input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        els.form.requestSubmit();
      }
    });

    setInterval(tickTimers, 200);
    buildDemoButtons();
    return loadHistory();
  }

  // -----------------------------------------------------------------------------------
  // History. SPEC §11.4: the turn persists the JOB, not just the text, because the
  // refinement lands after the turn ends. Reload and the 34-second answer is still here,
  // still stacked under the provisional it did not replace.
  // -----------------------------------------------------------------------------------
  function loadHistory() {
    return SPARK.data.loadHistory().then(function (h) {
      (h.turns || [])
        .slice()
        .sort(function (a, b) {
          // UTC epoch, always. Local strings are display-only and compare wrong inside
          // a DST fall-back hour (PEP 495).
          return SPARK.time.epochMs(a.ts) - SPARK.time.epochMs(b.ts);
        })
        .forEach(function (turn) {
          renderTurn(turn);
          var job = turn.job_id && h.jobs ? h.jobs[turn.job_id] : null;
          renderProvisional(turn, { job: job, dedupe_of: null, restored: true });
          if (job && job.state === "done") appendRefinement(turn.turn_id, job);
        });
      scrollToEnd();
    });
  }

  function buildDemoButtons() {
    SPARK.data.demoQuestions().then(function (list) {
      if (!list.length) return;
      list.forEach(function (d) {
        var b = button(
          (d.grounded === false ? "⚠ " : "✓ ") + shorten(d.question, 44),
          "chip chip--demo",
          function () {
            submit(d.question);
          }
        );
        b.title = d.question;
        els.demos.appendChild(b);
      });
      var dd = button("⧉ double-ask (dedupe)", "chip chip--demo", function () {
        var esc = list.filter(function (d) {
          return d.grounded === false;
        })[0];
        if (!esc) return;
        submit(esc.question);
        setTimeout(function () {
          submit(esc.question);
        }, 600 / SPARK.data.speed());
      });
      dd.title = "asks the same question twice — the second turn must say 'already running'";
      els.demos.appendChild(dd);
    });
  }

  // -----------------------------------------------------------------------------------
  // Submit
  // -----------------------------------------------------------------------------------
  function submit(question) {
    SPARK.data.ask(question, {
      onSubmitted: function (turn) {
        renderTurn(turn);
        pending(turn.turn_id, true);
        scrollToEnd();
      },
      onProvisional: function (turn, meta, submittedTurnId) {
        var id = submittedTurnId || turn.turn_id;
        adoptTurnId(id, turn.turn_id);
        pending(turn.turn_id, false);
        renderProvisional(turn, meta || {});
        scrollToEnd();
      },
      onRefined: function (turnId, job) {
        appendRefinement(turnId, job);
        scrollToEnd();
      },
      onFailed: function (turnId, message) {
        pending(turnId, false);
        var node = turnNode(turnId);
        if (!node) return;
        var err = div("answer-error");
        err.textContent = "✗ " + message;
        node.appendChild(err);
      },
    });
  }

  /** The server owns turn_id in live mode; re-key the card we already drew. */
  function adoptTurnId(oldId, newId) {
    if (!newId || oldId === newId) return;
    var node = turnNode(oldId);
    if (node) node.dataset.turnId = newId;
  }

  // -----------------------------------------------------------------------------------
  // Rendering
  // -----------------------------------------------------------------------------------
  function renderTurn(turn) {
    if (turnNode(turn.turn_id)) return turnNode(turn.turn_id);
    var node = document.createElement("article");
    node.className = "turn";
    node.dataset.turnId = turn.turn_id;

    var user = div("bubble bubble--user");
    var head = div("bubble-head");
    head.textContent = "You";
    var time = document.createElement("span");
    time.className = "bubble-time";
    time.textContent = SPARK.time.fmt(turn.ts);
    time.title = turn.ts + " (UTC, as stored)";
    head.appendChild(time);
    var body = document.createElement("p");
    body.className = "bubble-body";
    body.textContent = turn.question;
    user.appendChild(head);
    user.appendChild(body);
    node.appendChild(user);

    els.log.appendChild(node);
    return node;
  }

  function pending(turnId, on) {
    var node = turnNode(turnId);
    if (!node) return;
    var existing = node.querySelector(".pending");
    if (on) {
      if (existing) return;
      var p = div("pending");
      p.textContent = "⌕ searching index …";
      node.appendChild(p);
    } else if (existing) {
      existing.remove();
    }
  }

  /** The retrieval line from the mockup: "⌕ searched index · 20 → 5 chunks · range".
   *  The two counts come from config (index.search.*), not from a literal — and the
   *  range is derived from the cited chunk ids, so no schema field was invented for it. */
  function retrievalLine(turn) {
    var line = div("retrieval");
    var counts =
      annK && rerankN ? " · " + annK + " → " + rerankN + " chunks" : "";
    var label = document.createElement("span");
    label.textContent = "⌕ searched index" + counts;
    line.appendChild(label);
    var span = citedSpan(turn.cited_chunk_ids);
    if (span) {
      line.appendChild(document.createTextNode(" · "));
      line.appendChild(citeChip(span.t_start, span.t_end, "retrieved context"));
    }
    return line;
  }

  function renderProvisional(turn, meta) {
    var node = turnNode(turn.turn_id) || renderTurn(turn);
    if (node.querySelector(".answer--provisional")) return;

    node.appendChild(retrievalLine(turn));

    var card = document.createElement("section");
    card.className = "card answer answer--provisional";
    card.appendChild(cardTitle("Provisional"));
    card.appendChild(cardBadge(SPARK.time.secs(turn.latency_s)));

    var body = document.createElement("p");
    body.className = "answer-body";
    body.textContent = turn.provisional_answer;
    card.appendChild(body);

    // Freeze it. appendRefinement() re-checks this exact string afterwards.
    card.dataset.frozenAnswer = turn.provisional_answer;

    card.appendChild(groundBadge(turn.grounded));

    var job = meta.job || null;
    if (meta.dedupe_of) {
      card.appendChild(dedupeNotice(meta.dedupe_of, turn));
      subscribe(meta.dedupe_of, turn.turn_id, card, null);
    } else if (job) {
      card.appendChild(jobLine(job));
      subscribe(job.job_id, turn.turn_id, card, job);
    }

    node.appendChild(card);
  }

  /** THE most important pixel in the build (SPEC §11.2). §4.2's gate returns a literal
   *  yes/no; this prints it. */
  function groundBadge(grounded) {
    var b = div("ground");
    if (grounded === true) {
      b.className += " ground--indexed";
      b.textContent = "✓ ANSWERED FROM INDEX";
    } else if (grounded === false) {
      b.className += " ground--escalated";
      b.textContent = "⚠ NOT ANSWERABLE FROM INDEX → escalated";
    } else {
      b.className += " ground--unknown";
      b.textContent = "· groundedness gate did not report";
    }
    return b;
  }

  function jobLine(job) {
    var wrap = div("jobline");
    var l1 = div("jobline-row");
    l1.appendChild(strong("job " + job.job_id));
    l1.appendChild(document.createTextNode(" · re-watching "));
    l1.appendChild(citeChip(job.t_start, job.t_end, "deep range"));
    wrap.appendChild(l1);

    var fps = SPARK.data.cfg(cfg, "vlm.profiles.deep.sample_fps", 4);
    var native = SPARK.data.cfg(cfg, "vlm.profiles.deep.native_resolution", true);
    var l2 = div("jobline-row jobline-row--meta");
    l2.appendChild(
      document.createTextNode((native ? "native res" : "downscaled") + ", " + fps + " fps · ⏱ ")
    );
    var timer = document.createElement("span");
    timer.className = "timer";
    timer.dataset.jobId = job.job_id;
    timer.textContent = "0s";
    l2.appendChild(timer);
    l2.appendChild(document.createTextNode(" / " + timeoutSeconds + "s timeout"));
    wrap.appendChild(l2);
    return wrap;
  }

  /** SPEC §11.2: "already running — job 7f3a". A silent no-op reads as a bug in
   *  rehearsal, and §4.3 requires the second identical range not be queued twice. */
  function dedupeNotice(jobId, turn) {
    var wrap = div("jobline jobline--dedupe");
    var l1 = div("jobline-row");
    l1.appendChild(strong("⧉ already running — job " + jobId));
    wrap.appendChild(l1);
    var l2 = div("jobline-row jobline-row--meta");
    l2.appendChild(
      document.createTextNode("identical time range · not queued twice · ⏱ ")
    );
    var timer = document.createElement("span");
    timer.className = "timer";
    timer.dataset.jobId = jobId;
    timer.textContent = "0s";
    l2.appendChild(timer);
    l2.appendChild(document.createTextNode(" / " + timeoutSeconds + "s timeout"));
    wrap.appendChild(l2);
    return wrap;
  }

  // -----------------------------------------------------------------------------------
  // Elapsed timers. Driven off the job's absolute requested_at, so a busy tab, a slow
  // frame or a paused rehearsal can never make the wait look shorter than it was.
  // -----------------------------------------------------------------------------------
  function subscribe(jobId, turnId, card, job) {
    var rec = jobs[jobId];
    if (!rec) {
      rec = jobs[jobId] = {
        requestedAtMs: job ? SPARK.time.epochMs(job.requested_at) : Date.now(),
        turnIds: [],
        done: false,
      };
    }
    if (job && job.requested_at) rec.requestedAtMs = SPARK.time.epochMs(job.requested_at);
    // A restored turn marks its job done on load. If the same job id then shows up again
    // still running — which the demo script does on purpose, so the audience sees the
    // familiar "job 7f3a" — the registry has to un-finish, or the elapsed timer sits at
    // 0s through the whole 34-second wait it exists to cover.
    if (job && job.state && job.state !== "done") {
      rec.done = false;
      rec.timedOut = false;
    }
    if (rec.turnIds.indexOf(turnId) === -1) rec.turnIds.push(turnId);
  }

  function tickTimers() {
    var now = Date.now();
    Object.keys(jobs).forEach(function (jobId) {
      var rec = jobs[jobId];
      if (rec.done) return;
      var elapsed = (now - rec.requestedAtMs) / 1000;
      var timedOut = elapsed > timeoutSeconds;
      document.querySelectorAll('.timer[data-job-id="' + cssEscape(jobId) + '"]').forEach(function (el) {
        if (el.classList.contains("timer--frozen")) return;
        el.textContent = Math.floor(elapsed) + "s";
        el.classList.toggle("timer--warn", elapsed > timeoutSeconds * 0.66 && !timedOut);
        el.classList.toggle("timer--timeout", timedOut);
      });
      if (timedOut && !rec.timedOut) {
        rec.timedOut = true;
        rec.turnIds.forEach(function (tid) {
          var node = turnNode(tid);
          if (!node || node.querySelector(".answer--timeout")) return;
          var card = document.createElement("section");
          card.className = "card answer answer--timeout";
          card.appendChild(cardTitle("Timed out"));
          card.appendChild(cardBadge(timeoutSeconds + " s"));
          var p = document.createElement("p");
          p.className = "answer-body";
          p.textContent =
            "Deep analysis job " + jobId + " passed the " + timeoutSeconds +
            " s timeout. The provisional answer above still stands — it was never blocked on this.";
          card.appendChild(p);
          node.appendChild(card);
        });
      }
    });
  }

  // -----------------------------------------------------------------------------------
  // THE APPEND. Read the file header before touching this function.
  // -----------------------------------------------------------------------------------
  function appendRefinement(turnId, job) {
    var rec = jobs[job.job_id];
    if (rec) rec.done = true;
    freezeTimers(job);

    // Fan out to every turn waiting on this job. A turn that was deduped onto an
    // in-flight job (§4.3) is still owed the answer — leaving it showing only "already
    // running" would turn the dedupe brake into a dropped question.
    var targets = [turnId];
    if (rec) {
      rec.turnIds.forEach(function (id) {
        if (targets.indexOf(id) === -1) targets.push(id);
      });
    }
    targets.forEach(function (id) {
      renderRefinedInto(id, job);
    });
  }

  function renderRefinedInto(turnId, job) {
    var node = turnNode(turnId);
    if (!node) return;
    var prov = node.querySelector(".answer--provisional");

    if (node.querySelector(".answer--refined")) return; // idempotent on reconnect

    var connector = div("append-connector");
    connector.textContent = "↓ refinement appended — the provisional above is unchanged";
    node.appendChild(connector);

    var card = document.createElement("section");
    card.className = "card answer answer--refined";
    card.appendChild(cardTitle("Refined"));
    var elapsed =
      job.completed_at && job.requested_at
        ? SPARK.time.durationSeconds(job.requested_at, job.completed_at)
        : null;
    card.appendChild(cardBadge(SPARK.time.secs(elapsed)));

    var body = document.createElement("p");
    body.className = "answer-body";
    body.textContent = job.answer || "";
    card.appendChild(body);

    var cites = (job.cited_chunk_ids || []).filter(function (id) {
      return chunkById[id];
    });
    if (cites.length) {
      var row = div("cites");
      row.appendChild(labelSpan("cited:"));
      cites.forEach(function (id) {
        var c = chunkById[id];
        row.appendChild(citeChip(c.t_start, c.t_end, c.caption));
      });
      card.appendChild(row);
    }

    var actions = div("answer-actions");
    if (job.evidence_clip) {
      var clip = button("▶ clip", "chip chip--action", function () {
        SPARK.player.scrubTo(job.t_start, job.t_end, { source: "evidence clip · job " + job.job_id });
      });
      clip.title = job.evidence_clip + " — opened by time range, not by filename";
      actions.appendChild(clip);
    }
    if (job.reasoning) {
      var det = document.createElement("details");
      det.className = "reasoning";
      var sum = document.createElement("summary");
      var maxTok = SPARK.data.cfg(cfg, "vlm.profiles.deep.max_tokens", null);
      sum.textContent = "reasoning ▾" + (maxTok ? " (≤" + maxTok + " tok)" : "");
      det.appendChild(sum);
      var pre = document.createElement("pre");
      pre.className = "reasoning-body";
      pre.textContent = job.reasoning;
      det.appendChild(pre);
      actions.appendChild(det);
    }
    if (job.confidence !== null && job.confidence !== undefined) {
      actions.appendChild(labelSpan("confidence " + Number(job.confidence).toFixed(2)));
    }
    card.appendChild(actions);

    node.appendChild(card);

    assertProvisionalIntact(node, prov);
  }

  function freezeTimers(job) {
    var elapsed =
      job.completed_at && job.requested_at
        ? SPARK.time.durationSeconds(job.requested_at, job.completed_at)
        : null;
    document.querySelectorAll('.timer[data-job-id="' + cssEscape(job.job_id) + '"]').forEach(function (el) {
      el.classList.add("timer--frozen", "timer--done");
      el.classList.remove("timer--warn", "timer--timeout");
      el.textContent = elapsed === null ? "done" : elapsed.toFixed(1) + "s";
    });
  }

  /** Guard rail, not decoration. If a future change ever rewrites the provisional in
   *  place, this puts the failure on screen instead of letting the demo quietly become
   *  a slow chatbot. */
  function assertProvisionalIntact(node, prov) {
    if (!prov) return;
    var body = prov.querySelector(".answer-body");
    var expected = prov.dataset.frozenAnswer;
    if (body && expected !== undefined && body.textContent !== expected) {
      console.error("[ask] INVARIANT 4 VIOLATED: provisional answer was mutated in place");
      var warn = div("answer-error");
      warn.textContent =
        "✗ invariant 4 violated: the provisional answer was overwritten. " +
        "The refinement must be appended, never substituted (SPEC §11.2).";
      node.insertBefore(warn, prov);
    }
    if (!node.querySelector(".answer--provisional")) {
      console.error("[ask] INVARIANT 4 VIOLATED: provisional card removed on refinement");
    }
  }

  // -----------------------------------------------------------------------------------
  // helpers
  // -----------------------------------------------------------------------------------
  function turnNode(turnId) {
    if (!turnId) return null;
    return els.log.querySelector('[data-turn-id="' + cssEscape(turnId) + '"]');
  }

  function citedSpan(ids) {
    var hits = (ids || [])
      .map(function (id) {
        return chunkById[id];
      })
      .filter(Boolean);
    if (!hits.length) return null;
    var start = hits[0].t_start;
    var end = hits[0].t_end;
    hits.forEach(function (c) {
      if (SPARK.time.epochMs(c.t_start) < SPARK.time.epochMs(start)) start = c.t_start;
      if (SPARK.time.epochMs(c.t_end) > SPARK.time.epochMs(end)) end = c.t_end;
    });
    return { t_start: start, t_end: end };
  }

  /** A cited range, clickable. The chunk carries segment + pts_offset (SPEC §3.1); the
   *  click hands the player a TIME RANGE and the player resolves the files. */
  function citeChip(tStart, tEnd, tooltip) {
    var b = button(SPARK.time.range(tStart, tEnd), "chip chip--cite", function () {
      SPARK.player.scrubTo(tStart, tEnd, { source: tooltip || "" });
    });
    b.title = (tooltip ? tooltip + "\n" : "") + tStart + " → " + tEnd + " (UTC, as stored)";
    return b;
  }

  function cardTitle(text) {
    var s = document.createElement("span");
    s.className = "card-title";
    s.textContent = text;
    return s;
  }
  function cardBadge(text) {
    var s = document.createElement("span");
    s.className = "card-badge";
    s.textContent = text;
    return s;
  }
  function div(cls) {
    var d = document.createElement("div");
    d.className = cls;
    return d;
  }
  function strong(text) {
    var s = document.createElement("strong");
    s.textContent = text;
    return s;
  }
  function labelSpan(text) {
    var s = document.createElement("span");
    s.className = "muted";
    s.textContent = text;
    return s;
  }
  function button(text, cls, onClick) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }
  function shorten(text, n) {
    return text.length > n ? text.slice(0, n - 1) + "…" : text;
  }
  function cssEscape(value) {
    return String(value).replace(/["\\]/g, "\\$&");
  }
  function scrollToEnd() {
    els.log.scrollTop = els.log.scrollHeight;
  }

  return { init: init, submit: submit, appendRefinement: appendRefinement };
})();
