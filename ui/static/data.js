/* Data layer — the ONE place that knows whether we are on fixtures or on M3.
 *
 *   ┌──────────────────────────────────────────────────────────────────────────┐
 *   │  THE SWITCH.  Flip MODE to "live" when the M3 FastAPI process is up.     │
 *   │  Nothing else in the UI needs to change: every pane calls the same       │
 *   │  functions below and neither knows nor cares which branch served it.     │
 *   │  ?mode=live / ?mode=mock in the URL overrides it for a quick check.      │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * The endpoint contract the live branch expects is written out in ENDPOINTS below.
 * That list, plus ui/mock/*.json, is the whole handshake with M3 — the fixtures are
 * exact `to_dict()` output from shared/schema.py, asserted by tests/test_ui.py.
 */
window.SPARK = window.SPARK || {};

SPARK.data = (function () {
  "use strict";

  // Resolved by loadConfig(): if /api/config answers, this page is being served by M3
  // and everything below is real. Mock fixtures are the FALLBACK, not the default.
  //
  // It used to be the other way round, from when there was no backend to talk to. The
  // cost of that default once M3 existed: opening the page without ?mode=live showed a
  // fully populated console — a live camera pane saying "unavailable", a Timeline of six
  // invented alerts, a funnel mid-cooldown — all fixtures, with one small pill as the
  // only warning. A demo surface that convincingly shows fake data by default is worse
  // than one that fails loudly.
  var MODE = "mock";
  var MODE_PINNED = false;

  // ---------------------------------------------------------------------------------
  // Endpoint contract (live mode). Relative paths only — no absolute origins anywhere
  // in this UI, which is also how the no-remote-assets grep stays clean.
  // ---------------------------------------------------------------------------------
  var ENDPOINTS = {
    config: "/api/config", //                     GET  -> config/settings.yaml (subset ok)
    chunks: "/api/chunks", //                     GET  ?t_from&t_to -> {chunks: [ChunkRecord]}
    index: "/api/index", //                       GET  ?offset&limit&q&tier&gated&t_from&t_to
    //                                                 -> {chunks, total, offset, limit, page,
    //                                                     pages, filters, stats}
    history: "/api/chat/history", //              GET  -> {turns: [ChatTurn], jobs: {id: DeepJob}}
    ask: "/api/ask", //                           POST {question} -> ChatTurn + {dedupe_of?}
    ws: "/ws", //                                 WS   -> {type: "refinement"|"monitor_state"|"action", ...}
    tasks: "/api/tasks", //                       GET  -> {tasks: [Task]}
    registerTask: "/api/register_task", //        POST Task fields -> Task  (SPEC §10 D5)
    task: "/api/tasks/", //                       DELETE /<id>, PATCH /<id> -> Task
    monitorState: "/api/monitor/state", //        GET  -> funnel state, shape per ui/mock/monitor_state.json
    actions: "/api/actions", //                   GET  ?t_from&t_to -> {entries: [ActionLogEntry]}
    retention: "/api/retention", //               GET  -> the plan; POST {confirm} -> deletes it
    video: "/api/video", //                       GET  ?t_from&t_to -> stitched stream (NEVER ?file=)
    model: "/api/model", //                       GET  -> {active, model, sources: [ModelSource]}
    //                                                 POST {source} -> the same, switched
  };

  var MOCK_BASE = "mock/";

  var params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "live" || params.get("mode") === "mock") {
    MODE = params.get("mode");
    MODE_PINNED = true; // an explicit ?mode= always wins over detection
  }
  // Rehearsal aid: ?speed=4 divides the scripted mock delays. MUST be 1 on stage —
  // the 34.8 s of dead air is part of what the elapsed timer exists to cover.
  var SPEED = Math.max(1, parseFloat(params.get("speed") || "1") || 1);

  function mode() {
    return MODE;
  }
  function speed() {
    return SPEED;
  }
  function isMock() {
    return MODE === "mock";
  }

  // ---------------------------------------------------------------------------------
  // fetch helpers
  // ---------------------------------------------------------------------------------
  function getJSON(url) {
    return fetch(url, { headers: { Accept: "application/json" } }).then(function (r) {
      if (!r.ok) throw new Error(url + " -> HTTP " + r.status);
      return r.json();
    });
  }

  /** DELETE and PATCH share postJSON's error handling; only the verb differs. */
  function sendJSON(method, url, body) {
    var init = {
      method: method,
      headers: { Accept: "application/json" },
    };
    if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    return fetch(url, init).then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok) throw new Error((data && data.detail) || url + " -> HTTP " + r.status);
        return data;
      });
    });
  }

  /** Stop evaluating a standing task. Its history stays on the Timeline — the action
   *  log is append-only, and "why did you alert at 21:11?" must keep working for a task
   *  that no longer exists (SPEC §6.4). */
  function deleteTask(taskId) {
    if (isMock()) return Promise.resolve({ deleted: { task_id: taskId } });
    return sendJSON("DELETE", ENDPOINTS.task + encodeURIComponent(taskId));
  }

  function patchTask(taskId, changes) {
    if (isMock()) return Promise.resolve(changes);
    return sendJSON("PATCH", ENDPOINTS.task + encodeURIComponent(taskId), changes);
  }

  // ---------------------------------------------------------------------------------
  // Retention — the only destructive call this UI can make.
  //
  // Two functions, never one, because the plan is what makes the button honest: the page
  // shows the file count, the caption count and the bytes BEFORE anything is unlinked.
  // Both refuse outright in mock mode. There is no archive behind the fixtures, so a
  // scripted "deleted 42 files" would be a lie about a destructive action — the one
  // category of mock this page must not have.
  // ---------------------------------------------------------------------------------
  function retentionPlan(olderThanSeconds) {
    if (isMock()) return Promise.reject(new Error("retention needs live M3; this page is on fixtures"));
    var qs = olderThanSeconds ? "?older_than_seconds=" + encodeURIComponent(olderThanSeconds) : "";
    return getJSON(ENDPOINTS.retention + qs);
  }

  /** Irreversible. Only called from a confirmed click in retention.js. */
  function applyRetention(olderThanSeconds) {
    if (isMock()) return Promise.reject(new Error("retention needs live M3; this page is on fixtures"));
    var body = { confirm: true };
    if (olderThanSeconds) body.older_than_seconds = olderThanSeconds;
    return postJSON(ENDPOINTS.retention, body);
  }

  // ---------------------------------------------------------------------------------
  // Model source — which model answers, chosen from the topbar.
  //
  // Mock mode reports the two options and refuses the switch. The fixtures answer from
  // a scripted script, not a model, so "now using LM Studio" would be a claim about
  // something this page is not doing.
  // ---------------------------------------------------------------------------------
  function loadModel() {
    if (isMock()) {
      return Promise.resolve({
        active: "default",
        model: "mock fixtures",
        backend: "mock",
        sources: [
          { id: "default", label: "gemma-4-E2B-it", model: "fixtures", available: true, detail: "mock", note: "" },
          { id: "lmstudio", label: "LM Studio", model: "—", available: false, detail: "live mode only", note: "" },
        ],
      });
    }
    return getJSON(ENDPOINTS.model);
  }

  function selectModel(source) {
    if (isMock()) {
      return Promise.reject(new Error("switching models needs live M3; this page is on fixtures"));
    }
    return postJSON(ENDPOINTS.model, { source: source });
  }

  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    }).then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok) throw new Error((data && data.detail) || url + " -> HTTP " + r.status);
        return data;
      });
    });
  }

  var _cache = {};
  function mockJSON(name) {
    if (!_cache[name]) {
      _cache[name] = getJSON(MOCK_BASE + name).catch(function (err) {
        delete _cache[name];
        throw new Error(
          "could not load fixture " + MOCK_BASE + name + " (" + err.message + "). " +
            "If you opened index.html straight off disk, file:// blocks fetch — " +
            "run `python3 ui/serve.py` and open the localhost URL it prints."
        );
      });
    }
    return _cache[name];
  }

  // ---------------------------------------------------------------------------------
  // config — tries the real endpoint first even in mock mode, so a half-up stack still
  // renders with the operator's actual tunables.
  // ---------------------------------------------------------------------------------
  function loadConfig() {
    if (MODE_PINNED && isMock()) return mockJSON("config.json");
    // Liveness is decided by an endpoint ONLY M3 serves. /api/config is deliberately NOT
    // that endpoint: ui/serve.py serves it as well, so probing it would call the mock
    // previewer "live" and then every other pane would 404 against a server that has no
    // chunks, no history and no tasks.
    return getJSON(ENDPOINTS.tasks)
      .then(function () {
        if (!MODE_PINNED) MODE = "live";
      })
      .catch(function () {
        // No M3: opened from file://, or ui/serve.py, or the agent is down. Fixtures let
        // the page still be read and rehearsed, and the mode pill says which it is.
        if (!MODE_PINNED) MODE = "mock";
      })
      .then(function () {
        // Real settings when anything can supply them — ui/serve.py serves this route
        // precisely so a mock preview still renders the true timezone and thresholds.
        return getJSON(ENDPOINTS.config).catch(function () {
          return mockJSON("config.json");
        });
      });
  }

  /** Nested lookup with the same dotted-path feel as shared/config.get(). */
  function cfg(root, dotted, fallback) {
    var node = root;
    var parts = dotted.split(".");
    for (var i = 0; i < parts.length; i++) {
      if (node === null || typeof node !== "object" || !(parts[i] in node)) return fallback;
      node = node[parts[i]];
    }
    return node === null || node === undefined ? fallback : node;
  }

  // ---------------------------------------------------------------------------------
  // Reads
  // ---------------------------------------------------------------------------------
  function loadChunks() {
    if (isMock()) return mockJSON("chunks.json").then(pluck("chunks"));
    return getJSON(ENDPOINTS.chunks).then(pluck("chunks"));
  }

  /** One page of the index, for ui/browse.html.
   *
   *  Live mode hands the filters to M3 and renders what comes back — the server owns
   *  ordering, paging and the corpus totals, because it is the only side that can see
   *  the whole corpus. Mock mode paginates ui/mock/chunks.json through the SAME filter
   *  semantics, so the page is rehearsable with the stack down; the fixture is a dozen
   *  rows, which is enough to prove the pager moves and nowhere near enough to prove
   *  it scales.
   */
  function loadIndexPage(params) {
    var p = params || {};
    if (isMock()) return mockIndexPage(p);
    var qs = new URLSearchParams();
    if (p.offset) qs.set("offset", p.offset);
    if (p.limit) qs.set("limit", p.limit);
    if (p.q) qs.set("q", p.q);
    if (p.tier) qs.set("tier", p.tier);
    if (p.gated !== null && p.gated !== undefined && p.gated !== "") qs.set("gated", p.gated);
    if (p.t_from) qs.set("t_from", p.t_from);
    if (p.t_to) qs.set("t_to", p.t_to);
    if (p.newest_first === false) qs.set("newest_first", "false");
    var query = qs.toString();
    return getJSON(ENDPOINTS.index + (query ? "?" + query : ""));
  }

  function mockIndexPage(p) {
    return mockJSON("chunks.json").then(function (doc) {
      var rows = (doc.chunks || []).slice();
      var needle = (p.q || "").trim().toLowerCase();
      var wantGated = p.gated === true || p.gated === "true";
      var wantCaptioned = p.gated === false || p.gated === "false";
      var fromMs = p.t_from ? SPARK.time.epochMs(p.t_from) : null;
      var toMs = p.t_to ? SPARK.time.epochMs(p.t_to) : null;

      var matched = rows.filter(function (c) {
        if (p.tier && c.tier !== p.tier) return false;
        if (wantGated && !c.gated) return false;
        if (wantCaptioned && c.gated) return false;
        // Overlap, not containment — same rule the backend applies (invariant 3).
        if (fromMs !== null && SPARK.time.epochMs(c.t_end) < fromMs) return false;
        if (toMs !== null && SPARK.time.epochMs(c.t_start) > toMs) return false;
        if (needle && (c.caption || "").toLowerCase().indexOf(needle) === -1) return false;
        return true;
      });
      matched.sort(function (a, b) {
        var d = SPARK.time.epochMs(a.t_start) - SPARK.time.epochMs(b.t_start);
        if (d === 0) d = a.chunk_id < b.chunk_id ? -1 : a.chunk_id > b.chunk_id ? 1 : 0;
        return p.newest_first === false ? d : -d;
      });

      var limit = p.limit || matched.length || 1;
      var offset = Math.max(0, p.offset || 0);
      var all = doc.chunks || [];
      var gatedCount = all.filter(function (c) {
        return c.gated;
      }).length;
      return {
        chunks: matched.slice(offset, offset + limit),
        total: matched.length,
        offset: offset,
        limit: limit,
        page: matched.length ? Math.floor(offset / limit) + 1 : 0,
        pages: Math.ceil(matched.length / limit),
        filters: p,
        stats: {
          total: all.length,
          captioned: all.length - gatedCount,
          gated: gatedCount,
          skip_rate: all.length ? gatedCount / all.length : 0,
          // The fixture is a hand-written demo corpus, not a gate measurement. Saying
          // "ok"/"low" about it would be inventing a health signal out of test data.
          gate_health: "fixture",
        },
      };
    });
  }

  function loadTasks() {
    if (isMock()) {
      return mockJSON("tasks.json").then(function (d) {
        return d.tasks.concat(_registered);
      });
    }
    return getJSON(ENDPOINTS.tasks).then(pluck("tasks"));
  }

  function loadActions() {
    if (isMock()) return mockJSON("actions.json").then(pluck("entries"));
    return getJSON(ENDPOINTS.actions).then(pluck("entries"));
  }

  function loadHistory() {
    if (isMock()) {
      return Promise.all([mockJSON("chat_turns.json"), mockJSON("jobs.json")]).then(function (r) {
        return { turns: r[0].turns, jobs: r[1].jobs };
      });
    }
    return getJSON(ENDPOINTS.history);
  }

  function loadMonitorState() {
    if (isMock()) {
      return mockJSON("monitor_state.json").then(function (d) {
        return rebaseMonitorState(d);
      });
    }
    return getJSON(ENDPOINTS.monitorState);
  }

  function pluck(key) {
    return function (d) {
      return d[key] || [];
    };
  }

  // ---------------------------------------------------------------------------------
  // MOCK ONLY — rebase the funnel fixture so its clock is "now".
  //
  // Without this the cooldown in the fixture expired months ago and the Watch pane
  // renders a dead panel. With it, the cooldown counts visibly down from 247 s while
  // stages 1 and 2 keep matching — which is the SPEC §6.4 brake being proved rather
  // than asserted. Live mode never touches server timestamps.
  // ---------------------------------------------------------------------------------
  var _rebaseDelta = null;
  function rebaseMonitorState(doc) {
    if (_rebaseDelta === null) {
      _rebaseDelta = Date.now() - SPARK.time.epochMs(doc.generated_at);
    }
    var shift = function (iso) {
      if (!iso) return iso;
      return new Date(SPARK.time.epochMs(iso) + _rebaseDelta).toISOString().replace(".000Z", "Z");
    };
    var out = JSON.parse(JSON.stringify(doc));
    out.generated_at = shift(out.generated_at);
    out.tasks.forEach(function (t) {
      t.last_fired_ts = shift(t.last_fired_ts);
      if (t.stage2) t.stage2.since = shift(t.stage2.since);
      if (t.match_range) {
        t.match_range.t_start = shift(t.match_range.t_start);
        t.match_range.t_end = shift(t.match_range.t_end);
      }
    });
    out._rebased = true;
    return out;
  }

  // ---------------------------------------------------------------------------------
  // register_task — SPEC §11.3 / §10 D5.
  // The form POSTs the six §6.1 fields. Binding M3 to the same endpoint later is a tool
  // schema, not new plumbing, which is exactly why the form talks to an endpoint rather
  // than to an in-memory list.
  // ---------------------------------------------------------------------------------
  var _registered = [];

  function registerTask(task) {
    if (!isMock()) return postJSON(ENDPOINTS.registerTask, task);
    return new Promise(function (resolve, reject) {
      var dup = _registered.some(function (t) {
        return t.task_id === task.task_id;
      });
      setTimeout(function () {
        if (dup) return reject(new Error("task_id already registered: " + task.task_id));
        var stored = JSON.parse(JSON.stringify(task));
        stored.embedding = []; // M5 embeds `describe` once at registration (SPEC §6.2)
        _registered.push(stored);
        resolve(stored);
      }, 250 / SPEED);
    });
  }

  /** Funnel row for a task registered this session, so it appears in the Watch pane. */
  function syntheticMonitorRow(task) {
    return {
      task_id: task.task_id,
      state: "armed",
      in_active_window: true,
      stage1: { score: 0.0, threshold: null, matched: false, chunk_id: null },
      stage2: { verdict: null, since: null, sustain_window_s: task.window, last_chunk_id: null },
      stage3: { state: "idle", job_id: null, verdict: null },
      last_fired_ts: null,
      cooldown_seconds: task.cooldown,
      match_range: null,
      _just_registered: true,
    };
  }

  // ---------------------------------------------------------------------------------
  // ask() — the escalation arc.
  //
  // Callbacks, in the order the SPEC §4.3 shape guarantees:
  //   onSubmitted(turn)                     user bubble, immediately
  //   onProvisional(turn, {job, dedupe_of}) the turn ENDS here — never blocks on deep
  //   onRefined(turn_id, job)               later, over the WebSocket, APPENDED
  //   onFailed(turn_id, message)
  //
  // The turn ending before the job does is the whole point; anything that awaits the
  // refinement violates CLAUDE.md invariant 4.
  // ---------------------------------------------------------------------------------
  var _inflight = {}; // "t_start|t_end" -> job_id   (SPEC §4.3 dedupe key)
  var _seq = 0;

  function nowIso() {
    return new Date().toISOString().replace(/\.(\d{3})Z$/, ".$1000Z");
  }

  function newId(prefix) {
    _seq += 1;
    return prefix + "-" + String(Date.now()).slice(-6) + "-" + _seq;
  }

  function ask(question, h, opts) {
    return isMock() ? askMock(question, h) : askLive(question, h, opts);
  }

  function askLive(question, h, opts) {
    var turn = {
      turn_id: newId("turn"),
      ts: nowIso(),
      question: question,
      provisional_answer: "",
      grounded: null,
      cited_chunk_ids: [],
      job_id: null,
      latency_s: null,
    };
    h.onSubmitted(turn);
    // `widen` is the user answering the previous turn's offer — "nothing in the last
    // 30 minutes covers that; look further back?". Only a click sets it.
    var body = { question: question };
    if (opts && opts.widen) body.widen = true;
    return postJSON(ENDPOINTS.ask, body)
      .then(function (resp) {
        // The server owns turn_id; adopt it so WebSocket refinements address the
        // right card. Everything else is ChatTurn.to_dict().
        h.onProvisional(
          resp,
          {
            dedupe_of: resp.dedupe_of || null,
            job: resp.job || null,
            widen_offer: resp.widen_offer || null,
          },
          turn.turn_id
        );
        return resp;
      })
      .catch(function (err) {
        h.onFailed(turn.turn_id, err.message);
      });
  }

  function askMock(question, h) {
    var turnId = newId("turn");
    var stub = {
      turn_id: turnId,
      ts: nowIso(),
      question: question,
      provisional_answer: "",
      grounded: null,
      cited_chunk_ids: [],
      job_id: null,
      latency_s: null,
    };
    h.onSubmitted(stub);

    return mockJSON("ask_script.json").then(function (doc) {
      var script = pickScript(doc, question);
      var jobSpec = script.job ? JSON.parse(JSON.stringify(script.job)) : null;
      var dedupeOf = null;
      var job = null;

      if (jobSpec) {
        var key = jobSpec.t_start + "|" + jobSpec.t_end;
        if (_inflight[key]) {
          // SPEC §4.3 / §11.2: an impatient second click must not queue the work twice,
          // and must not look like nothing happened either.
          dedupeOf = _inflight[key];
        } else {
          job = jobSpec;
          job.job_id = jobSpec.job_id || newId("job");
          job.requested_at = nowIso();
          job.completed_at = null;
          job.state = "queued";
          _inflight[key] = job.job_id;
        }
      }

      var turn = JSON.parse(JSON.stringify(script.turn));
      turn.turn_id = turnId;
      turn.ts = stub.ts;
      turn.question = question; // the user's words, not the script's
      turn.job_id = job ? job.job_id : dedupeOf;

      setTimeout(function () {
        h.onProvisional(turn, { job: job, dedupe_of: dedupeOf }, turnId);
        if (!job) return;
        job.state = "running";
        var remaining = Math.max(0, (script.refined_ms || 0) - (script.provisional_ms || 0));
        setTimeout(function () {
          job.state = "done";
          job.completed_at = nowIso();
          delete _inflight[jobSpec.t_start + "|" + jobSpec.t_end];
          h.onRefined(turnId, job);
        }, remaining / SPEED);
      }, (script.provisional_ms || 0) / SPEED);

      return turn;
    });
  }

  function pickScript(doc, question) {
    var q = (question || "").toLowerCase();
    var chosen = null;
    doc.scripts.forEach(function (s) {
      if (chosen) return;
      var hit = (s.match || []).some(function (m) {
        return q.indexOf(m.toLowerCase()) !== -1;
      });
      if (hit) chosen = s;
    });
    if (chosen) return chosen;
    var fallback = doc.scripts.filter(function (s) {
      return s.id === doc.default_script;
    })[0];
    return fallback || doc.scripts[0];
  }

  /** The scripted demo questions, for the rehearsal buttons in the Ask pane. */
  function demoQuestions() {
    if (!isMock()) return Promise.resolve([]);
    return mockJSON("ask_script.json").then(function (doc) {
      return doc.scripts.map(function (s) {
        return { id: s.id, question: s.turn.question, grounded: s.turn.grounded };
      });
    });
  }

  // ---------------------------------------------------------------------------------
  // Live push channel. In mock mode there is nothing to connect to — the scripted
  // timers above play the same events the socket would.
  // ---------------------------------------------------------------------------------
  function connect(handlers) {
    if (isMock()) return null;
    var proto = window.location.protocol === "https:" ? "wss://" : "ws://";
    var ws = new WebSocket(proto + window.location.host + ENDPOINTS.ws);
    ws.onmessage = function (ev) {
      var msg = JSON.parse(ev.data);
      if (msg.type === "refinement" && handlers.onRefined) handlers.onRefined(msg.turn_id, msg.job);
      else if (msg.type === "monitor_state" && handlers.onMonitorState) handlers.onMonitorState(msg.state);
      else if (msg.type === "action" && handlers.onAction) handlers.onAction(msg.entry);
    };
    ws.onclose = function () {
      if (handlers.onClose) handlers.onClose();
    };
    return ws;
  }

  return {
    mode: mode,
    isMock: isMock,
    speed: speed,
    endpoints: ENDPOINTS,
    cfg: cfg,
    loadConfig: loadConfig,
    loadChunks: loadChunks,
    loadIndexPage: loadIndexPage,
    loadTasks: loadTasks,
    loadActions: loadActions,
    loadHistory: loadHistory,
    loadMonitorState: loadMonitorState,
    registerTask: registerTask,
    deleteTask: deleteTask,
    patchTask: patchTask,
    retentionPlan: retentionPlan,
    applyRetention: applyRetention,
    loadModel: loadModel,
    selectModel: selectModel,
    syntheticMonitorRow: syntheticMonitorRow,
    ask: ask,
    demoQuestions: demoQuestions,
    connect: connect,
  };
})();
