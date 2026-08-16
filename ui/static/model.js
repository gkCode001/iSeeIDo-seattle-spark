/* The model selector: which model answers on the Ask surface.
 *
 * Two options, both already running somewhere — this control never loads or unloads
 * anything. "default" is the llama-server `make serve` started; "LM Studio" is whatever
 * model LM Studio happens to have loaded, resolved by asking it rather than configured
 * by filename, so the GUI stays the source of truth for what is loaded there.
 *
 * Three things this deliberately shows rather than hides:
 *
 *   1. The MODEL ID, not just the label. The whole reason to switch is to compare two
 *      models on the same question; a selector that says "LM Studio" without saying
 *      which model is loaded turns the comparison into a guess.
 *   2. WHY an option is unavailable, in the option itself. LM Studio not running, no
 *      model loaded and a too-small context are three different fixes, and the server
 *      already distinguishes them — dropping that to a greyed-out row wastes it.
 *   3. That both servers being up is a PROBLEM. They share one 128 GB pool
 *      (CLAUDE.md invariant 1), and this control can start neither and stop neither.
 *
 * Scope, stated on the control because it is not guessable: this rebinds M3 only. The
 * captioner runs in the ingest process against vlm.endpoint and does not move.
 */
window.SPARK = window.SPARK || {};

SPARK.model = (function () {
  "use strict";

  var refs = null;
  var state = null;
  var busy = false;

  function init() {
    refs = {
      root: document.querySelector("[data-model]"),
      button: document.querySelector("[data-model-button]"),
      panel: document.querySelector("[data-model-panel]"),
      list: document.querySelector("[data-model-list]"),
      note: document.querySelector("[data-model-note]"),
      scope: document.querySelector("[data-model-scope]"),
    };
    if (!refs.button) return Promise.resolve();

    refs.button.addEventListener("click", toggle);
    // Any click outside closes it. A dropdown pinned open behind the Ask pane is the
    // kind of thing nobody notices until it is on a projector.
    document.addEventListener("click", function (event) {
      if (refs.root && !refs.root.contains(event.target)) close();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") close();
    });

    return refresh();
  }

  function refresh() {
    return SPARK.data
      .loadModel()
      .then(function (payload) {
        state = payload;
        render();
        return payload;
      })
      .catch(function (err) {
        state = null;
        refs.button.textContent = "model ?";
        refs.button.title = "could not read /api/model: " + err.message;
      });
  }

  // -----------------------------------------------------------------------------------
  // render
  // -----------------------------------------------------------------------------------
  function render() {
    if (!state) return;
    var active = activeSource();

    // The pill carries the model id, not the source id. "what is answering me" is the
    // question someone reading a transcript actually has, and the source name does not
    // answer it — especially for LM Studio, where the model is whatever is loaded.
    refs.button.textContent = shortModel(state.model || (active && active.label) || "model");
    refs.button.title =
      "answering with " +
      (state.model || "?") +
      (state.backend ? " (" + state.backend + ")" : "") +
      "\nclick to switch the Ask surface to another model";
    refs.button.className = "pill pill--model" + (state.warning ? " pill--warn" : "");

    refs.list.textContent = "";
    (state.sources || []).forEach(function (source) {
      refs.list.appendChild(row(source));
    });

    refs.note.textContent = state.warning || "";
    refs.note.hidden = !state.warning;
    refs.scope.textContent = state.scope || "";
  }

  function row(source) {
    var selected = source.id === state.active;
    var el = document.createElement("button");
    el.type = "button";
    el.className =
      "model-row" +
      (selected ? " model-row--on" : "") +
      (source.available ? "" : " model-row--off");
    el.disabled = busy || (!source.available && !selected);

    var head = document.createElement("span");
    head.className = "model-row-head";
    head.textContent = source.label;
    if (selected) {
      var tick = document.createElement("span");
      tick.className = "model-row-tick";
      tick.textContent = "answering";
      head.appendChild(tick);
    }
    el.appendChild(head);

    var id = document.createElement("span");
    id.className = "model-row-id";
    id.textContent = source.model || "—";
    el.appendChild(id);

    // The reason line. For a reachable option this is the endpoint and context; for an
    // unreachable one it is the sentence explaining what to fix, straight from the
    // server — which knows whether LM Studio is closed, empty or loaded too small.
    var detail = source.note || source.detail;
    if (detail) {
      var why = document.createElement("span");
      why.className = "model-row-why" + (source.available ? "" : " model-row-why--bad");
      why.textContent = detail;
      el.appendChild(why);
    }

    if (!selected) el.addEventListener("click", function () { choose(source); });
    return el;
  }

  function activeSource() {
    return (state.sources || []).filter(function (s) {
      return s.id === state.active;
    })[0];
  }

  /** Filename -> the name a person would say. Matches the server's own labelling. */
  function shortModel(name) {
    var stem = String(name).split("/").pop().replace(/\.gguf$/, "");
    return stem.split("-UD-")[0] || stem;
  }

  // -----------------------------------------------------------------------------------
  // choose — the switch
  // -----------------------------------------------------------------------------------
  function choose(source) {
    if (busy || !source.available) return;
    busy = true;
    refs.note.hidden = false;
    refs.note.textContent = "switching to " + source.label + "…";

    SPARK.data
      .selectModel(source.id)
      .then(function (payload) {
        busy = false;
        state = payload;
        render();
        // Turns already on the page were answered by the previous model and still say
        // so. Only later questions move — the same rule the server applies to a job
        // already in flight.
        refs.note.hidden = false;
        refs.note.textContent =
          "now answering with " +
          (payload.model || source.label) +
          (payload.switched_from ? " (was " + shortModel(payload.switched_from) + ")" : "") +
          ". Answers already on the page came from the previous model.";
      })
      .catch(function (err) {
        busy = false;
        render();
        refs.note.hidden = false;
        refs.note.textContent = "could not switch: " + err.message;
      });
  }

  // -----------------------------------------------------------------------------------
  // open/close
  // -----------------------------------------------------------------------------------
  function toggle(event) {
    event.stopPropagation();
    if (refs.panel.hidden) open();
    else close();
  }

  function open() {
    refs.panel.hidden = false;
    // Re-probe on open: LM Studio is loaded and unloaded from a GUI this page cannot
    // see, so availability read at page load is already stale by the time it is offered.
    refresh();
  }

  function close() {
    if (refs.panel) refs.panel.hidden = true;
  }

  return { init: init, refresh: refresh };
})();
