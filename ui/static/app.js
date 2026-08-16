/* Bootstrap. Load config first (it carries the display timezone and every tunable the
 * panes read), then wire the three panes over the one shared player.
 *
 * Order matters for exactly one reason: SPARK.time must be configured before anything
 * renders a timestamp, or the first paint is in UTC and the second is not.
 */
(function () {
  "use strict";

  var banner = document.querySelector("[data-banner]");

  function showBanner(text, kind) {
    banner.hidden = false;
    banner.className = "banner" + (kind ? " banner--" + kind : "");
    banner.textContent = text;
  }

  function header(cfg) {
    var mode = document.querySelector("[data-mode-pill]");
    if (SPARK.data.isMock()) {
      mode.textContent = "MOCK FIXTURES";
      mode.className = "pill pill--mock";
      mode.title =
        "Rendering ui/mock/*.json. Flip MODE in ui/static/data.js (or add ?mode=live) " +
        "once M3 is serving the endpoints listed there.";
      if (SPARK.data.speed() !== 1) {
        mode.textContent += " · " + SPARK.data.speed() + "× SPEED";
        mode.title += "\nScripted delays are divided by " + SPARK.data.speed() + ". Set 1× before rehearsing.";
      }
    } else {
      mode.textContent = "LIVE · M3";
      mode.className = "pill pill--live";
    }

    var tz = document.querySelector("[data-tz-pill]");
    tz.textContent = SPARK.time.tzLabel();
    tz.title =
      "SPEC §11.5: UTC everywhere underneath, converted once at render. " +
      "Hover any timestamp to see the Z-suffixed value it came from.";

    document.querySelector("[data-camera-id]").textContent = SPARK.data.cfg(cfg, "camera.id", "cam01");

    var clock = document.querySelector("[data-clock-pill]");
    setInterval(function () {
      clock.textContent = SPARK.time.fmt(new Date().toISOString());
    }, 500);
  }

  SPARK.data
    .loadConfig()
    .then(function (cfg) {
      SPARK.time.configure(cfg);
      header(cfg);

      return SPARK.data.loadChunks().then(function (chunks) {
        SPARK.player.init(document.querySelector("[data-player]"), cfg, chunks);
        // What the camera sees right now, beside what the system understood from it.
        cfg.__mode = SPARK.data.mode ? SPARK.data.mode() : "live";
        SPARK.live.init(document.querySelector("[data-live]"), cfg);

        var jobs = [
          SPARK.ask.init(document.querySelector("[data-ask]"), cfg, chunks),
          SPARK.watch.init(document.querySelector("[data-watch]"), cfg),
          SPARK.timeline.init(document.querySelector("[data-timeline]"), cfg),
          // The topbar's delete control. Wired last and separately from the panes: it
          // reads nothing they render and writes nothing they hold.
          SPARK.retention.init(cfg),
          // The topbar's model selector. Same reasoning: it rebinds which model the
          // Ask surface calls and touches nothing any pane is holding.
          SPARK.model.init(),
        ];

        // Live push channel. In mock mode this is a no-op: the scripted timers in
        // data.js play the same events the socket would.
        SPARK.data.connect({
          onRefined: function (turnId, job) {
            SPARK.ask.appendRefinement(turnId, job);
          },
          onAction: function () {
            SPARK.timeline.refresh();
          },
          onMonitorState: function () {
            SPARK.watch.refresh();
          },
          onClose: function () {
            showBanner("WebSocket closed — refinements will stop arriving until it reconnects.", "warn");
          },
        });

        return Promise.all(jobs);
      });
    })
    .catch(function (err) {
      showBanner(err.message, "bad");
      console.error(err);
    });
})();
