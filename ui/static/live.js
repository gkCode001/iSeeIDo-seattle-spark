/* Live camera view — SPEC §11, addendum.
 *
 * Polls GET /api/live.jpg, a rolling JPEG the recorder's own ffmpeg rewrites ~2x/second
 * as a SECOND OUTPUT of the process that writes the archive. v4l2 access to /dev/video0
 * is exclusive, so a live view cannot come from a second reader; it comes from the
 * recorder or not at all.
 *
 * Deliberately a polled still, not a <video> or an MJPEG stream: one small GET per tick,
 * it survives the recorder restarting, and it cannot wedge a socket open against the one
 * process whose whole job is to keep recording (SPEC §2.1).
 *
 * This pane shows what the camera SEES. Every other pane shows what the system
 * UNDERSTOOD, which is deliberately later and lossier — that gap is the design, so the
 * two are kept visually distinct rather than blended into one "feed".
 */
window.SPARK = window.SPARK || {};

SPARK.live = (function () {
  "use strict";

  var POLL_MS = 500; // the recorder writes at 2 fps; polling faster only re-reads bytes

  function init(root, config) {
    if (!root) return null;
    var img = root.querySelector("[data-live-img]");
    var badge = root.querySelector("[data-live-badge]");
    var clock = root.querySelector("[data-live-clock]");
    var source = root.querySelector("[data-live-source]");
    var offline = root.querySelector("[data-live-offline]");
    if (!img) return null;

    var cfg = config || {};
    var cameraId = (cfg.camera && cfg.camera.id) || "cam01";
    var timer = null;
    var misses = 0;

    function online() {
      misses = 0;
      badge.textContent = "live";
      badge.classList.add("badge--live");
      offline.hidden = true;
      img.hidden = false;
    }

    function offlineNow(detail) {
      misses += 1;
      // One miss is a read that caught ffmpeg mid-rewrite — normal, and it must not make
      // the badge flap. Two consecutive misses means no frames are being produced.
      if (misses < 2) return;
      badge.textContent = "offline";
      badge.classList.remove("badge--live");
      offline.hidden = false;
      img.hidden = true;
      if (detail && source) source.textContent = detail;
    }

    function tick() {
      // The URL is stable but the bytes are not, and a cached live view is a photograph.
      var probe = new Image();
      probe.onload = function () {
        img.src = probe.src;
        online();
        if (clock) {
          clock.textContent = SPARK.time && SPARK.time.isConfigured()
            ? SPARK.time.fmt(new Date().toISOString())
            : new Date().toISOString().slice(11, 19);
        }
        if (source) {
          source.textContent = cameraId + " · " + probe.naturalWidth + "×" + probe.naturalHeight;
        }
      };
      probe.onerror = function () {
        offlineNow("waiting for the recorder");
      };
      probe.src = "/api/live.jpg?t=" + Date.now();
    }

    function stop() {
      if (timer) { window.clearInterval(timer); timer = null; }
    }

    function start() {
      if (timer) return;
      tick();
      timer = window.setInterval(tick, POLL_MS);
    }

    if (cfg.__mode === "mock") {
      // Mock mode has no camera. Say so rather than showing a stale or invented frame —
      // fabricating a "live" view is the one thing this pane must never do.
      badge.textContent = "mock mode";
      offline.hidden = false;
      offline.querySelector("p").textContent = "Live camera is unavailable in mock mode.";
      img.hidden = true;
      return { start: function () {}, stop: function () {} };
    }

    start();
    // Polling a hidden tab spends bandwidth on the same box that is running the model.
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) stop(); else start();
    });

    return { start: start, stop: stop };
  }

  return { init: init };
})();
