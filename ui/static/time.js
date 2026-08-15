/* SPEC §11.5 — THE conversion helper. The only place in this UI where a UTC instant
 * becomes a local string.
 *
 * Rules this file enforces:
 *   - Everything underneath is UTC with a Z suffix. Payloads are never rewritten.
 *   - Conversion happens at render, here, and nowhere else. If you find yourself
 *     reaching for Date#toLocaleTimeString in a pane, that is the bug.
 *   - The conversion runs one way only, UTC -> local. There is deliberately no
 *     local -> UTC inverse: two directions is two chances to be wrong on a DST edge,
 *     and nothing in the UI needs it (citations carry their own Z-suffixed range).
 *
 * Timezone and format come from config (ui.display_timezone / ui.time_format) via
 * configure(). Before configure() runs, this module formats in UTC and says so.
 */
window.SPARK = window.SPARK || {};

SPARK.time = (function () {
  "use strict";

  var TZ = "UTC";
  var FMT = "%H:%M:%S";
  var configured = false;

  /** Wire in ui.display_timezone / ui.time_format from config/settings.yaml. */
  function configure(cfg) {
    var ui = (cfg && cfg.ui) || {};
    if (ui.display_timezone) TZ = ui.display_timezone;
    if (ui.time_format) FMT = ui.time_format;
    // Fail loudly on an unusable timezone rather than silently rendering UTC and
    // letting someone read the wrong minute off a demo screen.
    try {
      new Intl.DateTimeFormat("en-GB", { timeZone: TZ }).format(new Date());
    } catch (e) {
      console.error("[time] unusable ui.display_timezone " + TZ + "; falling back to UTC");
      TZ = "UTC";
    }
    configured = true;
  }

  function timezone() {
    return TZ;
  }

  function isConfigured() {
    return configured;
  }

  /** Parse an ISO-8601 Z timestamp. Fractional seconds are trimmed to milliseconds,
   *  because shared/schema.py to_iso() emits microseconds when they are non-zero and
   *  not every engine parses six digits. */
  function parse(iso) {
    if (iso instanceof Date) return iso;
    if (typeof iso !== "string") return null;
    var normalized = iso.replace(/\.(\d{3})\d+/, ".$1");
    var d = new Date(normalized);
    return isNaN(d.getTime()) ? null : d;
  }

  function epochMs(iso) {
    var d = parse(iso);
    return d ? d.getTime() : NaN;
  }

  var PART_OPTS = {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  };

  function partsFor(date, tz) {
    var opts = { timeZone: tz || TZ };
    for (var k in PART_OPTS) opts[k] = PART_OPTS[k];
    var out = {};
    new Intl.DateTimeFormat("en-GB", opts).formatToParts(date).forEach(function (p) {
      out[p.type] = p.value;
    });
    // en-GB renders midnight as "24" in some engines; normalize to strftime's 00.
    if (out.hour === "24") out.hour = "00";
    return out;
  }

  var _tzLabel = null;
  function tzLabel() {
    if (_tzLabel) return _tzLabel;
    var label = TZ;
    try {
      new Intl.DateTimeFormat("en-GB", { timeZone: TZ, timeZoneName: "shortOffset" })
        .formatToParts(new Date())
        .forEach(function (p) {
          if (p.type === "timeZoneName") label = TZ + " (" + p.value + ")";
        });
    } catch (e) {
      /* shortOffset unsupported; the bare tz name is still honest */
    }
    _tzLabel = label;
    return label;
  }

  /** strftime-lite. Supports the subset config/settings.yaml can plausibly hold. */
  function fmt(iso, pattern, tz) {
    var d = parse(iso);
    if (!d) return "--:--:--";
    var p = partsFor(d, tz);
    var h24 = parseInt(p.hour, 10);
    var h12 = h24 % 12 === 0 ? 12 : h24 % 12;
    var map = {
      Y: p.year,
      m: p.month,
      d: p.day,
      H: p.hour,
      M: p.minute,
      S: p.second,
      I: String(h12).padStart(2, "0"),
      p: h24 < 12 ? "AM" : "PM",
      Z: tz || TZ,
      "%": "%",
    };
    return String(pattern || FMT).replace(/%(.)/g, function (whole, key) {
      return key in map ? map[key] : whole;
    });
  }

  /** "21:11:07–21:11:52" — the clickable-citation label from SPEC §11.2. */
  function range(isoStart, isoEnd, pattern) {
    return fmt(isoStart, pattern) + "–" + fmt(isoEnd, pattern);
  }

  /** Render an instant in UTC. This is NOT the §11.5 conversion — it converts nothing.
   *  It exists for one caller: the wall-clock overlay the player mimics, which ingest
   *  burns into the frame in UTC (ingest.overlay.format). Rendering that stamp in local
   *  time would misrepresent what the VLM actually reads off the pixels. */
  function fmtUtc(iso, pattern) {
    return fmt(iso, pattern, "UTC");
  }

  /** The underlying UTC value, verbatim, for tooltips and title attributes. Handing a
   *  reader the Z-suffixed original next to the local render is what makes §11.5
   *  auditable on stage instead of a claim in a doc. */
  function utc(iso) {
    return typeof iso === "string" ? iso : "";
  }

  function durationSeconds(isoStart, isoEnd) {
    var a = epochMs(isoStart);
    var b = epochMs(isoEnd);
    if (isNaN(a) || isNaN(b)) return NaN;
    return (b - a) / 1000;
  }

  /** "34.8 s" / "2.1 s" — one decimal, the way SPEC §11.2 prints latencies. */
  function secs(value, decimals) {
    if (value === null || value === undefined || isNaN(value)) return "--";
    var dp = decimals === undefined ? 1 : decimals;
    return Number(value).toFixed(dp) + " s";
  }

  /** "247s" — integer seconds, for the cooldown countdown. */
  function countdown(value) {
    return Math.max(0, Math.ceil(value)) + "s";
  }

  return {
    configure: configure,
    isConfigured: isConfigured,
    timezone: timezone,
    tzLabel: tzLabel,
    fmt: fmt,
    fmtUtc: fmtUtc,
    range: range,
    utc: utc,
    epochMs: epochMs,
    durationSeconds: durationSeconds,
    secs: secs,
    countdown: countdown,
  };
})();
