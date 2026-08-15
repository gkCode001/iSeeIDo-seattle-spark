"""SPEC §9, block 0 — *time a single 5-frame caption*. ``make bench``.

    | 0–2 h | **Benchmark one chunk** | ``time`` a single 5-frame caption. If >4 s, D1 is forced. |

This is the first number in the build and it governs the rest of it. SPEC §2.4 expects
~2 s per captioned chunk; SPEC §8's whole event→alert budget is built on that; and D1 —
which Cosmos 3 variant runs on the live path — is decided by this measurement and not by
taste (CLAUDE.md: benchmark before optimizing).

So the one thing this module must never do is print a credible-looking number that
measures nothing. ``vlm.backend`` is ``stub`` on this box and a stub "caption" is a
dictionary lookup: it will happily report 0.4 ms per chunk, which would settle D1 in
favour of a model that does not exist. **Every stub run is fenced with a banner and the
verdict is withheld**, not softened.

What is and is not separable
----------------------------
An OpenAI-compatible completion arrives whole. Without streaming there is no
time-to-first-token, so prefill and decode cannot be split from one response, and this
reports what it can actually measure rather than a plausible decomposition:

* **frame time** — decode, resize and overlay burn, measured directly here.
* **caption wall time** — prefill + decode + transport, from the client's own clock.
* **tokens/s** — completion tokens over the caption wall time. A *lower bound* on decode
  speed, because prefill and transport are inside the denominator. Labelled as such.

Frames come from the real archive when there is one, because prefill scales with what
the model is actually shown. With an empty archive it falls back to synthetic frames
from ``lavfi testsrc`` and says so — a bench that refuses to run on a fresh box is a
bench nobody runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from shared.queue import Priority, VLMQueue
from shared.schema import to_iso, utcnow
from shared.timecode import resolve_range
from shared.vlm_client import VLMChunk, encode_frame

from .captioner import STUB_MODEL, build_captioner
from .ffmpeg import BASE_ARGS, FFmpegDecodeError, resolve_ffmpeg, run_ffmpeg
from .frames import FrameExtractor, filter_chain, split_jpeg_stream
from .settings import IngestSettings
from .telemetry import log_event
from .windows import Window, archive_bounds

__all__ = ["BenchResult", "bench_once", "main"]

LOGGER = logging.getLogger("services.ingest.bench")

#: Printed around any measurement a stub produced. Loud on purpose: this number's only
#: job is to settle D1, and a stub cannot settle it.
_STUB_BANNER = "=" * 78


@dataclass
class BenchResult:
    """One benchmarked caption, plus everything needed to judge whether it counts."""

    backend: str
    model: str
    frames: int
    frame_bytes: int
    frame_ms: float
    caption_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    source: str = ""
    target_seconds: float = 0.0

    @property
    def synthetic(self) -> bool:
        """True when a stub produced the timing, i.e. when it means nothing."""
        return self.model == STUB_MODEL or self.backend == STUB_MODEL

    @property
    def median_ms(self) -> float:
        return statistics.median(self.caption_ms) if self.caption_ms else 0.0

    @property
    def best_ms(self) -> float:
        return min(self.caption_ms) if self.caption_ms else 0.0

    @property
    def decode_tok_per_s(self) -> float:
        """Completion tokens over caption wall time — a **lower bound** on decode speed.

        Prefill and transport sit inside the denominator and cannot be subtracted from a
        non-streamed response. Reported as a floor rather than dressed up as a rate.
        """
        if not self.completion_tokens or self.median_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.median_ms / 1000.0)

    @property
    def verdict(self) -> str:
        """SPEC §9: *if >4 s, D1 is forced.* Withheld entirely for a stub."""
        if self.synthetic:
            return "no verdict — a stub timing cannot settle SPEC §10 D1"
        if self.median_ms / 1000.0 > self.target_seconds:
            return (
                f"OVER BUDGET: {self.median_ms / 1000.0:.2f}s > {self.target_seconds:.1f}s. "
                f"SPEC §9 block 0 — D1 is forced: take the smaller Cosmos 3 variant."
            )
        return (
            f"within budget: {self.median_ms / 1000.0:.2f}s <= {self.target_seconds:.1f}s. "
            f"SPEC §2.4's ~2 s per captioned chunk holds."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "synthetic": self.synthetic,
            "frames": self.frames,
            "frame_bytes": self.frame_bytes,
            "frame_ms": round(self.frame_ms, 2),
            "caption_ms": [round(v, 2) for v in self.caption_ms],
            "caption_median_ms": round(self.median_ms, 2),
            "caption_best_ms": round(self.best_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "decode_tok_per_s_floor": round(self.decode_tok_per_s, 2),
            "source": self.source,
            "target_seconds": self.target_seconds,
            "verdict": self.verdict,
            "measured_at": to_iso(utcnow()),
        }


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------


def _frames_from_archive(settings: IngestSettings) -> tuple[list[bytes], str] | None:
    """The most recent complete window in the archive, sampled exactly as ingest would.

    Most recent rather than first: the tail of the archive is the footage someone just
    shot to test with, and benchmarking against a scene nobody remembers is how a
    surprising prefill number goes unexplained.
    """
    bounds = archive_bounds(str(settings.archive_dir), settings.camera_id)
    if bounds is None:
        return None
    width = timedelta(seconds=settings.window_seconds)
    # Walk backwards a window at a time; the final segment of a hard-killed recorder is
    # often unreadable, and the bench should step over it rather than report a failure.
    cursor = bounds[1] - width
    while cursor >= bounds[0]:
        spans = resolve_range(
            cursor, cursor + width, str(settings.archive_dir), settings.camera_id
        )
        if any(not span.is_gap for span in spans):
            try:
                frames = FrameExtractor(settings).extract(spans)
            except FFmpegDecodeError:
                frames = []
            if frames:
                window = Window(index=0, t_start=cursor, t_end=cursor + width)
                return frames, f"archive {window}"
        cursor -= width
    return None


def _frames_synthetic(settings: IngestSettings) -> tuple[list[bytes], str]:
    """``lavfi testsrc`` through the real filter chain, for a box with no archive.

    Same scale, same overlay, same JPEG quality — so the prefill side of the measurement
    is honest even though the pixels are not. Only the caption's *content* is meaningless,
    and the bench does not read it.
    """
    duration = settings.window_seconds
    argv = [
        settings.ffmpeg_bin,
        *BASE_ARGS,
        "-f",
        "lavfi",
        "-i",
        f"testsrc=s=1920x1080:r=30:d={duration}",
        "-vf",
        # A datetime, not an epoch float: filter_chain hands this to the shared drawtext
        # escaping in services/worker/decode.py, which requires an aware datetime and
        # raises AttributeError on a float. This is the fresh-box path — no archive yet —
        # so it is exactly the path that must not crash.
        filter_chain(settings, utcnow(), FrameExtractor(settings).fontsize),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        str(settings.frame_jpeg_quality),
        "-",
    ]
    raw = run_ffmpeg(argv, timeout=settings.ffmpeg_timeout_seconds)
    return split_jpeg_stream(raw), "synthetic lavfi testsrc (no archive found)"


# --------------------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------------------


def bench_once(settings: IngestSettings, *, repeat: int = 1) -> BenchResult:
    """Sample one window's frames, then caption them ``repeat`` times.

    The frames are sampled once and reused across repeats on purpose: SPEC §9 asks how
    long a *caption* takes, and re-decoding the same five frames each time would fold the
    frame cost into the number that decides D1.

    The caption goes through ``shared/queue.py`` at ``Priority.INGEST`` like every other
    ingest caption (invariant 1 — nothing reaches the endpoint any other way). With an
    idle queue that adds no measurable time, and it keeps the bench measuring the path
    ingest actually uses rather than a shortcut around it.
    """
    resolve_ffmpeg(settings.ffmpeg_bin)

    started = time.perf_counter()
    found = _frames_from_archive(settings)
    frames, source = found if found is not None else _frames_synthetic(settings)
    frame_ms = (time.perf_counter() - started) * 1000.0
    if not frames:
        raise FFmpegDecodeError("no frames could be sampled for the benchmark")

    captioner = build_captioner(settings)
    chunk = VLMChunk(
        chunk_id=f"bench_{utcnow():%Y%m%dT%H%M%S}",
        frames=[encode_frame(frame) for frame in frames],
    )

    result = BenchResult(
        backend=settings.vlm_backend,
        model=captioner.model,
        frames=len(frames),
        frame_bytes=sum(len(f) for f in frames),
        frame_ms=frame_ms,
        source=source,
        target_seconds=settings.bench_target_seconds,
    )

    queue = VLMQueue()
    queue.start()
    try:
        for _ in range(max(1, repeat)):
            call_started = time.perf_counter()
            # A list of one — invariant 9, and the same call ingest makes.
            job = queue.submit(
                Priority.INGEST, lambda: captioner.caption([chunk]), label="bench"
            )
            results = job.result(timeout=settings.caption_timeout_seconds)
            result.caption_ms.append((time.perf_counter() - call_started) * 1000.0)
            if results:
                result.prompt_tokens = results[0].prompt_tokens
                result.completion_tokens = results[0].completion_tokens
    finally:
        queue.stop(drain=False)

    log_event("bench.caption", **result.to_dict())
    return result


def format_report(result: BenchResult) -> str:
    """The human-readable block ``make bench`` prints."""
    lines: list[str] = []
    if result.synthetic:
        lines += [
            _STUB_BANNER,
            "  THIS NUMBER IS MEANINGLESS.",
            f"  vlm.backend={result.backend!r}, model={result.model!r} — the caption was",
            "  a dictionary lookup, not inference. It does not settle SPEC §10 D1 and it",
            "  must not be quoted as a latency figure. Set vlm.backend: vllm and point",
            "  vlm.model at a real Cosmos 3 variant, then run `make bench` again.",
            _STUB_BANNER,
            "",
        ]

    lines += [
        "SPEC §9 block 0 — single caption benchmark",
        f"  backend           {result.backend}",
        f"  model             {result.model}",
        f"  frames            {result.frames} ({result.frame_bytes / 1024:.0f} KiB of JPEG)",
        f"  source            {result.source}",
        "",
        f"  frame sampling    {result.frame_ms:8.1f} ms   decode + resize + overlay burn",
        f"  caption (median)  {result.median_ms:8.1f} ms   prefill + decode + transport",
        f"  caption (best)    {result.best_ms:8.1f} ms",
        f"  runs              {len(result.caption_ms)}",
        "",
        f"  prompt tokens     {result.prompt_tokens}",
        f"  completion tokens {result.completion_tokens}",
    ]
    if result.decode_tok_per_s:
        lines.append(
            f"  decode floor      {result.decode_tok_per_s:8.1f} tok/s  "
            f"(lower bound — prefill is inside this)"
        )
    else:
        lines.append(
            "  decode floor           n/a  no completion tokens reported; "
            "prefill/decode cannot be split from a non-streamed response"
        )
    lines += ["", f"  verdict           {result.verdict}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m services.ingest.bench",
        description=(
            "Time a single 5-frame caption (SPEC §9, block 0). The number that decides "
            "SPEC §10 D1."
        ),
    )
    parser.add_argument("--archive", help="override paths.archive for this run")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="caption the same frames N times and report the median (default: 1)",
    )
    parser.add_argument("--json", action="store_true", help="emit the result as JSON only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.json else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    settings = IngestSettings.from_config(args.archive)
    result = bench_once(settings, repeat=args.repeat)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(result))
    # Exit non-zero when the number is real and over budget: `make bench` is the gate on
    # block 0, and a build step that reports failure only in prose gets read as passing.
    return 0 if result.synthetic or result.median_ms / 1000.0 <= result.target_seconds else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
