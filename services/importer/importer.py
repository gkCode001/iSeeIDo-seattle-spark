"""Slice a recorded file into archive segments, then ingest the range it now occupies.

See the package docstring for why this is a filename problem rather than a pipeline one.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared import config
from shared.timecode import list_segments, segment_name_for

from .telemetry import log_event

__all__ = [
    "ImportError_",
    "ImportResult",
    "VideoInfo",
    "default_camera_id",
    "import_video",
    "inbox_files",
    "probe_video",
    "slice_into_archive",
]


class ImportError_(RuntimeError):
    """A file that cannot become archive segments, with the reason a human needs."""


# A camera id ends up inside a filename that ``recorder.filename_pattern`` has to parse
# back out. The pattern's capture is non-greedy and the date fields are fixed width, so an
# id carrying the separator would still parse — but it would parse *ambiguously*, and an
# archive is not the place to rely on a regex being clever.
_SAFE_ID = re.compile(r"^[A-Za-z0-9-]+$")


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe says about a candidate file. ``duration`` is the only load-bearing
    field — it decides where on the timeline the recording lands."""

    path: Path
    duration: float
    codec: str
    width: int
    height: int
    fps: float

    @property
    def copyable(self) -> bool:
        """Can this stream go into our mp4 segments untouched?

        h264 only. Anything else (hevc from a phone, vp9 from a web download, mpeg4 from
        an old dataset) is re-encoded — mp4 would take some of them, but the archive is
        what M4 re-reads and a container that plays here and not in the decoder is a
        failure that surfaces an hour later as "no decodable spans".
        """
        return self.codec == "h264"


@dataclass(frozen=True)
class ImportResult:
    """One imported recording, as it now exists on the timeline."""

    source: Path
    camera_id: str
    t_start: datetime
    t_end: datetime
    segments: list[Path]
    info: VideoInfo
    reencoded: bool

    def to_dict(self) -> dict[str, object]:
        from shared.schema import to_iso

        return {
            "source": str(self.source),
            "camera_id": self.camera_id,
            "t_start": to_iso(self.t_start),
            "t_end": to_iso(self.t_end),
            "segments": [p.name for p in self.segments],
            "seconds": round(self.info.duration, 2),
            "codec": self.info.codec,
            "resolution": f"{self.info.width}x{self.info.height}",
            "reencoded": self.reencoded,
        }


# --------------------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------------------


def probe_video(path: Path, *, ffprobe_bin: str | None = None) -> VideoInfo:
    """Duration, codec and shape of ``path``'s first video stream.

    Duration is read from the *format*, not the stream: a stream that reports no duration
    is common in files remuxed by a phone or a downloader, while the container almost
    always knows. A file we cannot time cannot be placed on a timeline, so that is an
    error rather than a guess.
    """
    binary = ffprobe_bin or _ffprobe_bin()
    argv = [
        binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    except FileNotFoundError as exc:
        raise ImportError_(f"{binary} not found; ffprobe ships with ffmpeg") from exc
    except subprocess.TimeoutExpired as exc:
        raise ImportError_(f"ffprobe timed out on {path.name}") from exc
    if out.returncode != 0:
        raise ImportError_(f"ffprobe could not read {path.name}: {out.stderr.strip()[:300]}")

    try:
        payload = json.loads(out.stdout)
        stream = (payload.get("streams") or [{}])[0]
        duration = float(payload.get("format", {}).get("duration"))
    except (json.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
        raise ImportError_(
            f"{path.name} has no readable video stream or duration — ffprobe said: "
            f"{out.stdout.strip()[:200]}"
        ) from exc

    if duration <= 0:
        raise ImportError_(f"{path.name} reports a duration of {duration}s")

    return VideoInfo(
        path=path,
        duration=duration,
        codec=str(stream.get("codec_name") or "?"),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=_parse_fps(str(stream.get("avg_frame_rate") or "0/0")),
    )


def _parse_fps(raw: str) -> float:
    """ffprobe reports frame rate as ``num/den``; ``0/0`` for streams that do not say."""
    try:
        num, _, den = raw.partition("/")
        return round(float(num) / float(den), 3) if float(den) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


# --------------------------------------------------------------------------------------
# Slicing
# --------------------------------------------------------------------------------------


def slice_into_archive(
    info: VideoInfo,
    t_start: datetime,
    camera_id: str,
    *,
    archive_dir: Path | None = None,
    force_reencode: bool = False,
) -> tuple[list[Path], bool]:
    """Cut ``info.path`` into archive segments starting at ``t_start``.

    Returns the segment paths in time order and whether the video had to be re-encoded.

    The whole file is segmented into a scratch directory first and only then moved into
    the archive, one ``os.replace`` per file. M1 may be following the archive while this
    runs, and a half-written segment that appears under a name implying 60 s of footage is
    exactly the corruption CLAUDE.md warns about — the analysis windows overlapping it all
    fail to decode, and they fail *quietly*.
    """
    archive = archive_dir or config.repo_path("paths.archive")
    archive.mkdir(parents=True, exist_ok=True)
    segment_seconds = float(config.get("recorder.segment_seconds"))
    reencode = force_reencode or not info.copyable

    with tempfile.TemporaryDirectory(dir=str(archive), prefix=".import-") as tmp:
        tmpdir = Path(tmp)
        listing = tmpdir / "segments.csv"
        argv = [
            _ffmpeg_bin(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-i",
            str(info.path),
            # Video only. Audio would ride along harmlessly in the container, but nothing
            # downstream reads it and some source files carry codecs mp4 will refuse.
            "-map",
            "0:v:0",
            "-an",
        ]
        if reencode:
            gop = max(1, round((info.fps or 30.0) * _keyframe_interval_seconds()))
            argv += ["-c:v", str(config.get("recorder.device.encoder")), "-g", str(gop)]
        else:
            argv += ["-c:v", "copy"]
        argv += [
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-segment_format",
            str(config.get("recorder.container")),
            # PTS restarts at zero in every file, exactly as the recorder writes them.
            # Invariant 2 is built on that: pts_offset is meaningless without it.
            "-reset_timestamps",
            "1",
            "-segment_list",
            str(listing),
            "-segment_list_type",
            "csv",
            str(tmpdir / "part_%05d.mp4"),
        ]

        log_event(
            "import.slice",
            source=info.path.name,
            reencode=reencode,
            codec=info.codec,
            seconds=round(info.duration, 2),
        )
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise ImportError_(
                f"ffmpeg could not segment {info.path.name}: {proc.stderr.strip()[-400:]}"
            )
        if not listing.is_file():
            raise ImportError_(f"ffmpeg wrote no segment list for {info.path.name}")

        placed: list[Path] = []
        for part, offset in _read_segment_list(listing):
            source = tmpdir / part
            if not source.is_file():
                raise ImportError_(f"segment list names {part}, which ffmpeg did not write")
            # Named from ffmpeg's own reported offset, not from index * segment_time —
            # see the package docstring on cut points.
            start = t_start + timedelta(seconds=offset)
            destination = archive / segment_name_for(camera_id, start)
            if destination.exists():
                raise ImportError_(
                    f"{destination.name} already exists; importing here would overwrite "
                    f"footage. Use --camera-id or --start to place this elsewhere."
                )
            source.replace(destination)
            placed.append(destination)

    if not placed:
        raise ImportError_(f"{info.path.name} produced no segments")
    return placed, reencode


def _read_segment_list(path: Path) -> list[tuple[str, float]]:
    """``file,start,end`` rows → ``(filename, start_seconds)``, in time order."""
    rows: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            rows.append((Path(fields[0]).name, float(fields[1])))
        except ValueError:
            continue
    rows.sort(key=lambda row: row[1])
    return rows


# --------------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------------


def import_video(
    path: str | Path,
    *,
    camera_id: str | None = None,
    start: datetime | None = None,
    archive_dir: str | Path | None = None,
    force_reencode: bool = False,
    keep_source: bool = True,
) -> ImportResult:
    """Place one recording on the timeline. Does **not** ingest it — see ``__main__``.

    ``start`` defaults to ``now - duration``, so the clip ends at the moment of import.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise ImportError_(f"no such file: {source}")

    archive = Path(archive_dir) if archive_dir is not None else config.repo_path("paths.archive")
    info = probe_video(source)
    cam = camera_id or default_camera_id(archive)
    if not _SAFE_ID.match(cam):
        raise ImportError_(
            f"camera id {cam!r} must be letters, digits and dashes only — it has to "
            f"round-trip through recorder.filename_pattern"
        )

    t_start = start or (_now() - timedelta(seconds=info.duration))
    t_start = t_start.astimezone(timezone.utc).replace(microsecond=0)
    t_end = t_start + timedelta(seconds=info.duration)

    segments, reencoded = slice_into_archive(
        info, t_start, cam, archive_dir=archive, force_reencode=force_reencode
    )
    result = ImportResult(
        source=source,
        camera_id=cam,
        t_start=t_start,
        t_end=t_end,
        segments=segments,
        info=info,
        reencoded=reencoded,
    )
    _record(result)
    if not keep_source:
        source.unlink(missing_ok=True)
    log_event("import.placed", **result.to_dict())
    return result


def default_camera_id(archive_dir: Path | None = None) -> str:
    """``clip01``, ``clip02``, … — the next id not already present in the archive.

    Derived from what is on disk rather than from a counter file: the archive is the
    thing that would collide, so it is the thing worth asking.
    """
    prefix = str(config.get("importer.camera_id_prefix", "clip"))
    archive = archive_dir or config.repo_path("paths.archive")
    used: set[str] = set()
    if archive.is_dir():
        pattern = re.compile(rf"^({re.escape(prefix)}\d+)_")
        for entry in archive.iterdir():
            match = pattern.match(entry.name)
            if match:
                used.add(match.group(1))
    for n in range(1, 1000):
        candidate = f"{prefix}{n:02d}"
        if candidate not in used:
            return candidate
    raise ImportError_(f"1000 {prefix}NN ids already in use; pass --camera-id")


def inbox_files(inbox: str | Path | None = None) -> list[Path]:
    """Importable files sitting in the drop folder, oldest first.

    Hidden files and anything still being written into the folder by a copy that has not
    finished are skipped: a file whose size is still changing is not ready, and importing
    half of it produces segments that decode to nothing.
    """
    directory = Path(inbox) if inbox is not None else config.repo_path("importer.inbox")
    if not directory.is_dir():
        return []
    suffixes = {
        f".{str(ext).lstrip('.').lower()}"
        for ext in config.get("importer.extensions", ["mp4", "mov", "mkv", "avi", "webm"])
    }
    found = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and not entry.name.startswith(".") and entry.suffix.lower() in suffixes
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def ingest_settings_for(result: ImportResult, archive_dir: str | Path | None = None):
    """M1's settings, pointed at the imported recording's camera id.

    Everything else — window, stride, gate threshold, the caption prompt — is deliberately
    the live configuration. An import that was analysed by different rules than the camera
    would not be evidence of anything the demo claims.
    """
    from services.ingest.settings import IngestSettings

    return replace(IngestSettings.from_config(archive_dir), camera_id=result.camera_id)


def _record(result: ImportResult) -> None:
    """Append to ``data/imports.jsonl`` — what is on the timeline and where it came from.

    The archive holds segments named ``clip01_...`` and nothing in them says which file
    they were cut from. Without this, "what is this footage?" is answerable only by
    someone who remembers.
    """
    path = config.repo_path("importer.manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def imported_cameras(archive_dir: str | Path | None = None) -> list[str]:
    """Camera ids present in the archive that are imports rather than the live camera."""
    live = str(config.get("camera.id"))
    archive = Path(archive_dir) if archive_dir is not None else config.repo_path("paths.archive")
    if not archive.is_dir():
        return []
    cameras: set[str] = set()
    prefix = str(config.get("importer.camera_id_prefix", "clip"))
    pattern = re.compile(rf"^({re.escape(prefix)}\d+)_")
    for entry in archive.iterdir():
        match = pattern.match(entry.name)
        if match and match.group(1) != live:
            cameras.add(match.group(1))
    return sorted(cameras)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ffmpeg_bin() -> str:
    binary = str(config.get("recorder.ffmpeg_bin", "ffmpeg"))
    return shutil.which(binary) or binary


def _ffprobe_bin() -> str:
    binary = str(config.get("importer.ffprobe_bin", "ffprobe"))
    return shutil.which(binary) or binary


def _keyframe_interval_seconds() -> float:
    return float(config.get("recorder.device.keyframe_interval_seconds", 1.0))
