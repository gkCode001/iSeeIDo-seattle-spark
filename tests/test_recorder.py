"""Tests for the recorder (SPEC §2.1).

Stdlib ``unittest``, no third-party packages:

    python3 -m unittest discover -s tests -t . -v

**Nothing here needs ffmpeg**, which is not installed on this box (CLAUDE.md machine
state). That is the whole reason command construction is a pure function: what gets
asserted is the argv list, and the supervision logic runs against a fake process, a fake
clock and a fake sleep, so the restart tests finish in microseconds instead of minutes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import subprocess
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

from shared import config

from services.recorder import (
    FFmpegMissingError,
    RecorderError,
    RecorderSettings,
    RecorderSupervisor,
    SourceKind,
    SourceUnsetError,
    backoff_delay,
    build_ffmpeg_command,
    classify_source,
    describe_command,
    normalize_source,
    redact_source,
    resolve_ffmpeg,
)
from services.recorder import log as recorder_log
from services.recorder import settings as recorder_settings

RTSP = "rtsp://user:secret@10.0.0.9:554/stream1"


def make_settings(archive: Path, **overrides: object) -> RecorderSettings:
    """A settings object with no config dependency, for the pure-construction tests."""
    base = RecorderSettings(
        camera_id="cam01",
        source=RTSP,
        archive_dir=archive,
        segment_seconds=60,
        filename_pattern="{camera_id}_%Y%m%d_%H%M%S.mp4",
        container="mp4",
        copy_codec=True,
        rtsp_transport="tcp",
        realtime_file_playback=True,
        ffmpeg_bin="ffmpeg",
        loglevel="warning",
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Source handling — SPEC §10 D2 is open, so both kinds must work and neither is chosen
# --------------------------------------------------------------------------------------


class TestSourceClassification(unittest.TestCase):
    def test_rtsp_urls_are_streams(self) -> None:
        self.assertIs(classify_source(RTSP), SourceKind.RTSP)
        self.assertIs(classify_source("rtsps://cam.local/stream"), SourceKind.RTSP)

    def test_paths_are_files(self) -> None:
        self.assertIs(classify_source("tests/fixtures/demo.mp4"), SourceKind.FILE)
        self.assertIs(classify_source("/data/demo.mp4"), SourceKind.FILE)
        self.assertIs(classify_source("file:///data/demo.mp4"), SourceKind.FILE)

    def test_unknown_scheme_is_rejected_with_both_options_named(self) -> None:
        with self.assertRaises(RecorderError) as ctx:
            classify_source("rtp://cam.local/stream")
        message = str(ctx.exception)
        self.assertIn("rtp", message)
        self.assertIn("rtsp://", message)
        self.assertIn("file on disk", message)

    def test_empty_source_is_unset_not_invalid(self) -> None:
        with self.assertRaises(SourceUnsetError):
            classify_source("   ")

    def test_credentials_never_reach_a_log_line(self) -> None:
        redacted = redact_source(RTSP)
        self.assertNotIn("secret", redacted)
        self.assertIn("10.0.0.9", redacted)
        # A path has no credentials to strip and must survive untouched.
        self.assertEqual(redact_source("/data/demo.mp4"), "/data/demo.mp4")

    def test_relative_file_source_is_anchored_to_the_repo(self) -> None:
        resolved = normalize_source("tests/fixtures/demo.mp4")
        self.assertTrue(Path(resolved).is_absolute())
        self.assertTrue(resolved.endswith("tests/fixtures/demo.mp4"))

    def test_file_url_is_unwrapped_for_ffmpeg(self) -> None:
        self.assertEqual(normalize_source("file:///data/demo.mp4"), "/data/demo.mp4")

    def test_stream_url_is_passed_through_verbatim(self) -> None:
        self.assertEqual(normalize_source(RTSP), RTSP)


class TestDeviceCapture(unittest.TestCase):
    """SPEC §10 D2 resolved to a USB webcam. A v4l2 device is a path by spelling but a
    live capture by behaviour, and the two must not be confused."""

    def setUp(self) -> None:
        self.archive = Path(tempfile.mkdtemp(prefix="archive-"))

    def device_settings(self, **overrides: object) -> RecorderSettings:
        return make_settings(self.archive, source="/dev/video0", **overrides)

    def test_a_device_node_is_not_a_file(self) -> None:
        self.assertIs(classify_source("/dev/video0"), SourceKind.DEVICE)
        self.assertIs(classify_source("/dev/video2"), SourceKind.DEVICE)
        # A real file that merely lives near one is still a file.
        self.assertIs(classify_source("/data/video0.mp4"), SourceKind.FILE)

    def test_device_path_is_not_resolved_against_the_repo(self) -> None:
        """/dev/video0 is frequently a symlink; ffmpeg wants the configured name."""
        self.assertEqual(normalize_source("/dev/video0"), "/dev/video0")

    def test_v4l2_input_format_and_capture_mode_are_explicit(self) -> None:
        argv = build_ffmpeg_command(self.device_settings())
        pairs = {a: b for a, b in zip(argv, argv[1:], strict=False) if a.startswith("-")}
        # Without -f v4l2 ffmpeg probes the node as a file and dies on a demuxer error.
        self.assertEqual(argv[argv.index("-f")], "-f")
        self.assertEqual(argv[argv.index("-f") + 1], "v4l2")
        # A camera asked for nothing in particular hands back 640x480 and says nothing.
        self.assertEqual(pairs["-video_size"], "1280x720")
        self.assertEqual(pairs["-framerate"], "30")

    def test_a_webcam_is_encoded_not_stream_copied(self) -> None:
        """rawvideo/mjpeg do not stream-copy into mp4 as anything a player will open."""
        argv = build_ffmpeg_command(self.device_settings())
        self.assertNotIn("-c", argv, "a webcam must not be stream-copied")
        self.assertEqual(argv[argv.index("-c:v") + 1], "h264_nvenc")

    def test_encoding_still_never_downscales_the_archive(self) -> None:
        """Invariant 7 forbids downscaling the archive, not encoding it. There must be
        no scale filter, no size flag and no bitrate cap anywhere on this path."""
        argv = build_ffmpeg_command(self.device_settings())
        for forbidden in ("-vf", "-filter:v", "-s", "-b:v", "-crf"):
            self.assertNotIn(forbidden, argv, f"{forbidden} would degrade the archive")

    def test_input_format_is_passed_through_when_set(self) -> None:
        argv = build_ffmpeg_command(self.device_settings(device_input_format="mjpeg"))
        self.assertEqual(argv[argv.index("-input_format") + 1], "mjpeg")

    def test_no_input_format_flag_when_unset(self) -> None:
        argv = build_ffmpeg_command(self.device_settings(device_input_format=None))
        self.assertNotIn("-input_format", argv)

    def test_a_device_is_never_paced_with_re(self) -> None:
        """-re is for reading a file faster than realtime. A camera is already realtime;
        pacing it would drift the archive away from wall clock and break invariant 2."""
        argv = build_ffmpeg_command(self.device_settings(realtime_file_playback=True))
        self.assertNotIn("-re", argv)

    def test_the_segmenter_flags_survive_on_the_device_path(self) -> None:
        argv = build_ffmpeg_command(self.device_settings())
        pairs = {a: b for a, b in zip(argv, argv[1:], strict=False) if a.startswith("-")}
        self.assertEqual(pairs["-strftime"], "1")
        self.assertEqual(pairs["-segment_time"], "60")
        self.assertEqual(pairs["-reset_timestamps"], "1")

    def test_keyframe_interval_bounds_clip_accuracy(self) -> None:
        """A stream-copied clip can only begin on a keyframe, so the GOP is what makes
        invariant 3's fetch-by-time-range accurate to the second rather than to the
        nearest keyframe. Measured: an 8.3 s GOP turned a 2.0 s request into 8.07 s."""
        argv = build_ffmpeg_command(
            self.device_settings(device_framerate=30, device_keyframe_interval_seconds=1.0)
        )
        self.assertEqual(argv[argv.index("-g") + 1], "30")

    def test_keyframe_interval_scales_with_framerate(self) -> None:
        argv = build_ffmpeg_command(
            self.device_settings(device_framerate=60, device_keyframe_interval_seconds=0.5)
        )
        self.assertEqual(argv[argv.index("-g") + 1], "30")

    def test_keyframe_interval_never_rounds_to_zero(self) -> None:
        """-g 0 disables keyframes entirely, which would make every clip start at the
        beginning of its segment."""
        argv = build_ffmpeg_command(
            self.device_settings(device_framerate=30, device_keyframe_interval_seconds=0.001)
        )
        self.assertEqual(argv[argv.index("-g") + 1], "1")

    def test_a_fallback_encoder_can_be_configured(self) -> None:
        argv = build_ffmpeg_command(
            self.device_settings(encoder="libx264", encoder_args=("-preset", "veryfast"))
        )
        self.assertEqual(argv[argv.index("-c:v") + 1], "libx264")
        self.assertIn("-preset", argv)


class TestSettingsFromConfig(unittest.TestCase):
    def test_configured_source_is_the_webcam(self) -> None:
        """SPEC §10 D2 resolved to a USB webcam as the primary source."""
        settings = RecorderSettings.from_config()
        self.assertIs(settings.kind, SourceKind.DEVICE)
        self.assertTrue(settings.source.startswith("/dev/video"))

    def test_unset_source_still_names_the_open_decision(self) -> None:
        """D2 is answered, but the recorder must still refuse to invent a source if the
        setting is ever cleared — a silent default would record nothing and say nothing."""
        with override_settings({"recorder.source": None}), self.assertRaises(
            SourceUnsetError
        ) as ctx:
            RecorderSettings.from_config()
        message = str(ctx.exception)
        self.assertIn("D2", message)
        self.assertIn("--source", message)
        # Both options are offered; neither is picked.
        self.assertIn("rtsp://", message)
        self.assertIn(".mp4", message)

    def test_override_uses_the_configured_archive_and_segment_length(self) -> None:
        settings = RecorderSettings.from_config(source=RTSP)
        self.assertEqual(settings.camera_id, "cam01")
        self.assertEqual(settings.segment_seconds, 60)
        self.assertTrue(settings.copy_codec, "invariant 7: the archive is stream-copied")
        self.assertTrue(settings.archive_dir.is_absolute())
        self.assertIs(settings.kind, SourceKind.RTSP)

    def test_env_var_can_point_at_another_ffmpeg_build(self) -> None:
        with mock.patch.dict("os.environ", {"SPARK_FFMPEG": "/opt/ffmpeg-cuda/bin/ffmpeg"}):
            settings = RecorderSettings.from_config(source=RTSP)
        self.assertEqual(settings.ffmpeg_bin, "/opt/ffmpeg-cuda/bin/ffmpeg")


# --------------------------------------------------------------------------------------
# Command construction — the argv list is the unit under test, never a live process
# --------------------------------------------------------------------------------------


class TestCommandConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.archive = Path(tempfile.mkdtemp(prefix="archive-"))

    def _pairs(self, argv: list[str]) -> dict[str, str]:
        """Flag -> value, for the flags that take one."""
        return {a: b for a, b in zip(argv, argv[1:], strict=False) if a.startswith("-")}

    def test_segmenter_flags_are_present(self) -> None:
        argv = build_ffmpeg_command(make_settings(self.archive))
        pairs = self._pairs(argv)
        self.assertEqual(pairs["-f"], "segment")
        self.assertEqual(pairs["-strftime"], "1")
        self.assertEqual(pairs["-segment_time"], "60")
        self.assertEqual(pairs["-segment_format"], "mp4")
        # PTS restarts at zero every segment — invariant 2 depends on this being true.
        self.assertEqual(pairs["-reset_timestamps"], "1")

    def test_stream_copy_means_no_re_encode(self) -> None:
        """Invariant 7: the archive stays at native resolution. No scale filter, no
        encoder, no bitrate — if any of these appear the archive has been re-encoded."""
        argv = build_ffmpeg_command(make_settings(self.archive))
        self.assertIn("-c", argv)
        self.assertEqual(argv[argv.index("-c") + 1], "copy")
        for forbidden in ("-vf", "-filter:v", "-s", "-b:v", "-crf", "-c:v"):
            self.assertNotIn(forbidden, argv, f"{forbidden} would re-encode the archive")

    def test_copy_codec_false_drops_the_copy_flag(self) -> None:
        argv = build_ffmpeg_command(make_settings(self.archive, copy_codec=False))
        self.assertNotIn("-c", argv)

    def test_output_is_an_strftime_template_under_the_archive(self) -> None:
        argv = build_ffmpeg_command(make_settings(self.archive))
        output = argv[-1]
        self.assertEqual(Path(output).parent, self.archive)
        # camera_id is substituted by us; the %-escapes are left for ffmpeg -strftime,
        # which is what puts the segment's wall-clock start in its filename.
        self.assertEqual(Path(output).name, "cam01_%Y%m%d_%H%M%S.mp4")

    def test_rtsp_source_forces_tcp_and_does_not_pace(self) -> None:
        argv = build_ffmpeg_command(make_settings(self.archive, source=RTSP))
        self.assertEqual(self._pairs(argv)["-rtsp_transport"], "tcp")
        self.assertNotIn("-re", argv, "-re would throttle a live stream")

    def test_file_source_is_paced_at_native_rate(self) -> None:
        argv = build_ffmpeg_command(make_settings(self.archive, source="/data/demo.mp4"))
        self.assertIn("-re", argv)
        self.assertNotIn("-rtsp_transport", argv)

    def test_realtime_playback_can_be_turned_off(self) -> None:
        settings = make_settings(
            self.archive, source="/data/demo.mp4", realtime_file_playback=False
        )
        self.assertNotIn("-re", build_ffmpeg_command(settings))

    def test_binary_can_be_overridden_for_a_cuda_build(self) -> None:
        argv = build_ffmpeg_command(make_settings(self.archive), ffmpeg_bin="/opt/ff/ffmpeg")
        self.assertEqual(argv[0], "/opt/ff/ffmpeg")

    def test_container_and_extension_must_agree(self) -> None:
        settings = make_settings(self.archive, container="matroska")
        with self.assertRaises(RecorderError) as ctx:
            build_ffmpeg_command(settings)
        self.assertIn("matroska", str(ctx.exception))

    def test_segment_length_must_be_positive(self) -> None:
        with self.assertRaises(RecorderError):
            build_ffmpeg_command(make_settings(self.archive, segment_seconds=0))

    def test_unknown_placeholder_in_filename_pattern_is_explained(self) -> None:
        settings = make_settings(self.archive, filename_pattern="{site}_%H%M%S.mp4")
        with self.assertRaises(RecorderError) as ctx:
            build_ffmpeg_command(settings)
        self.assertIn("camera_id", str(ctx.exception))

    def test_describe_command_quotes_what_a_shell_would_eat(self) -> None:
        rendered = describe_command(["ffmpeg", "-i", "rtsp://a b/c?d=1"])
        self.assertIn("'rtsp://a b/c?d=1'", rendered)

    def test_the_logged_command_carries_no_camera_password(self) -> None:
        """The whole command line is logged at startup, and the source is one of its
        arguments — redacting only the ``source`` field would leak it right beside."""
        argv = build_ffmpeg_command(make_settings(self.archive, source=RTSP))
        self.assertNotIn("secret", describe_command(argv, redact=True))
        # Un-redacted stays verbatim: --dry-run exists to be pasted into a shell.
        self.assertIn("secret", describe_command(argv))


class TestFFmpegResolution(unittest.TestCase):
    def test_missing_binary_is_actionable_not_a_filenotfounderror(self) -> None:
        with self.assertRaises(FFmpegMissingError) as ctx:
            resolve_ffmpeg("ffmpeg-that-does-not-exist")
        message = str(ctx.exception)
        self.assertIn("apt install ffmpeg", message)
        self.assertIn("SPARK_FFMPEG", message)
        self.assertIn("doctor.py", message)

    def test_absolute_path_that_is_not_executable_is_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix="-ffmpeg") as handle:
            with self.assertRaises(FFmpegMissingError):
                resolve_ffmpeg(handle.name)

    def test_a_binary_on_path_resolves(self) -> None:
        # Any executable will do; this asserts the lookup, not ffmpeg's presence.
        self.assertTrue(Path(resolve_ffmpeg("python3")).is_absolute())


# --------------------------------------------------------------------------------------
# Supervision — a recorder that dies quietly at hour 30 loses the demo footage
# --------------------------------------------------------------------------------------


class FakeClock:
    """Monotonic time the test advances by hand. No test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeProcess:
    """Stands in for ``subprocess.Popen``.

    ``runtime`` is how long it stays up before ``wait`` returns; ``stubborn`` means it
    ignores SIGTERM and must be killed, which is the path that has to work when a camera
    wedges ffmpeg.
    """

    def __init__(
        self,
        clock: FakeClock,
        returncode: int = 1,
        runtime: float = 0.0,
        hangs: bool = False,
        stubborn: bool = False,
    ) -> None:
        self.pid = 4242
        self._clock = clock
        self._rc = returncode
        self._runtime = runtime
        self._hangs = hangs
        self._stubborn = stubborn
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is not None:
            return self.returncode
        if self._hangs and not (self.killed or (self.terminated and not self._stubborn)):
            self._clock.now += timeout or 0.0
            raise subprocess.TimeoutExpired("ffmpeg", timeout or 0.0)
        self._clock.now += self._runtime
        self.returncode = -9 if self.killed else (-15 if self.terminated else self._rc)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


@contextlib.contextmanager
def override_settings(values: dict[str, object]) -> Iterator[None]:
    """Temporarily override *effective* settings.

    Patching ``PENDING_SETTINGS`` is not enough. ``setting()`` prefers
    ``config/settings.yaml``, so the moment a key lands there — which was always the
    plan — the fallback becomes dead code and the patch silently stops applying. For
    ``max_consecutive_failures`` that silence is an infinite loop rather than a failed
    assertion, so these tests override the loaded config itself.
    """
    root = config.load()
    missing = object()
    saved: list[tuple[dict, str, object]] = []
    for dotted, value in values.items():
        parts = dotted.split(".")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        saved.append((node, parts[-1], node.get(parts[-1], missing)))
        node[parts[-1]] = value
    try:
        yield
    finally:
        for node, key, old in reversed(saved):
            if old is missing:
                node.pop(key, None)
            else:
                node[key] = old


FAST_SUPERVISION = {
    "recorder.restart.initial_backoff_seconds": 1.0,
    "recorder.restart.max_backoff_seconds": 8.0,
    "recorder.restart.backoff_multiplier": 2.0,
    "recorder.restart.healthy_seconds": 30.0,
    "recorder.restart.max_consecutive_failures": 0,
    "recorder.restart.stall_timeout_seconds": 10.0,
    "recorder.restart.poll_interval_seconds": 1.0,
    "recorder.stop_timeout_seconds": 5.0,
}


class TestBackoff(unittest.TestCase):
    def test_first_restart_is_immediate_ish(self) -> None:
        self.assertEqual(backoff_delay(1, 1.0, 60.0, 2.0), 1.0)

    def test_growth_is_exponential_and_capped(self) -> None:
        delays = [backoff_delay(n, 1.0, 8.0, 2.0) for n in range(1, 7)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0, 8.0, 8.0])

    def test_zeroth_attempt_does_not_go_negative(self) -> None:
        self.assertEqual(backoff_delay(0, 1.5, 60.0, 2.0), 1.5)


class SupervisorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.archive = Path(tempfile.mkdtemp(prefix="archive-"))
        self.clock = FakeClock()
        self.settings = make_settings(self.archive)
        overrides = override_settings(FAST_SUPERVISION)
        overrides.__enter__()
        self.addCleanup(overrides.__exit__, None, None, None)
        self.logger = logging.getLogger("recorder.test")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False

    def supervisor(self, spawn: object) -> RecorderSupervisor:
        return RecorderSupervisor(
            self.settings,
            spawn=spawn,  # type: ignore[arg-type]
            sleep=self.clock.sleep,
            clock=self.clock,
            logger=self.logger,
            # No ffmpeg on this box; the argv is asserted elsewhere.
            resolve_binary=False,
        )


class TestSupervisorRestarts(SupervisorTestCase):
    def test_a_dying_ffmpeg_is_restarted_with_growing_backoff(self) -> None:
        spawned: list[list[str]] = []

        def spawn(argv: list[str]) -> FakeProcess:
            spawned.append(list(argv))
            return FakeProcess(self.clock, returncode=1)

        self.supervisor(spawn).run_forever(max_starts=4)

        self.assertEqual(len(spawned), 4, "recording must not stop because ffmpeg died")
        self.assertEqual(self.clock.slept, [1.0, 2.0, 4.0])
        self.assertIn("-i", spawned[0])

    def test_the_supervisor_spawns_exactly_the_built_command(self) -> None:
        seen: list[list[str]] = []

        def spawn(argv: list[str]) -> FakeProcess:
            seen.append(list(argv))
            return FakeProcess(self.clock, returncode=0)

        self.supervisor(spawn).run_forever(max_starts=1)
        self.assertEqual(seen[0], build_ffmpeg_command(self.settings))

    def test_a_long_healthy_run_resets_the_failure_streak(self) -> None:
        """Hour 29 of clean recording must not inherit a backoff from hour 3."""
        runtimes = iter([0.0, 0.0, 0.0, 120.0, 0.0])

        def spawn(argv: list[str]) -> FakeProcess:
            return FakeProcess(self.clock, returncode=1, runtime=next(runtimes))

        self.supervisor(spawn).run_forever(max_starts=5)
        # 1, 2, 4 while flapping; the 120 s run resets, so the next wait is 1 s again.
        self.assertEqual(self.clock.slept, [1.0, 2.0, 4.0, 1.0])

    def test_giving_up_is_opt_in_and_reports_failure(self) -> None:
        with override_settings({"recorder.restart.max_consecutive_failures": 3}):
            # max_starts bounds the loop so that a regression fails this assertion
            # instead of hanging the whole suite, which is how this bug first showed up.
            status = self.supervisor(
                lambda argv: FakeProcess(self.clock, returncode=1)
            ).run_forever(max_starts=10)
        self.assertEqual(status, 1)

    def test_a_spawn_that_cannot_start_does_not_crash_the_supervisor(self) -> None:
        def spawn(argv: list[str]) -> FakeProcess:
            raise OSError("no such file")

        status = self.supervisor(spawn).run_forever(max_starts=2)
        self.assertEqual(status, 0)
        self.assertEqual(self.clock.slept, [1.0])

    def test_the_archive_directory_is_created(self) -> None:
        nested = self.archive / "deeper"
        self.settings = dataclasses.replace(self.settings, archive_dir=nested)
        self.supervisor(lambda argv: FakeProcess(self.clock)).run_forever(max_starts=1)
        self.assertTrue(nested.is_dir())


class TestSupervisorStall(SupervisorTestCase):
    def test_a_hung_ffmpeg_that_writes_nothing_is_ended(self) -> None:
        """Alive but archiving nothing is the failure that loses the footage, because
        nothing about it looks like a failure from the outside."""
        procs: list[FakeProcess] = []

        def spawn(argv: list[str]) -> FakeProcess:
            proc = FakeProcess(self.clock, hangs=True)
            procs.append(proc)
            return proc

        self.supervisor(spawn).run_forever(max_starts=1)

        self.assertTrue(procs[0].terminated, "a stalled recorder must be ended")
        # Grace before the first segment closes is one segment length plus the timeout.
        self.assertGreaterEqual(self.clock.now - 1000.0, 60.0 + 10.0)

    def test_a_stalled_process_that_ignores_sigterm_is_killed(self) -> None:
        proc = FakeProcess(self.clock, hangs=True, stubborn=True)
        self.supervisor(lambda argv: proc).run_forever(max_starts=1)
        self.assertTrue(proc.killed, "ffmpeg wedged on a dead camera must not survive")

    def test_a_growing_archive_is_not_a_stall(self) -> None:
        """The watchdog watches the archive, not the clock: a process writing segments is
        healthy no matter how long it has been up."""
        proc = FakeProcess(self.clock, hangs=True)
        supervisor = self.supervisor(lambda argv: proc)
        writes = {"n": 0}

        def marker() -> tuple[str, float, int]:
            """The archive grows on every poll, as it would while ffmpeg is recording."""
            writes["n"] += 1
            if writes["n"] > 100:  # long past the 10 s stall timeout
                supervisor.request_stop()
            return (f"cam01_seg{writes['n'] // 5}.mp4", float(writes["n"]), 1024)

        supervisor._archive_marker = marker  # type: ignore[method-assign]
        supervisor.run_forever(max_starts=1)

        # It ran far longer than the stall timeout and was only ever ended by the stop
        # request — a growing archive is never declared dead.
        self.assertGreater(self.clock.now - 1000.0, 10.0)
        self.assertTrue(proc.terminated)
        self.assertFalse(proc.killed)


class TestSupervisorShutdown(SupervisorTestCase):
    def test_stop_request_terminates_ffmpeg_and_ends_the_loop(self) -> None:
        """SIGTERM first: ffmpeg needs it to finalise the open mp4, and a segment with no
        moov atom will not play."""
        procs: list[FakeProcess] = []
        supervisor: list[RecorderSupervisor] = []

        def spawn(argv: list[str]) -> FakeProcess:
            proc = FakeProcess(self.clock, hangs=True)
            procs.append(proc)
            supervisor[0].request_stop()
            return proc

        supervisor.append(self.supervisor(spawn))
        status = supervisor[0].run_forever()

        self.assertEqual(status, 0)
        self.assertEqual(len(procs), 1, "a stopped supervisor must not respawn")
        self.assertTrue(procs[0].terminated)
        self.assertFalse(procs[0].killed, "a polite exit should not need SIGKILL")


# --------------------------------------------------------------------------------------
# Structured logging — we cannot tune what we cannot see
# --------------------------------------------------------------------------------------


class TestStructuredLogging(unittest.TestCase):
    def test_events_are_one_json_object_per_line_with_utc_timestamps(self) -> None:
        record = logging.LogRecord(
            "recorder", logging.INFO, __file__, 1, "recorder.started", None, None
        )
        record.pid = 17
        line = recorder_log.JsonFormatter().format(record)
        payload = json.loads(line)
        self.assertEqual(payload["event"], "recorder.started")
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["pid"], 17)
        self.assertTrue(payload["ts"].endswith("Z"), "timestamps are UTC with a Z suffix")


if __name__ == "__main__":
    unittest.main()


class TestLivePreview(unittest.TestCase):
    """The UI's live view is a SECOND OUTPUT of the recorder's ffmpeg, not a second
    camera reader — v4l2 access to /dev/video0 is exclusive."""

    def setUp(self) -> None:
        self.archive = Path(tempfile.mkdtemp(prefix="archive-"))

    def settings(self, **overrides: object) -> RecorderSettings:
        base = make_settings(
            self.archive,
            source="/dev/video0",
            preview_enabled=True,
            preview_path=self.archive / "live.jpg",
            preview_fps=2.0,
            preview_width=640,
            preview_quality=6,
        )
        return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]

    def test_preview_only_ever_appends_to_the_archive_command(self) -> None:
        """The archive is what the deep worker re-reads (invariant 7) and what SPEC §2.1
        says must keep being written no matter what. A cosmetic preview must not be able
        to change one byte of how it is produced."""
        with_preview = build_ffmpeg_command(self.settings())
        without = build_ffmpeg_command(self.settings(preview_enabled=False))
        self.assertEqual(with_preview[: len(without)], without)
        self.assertGreater(len(with_preview), len(without))

    def test_preview_writes_one_rolling_file(self) -> None:
        """-update 1 rewrites a single file rather than accumulating frames; without it
        a day of preview is 172,800 JPEGs."""
        argv = build_ffmpeg_command(self.settings())
        self.assertEqual(argv[argv.index("-update") + 1], "1")
        self.assertEqual(argv[-1], str(self.archive / "live.jpg"))

    def test_the_preview_scale_never_touches_the_archive(self) -> None:
        """The preview is downscaled; the archive is not. The -vf must therefore land
        AFTER the archive output, applied only to the second -map."""
        argv = build_ffmpeg_command(self.settings())
        self.assertLess(argv.index(str(self.archive / "cam01_%Y%m%d_%H%M%S.mp4")),
                        argv.index("-vf"))
        self.assertIn("scale=640:-2", argv[argv.index("-vf") + 1])

    def test_disabled_preview_adds_nothing(self) -> None:
        argv = build_ffmpeg_command(self.settings(preview_enabled=False))
        for flag in ("-update", "-vf", "-q:v"):
            self.assertNotIn(flag, argv)
