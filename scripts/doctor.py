#!/usr/bin/env python3
"""``make doctor`` — check this box against the CLAUDE.md machine-state table.

Run this first, and run it again whenever something behaves strangely. Every gap in that
table costs an hour when it is found by a container that will not start, and a minute
when it is found here.

Design rules, because this runs on a box where almost nothing is set up:

* **Never crash.** A check that raises reports itself as failed and the run continues.
  A doctor that dies on the first missing binary is a doctor nobody runs twice.
* **Every non-pass prints the exact command that fixes it.** Not a description of the
  fix — the command.
* **Stdlib only.** No third-party import, including PyYAML: the config loader is itself
  one of the things being checked.

Exit code is non-zero if any *hard* prerequisite fails. Warnings do not fail the run;
they are the things that degrade the demo rather than block it.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Run as `python3 scripts/doctor.py`, so sys.path[0] is scripts/. The optional
# `shared.config` import below needs the repo root, and it is optional on purpose —
# config loading is one of the things being checked.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Expectations from the CLAUDE.md machine-state table and SPEC §0. Read from
# settings.yaml when a ``doctor:`` block exists there, so the numbers stay in one place
# once someone adds it; see PENDING_SETTINGS in the report at the end of a run.
EXPECTED: dict[str, object] = {
    "doctor.expected_arch": "aarch64",
    "doctor.expected_compute_capability": "12.1",  # sm_121. Not sm_100, not sm_120.
    "doctor.min_unified_memory_gb": 120.0,  # 128 GB box; SPEC §7.1 budgets against it
    "doctor.min_free_disk_gb": 100.0,  # SPEC §2.1: 43 GB/day of archive
    "doctor.archive_gb_per_day": 43.0,
    "doctor.min_python": [3, 11],
    # Local runners that load models into the same unified memory (invariant 1).
    "doctor.model_runner_names": ["lm-studio", "lmstudio", "unsloth", "ollama", "vllm"],
    "doctor.model_runner_ports": [8888],
}


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class Check:
    """One line of the report."""

    name: str
    status: Status
    detail: str
    fix: str | None = None
    hard: bool = False
    extra: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------------------


def expected(key: str) -> object:
    """Expectation from ``settings.yaml`` if it carries one, else the table above."""
    try:
        from shared import config  # noqa: PLC0415 - optional, it is under test here

        return config.get(key, EXPECTED[key])
    except Exception:
        return EXPECTED[key]


def run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str] | None:
    """Run a command. Returns None if the binary is absent — never raises."""
    if shutil.which(cmd[0]) is None and not os.path.isabs(cmd[0]):
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def read_text(path: str | Path) -> str | None:
    """Read a file, or None for absent/unreadable. /etc and /proc both need this."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def gb(num_bytes: float) -> float:
    return num_bytes / 1024**3


# --------------------------------------------------------------------------------------
# Checks — hardware
# --------------------------------------------------------------------------------------


def check_python() -> Check:
    want = list(expected("doctor.min_python"))  # type: ignore[arg-type]
    have = list(sys.version_info[:2])
    ok = have >= want
    return Check(
        "python",
        Status.PASS if ok else Status.FAIL,
        f"{platform.python_version()} at {sys.executable}",
        fix=None if ok else f"need Python >= {want[0]}.{want[1]}; this is {have[0]}.{have[1]}",
        hard=True,
    )


def check_arch() -> Check:
    want = str(expected("doctor.expected_arch"))
    have = platform.machine()
    ok = have == want
    return Check(
        "arch",
        Status.PASS if ok else Status.FAIL,
        f"{have} ({platform.system()} {platform.release()})",
        fix=None
        if ok
        else f"expected {want}. This build targets ARM64 — wheels and containers "
        f"picked on an x86 box will not run here.",
        hard=True,
    )


def check_gpu() -> Check:
    query = "--query-gpu=name,compute_cap,driver_version"
    out = run(["nvidia-smi", query, "--format=csv,noheader"])
    if out is None:
        return Check(
            "gpu",
            Status.FAIL,
            "nvidia-smi not found",
            fix="install the NVIDIA driver / check the DGX base image: nvidia-smi",
            hard=True,
        )
    rc, stdout, stderr = out
    if rc != 0 or not stdout:
        return Check(
            "gpu",
            Status.FAIL,
            f"nvidia-smi exited {rc}: {stderr or stdout}",
            fix="check the driver: sudo dmesg | grep -i nvidia",
            hard=True,
        )
    rows = [r.strip() for r in stdout.splitlines() if r.strip()]
    return Check("gpu", Status.PASS, "; ".join(rows), hard=True)


def check_compute_capability() -> Check:
    want = str(expected("doctor.expected_compute_capability"))
    out = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if out is None or out[0] != 0 or not out[1]:
        return Check(
            "compute_cap",
            Status.SKIP,
            "no nvidia-smi; cannot read compute capability",
            fix="resolve the gpu check first",
        )
    have = out[1].splitlines()[0].strip()
    if have == want:
        sm = "sm_" + have.replace(".", "")
        return Check("compute_cap", Status.PASS, f"{have} ({sm})", hard=True)
    return Check(
        "compute_cap",
        Status.FAIL,
        f"{have}, expected {want}",
        fix=(
            "this box is not the target hardware. sm_121 is not sm_100 (datacenter "
            "Blackwell) and not sm_120 (RTX 50xx); prebuilt wheels for either will not "
            "load here."
        ),
        hard=True,
    )


def check_cuda() -> Check:
    extra: list[str] = []
    smi = run(["nvidia-smi"])
    if smi and smi[0] == 0:
        match = re.search(r"CUDA Version:\s*([\d.]+)", smi[1])
        if match:
            extra.append(f"driver CUDA {match.group(1)}")
    nvcc = run(["nvcc", "--version"]) or run(["/usr/local/cuda/bin/nvcc", "--version"])
    if nvcc and nvcc[0] == 0:
        match = re.search(r"release ([\d.]+), V([\d.]+)", nvcc[1])
        extra.append(f"nvcc {match.group(2)}" if match else "nvcc present")
    if not extra:
        return Check(
            "cuda",
            Status.WARN,
            "no CUDA toolkit found",
            fix="only needed to build things; the runtime lives in the driver. "
            "PATH=$PATH:/usr/local/cuda/bin if it is installed but unlisted.",
        )
    return Check("cuda", Status.PASS, ", ".join(extra))


def check_memory() -> Check:
    """Unified memory. ``nvidia-smi`` reports [N/A] for memory.total on GB10 — there is
    no separate VRAM to report, which is the whole premise of invariant 1."""
    meminfo = read_text("/proc/meminfo")
    if not meminfo:
        return Check("memory", Status.SKIP, "cannot read /proc/meminfo")
    values: dict[str, float] = {}
    for line in meminfo.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in {"MemTotal", "MemAvailable"}:
            values[parts[0].rstrip(":")] = float(parts[1]) / 1024**2  # kB -> GiB
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", 0.0)
    floor = float(expected("doctor.min_unified_memory_gb"))  # type: ignore[arg-type]
    ok = total >= floor
    return Check(
        "memory",
        Status.PASS if ok else Status.FAIL,
        f"{total:.1f} GiB unified, {available:.1f} GiB available",
        fix=None if ok else f"expected >= {floor:.0f} GiB (SPEC §7.1 budgets against 128)",
        hard=True,
    )


def check_disk() -> Check:
    archive = REPO_ROOT / "data" / "archive"
    try:
        from shared import config  # noqa: PLC0415

        archive = config.repo_path("paths.archive")
    except Exception:
        pass
    probe = archive
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Check("disk", Status.WARN, f"cannot stat {probe}: {exc}")
    free = gb(usage.free)
    floor = float(expected("doctor.min_free_disk_gb"))  # type: ignore[arg-type]
    per_day = float(expected("doctor.archive_gb_per_day"))  # type: ignore[arg-type]
    ok = free >= floor
    return Check(
        "disk",
        Status.PASS if ok else Status.FAIL,
        f"{free:.0f} GiB free at {probe} — {free / per_day:.0f} days of archive "
        f"at {per_day:.0f} GB/day",
        fix=None if ok else f"free up space or repoint paths.archive; want >= {floor:.0f} GiB",
        hard=True,
    )


# --------------------------------------------------------------------------------------
# Checks — containers
# --------------------------------------------------------------------------------------


def check_docker() -> Check:
    if shutil.which("docker") is None:
        return Check(
            "docker",
            Status.FAIL,
            "docker CLI not found",
            fix="curl -fsSL https://get.docker.com | sh",
            hard=True,
        )
    out = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=20.0)
    if out and out[0] == 0:
        return Check("docker", Status.PASS, f"daemon reachable, server {out[1]}", hard=True)

    stderr = (out[2] if out else "") or (out[1] if out else "")
    in_group = "docker" in _my_groups()
    daemon_up = (run(["systemctl", "is-active", "docker"]) or (1, "", ""))[1] == "active"

    if "permission denied" in stderr.lower() or not in_group:
        return Check(
            "docker",
            Status.FAIL,
            "daemon is running but this user cannot talk to it (not in the docker group)",
            fix="sudo usermod -aG docker $USER   # then log out and back in, or: newgrp docker",
            hard=True,
            extra=[f"groups: {' '.join(_my_groups())}", f"daemon active: {daemon_up}"],
        )
    return Check(
        "docker",
        Status.FAIL,
        f"docker info failed: {stderr.splitlines()[0] if stderr else 'unknown error'}",
        fix="sudo systemctl start docker",
        hard=True,
    )


def _my_groups() -> list[str]:
    try:
        import grp  # noqa: PLC0415 - POSIX only

        gids = os.getgroups()
        return sorted({grp.getgrgid(g).gr_name for g in gids})
    except Exception:
        return []


def check_nvidia_runtime() -> Check:
    path = Path("/etc/docker/daemon.json")
    fix = "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    have_ctk = shutil.which("nvidia-ctk") is not None
    extra = [] if have_ctk else ["nvidia-ctk absent — install nvidia-container-toolkit first"]
    raw = read_text(path)
    if raw is None:
        return Check(
            "nvidia_runtime",
            Status.FAIL,
            f"{path} absent — the nvidia runtime is not registered with docker",
            fix=fix,
            hard=True,
            extra=extra,
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return Check(
            "nvidia_runtime",
            Status.FAIL,
            f"{path} is not valid JSON: {exc}",
            fix=f"fix the file, then: {fix}",
            hard=True,
        )
    runtimes = data.get("runtimes") or {}
    if "nvidia" in runtimes:
        default = data.get("default-runtime", "runc")
        return Check(
            "nvidia_runtime",
            Status.PASS,
            f"nvidia runtime registered (default-runtime: {default})",
            hard=True,
        )
    return Check(
        "nvidia_runtime",
        Status.FAIL,
        f"{path} exists but declares no nvidia runtime",
        fix=fix,
        hard=True,
        extra=extra,
    )


def check_ngc() -> Check:
    """NGC is the external dependency — Cosmos, the Nemotron NIMs, embed, rerank and
    DeepStream all come from nvcr.io. CLAUDE.md flags it as the one to start early."""
    found: list[str] = []
    ngc_dir = Path.home() / ".ngc"
    if ngc_dir.exists():
        found.append(str(ngc_dir))
    docker_cfg = read_text(Path.home() / ".docker" / "config.json")
    if docker_cfg:
        try:
            auths = (json.loads(docker_cfg).get("auths") or {}).keys()
        except json.JSONDecodeError:
            auths = []
        if any("nvcr.io" in a for a in auths):
            found.append("~/.docker/config.json (nvcr.io)")
    if found:
        detail = "credentials at " + ", ".join(found)
        return Check("ngc_credentials", Status.PASS, detail, hard=True)
    return Check(
        "ngc_credentials",
        Status.FAIL,
        "no ~/.ngc and no nvcr.io entry in ~/.docker/config.json — nvcr.io will 401",
        fix=(
            "get a key at https://ngc.nvidia.com/setup/api-key, then:\n"
            "        docker login nvcr.io -u '$oauthtoken' -p <API_KEY>"
        ),
        hard=True,
        extra=[
            "external dependency — start this early; it gates Cosmos, the Nemotron NIMs, "
            "embed, rerank and DeepStream",
            "HuggingFace and PyPI are reachable, so Cosmos weights may be pullable "
            "without NGC; the NIM containers are not",
        ],
    )


# --------------------------------------------------------------------------------------
# Checks — media
# --------------------------------------------------------------------------------------


def check_ffmpeg() -> list[Check]:
    binary = os.environ.get("SPARK_FFMPEG", "ffmpeg")
    out = run([binary, "-version"])
    if out is None or out[0] != 0:
        return [
            Check(
                "ffmpeg",
                Status.FAIL,
                f"{binary} not found — the recorder (SPEC §2.1) cannot run",
                fix="sudo apt install ffmpeg",
                hard=True,
                extra=[
                    "the apt candidate (6.1.1) advertises no NVDEC; fine for the recorder, "
                    "which stream-copies, but SPEC §2.4 GPU decode wants a CUDA build",
                ],
            )
        ]
    version = out[1].splitlines()[0] if out[1] else "unknown"
    checks = [Check("ffmpeg", Status.PASS, version, hard=True)]

    accels = run([binary, "-hide_banner", "-hwaccels"])
    names = set()
    if accels and accels[0] == 0:
        names = {line.strip() for line in accels[1].splitlines()[1:] if line.strip()}
    accelerated = names & {"cuda", "nvdec"}
    if accelerated:
        checks.append(
            Check("ffmpeg_nvdec", Status.PASS, "hwaccels: " + ", ".join(sorted(accelerated)))
        )
    else:
        checks.append(
            Check(
                "ffmpeg_nvdec",
                Status.WARN,
                f"no cuda/nvdec hwaccel (has: {', '.join(sorted(names)) or 'none'})",
                fix=(
                    "not needed by the recorder — it stream-copies and never decodes. "
                    "SPEC §2.4 wants NVDEC for the ingest path; a CUDA-enabled build or "
                    "DeepStream covers it."
                ),
            )
        )
    return checks


# --------------------------------------------------------------------------------------
# Checks — memory budget (invariant 1)
# --------------------------------------------------------------------------------------


def _process_names(pid: str) -> tuple[list[str], float]:
    """Identifying name fragments for a process, plus its RSS in GiB.

    Deliberately not the whole command line: a browser tab open at lmstudio.ai contains
    the string "lmstudio" and is not a model runner, and one false alarm trains people to
    ignore the check. What we use instead is the *path components of argv[0]* — which
    catches the workers a runner forks under its own install directory, where argv[0] is
    a plain "python" — plus the first couple of argument basenames and comm.
    """
    base = Path("/proc") / pid
    raw = read_text(base / "cmdline") or ""
    argv = [a for a in raw.split("\0") if a]
    names: list[str] = []
    if argv:
        names += [part.lower() for part in Path(argv[0]).parts]
    names += [Path(a).name.lower() for a in argv[1:3]]
    comm = (read_text(base / "comm") or "").strip().lower()
    if comm:
        names.append(comm)
    rss = 0.0
    status = read_text(base / "status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                rss = float(parts[1]) / 1024**2
            break
    return names, rss


def check_model_runners() -> Check:
    names_wanted: list[str] = list(expected("doctor.model_runner_names"))  # type: ignore[arg-type]
    wanted = [str(n).lower() for n in names_wanted]
    ports = [int(p) for p in expected("doctor.model_runner_ports")]  # type: ignore[arg-type]
    hits: dict[str, float] = {}
    try:
        pids = [p.name for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        pids = []
    for pid in pids:
        try:
            names, rss = _process_names(pid)
        except Exception:
            continue
        for want in wanted:
            if any(want in n for n in names):
                hits[want] = hits.get(want, 0.0) + rss
                break

    listening = [p for p in ports if _port_open(p)]
    if not hits and not listening:
        return Check("model_runners", Status.PASS, "no local model runner holding memory")

    detail_parts = [f"{name} ({rss:.1f} GiB RSS)" for name, rss in sorted(hits.items())]
    if listening:
        detail_parts.append("listening on " + ", ".join(str(p) for p in listening))
    return Check(
        "model_runners",
        Status.WARN,
        "; ".join(detail_parts),
        fix=(
            "quit LM Studio, and: pkill -f 'unsloth studio'\n"
            "        one VLM process, ever — 128 GB is shared between CPU and GPU and a "
            "second model instance OOMs the box (invariant 1, SPEC §7.1)"
        ),
    )


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------------------
# Checks — repo
# --------------------------------------------------------------------------------------


def check_config() -> Check:
    """settings.yaml must load, and its UNRESOLVED nulls are worth naming out loud."""
    try:
        from shared import config  # noqa: PLC0415
    except Exception as exc:
        return Check(
            "config",
            Status.FAIL,
            f"cannot import shared.config: {exc}",
            fix="pip install --user pyyaml   # PyYAML is the one runtime dependency",
            hard=True,
        )
    try:
        config.load()
    except Exception as exc:
        return Check("config", Status.FAIL, f"settings.yaml did not load: {exc}", hard=True)

    unresolved = [
        (key, ref)
        for key, ref in (
            ("recorder.source", "D2 live camera vs pre-ingested recording"),
            ("vlm.model", "D1 which Cosmos 3 variant"),
            ("agent.model", "D3 Nemotron 3 Nano vs 3.5 Lightning"),
        )
        if config.get(key, None) is None
    ]
    if not unresolved:
        return Check("config", Status.PASS, "settings.yaml loads, no unresolved values", hard=True)
    return Check(
        "config",
        Status.WARN,
        f"{len(unresolved)} setting(s) still null — SPEC §10 is open",
        fix="decide, then set them in config/settings.yaml",
        extra=[f"{key} — SPEC §10 {ref}" for key, ref in unresolved],
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

CHECKS: list[Callable[[], Check | list[Check]]] = [
    check_python,
    check_arch,
    check_gpu,
    check_compute_capability,
    check_cuda,
    check_memory,
    check_disk,
    check_docker,
    check_nvidia_runtime,
    check_ngc,
    check_ffmpeg,
    check_model_runners,
    check_config,
]

_COLOR = {
    Status.PASS: "\033[32m",
    Status.WARN: "\033[33m",
    Status.FAIL: "\033[31m",
    Status.SKIP: "\033[90m",
}


def _paint(status: Status, use_color: bool) -> str:
    label = f"[ {status.value} ]"
    return f"{_COLOR[status]}{label}\033[0m" if use_color else label


def collect() -> list[Check]:
    """Run every check. A check that raises becomes a failed check, not a traceback."""
    results: list[Check] = []
    for fn in CHECKS:
        try:
            produced: Check | list[Check] = fn()
        except Exception as exc:  # noqa: BLE001 - a doctor that dies is useless
            results.append(
                Check(
                    fn.__name__.removeprefix("check_"),
                    Status.FAIL,
                    f"check raised {type(exc).__name__}: {exc}",
                    fix="this is a bug in scripts/doctor.py",
                )
            )
            continue
        results.extend(produced if isinstance(produced, list) else [produced])
    return results


def render(results: Iterable[Check], use_color: bool = False) -> str:
    results = list(results)
    width = max((len(r.name) for r in results), default=10)
    lines = ["", "  DGX Spark preflight — CLAUDE.md machine state", ""]
    for r in results:
        lines.append(f"  {_paint(r.status, use_color)}  {r.name.ljust(width)}  {r.detail}")
        for note in r.extra:
            lines.append(f"{' ' * (width + 14)}· {note}")

    blocked = [r for r in results if r.status is Status.FAIL and r.hard]
    todo = [r for r in results if r.status is not Status.PASS and r.fix]
    if todo:
        lines += ["", "  Fixes, in the order they bite:", ""]
        for r in todo:
            mark = "!" if (r.status is Status.FAIL and r.hard) else "-"
            lines.append(f"  {mark} {r.name}")
            lines.append(f"        {r.fix}")
    counts = {s: sum(1 for r in results if r.status is s) for s in Status}
    lines += [
        "",
        f"  {counts[Status.PASS]} pass, {counts[Status.WARN]} warn, "
        f"{counts[Status.FAIL]} fail, {counts[Status.SKIP]} skipped",
    ]
    if blocked:
        lines.append(f"  BLOCKED on: {', '.join(r.name for r in blocked)}")
    else:
        lines.append("  No hard prerequisite is missing.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    results = collect()
    if "--json" in args:
        print(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "status": r.status.value,
                        "detail": r.detail,
                        "fix": r.fix,
                        "hard": r.hard,
                        "extra": r.extra,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        print(render(results, use_color=use_color))
    return 1 if any(r.status is Status.FAIL and r.hard for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
