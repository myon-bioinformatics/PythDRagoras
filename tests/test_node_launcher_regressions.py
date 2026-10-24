"""Real-process regressions for the launcher inherited only from Aoi."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ["littlebit_survey", "season_sensitivity", "score_sensitivity"]


def _node() -> str | None:
    node = os.environ.get("NODE") or shutil.which("node")
    if node:
        try:
            result = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.match(r"v(\d+)", result.stdout.strip())
        if result.returncode == 0 and match and int(match[1]) >= 24:
            return node
    return None


NODE = _node()
pytestmark = pytest.mark.skipif(NODE is None, reason="Node 24+ not available (set NODE)")


def _copy_entry(tmp_path: Path, tool: str) -> dict[str, str]:
    (tmp_path / "vendor").mkdir()
    shutil.copyfile(ROOT / "vendor/py_launch.cjs", tmp_path / "vendor/py_launch.cjs")
    shutil.copyfile(ROOT / f"{tool}.cjs", tmp_path / f"{tool}.cjs")
    env = dict(os.environ)
    env[f"{tool.upper()}_PYTHON"] = sys.executable
    env.pop("PY_LAUNCH_ERROR_FORMAT", None)
    return env


def _site_hook(tmp_path: Path, env: dict[str, str]) -> Path:
    marker = tmp_path / "site-hook-ran"
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "with open(os.environ['PY_LAUNCH_SITE_MARKER'], 'w') as out:\n"
        "    out.write('ran')\n"
        "raise SystemExit(42)\n", encoding="utf-8",
    )
    env.update(PYTHONPATH=str(tmp_path), PY_LAUNCH_SITE_MARKER=str(marker))
    return marker


@pytest.mark.parametrize("tool", ENTRIES)
def test_every_entry_suppresses_site_during_discovery(tmp_path, tool):
    env = _copy_entry(tmp_path, tool)
    marker = _site_hook(tmp_path, env)
    (tmp_path / f"{tool}.py").write_text(
        "import sys\nprint(sys.flags.no_site)\n", encoding="utf-8",
    )
    result = subprocess.run(
        [NODE, str(tmp_path / f"{tool}.cjs")], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=10,
    )
    assert not marker.exists(), "interpreter discovery must also use -S"
    assert (result.returncode, result.stdout, result.stderr) == (0, "1\n", "")


@pytest.mark.parametrize("mode", ["resolve", "run", "main"])
@pytest.mark.parametrize("flags", [{}, {"noSite": True}, {"isolated": True}, {"noSite": True, "isolated": True}],
                         ids=["default", "noSite", "isolated", "both"])
def test_library_discovery_flags_are_independent(tmp_path, mode, flags):
    env = dict(os.environ)
    env.pop("PY_LAUNCH_ERROR_FORMAT", None)
    marker = _site_hook(tmp_path, env)
    script = tmp_path / "echo.py"
    script.write_text("print('safe')\n", encoding="utf-8")
    options = json.dumps({"python": sys.executable, **flags})
    script_arg = json.dumps(str(script))
    calls = {
        "resolve": f"launcher.resolvePython({options}); console.log('safe');",
        "run": f"process.stdout.write(launcher.run({script_arg}, [], {options}).stdout);",
        "main": f"launcher.main({script_arg}, {options});",
    }
    program = f"const launcher = require({json.dumps(str(ROOT / 'vendor/py_launch.cjs'))}); {calls[mode]}"
    result = subprocess.run([NODE, "-e", program], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    if not flags:
        assert marker.exists(), "control must prove the hostile site hook is reachable"
        assert result.returncode != 0
        assert "not suitable" in result.stderr
    else:
        assert not marker.exists(), "startup flags must also protect the version probe"
        assert (result.returncode, result.stdout, result.stderr) == (0, "safe\n", "")


def _expect_line(process: subprocess.Popen, expected: bytes) -> None:
    ready, _, _ = select.select([process.stdout], [], [], 5)
    assert ready, f"timed out waiting for child acknowledgement: {expected!r}"
    actual = process.stdout.readline()
    assert actual == expected, (actual, expected, process.poll())


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX signals and process groups")
@pytest.mark.parametrize("tool", ENTRIES)
@pytest.mark.parametrize("ignored,terminal", [("SIGINT", "SIGTERM"), ("SIGTERM", "SIGINT")])
@pytest.mark.parametrize("repetitions", [0, 2], ids=["direct", "repeated"])
@pytest.mark.parametrize("error_format", ["text", "json"])
def test_every_entry_forwards_acknowledged_signals(tmp_path, tool, ignored, terminal, repetitions, error_format):
    env = _copy_entry(tmp_path, tool)
    env["PY_LAUNCH_ERROR_FORMAT"] = error_format
    (tmp_path / f"{tool}.py").write_text(
        "import signal, time\n"
        "def acknowledge(_signal, _frame):\n"
        f"    print('IGNORED {ignored}', flush=True)\n"
        f"signal.signal(signal.{ignored}, acknowledge)\n"
        f"signal.signal(signal.{terminal}, signal.SIG_DFL)\n"
        "print('READY', flush=True)\n"
        "while True:\n"
        "    time.sleep(60)\n", encoding="utf-8",
    )
    process = subprocess.Popen(
        [NODE, str(tmp_path / f"{tool}.cjs")], cwd=tmp_path, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _expect_line(process, b"READY\n")
        for _ in range(repetitions):
            # Only signal the launcher PID, never the Python child or its group.
            process.send_signal(getattr(signal, ignored))
            _expect_line(process, f"IGNORED {ignored}\n".encode())
            assert process.poll() is None
        process.send_signal(getattr(signal, terminal))
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 2
        assert stdout == b""
        assert stderr.count(b"\n") == 1
        if error_format == "json":
            payload = json.loads(stderr)
            assert payload["ok"] is False
            assert payload["code"] == "ESIGNAL"
            assert payload["exit_code"] == 2
            assert payload["message"].endswith(f"terminated by signal {terminal}")
        else:
            assert stderr.decode().strip().endswith(f"terminated by signal {terminal}")
    finally:
        # Also reap an orphaned Python process when testing a broken launcher.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
