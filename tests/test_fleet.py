"""🌐 Fleet contracts through the parent's shared checks (vendor/fleet_check.py): vendor lock, parent pins, CI setup,
Node entries through the vendored py_launch.cjs, launcher contract.

共通の検査（lock のバイト・親の参照・ジョブの切り出し・Node と Python の一致・ランチャーの契約）は親の fleet_check に任せ、
ここにはこのリポジトリの期待値（ピン・ツール・ケース・文言）と、fleet_check にまだ無い検査だけを残す。

Node entry: `node vendor/py_launch.cjs --env-var <TOOL>_PYTHON -- <core|npb>/<tool>.py ARGS` (no per-tool wrapper).
"""
from __future__ import annotations

import importlib.util, json, os, re, select, shutil, signal, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Aoi は子を全部 1 回の pytest で回す。トップの `vendor` パッケージは子どうしでぶつかるので、ファイルから固有の名前で読む
_spec = importlib.util.spec_from_file_location("pythdragoras_fleet_check", ROOT / "vendor/fleet_check.py")
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)

PARENT_WORKFLOW_PIN = "9e3da7ae1846e9604ebc24bb5178f7de094a4e77"
PARENT_REPOSITORY_ID = "638801160"
LIVE_PARENT_REF = "${{ steps.parent.outputs.sha }}"
ACCEPTED = {
    "vendor/py_launch.cjs": "9e0147a17695ce63fa73daf75d8195a4f49095bb",
    "vendor/uv_tool_build.py": "f15063440ee7f5f869cef1546ae651fa32a07505",
    "vendor/fleet_check.py": "6c29503cf17eda856766ddfdf4d66a5b0965a8b1",
}
# Node 入口: ツール名 -> 本体（トップにはメインだけ。ツールは core/ と npb/ にいる）。環境変数は <TOOL>_PYTHON
SCRIPTS = {
    "littlebit_survey": "core/littlebit_survey.py",
    "score_sensitivity": "npb/score_sensitivity.py",
    "season_sensitivity": "npb/season_sensitivity.py",
}
CASES = [{"id": "help", "args": ["--help"], "rc": 0}, {"id": "bogus", "args": ["--bogus"], "rc": 2}]
NODE = fleet.node()
needs_node = pytest.mark.skipif(NODE is None, reason="Node 26+ not available (set NODE)")
every_tool = pytest.mark.parametrize("tool", list(SCRIPTS))


def _env_var(tool: str) -> str:
    return f"{tool.upper()}_PYTHON"


# ==== vendor lock: placed bytes and accepted commits (adopted vendor locks are deliberately independent of the parent pin)

def test_vendor_lock_matches_checked_in_bytes_and_accepted_commits():
    rows = fleet.check_lock(ROOT, commits=ACCEPTED)
    launcher = rows["vendor/py_launch.cjs"]
    assert (launcher["repository"], launcher["source"]) == ("myon-bioinformatics/Aoi", "vendor/py_launch.cjs")
    shared = rows["vendor/fleet_check.py"]
    assert (shared["repository"], shared["source"]) == (fleet.PARENT, "fleet_check.py")


# ==== parent pins: local rollout contract

def test_parent_workflow_references_use_reviewed_rollout_commit():
    # Mutable and shortened references are returned too, rather than ignored.
    refs = fleet.parent_refs(ROOT)
    assert refs.count(LIVE_PARENT_REF) == 1, refs
    assert {ref for ref in refs if ref != LIVE_PARENT_REF} == {PARENT_WORKFLOW_PIN}, refs


def test_vendor_dispatch_verifies_lock_on_resolved_parent_main():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "git ls-remote" in text
    assert PARENT_REPOSITORY_ID in text
    assert "pilot/vendor_dispatch.py" in text
    assert "--manifest vendor.lock.json" in text
    assert "--record" not in text
    assert 'node-version: "26"' in text
    assert 'python-version: "3.15"' in text
    assert "actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1 # v7.1.0" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" in text
    for line in text.splitlines():
        if "vendor_dispatch.py" in line:
            assert "-S" not in line
            assert "--record" not in line


def test_vendor_dispatch_node_floor_matches_setup_node():
    """Execute the defensive guard at the Node 26 boundary."""
    job = fleet.workflow_job(ROOT, "vendor-dispatch")
    guard = re.search(r'if \[\[ "\$node_major" -lt \d+ \]\]; then\n.*?\n          fi', job, re.S)
    assert guard is not None
    assert 'node-version: "26"' in job
    for major, expected in ((24, 1), (25, 1), (26, 0), (27, 0)):
        result = subprocess.run(
            ["bash", "-c", f"node_major={major}\n" + guard.group()],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == expected, (major, result.stdout, result.stderr)


# ==== CI setup: offline guards for the CI interpreter preparation order

def test_python_setup_precedes_both_metadata_resolvers():
    for name, version in (("test", "${{ matrix.python }}"), ("vendor-dispatch", "3.15")):
        job = fleet.workflow_job(ROOT, name)
        setup = job.index("uses: astral-sh/setup-uv@")
        python = re.search(r"\bpython3?\s+(?:-c|-\s|[A-Za-z_])", job)
        assert python is not None
        assert setup < python.start(), name
        preparation = job[setup:python.start()]
        assert 'python-version: "' + version + '"' in preparation
        assert "activate-environment: true" in preparation
        assert job.count("uses: astral-sh/setup-uv@") == 1


def test_ci_requires_node_for_node_entry_parity():
    # The test job must set up Node before running its pytest suite.
    job = fleet.workflow_job(ROOT, "test")
    assert "actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1" in job
    assert "node-version: '26'" in job
    assert "Require supported Node in parity CI" in job
    if os.environ.get("GITHUB_ACTIONS") == "true":
        assert NODE is not None, "Node 26+ is required for the Node parity suite"


# ==== Node entry: parity with each tool's .py, and the launcher contract under each tool's <TOOL>_PYTHON

@every_tool
def test_no_per_tool_wrapper_and_readme_documents_the_direct_command(tool):
    assert not (ROOT / f"{tool}.cjs").exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"node vendor/py_launch.cjs --env-var {_env_var(tool)} -- {SCRIPTS[tool]}" in readme
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert pkg["engines"]["node"] == ">=26"


@needs_node
@pytest.mark.parametrize("args,expected_rc", [(c["args"], c["rc"]) for c in CASES],
                         ids=[c["id"] for c in CASES])
@every_tool
def test_node_matches_python(tool, args, expected_rc):
    via_node, via_python = fleet.node_parity(ROOT, SCRIPTS[tool], _env_var(tool), *args)
    assert via_python[0] == expected_rc
    assert via_node == via_python


@needs_node
@every_tool
def test_vendored_launcher_keeps_its_contract(tool):
    """Missing interpreter, argv/stdin/exit passthrough under -S, -S during discovery, resolvePython flags, SIGINT then SIGTERM."""
    fleet.launcher_contract(ROOT, _env_var(tool))


# ==== launcher regressions not in fleet_check yet (candidates for the parent): real processes, launcher inherited only from Aoi

def _launcher_env(tmp_path: Path, tool: str) -> dict[str, str]:
    (tmp_path / "vendor").mkdir()
    shutil.copyfile(ROOT / "vendor/py_launch.cjs", tmp_path / "vendor/py_launch.cjs")
    env = {**os.environ, _env_var(tool): sys.executable}
    env.pop("PY_LAUNCH_ERROR_FORMAT", None)
    return env


@needs_node
@pytest.mark.parametrize("mode", ["run", "main"])
@pytest.mark.parametrize("flags", [{}, {"noSite": True}, {"isolated": True}, {"noSite": True, "isolated": True}],
                         ids=["default", "noSite", "isolated", "both"])
def test_library_discovery_flags_are_independent(tmp_path, mode, flags):
    """`run` and `main` against a hostile site hook (fleet.launcher_contract covers `resolvePython`)."""
    env = dict(os.environ)
    env.pop("PY_LAUNCH_ERROR_FORMAT", None)
    marker = tmp_path / "site-hook-ran"
    (tmp_path / "sitecustomize.py").write_text(
        "import os\n"
        "with open(os.environ['PY_LAUNCH_SITE_MARKER'], 'w') as out:\n"
        "    out.write('ran')\n"
        "raise SystemExit(42)\n", encoding="utf-8",
    )
    env.update(PYTHONPATH=str(tmp_path), PY_LAUNCH_SITE_MARKER=str(marker))
    script = tmp_path / "echo.py"
    script.write_text("print('safe')\n", encoding="utf-8")
    options = json.dumps({"python": sys.executable, **flags})
    script_arg = json.dumps(str(script))
    calls = {
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


@needs_node
@pytest.mark.skipif(os.name == "nt", reason="requires POSIX signals and process groups")
@every_tool
@pytest.mark.parametrize("repetitions", [0, 2], ids=["direct", "repeated"])
@pytest.mark.parametrize("error_format", ["text", "json"])
def test_every_entry_forwards_acknowledged_sigterm_then_sigint(tmp_path, tool, repetitions, error_format):
    """The reverse order of fleet.launcher_contract (acknowledged SIGINT, terminal SIGTERM), with the exact error line."""
    ignored, terminal = "SIGTERM", "SIGINT"
    env = _launcher_env(tmp_path, tool)
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
        [NODE, str(tmp_path / "vendor/py_launch.cjs"), "--env-var", _env_var(tool), "--", str(tmp_path / f"{tool}.py")],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
