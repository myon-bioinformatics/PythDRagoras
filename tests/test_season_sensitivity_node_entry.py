"""Node entry parity with season_sensitivity.py, calling the vendored shared py_launch.cjs directly (no per-tool wrapper)."""
from __future__ import annotations

import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = "season_sensitivity"
SCRIPT = "npb/season_sensitivity.py"  # トップにはメインだけ。ツールは npb/ にいる
ENV_VAR = "SEASON_SENSITIVITY_PYTHON"
CASES = [{"id": "help", "args": ["--help"], "rc": 0}, {"id": "bogus", "args": ["--bogus"], "rc": 2}]


def _node() -> str | None:
    node = os.environ.get("NODE") or shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.match(r"v(\d+)", out.strip())
    return node if match and int(match.group(1)) >= 26 else None


NODE = _node()
needs_node = pytest.mark.skipif(NODE is None, reason="Node 26+ not available (set NODE)")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env[ENV_VAR] = sys.executable
    env.pop("PY_LAUNCH_ERROR_FORMAT", None)
    return env


def _run(argv: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(argv, cwd=cwd, env=_env(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    return p.returncode, p.stdout, p.stderr


def _direct(root: Path, *args: str) -> list[str]:
    """The documented command: node vendor/py_launch.cjs --env-var SEASON_SENSITIVITY_PYTHON -- season_sensitivity.py ARGS."""
    return [NODE, str(root / "vendor" / "py_launch.cjs"), "--env-var", ENV_VAR, "--",
            str(root / SCRIPT), *args]


def _parity(root: Path, args: list[str]) -> tuple[tuple[int, str, str], tuple[int, str, str]]:
    via_node = _run(_direct(root, *args), root)
    via_python = _run([sys.executable, "-S", str(ROOT / SCRIPT), *args], root)
    return via_node, via_python


def test_vendored_launcher_matches_lock():
    lock = json.loads((ROOT / "vendor.lock.json").read_text(encoding="utf-8"))
    entry = next(f for f in lock["files"] if f["destination"] == "vendor/py_launch.cjs")
    assert entry["repository"] == "myon-bioinformatics/Aoi"
    assert entry["commit"] == entry["ref"] == "9e0147a17695ce63fa73daf75d8195a4f49095bb"
    assert entry["source"] == "vendor/py_launch.cjs"
    data = (ROOT / "vendor" / "py_launch.cjs").read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == entry["blob_sha"]


def test_no_per_tool_wrapper_and_readme_documents_the_direct_command():
    assert not (ROOT / f"{TOOL}.cjs").exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"node vendor/py_launch.cjs --env-var {ENV_VAR} -- {SCRIPT}" in readme
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert pkg["engines"]["node"] == ">=26"


@needs_node
@pytest.mark.parametrize("args,expected_rc", [(c["args"], c["rc"]) for c in CASES],
                         ids=[c["id"] for c in CASES])
def test_node_matches_python(args, expected_rc):
    via_node, via_python = _parity(ROOT, args)
    assert via_python[0] == expected_rc
    assert via_node == via_python


@needs_node
def test_missing_interpreter_is_launcher_failure(tmp_path):
    env = _env()
    env[ENV_VAR] = str(tmp_path / "no-such-python")
    p = subprocess.run(_direct(ROOT, "--help"), env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert p.returncode == 2
    assert p.stdout == ""


@needs_node
@pytest.mark.parametrize("code", [0, 1, 3, 7])
def test_entry_forwards_stdin_argv_and_exit_code(tmp_path, code):
    (tmp_path / "vendor").mkdir()
    shutil.copy(ROOT / "vendor" / "py_launch.cjs", tmp_path / "vendor" / "py_launch.cjs")
    (tmp_path / SCRIPT).parent.mkdir(exist_ok=True)
    (tmp_path / SCRIPT).write_text(
        "import json, sys\n"
        "sys.stdout.write(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read(),"
        " 'nosite': sys.flags.no_site}))\n"
        f"sys.exit({code})\n", encoding="utf-8")
    p = subprocess.run(_direct(tmp_path, "a b", "--x=ü"), env=_env(),
                       input="こんにちは\n", capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert p.returncode == code
    assert json.loads(p.stdout) == {"argv": ["a b", "--x=ü"], "stdin": "こんにちは\n", "nosite": 1}
