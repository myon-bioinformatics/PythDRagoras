"""score_sensitivity.cjs: Node entry parity with score_sensitivity.py through the vendored shared py_launch.cjs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = "score_sensitivity"
ENV_VAR = "SCORE_SENSITIVITY_PYTHON"
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
    return node if match and int(match.group(1)) >= 24 else None


NODE = _node()
needs_node = pytest.mark.skipif(NODE is None, reason="Node 24+ not available (set NODE)")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env[ENV_VAR] = sys.executable
    env.pop("PY_LAUNCH_ERROR_FORMAT", None)
    return env


def _run(argv: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(argv, cwd=cwd, env=_env(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    return p.returncode, p.stdout, p.stderr


def _parity(root: Path, args: list[str]) -> tuple[tuple[int, str, str], tuple[int, str, str]]:
    via_node = _run([NODE, str(root / f"{TOOL}.cjs"), *args], root)
    via_python = _run([sys.executable, "-S", str(ROOT / f"{TOOL}.py"), *args], root)
    return via_node, via_python


def test_vendored_launcher_matches_lock():
    lock = json.loads((ROOT / "vendor.lock.json").read_text(encoding="utf-8"))
    entry = next(f for f in lock["files"] if f["destination"] == "vendor/py_launch.cjs")
    assert entry["repository"] == "myon-bioinformatics/Aoi"
    assert entry["commit"] == entry["ref"] == "bb5eb56f216eeff9135b7aa178197a79df152396"
    assert entry["source"] == "vendor/py_launch.cjs"
    data = (ROOT / "vendor" / "py_launch.cjs").read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == entry["blob_sha"]


def test_entry_uses_shared_launcher_only():
    text = (ROOT / f"{TOOL}.cjs").read_text(encoding="utf-8")
    assert 'require("./vendor/py_launch.cjs")' in text
    assert "child_process" not in text
    assert f'"{TOOL}.py"' in text
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert pkg["engines"]["node"] == ">=24"


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
    p = subprocess.run([NODE, str(ROOT / f"{TOOL}.cjs"), "--help"], env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert p.returncode == 2
    assert p.stdout == ""


@needs_node
@pytest.mark.parametrize("old,new", [
    (f'"{TOOL}.py"', '"missing_tool.py"'),
    ("main(PY_SCRIPT, PY_LAUNCH);", "main(PY_SCRIPT, { ...PY_LAUNCH, argv: [] });"),
    ("noSite: true", "noSite: true, isolated: true, argv: ['--help']"),
])
def test_mutated_entry_breaks_parity(tmp_path, old, new):
    (tmp_path / "vendor").mkdir()
    shutil.copy(ROOT / "vendor" / "py_launch.cjs", tmp_path / "vendor" / "py_launch.cjs")
    shutil.copy(ROOT / f"{TOOL}.py", tmp_path / f"{TOOL}.py")
    text = (ROOT / f"{TOOL}.cjs").read_text(encoding="utf-8")
    assert old in text
    (tmp_path / f"{TOOL}.cjs").write_text(text.replace(old, new), encoding="utf-8")
    assert any(_parity(tmp_path, c["args"])[0] != _parity(ROOT, c["args"])[1] for c in CASES)


@needs_node
@pytest.mark.parametrize("code", [0, 1, 3, 7])
def test_entry_forwards_stdin_argv_and_exit_code(tmp_path, code):
    (tmp_path / "vendor").mkdir()
    shutil.copy(ROOT / "vendor" / "py_launch.cjs", tmp_path / "vendor" / "py_launch.cjs")
    shutil.copy(ROOT / f"{TOOL}.cjs", tmp_path / f"{TOOL}.cjs")
    (tmp_path / f"{TOOL}.py").write_text(
        "import json, sys\n"
        "sys.stdout.write(json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read(),"
        " 'nosite': sys.flags.no_site}))\n"
        f"sys.exit({code})\n", encoding="utf-8")
    p = subprocess.run([NODE, str(tmp_path / f"{TOOL}.cjs"), "a b", "--x=ü"], env=_env(),
                       input="こんにちは\n", capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert p.returncode == code
    assert json.loads(p.stdout) == {"argv": ["a b", "--x=ü"], "stdin": "こんにちは\n", "nosite": 1}
