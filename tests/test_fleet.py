"""🌐 Fleet contracts through the parent's shared checks (vendor/fleet_check.py): vendor lock, parent pins, CI setup,
Node entries through the vendored py_launch.cjs, launcher contract.

共通の検査（lock のバイトと出どころ・親の参照・action のピン・ジョブの中身・Node と Python の一致・ランチャーの契約）は
親の fleet_check に任せ、ここにはこのリポジトリの期待値（ピン・ツール・ケース・文言）だけを残す。

Node entry: `node vendor/py_launch.cjs --env-var <TOOL>_PYTHON -- <core|npb>/<tool>.py ARGS` (no per-tool wrapper).
"""
from __future__ import annotations
import unittest

import importlib.util, json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Aoi は子を全部 1 回の pytest で回す。トップの `vendor` パッケージは子どうしでぶつかるので、ファイルから固有の名前で読む
_spec = importlib.util.spec_from_file_location("pythdragoras_fleet_check", ROOT / "vendor/fleet_check.py")
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)

PARENT_WORKFLOW_PIN = "fbf433b296ec047f1c9acdf4d5d12c81ff7f59e8"
LIVE_PARENT_REF = "${{ steps.parent.outputs.sha }}"
ACCEPTED = {
    "vendor/cli_args.py": "5ae22b1a1b808eca93e3c385a8db867fbeb8ee32",
    "vendor/py_launch.cjs": "9e0147a17695ce63fa73daf75d8195a4f49095bb",
    "vendor/fleet_check.py": "64db25eb0fd9025dbe3cbcbced8b88a39ecb9266",
}
# Node 入口: ツール名 -> 本体（トップにはメインだけ。ツールは core/ と npb/ にいる）。環境変数は <TOOL>_PYTHON
SCRIPTS = {
    "littlebit_survey": "core/littlebit_survey.py",
    "score_sensitivity": "npb/score_sensitivity.py",
    "season_sensitivity": "npb/season_sensitivity.py",
}
CASES = [{"id": "help", "args": ["--help"], "rc": 0}, {"id": "bogus", "args": ["--bogus"], "rc": 2}]
NODE = fleet.node(ci_jobs=None)
needs_node = pytest.mark.skipif(NODE is None, reason="Node 26+ not available (set NODE)")
every_tool = pytest.mark.parametrize("tool", list(SCRIPTS))


def _env_var(tool: str) -> str:
    return f"{tool.upper()}_PYTHON"


# ==== vendor lock: placed bytes, accepted commits and where each row comes from (independent of the parent pin)

def test_vendor_lock_matches_checked_in_bytes_and_accepted_commits():
    sources = {
        "vendor/cli_args.py": ("myon-bioinformatics/cli_args", "cli_args.py"),
        "vendor/py_launch.cjs": ("myon-bioinformatics/Aoi", "vendor/py_launch.cjs"),
        "vendor/fleet_check.py": (fleet.PARENT, "fleet_check.py"),
    }
    rows = fleet.check_lock(ROOT, commits=ACCEPTED, sources=sources)
    assert set(rows) == set(sources)
    assert not (ROOT / "vendor/uv_tool_build.py").exists()


# ==== parent pins and action pins: local rollout contract

def test_parent_workflow_references_use_reviewed_rollout_commit():
    # Mutable and shortened references are returned too, rather than ignored.
    refs = fleet.parent_refs(ROOT)
    assert refs.count(LIVE_PARENT_REF) == 1, refs
    assert {ref for ref in refs if ref != LIVE_PARENT_REF} == {PARENT_WORKFLOW_PIN}, refs


def test_every_action_is_pinned_to_its_reviewed_commit():
    rows = fleet.check_actions(ROOT, pins={
        "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
        "actions/setup-node": "949feb2413d6458794dcd2491c4babbbce0c15c1 # v7.1.0",
        "actions/upload-artifact": "cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7.0.2",
        "astral-sh/setup-uv": "68559e99cc807c1eb7e9238fd43a15bba460eec2 # v10.3.3",
    })
    assert {sha for action, sha, _ in rows if action.startswith(fleet.PARENT + "/")} == {PARENT_WORKFLOW_PIN}, rows


# ==== CI setup: interpreter prepared before the first python/uv command, Node 26 guard, verify-only vendor dispatch

def test_vendor_dispatch_verifies_lock_on_resolved_parent_main():
    fleet.workflow_job(ROOT, "vendor-dispatch", node_floor=26, dispatch_lock="vendor.lock.json",
                       setup="astral-sh/setup-uv@", inputs=('python-version: "3.15"', "activate-environment: true"))


def test_ci_requires_node_for_node_entry_parity():
    # The test job must set up Python, then Node, before running its pytest suite (NODE is required in every CI job).
    job = fleet.workflow_job(ROOT, "test", setup="astral-sh/setup-uv@",
                             inputs=('python-version: "${{ matrix.python }}"', "activate-environment: true"))
    assert "actions/setup-node@" in job
    assert "node-version: '26'" in job
    assert "Require supported Node in parity CI" in job


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
    """Missing interpreter, passthrough under -S, hostile sitecustomize x resolvePython/run/main, both signal orders."""
    fleet.launcher_contract(ROOT, _env_var(tool))


# 🧰 Scripts: 📝 each one describes itself and ⌨️ one that reads sys.argv declares its arguments (argparse /
# cli_args). Read with ast, never run; the portfolio's script catalog reads the same facts.

class ScriptsDescribeThemselves(unittest.TestCase):
    def test_every_script_has_a_description_and_declared_arguments(self):
        fleet.check_scripts(fleet.repo_root(__file__))
