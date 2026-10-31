"""Exact live-generator replay, separate from the frozen-model replay tests.

The source/input/artifact SHA guard is fail-closed. Source-hash drift against
Aoi's holdout.provenance.json warns instead of failing only when
HOLDOUT_SOURCE_PROVENANCE is report (pull requests). Unset, strict, and any
other value keep the failure. Input and artifact hashes always fail closed.
Strict regeneration is opt-in: the original record lacks its BLAS/CPU
configuration and its frozen coefficients are not bitwise portable. No test
updates the committed evidence. No scientific value, model coefficient, unit
ID, flag, or new field is ignored.
"""
from copy import deepcopy
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

import pytest

import aoi_paths

# 入力と証拠（cycles/）は Aoi（AOI_ROOT）から読む。ソースはこのテストのリポジトリのもの：
# PythDRagoras ではリポジトリ直下、Aoi では pythdragoras/。Aoi のコピーをハッシュしない。
_REPO_ROOT = aoi_paths.resolve_aoi_root(anchor=Path(__file__))
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREFIX = "pythdragoras/"
PINNED_RUNTIME = {"python": "3.14.8", "numpy": "2.5.4", "polars": "2.0.0"}
ARTIFACTS = ("holdout.json", "holdout.md")
CYCLE_PATH = f"cycles/{aoi_paths.DEFAULT_CYCLE}"
EVIDENCE_PATH = f"{CYCLE_PATH}/outputs/exploratory/littlebit-holdout"
PROVENANCE_PATH = f"{EVIDENCE_PATH}/holdout.provenance.json"
SOURCE_PATHS = tuple(f"{SOURCE_PREFIX}{name}.py" for name in (
    "littlebit_holdout", "littlebit_approx", "littlebit_prune", "rule_search",
    "propositions", "registration_audit", "pythdragoras"))
INPUT_PATHS = tuple(f"{CYCLE_PATH}/{name}" for name in (
    "outputs/season.jsonl", "rule_candidates.toml", "analysis.toml"))
ARTIFACT_PATHS = tuple(f"{EVIDENCE_PATH}/{name}" for name in ARTIFACTS)
PROVENANCE_GROUPS = {
    "source_sha256": SOURCE_PATHS,
    "inputs_sha256": INPUT_PATHS,
    "artifacts_sha256": ARTIFACT_PATHS,
}


def _source_divergence(manifest):
    """Reviewed source drift. Artifact and input hashes cannot be redirected."""
    rows = manifest.get("source_divergence") or []
    assert isinstance(rows, list)
    by_path = {}
    for row in rows:
        path = row["path"]
        assert path not in by_path
        assert path in SOURCE_PATHS
        assert row["artifacts_regenerated"] is False
        assert row["inputs_regenerated"] is False
        assert isinstance(row["reason"], str) and row["reason"].strip()
        assert row["evidence_sha256"] != row["current_sha256"]
        by_path[path] = row
    return by_path


def _source_file(source_root, name):
    assert name.startswith(SOURCE_PREFIX), name
    return source_root / name[len(SOURCE_PREFIX):]


_SOURCE_DRIFT = "SHA-256 drift; review provenance, never auto-rebaseline"
_REPORT_SOURCE_MODES = {"report", "warn", "report-only", "warning"}
_STRICT_SOURCE_MODES = {"", "strict", "fail", "enforce"}


def _source_hash_report_only():
    """True only for an explicit report-only flag. Anything else stays strict."""
    raw = os.environ.get("HOLDOUT_SOURCE_PROVENANCE", "strict")
    mode = raw.strip().lower()
    if mode in _STRICT_SOURCE_MODES:
        return False
    if mode in _REPORT_SOURCE_MODES:
        return True
    raise AssertionError(
        f"HOLDOUT_SOURCE_PROVENANCE={raw!r} is not a known mode; the source-hash check stays fail-closed")


def _warn_source_drift(message):
    warnings.warn(message, UserWarning, stacklevel=2)
    print(f"::warning title=holdout source provenance::{message}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"\n> holdout source provenance (report only): {message}\n")


def _verify_provenance(root, manifest, source_root=None, report_source_drift=False):
    """Fail closed on missing coverage, missing files, or any unreviewed byte change.

    Inputs and artifacts are read under ``root`` (Aoi). Sources are read under
    ``source_root`` (default ``root / "pythdragoras"``); manifest keys keep the
    ``pythdragoras/`` prefix. ``report_source_drift`` warns on source-hash
    mismatch only. Callers default to strict; the env flag is opt-in.
    """
    if source_root is None:
        source_root = root / SOURCE_PREFIX
    assert manifest["schema"] == "littlebit-holdout-provenance/1"
    diverged = _source_divergence(manifest)
    for group, paths in PROVENANCE_GROUPS.items():
        expected = manifest[group]
        assert set(expected) == set(paths), f"{group}: incomplete or unexpected coverage"
        for name in paths:
            path = _source_file(source_root, name) if group == "source_sha256" else root / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if name in diverged:
                assert group == "source_sha256"
                row = diverged[name]
                assert row["evidence_sha256"] == expected[name]
                recorded = row["current_sha256"]
            else:
                recorded = expected[name]
            if digest != recorded:
                message = f"{name}: {_SOURCE_DRIFT}"
                if group == "source_sha256" and report_source_drift:
                    _warn_source_drift(message)
                else:
                    raise AssertionError(message)


def test_littlebit_committed_provenance_matches_current_sources_and_untouched_evidence():
    root = _REPO_ROOT
    manifest = json.loads((root / PROVENANCE_PATH).read_text(encoding="utf-8"))
    _verify_provenance(root, manifest, SOURCE_ROOT, report_source_drift=_source_hash_report_only())
    report = json.loads((root / EVIDENCE_PATH / "holdout.json").read_text(encoding="utf-8"))
    assert manifest["historical_runtime"] == report["runtime"]
    assert {f"{CYCLE_PATH}/{name}": digest for name, digest in report["inputs_sha256"].items()} == manifest["inputs_sha256"]
    assert report["posthoc"] is True and report["exploratory"] is True
    assert report["independent_confirmation"] is False


def test_littlebit_sources_are_this_repositorys_own_files():
    """PythDRagoras: the repository root; inside Aoi: Aoi's pythdragoras/. Never AOI_ROOT's copy."""
    test_file = Path(__file__).resolve()
    assert SOURCE_ROOT == test_file.parent.parent
    for name in SOURCE_PATHS:
        assert _source_file(SOURCE_ROOT, name).is_file(), name
    assert (SOURCE_ROOT / "tests" / test_file.name).samefile(test_file)


def test_littlebit_change_to_own_source_is_detected_even_if_aoi_copy_is_unchanged(tmp_path):
    root = _REPO_ROOT
    manifest = json.loads((root / PROVENANCE_PATH).read_text(encoding="utf-8"))
    own = tmp_path / "own"
    own.mkdir()
    for name in SOURCE_PATHS:
        (own / name[len(SOURCE_PREFIX):]).write_bytes(_source_file(SOURCE_ROOT, name).read_bytes())
    _verify_provenance(root, manifest, own)
    (own / "littlebit_holdout.py").write_bytes((own / "littlebit_holdout.py").read_bytes() + b"# changed\n")
    with pytest.raises(AssertionError, match="littlebit_holdout.py: SHA-256 drift"):
        _verify_provenance(root, manifest, own)


def test_littlebit_inputs_come_from_root_and_sources_from_source_root(provenance_tree, tmp_path):
    root, manifest = provenance_tree
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in SOURCE_PATHS:
        (elsewhere / name[len(SOURCE_PREFIX):]).write_bytes((root / name).read_bytes())
        (root / name).write_bytes(b"stale copy under root\n")
    _verify_provenance(root, manifest, elsewhere)
    with pytest.raises(AssertionError, match="SHA-256 drift"):
        _verify_provenance(root, manifest)
    (elsewhere / "rule_search.py").unlink()
    with pytest.raises(FileNotFoundError):
        _verify_provenance(root, manifest, elsewhere)


@pytest.fixture
def provenance_tree(tmp_path):
    manifest = {"schema": "littlebit-holdout-provenance/1"}
    for group, paths in PROVENANCE_GROUPS.items():
        manifest[group] = {}
        for name in paths:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"synthetic provenance fixture: {name}\n".encode()
            path.write_bytes(content)
            manifest[group][name] = hashlib.sha256(content).hexdigest()
    return tmp_path, manifest


def test_littlebit_provenance_accepts_complete_identical_snapshot(provenance_tree):
    _verify_provenance(*provenance_tree)


@pytest.mark.parametrize("name", SOURCE_PATHS + INPUT_PATHS + ARTIFACT_PATHS)
def test_littlebit_provenance_rejects_each_generator_dependency_contract_and_artifact_change(provenance_tree, name):
    root, manifest = provenance_tree
    path = root / name
    path.write_bytes(path.read_bytes() + b"changed\n")
    with pytest.raises(AssertionError, match="SHA-256 drift"):
        _verify_provenance(root, manifest)


@pytest.mark.parametrize("group", PROVENANCE_GROUPS)
def test_littlebit_provenance_rejects_empty_coverage(provenance_tree, group):
    root, manifest = provenance_tree
    manifest[group] = {}
    with pytest.raises(AssertionError, match="coverage"):
        _verify_provenance(root, manifest)


def test_littlebit_provenance_rejects_missing_source(provenance_tree):
    root, manifest = provenance_tree
    (root / SOURCE_PATHS[0]).unlink()
    with pytest.raises(FileNotFoundError):
        _verify_provenance(root, manifest)


def test_littlebit_reviewed_source_divergence_stays_fail_closed(provenance_tree):
    root, manifest = provenance_tree
    path = SOURCE_PATHS[0]
    original = hashlib.sha256((root / path).read_bytes()).hexdigest()
    (root / path).write_bytes((root / path).read_bytes() + b"reviewed-diagnostic\n")
    current = hashlib.sha256((root / path).read_bytes()).hexdigest()
    manifest["source_divergence"] = [{
        "path": path,
        "evidence_sha256": original,
        "current_sha256": current,
        "artifacts_regenerated": False,
        "inputs_regenerated": False,
        "reason": "reviewed file drift; frozen artifacts stay at the evidence hash",
    }]
    _verify_provenance(root, manifest)
    (root / path).write_bytes((root / path).read_bytes() + b"again\n")
    with pytest.raises(AssertionError, match="SHA-256 drift"):
        _verify_provenance(root, manifest)


def test_littlebit_source_divergence_cannot_rebaseline_artifacts(provenance_tree):
    root, manifest = provenance_tree
    manifest["source_divergence"] = [{
        "path": ARTIFACT_PATHS[0],
        "evidence_sha256": "a" * 64,
        "current_sha256": "b" * 64,
        "artifacts_regenerated": False,
        "inputs_regenerated": False,
        "reason": "artifacts are not a source divergence",
    }]
    with pytest.raises(AssertionError):
        _verify_provenance(root, manifest)


def test_source_hash_mode_is_strict_unless_report_is_explicit(monkeypatch):
    monkeypatch.delenv("HOLDOUT_SOURCE_PROVENANCE", raising=False)
    assert _source_hash_report_only() is False
    for raw in ("", "strict", "fail", "enforce", " STRICT "):
        monkeypatch.setenv("HOLDOUT_SOURCE_PROVENANCE", raw)
        assert _source_hash_report_only() is False
    for raw in ("report", "warn", "report-only", "warning", " Report "):
        monkeypatch.setenv("HOLDOUT_SOURCE_PROVENANCE", raw)
        assert _source_hash_report_only() is True
    monkeypatch.setenv("HOLDOUT_SOURCE_PROVENANCE", "maybe")
    with pytest.raises(AssertionError, match="HOLDOUT_SOURCE_PROVENANCE"):
        _source_hash_report_only()


def test_report_only_source_drift_warns_and_records_summary(provenance_tree, tmp_path, monkeypatch):
    root, manifest = provenance_tree
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    path = root / SOURCE_PATHS[0]
    path.write_bytes(path.read_bytes() + b"pr-drift\n")
    with pytest.warns(UserWarning, match="SHA-256 drift"):
        _verify_provenance(root, manifest, report_source_drift=True)
    text = summary.read_text(encoding="utf-8")
    assert "report only" in text
    assert "SHA-256 drift" in text
    with pytest.raises(AssertionError, match="SHA-256 drift"):
        _verify_provenance(root, manifest)


def test_report_only_does_not_weaken_input_or_artifact_hashes(provenance_tree):
    root, manifest = provenance_tree
    target = root / INPUT_PATHS[0]
    target.write_bytes(target.read_bytes() + b"changed\n")
    with pytest.raises(AssertionError, match="SHA-256 drift"):
        _verify_provenance(root, manifest, report_source_drift=True)
    target.write_bytes(b"synthetic provenance fixture: " + INPUT_PATHS[0].encode() + b"\n")
    artifact = root / ARTIFACT_PATHS[0]
    artifact.write_bytes(artifact.read_bytes() + b"changed\n")
    with pytest.raises(AssertionError, match="SHA-256 drift"):
        _verify_provenance(root, manifest, report_source_drift=True)


def test_report_only_reviewed_source_divergence_still_warns_on_further_drift(provenance_tree, monkeypatch):
    root, manifest = provenance_tree
    monkeypatch.setenv("HOLDOUT_SOURCE_PROVENANCE", "report")
    path = SOURCE_PATHS[0]
    original = hashlib.sha256((root / path).read_bytes()).hexdigest()
    (root / path).write_bytes((root / path).read_bytes() + b"reviewed-diagnostic\n")
    current = hashlib.sha256((root / path).read_bytes()).hexdigest()
    manifest["source_divergence"] = [{
        "path": path,
        "evidence_sha256": original,
        "current_sha256": current,
        "artifacts_regenerated": False,
        "inputs_regenerated": False,
        "reason": "reviewed file drift; frozen artifacts stay at the evidence hash",
    }]
    _verify_provenance(root, manifest, report_source_drift=_source_hash_report_only())
    (root / path).write_bytes((root / path).read_bytes() + b"again\n")
    with pytest.warns(UserWarning, match="SHA-256 drift"):
        _verify_provenance(root, manifest, report_source_drift=_source_hash_report_only())


def test_ci_resolves_aoi_main_without_hard_pinning_its_sha():
    workflow = (SOURCE_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    readme = (SOURCE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "AOI_SHA:" not in workflow
    assert "git ls-remote" in workflow
    assert "https://github.com/myon-bioinformatics/Aoi" in workflow
    assert "refs/heads/main" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "aoi_ref" in workflow
    assert "schedule:" in workflow
    assert "HOLDOUT_SOURCE_PROVENANCE" in workflow
    assert "actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0" in workflow
    assert "astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2 # v10.3.3" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "git rev-parse HEAD" not in workflow
    assert "9249ab6e7bbec167cc75963ff315bc90ab5d9e0c" not in readme
    assert "git -C .test-deps/aoi checkout main" in readme


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _assert_matches(expected, actual):
    """Compare the *whole* serialized evidence, with no rounding/field removal."""
    for name in ARTIFACTS:
        before, after = expected[name], actual[name]
        if before != after:
            diff = difflib.unified_diff(
                before.decode("utf-8").splitlines(),
                after.decode("utf-8").splitlines(),
                fromfile=f"committed/{name}", tofile=f"regenerated/{name}",
                lineterm="")
            # Bound failure logs, not the comparison or the saved artifacts.
            import itertools
            excerpt = "\n".join(itertools.islice(diff, 60))
            raise AssertionError(f"{name} generator drift (no values normalized):\n{excerpt}")


def _read_artifacts(directory):
    return {name: (directory / name).read_bytes() for name in ARTIFACTS}


def _regenerate(generator, cycle, output):
    committed = cycle / "outputs/exploratory/littlebit-holdout"
    if output.resolve() == committed.resolve():
        raise ValueError("regeneration must not overwrite committed evidence")
    # Use the real CLI and its real run_cycle/default settings/serialization.
    # Do not feed it saved encoders, decoders, selected rules, or research masks.
    result = generator.main(["--cycle-dir", str(cycle), "--out", str(output)])
    assert result == 0, f"live generator returned {result!r}"
    return _read_artifacts(output)


@pytest.fixture
def evidence():
    # Tiny examples exercise the drift checker; they are not research results.
    return {
        "method": "littlebit-frozen-projection/1",
        "runtime": dict(PINNED_RUNTIME),
        "posthoc": True, "exploratory": True, "independent_confirmation": False,
        "settings": {"rank": 8, "seed": 0, "zero_tol": 1e-10, "rcond": 1e-12,
                     "train_through": 2022, "min_train_years": 5},
        "inputs_sha256": {"analysis.toml": "a" * 64},
        "cohort": {"excluded_seasons": [2020]},
        "primary": {
            "train_years": [2012, 2022], "test_years": [2023, 2025],
            "fitted": {
                "model": {"encoder": [[0.1]], "decoder": [[0.2]]},
                "selected": {"derived": {"formula": "x == True",
                                          "if_any": [[{"col": "x", "value": True}]]}}},
            "evaluation": {"mcc": 0.8, "tp": 1, "sets": {"tp": ["a-2023"], "fp": ["b-2023"]}}},
        "walk_forward": {"folds": [{"test_years": [2023]}], "pooled": {"mcc": 0.7}},
    }


def _example_artifacts(report, markdown=b"# Example\n"):
    return {"holdout.json": _json_bytes(report), "holdout.md": markdown}


def test_littlebit_drift_accepts_identical_evidence_without_mutation(evidence):
    expected = _example_artifacts(evidence)
    before = deepcopy(expected)
    _assert_matches(expected, deepcopy(expected))
    assert expected == before


@pytest.mark.parametrize(("path", "replacement"), [
    (("method",), "changed-method"),
    (("settings", "rank"), 9),
    (("settings", "seed"), 1),
    (("settings", "zero_tol"), 2e-10),
    (("settings", "rcond"), 2e-12),
    (("settings", "train_through"), 2021),
    (("settings", "min_train_years"), 6),
    (("inputs_sha256", "analysis.toml"), "b" * 64),
    (("cohort", "excluded_seasons"), []),
    (("primary", "train_years"), [2012, 2021]),
    (("primary", "test_years"), [2024, 2025]),
    (("primary", "fitted", "model", "encoder"), [[0.10000000000000002]]),
    (("primary", "fitted", "model", "decoder"), [[0.20000000000000004]]),
    (("primary", "fitted", "selected", "derived", "formula"), "x == False"),
    (("primary", "fitted", "selected", "derived", "if_any"), [[{"col": "x", "value": False}]]),
    (("primary", "evaluation", "mcc"), 0.8000000000000002),
    (("primary", "evaluation", "tp"), 2),
    # Same counts/MCC but different classification identities must still fail.
    (("primary", "evaluation", "sets"), {"tp": ["b-2023"], "fp": ["a-2023"]}),
    (("walk_forward", "folds"), [{"test_years": [2024]}]),
    (("walk_forward", "pooled", "mcc"), 0.71),
    (("posthoc",), False),
    (("exploratory",), False),
    (("independent_confirmation",), True),
    (("runtime", "numpy"), "different-environment"),
    (("new_scientific_field",), {"must_not_be_silently_ignored": True}),
])
def test_littlebit_drift_rejects_generator_and_research_contract_changes(evidence, path, replacement):
    changed = deepcopy(evidence)
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement
    with pytest.raises(AssertionError, match="holdout.json generator drift"):
        _assert_matches(_example_artifacts(evidence), _example_artifacts(changed))


def test_littlebit_drift_rejects_markdown_only_changes(evidence):
    with pytest.raises(AssertionError, match="holdout.md generator drift"):
        _assert_matches(_example_artifacts(evidence),
                        _example_artifacts(evidence, b"# Changed interpretation\n"))


@pytest.mark.parametrize("missing", ARTIFACTS)
def test_littlebit_drift_rejects_missing_output(evidence, missing):
    actual = _example_artifacts(evidence)
    del actual[missing]
    with pytest.raises(KeyError):
        _assert_matches(_example_artifacts(evidence), actual)


def test_littlebit_drift_calls_live_cli_and_protects_committed_files(tmp_path, evidence):
    cycle, output = tmp_path / "cycle", tmp_path / "regenerated"
    committed = cycle / "outputs/exploratory/littlebit-holdout"
    committed.mkdir(parents=True)
    expected = _example_artifacts(evidence)
    for name, data in expected.items():
        (committed / name).write_bytes(data)
    calls = []

    class Generator:
        @staticmethod
        def main(argv):
            calls.append(argv)
            output.mkdir()
            for name, data in expected.items():
                (output / name).write_bytes(data)
            return 0

    _assert_matches(expected, _regenerate(Generator, cycle, output))
    assert calls == [["--cycle-dir", str(cycle), "--out", str(output)]]
    with pytest.raises(ValueError, match="must not overwrite"):
        _regenerate(Generator, cycle, committed / ".." / committed.name)
    assert len(calls) == 1
    assert _read_artifacts(committed) == expected


@pytest.mark.parametrize("change", ["projection-tolerance", "mcc-contract"])
def test_littlebit_live_generator_and_dependency_changes_are_detected(tmp_path, monkeypatch, change):
    pl = pytest.importorskip("polars")
    import itertools
    import littlebit_holdout as lh

    # A tiny *freshly fitted* cycle tests mutations of real computation, not
    # just hand-edits to a stored report. The c001 evidence is never involved.
    rows = [{"season": year, "team": str(team),
             **{f"x{j}": bit for j, bit in enumerate(bits)}, "y": bool(bits[0] or bits[1])}
            for year in range(2010, 2018)
            for team, bits in enumerate(itertools.product((0, 1), repeat=3))]
    frame = pl.DataFrame(rows)
    spec = {"target": [{"col": "y", "op": "==", "value": True}],
            "candidate": [{"col": f"x{j}", "op": "==", "value": 1} for j in range(3)],
            "max_terms": 1, "max_conds": 1}
    monkeypatch.setattr(lh.la, "load_cycle", lambda cycle: (frame, spec, {}))
    for name in ("outputs/season.jsonl", "rule_candidates.toml", "analysis.toml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic test input\n", encoding="utf-8")

    def generate():
        report = lh.run_cycle(tmp_path, train_through=2015, min_train_years=2)
        return _example_artifacts(report, lh.render_markdown(report).encode("utf-8"))

    expected = generate()
    if change == "projection-tolerance":
        monkeypatch.setattr(lh, "ZERO_TOL", lh.ZERO_TOL * 2)
    else:
        real_mcc = lh.rs._mcc
        monkeypatch.setattr(lh.rs, "_mcc", lambda *args: real_mcc(*args) + 0.125)
    with pytest.raises(AssertionError, match="holdout.json generator drift"):
        _assert_matches(expected, generate())


@pytest.mark.skipif(os.environ.get("AOI_REGENERATE_LITTLEBIT_HOLDOUT") != "1",
                    reason="opt-in bitwise reconstruction; BLAS-dependent drift is recorded in provenance")
def test_littlebit_committed_holdout_regenerates_from_current_generator(tmp_path, monkeypatch):
    import numpy as np
    import polars as pl
    import littlebit_holdout as lh

    root = _REPO_ROOT
    cycle = aoi_paths.resolve_cycle_path(root)
    committed = cycle / "outputs/exploratory/littlebit-holdout"
    expected = _read_artifacts(committed)
    report = json.loads(expected["holdout.json"])
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__}
    assert report["runtime"] == PINNED_RUNTIME, "review the pinned environment explicitly; do not rebaseline research"
    assert runtime == PINNED_RUNTIME, f"use the pinned drift CI environment, got {runtime}"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        assert os.environ.get(key) == "1", f"set {key}=1 before starting Python"
    assert report["posthoc"] is True and report["exploratory"] is True
    assert report["independent_confirmation"] is False
    output = Path(os.environ.get("AOI_HOLDOUT_DRIFT_OUTPUT", str(tmp_path / "regenerated")))
    assert output.resolve() != committed.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fit_calls = []
    real_fit = lh.fit_rules

    def tracked_fit(*args, **kwargs):
        fit_calls.append(args[0].height)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(lh, "fit_rules", tracked_fit)
    started = time.monotonic()
    diagnostics = {"runtime": runtime, "comparison": "not-completed",
                   "normalization": "none; exact JSON and Markdown bytes",
                   "source_sha256": {SOURCE_PREFIX + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(SOURCE_ROOT.glob("*.py"))}}
    try:
        actual = _regenerate(lh, cycle, output)
        assert len(fit_calls) == 1 + len(report["walk_forward"]["folds"]), "every fold must refit, not replay saved models"
        _assert_matches(expected, actual)
        assert _read_artifacts(committed) == expected, "committed evidence must remain untouched"
        diagnostics["comparison"] = "identical"
    finally:
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        diagnostics["fit_training_rows"] = fit_calls
        diagnostics["expected_sha256"] = {n: hashlib.sha256(v).hexdigest() for n, v in expected.items()}
        diagnostics["actual_sha256"] = {n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                                         for n in ARTIFACTS if (output / n).is_file()}
        (output / "reproducibility.json").write_bytes(_json_bytes(diagnostics))
        print(json.dumps({k: v for k, v in diagnostics.items() if k != "source_sha256"}, sort_keys=True))
