"""Offline CI regression of the *exploratory*, unregistered Issue #9 candidates."""
from __future__ import annotations

import copy, hashlib, importlib.util, os
from pathlib import Path
import pytest

import aoi_paths

ROOT = aoi_paths.resolve_aoi_root(anchor=Path(__file__))
SCRIPT = ROOT / "scripts/check_c001_exploratory_candidates.py"
FIXTURE = Path(__file__).parent / "fixtures/c001_exploratory_snapshot.json"
spec = importlib.util.spec_from_file_location("c001_exploratory", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture(scope="module")
def snapshot():
    return mod.read_snapshot(FIXTURE)


def test_pinned_cohort_and_2020_sensitivity(snapshot):
    assert snapshot["research_sha"] == mod.RESEARCH_SHA
    report = mod.report(snapshot)
    assert report["status"] == "unregistered_posthoc_exploration"
    assert report["independently_validated"] is False
    assert report["causally_explained"] is False
    base, sensitivity = report["main"], report["include_2020"]
    assert base["cohort"] == {"years": "2013-2025", "exclude": [2020], "units": 144,
                               "outcome": "final rank >= 4 (B); not CS participation"}
    assert sensitivity["cohort"]["units"] == 156
    years = sorted(set(range(2013, 2026)) - {2020})
    for candidate, expected, restored in (("OR_A", 32, (33, 34)),
                                           ("OR_B", 54, (58, 59)),
                                           ("E7_AND_E22", 59, (64, 64))):
        a, b = base["candidates"][candidate], sensitivity["candidates"][candidate]
        assert (a["hold"], a["n"], a["undetermined"]) == (expected, expected, 0)
        assert a["focus_coverage"] == "12/12"
        assert a["focus_covered"] == years
        assert not a["counterexamples"]
        assert (b["hold"], b["n"], b["undetermined"]) == (*restored, 0)
        assert b["focus_coverage"] == "12/12"
    for candidate in ("OR_A", "OR_B"):
        cases = sensitivity["candidates"][candidate]["counterexamples"]
        assert len(cases) == 1
        assert (cases[0]["unit"], cases[0]["rank"], cases[0]["upper_half"]) == ("d-2020", 3, True)
        assert cases[0]["observed"]["rf_adv"] == -0.6466666666666665
        assert cases[0]["observed"]["inn_size_low_streak"] == 4
        assert cases[0]["matching_parents"] == (["P62", "P138", "P196"] if candidate == "OR_A"
                                                  else ["P62", "P196"])
    assert sensitivity["candidates"]["E7_AND_E22"]["counterexamples"] == []
    assert {x["unit"] for x in base["e7_excluded_by_e22"]} >= {"t-2015"}
    assert {x["unit"] for x in sensitivity["e7_excluded_by_e22"]} >= {"t-2015", "d-2020"}
    assert len(base["e7_excluded_by_e22"]) == 5
    assert len(sensitivity["e7_excluded_by_e22"]) == 8
    assert 2020 not in sensitivity["candidates"]["E7_AND_E22"]["focus_covered"]


def test_p62_2012_undefined_and_2020_streak_retained(snapshot):
    rows = {(r["season"], r["team"]): r for r in snapshot["rows"]}
    assert rows[2012, "d"]["inn_size_low_streak"] is None  # not false
    assert [rows[y, "d"]["inn_size_low_streak"] for y in (2019, 2020, 2021)] == [3, 4, 5]
    assert mod.verdict(rows[2020, "d"], "P62", snapshot["definitions"]) is True
    assert mod.verdict(rows[2020, "d"], "E22", snapshot["definitions"]) is False
    assert rows[2020, "d"]["q_wl_4"] == 8


def test_focus_coverage_paths_are_recalculated(snapshot):
    values = mod.report(snapshot)["main"]["candidates"]
    assert values["OR_A"]["focus_paths"]["2013"] == ["P196"]
    assert values["OR_A"]["focus_paths"]["2014"] == ["P62"]
    assert values["OR_B"]["focus_paths"]["2017"] == ["P189"]
    assert values["OR_A"]["focus_paths"]["2017"] == ["P138"]
    assert values["OR_B"]["focus_paths"]["2023"] == ["P62", "P189", "P196"]


def test_incomplete_cohort_fails_closed(snapshot):
    altered = copy.deepcopy(snapshot)
    altered["rows"] = altered["rows"][:-1]
    with pytest.raises(ValueError, match="168 unique"):
        mod.calculate(altered)


def test_missing_premise_is_unknown_not_false(snapshot):
    row = copy.deepcopy(next(x for x in snapshot["rows"] if x["season"] == 2020 and x["team"] == "d"))
    row["rf_adv"] = None
    assert mod.verdict(row, "P138", snapshot["definitions"]) is None
    row["course_rank_q3"] = None
    assert mod.verdict(row, "E22", snapshot["definitions"]) is None


def test_safe_ast_rejects_python_evaluation():
    with pytest.raises(ValueError, match="unsupported"):
        mod.evaluate("__import__('os').system('echo unsafe')", lambda _: True)


def test_snapshot_provenance_cannot_be_silently_repointed(snapshot, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(FIXTURE.read_text(encoding="utf-8").replace(mod.RESEARCH_SHA, "0" * 40), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        mod.read_snapshot(path)

def test_tracked_projection_blob_is_immutable():
    raw = FIXTURE.read_bytes()
    github_blob = b"blob " + str(len(raw)).encode("ascii") + bytes([0]) + raw
    assert hashlib.sha1(github_blob).hexdigest() == "8c5509c25c5377d0b740cdd870e163a5cc18dd06"


def test_optional_full_artifact_matches_tracked_projection(snapshot):
    archive = os.environ.get("AOI_C001_ARTIFACT_ZIP")
    if not archive:
        pytest.skip("AOI_C001_ARTIFACT_ZIP is not set; original artifact replay was not run")
    original = mod.from_artifact(Path(archive))
    assert mod.report(original) == mod.report(snapshot)
    assert mod.compact_snapshot(original)["rows"] == [
        [row[key] for key in mod.KEYS] for row in snapshot["rows"]]
