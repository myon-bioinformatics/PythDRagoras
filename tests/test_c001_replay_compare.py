import json, sys
from pathlib import Path

import aoi_paths

sys.path.insert(0, str(aoi_paths.resolve_aoi_root(anchor=Path(__file__)) / "scripts"))
from compare_c001_replay import compare_dirs, prepare_v2


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(hold=2, posthoc=None):
    r = {"id": "E1", "definition_sha256": "d", "forms": [{"form": "original", "n": 3, "hold": hold,
         "undetermined": 0, "rate": hold / 3, "verdict": "Supported", "code": 0,
         "counterexamples": [] if hold == 3 else [{"unit": "t-2015"}]}]}
    if posthoc is not None:
        r["posthoc"] = posthoc
    return r


def test_identical_replay_has_no_differences(tmp_path):
    for d in ("v1", "v2"):
        write(tmp_path/d/"sets.jsonl", [row()])
        write(tmp_path/d/"propositions.jsonl", [])
    assert compare_dirs(tmp_path/"v1", tmp_path/"v2") == []


def test_replay_reports_value_counterexample_and_metadata_drift(tmp_path):
    write(tmp_path/"v1"/"sets.jsonl", [row(2, False)])
    write(tmp_path/"v2"/"sets.jsonl", [row(3, True)])
    write(tmp_path/"v1"/"propositions.jsonl", [])
    write(tmp_path/"v2"/"propositions.jsonl", [])
    diffs = compare_dirs(tmp_path/"v1", tmp_path/"v2")
    fields = {d["field"] for d in diffs}
    assert {"hold", "rate", "counterexamples", "posthoc"} <= fields


def test_replay_reports_missing_ids_and_forms(tmp_path):
    a = row()
    b = row()
    b["id"] = "E2"
    for d, rows in (("v1", [a]), ("v2", [b])):
        write(tmp_path/d/"sets.jsonl", rows)
        write(tmp_path/d/"propositions.jsonl", [])
    diffs = compare_dirs(tmp_path/"v1", tmp_path/"v2")
    assert {d["id"] for d in diffs} == {"E1", "E2"}


def test_prepare_v2_never_copies_semantic_outputs(tmp_path):
    v1, v2 = tmp_path/"v1", tmp_path/"v2"
    write(v1/"propositions.jsonl", [{"id": "P1"}])
    write(v1/"sets.jsonl", [{"id": "E1"}])
    write(v1/"fetched"/"source.jsonl", [{"url": "saved"}])
    prepare_v2(v1, v2)
    assert not (v2/"propositions.jsonl").exists()
    assert not (v2/"sets.jsonl").exists()
    assert (v2/"fetched"/"source.jsonl").exists()
