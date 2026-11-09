"""No evaluation row or label may change fitted operators or rule selection."""
import hashlib, itertools, json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import littlebit_approx as la
import littlebit_holdout as lh
import littlebit_prune
from propositions import PropositionError
import rule_search as rs


@pytest.fixture
def sample():
    rows = []
    for year in range(2010, 2015):
        for team, bits in enumerate(itertools.product((0, 1), repeat=3)):
            rows.append({"season": year, "team": str(team),
                         **{f"x{j}": bit for j, bit in enumerate(bits)},
                         "y": bool(bits[0] and not bits[1] or bits[2] and year % 2)})
    df = pl.DataFrame(rows)
    candidates = [{"col": f"x{j}", "op": "==", "value": 1} for j in range(3)]
    target = [{"col": "y", "op": "==", "value": True}]
    return df, target, candidates


def test_fit_is_feature_only_and_transform_is_frozen_and_batch_independent(sample, monkeypatch):
    df, _, candidates = sample
    train = df.filter(pl.col("season") < 2014)
    model = lh.fit(train.drop("y"), candidates, rank=2, stages=2)
    assert model == lh.fit(train.with_columns(pl.col("y").not_()), candidates, rank=2, stages=2)
    assert model["candidates"]
    model = json.loads(json.dumps(model))
    before = json.dumps(model, sort_keys=True)

    def forbidden(*args, **kwargs):
        pytest.fail("evaluation must not fit or reuse a saved sample mask")
    monkeypatch.setattr(littlebit_prune, "littlebit", forbidden)
    monkeypatch.setattr(la, "derive", forbidden)
    monkeypatch.setattr(la, "attach_loadings", forbidden)
    test = df.filter(pl.col("season") == 2014)
    cols = [c["col"] for c in model["candidates"]]
    full = lh.transform(test, model).select(cols)
    assert full.equals(pl.concat([lh.transform(test.slice(i, 1), model).select(cols) for i in range(test.height)]))
    assert full.reverse().equals(lh.transform(test.reverse(), model).select(cols))
    renamed = test.with_columns(pl.lit("unseen").alias("team"), pl.lit(2099).alias("season")).drop("y")
    assert full.equals(lh.transform(renamed, model).select(cols))
    changed = test.with_columns((1 - pl.col("x0")).alias("x0"))
    assert full.head(1).equals(lh.transform(pl.concat([test.head(1), changed.tail(3)]), model).select(cols).head(1))
    assert json.dumps(model, sort_keys=True) == before
    assert "loading_units" not in before


def test_fit_uses_only_training_rows_and_test_changes_cannot_change_package(sample, monkeypatch):
    df, target, candidates = sample
    seen = []
    real = littlebit_prune.littlebit
    def spy(matrix, *args, **kwargs):
        seen.append(matrix.shape)
        return real(matrix, *args, **kwargs)
    monkeypatch.setattr(littlebit_prune, "littlebit", spy)
    kwargs = {"train_years": [2010, 2011, 2012], "test_years": [2013, 2014], "rank": 2, "stages": 2}
    first = lh.backtest(df, target, candidates, **kwargs)
    changed = df.with_columns(
        pl.when(pl.col("season") >= 2013).then(~pl.col("y")).otherwise(pl.col("y")).alias("y"),
        pl.when(pl.col("season") >= 2013).then(1 - pl.col("x0")).otherwise(pl.col("x0")).alias("x0"))
    second = lh.backtest(changed, target, candidates, **kwargs)
    assert first["fitted"] == second["fitted"]
    assert seen == [(24, 3), (24, 3)]
    assert set(first["training_units"]).isdisjoint(first["evaluation_units"])
    assert first["evaluation"] != second["evaluation"]
    assert first == lh.backtest(df.reverse(), target, candidates, **kwargs)


def test_projected_train_masks_are_used_for_selection_and_counts_are_exact(sample):
    df, target, candidates = sample
    fitted = lh.fit_rules(df, target, candidates, rank=2, stages=2, seed=0, max_terms=2, max_conds=2)
    model = fitted["model"]
    projected = lh.transform(df, model)
    for candidate in model["candidates"]:
        assert projected[candidate["col"]].sum() == candidate["n_loading"]
    extra = model["candidates"]
    all_rules = rs.search(projected, target, candidates + extra, top=100000, backend="bitset")
    derived = [r for r in all_rules if any(c["col"].startswith("littlebit_project_") for g in r["if_any"] for c in g)]
    assert fitted["selected"]["derived"]["if_any"] == derived[0]["if_any"]
    assert fitted["selected"]["augmented"]["if_any"] == all_rules[0]["if_any"]
    for selected in fitted["selected"].values():
        result = selected["training"]
        expected = rs._record(selected["if_any"], result["tp"], result["fp"],
                              result["tp"] + result["fn"], result["fp"] + result["tn"])[1]
        assert expected["mcc"] == result["mcc"]
        assert result["holds"] == result["tp"]
        assert sorted(sum(result["sets"].values(), [])) == sorted(la.unit_labels(df))


def test_degenerate_projection_has_no_fabricated_derived_rule(sample):
    df, target, candidates = sample
    constant = df.with_columns(pl.lit(1).alias("x0"), pl.lit(1).alias("x1"), pl.lit(1).alias("x2"))
    fitted = lh.fit_rules(constant, target, candidates, rank=2, stages=2, seed=0, max_terms=2, max_conds=2)
    assert fitted["model"]["candidates"] == []
    assert fitted["selected"]["derived"] is None
    assert fitted["selected"]["augmented"] == fitted["selected"]["baseline"]
    result = fitted["selected"]["baseline"]["training"]
    assert result["mcc"] == 0 and result["mcc_defined"] is False
    assert lh.pooled([{"evaluation": {"derived": None}}], "derived") is None


def test_invalid_cohorts_and_target_leak_are_rejected(sample):
    df, target, candidates = sample
    for train_years, test_years in [([], [2014]), ([2013], []), ([2014], [2014]), ([2014], [2013]), ([2001], [2014])]:
        with pytest.raises(PropositionError):
            lh.backtest(df, target, candidates, train_years=train_years, test_years=test_years)
    with pytest.raises(PropositionError, match="重複"):
        lh.backtest(pl.concat([df, df.head(1)]), target, candidates, train_years=[2010], test_years=[2014])
    with pytest.raises(PropositionError, match="目的列"):
        lh.fit_rules(df, target, candidates + target, rank=2, stages=1, seed=0, max_terms=2, max_conds=2)
    with pytest.raises(PropositionError, match="欠損"):
        lh.fit(df.with_columns(pl.lit(None).alias("x0")), candidates)
    with pytest.raises(PropositionError, match="loading_units"):
        lh.transform(df, {"derived_candidates": [{"loading_units": ["0-2014"]}]})


def test_pooled_mcc_uses_confusion_totals_and_refuses_duplicate_units():
    first = {"evaluation": {"baseline": {"sets": {"tp": ["a"], "fp": [], "fn": [], "tn": ["b"]}}}}
    second = {"evaluation": {"baseline": {"sets": {"tp": ["c"], "fp": ["d"], "fn": [], "tn": []}}}}
    result = lh.pooled([first, second], "baseline")
    assert result["mcc"] == rs._mcc(2, 1, 0, 1)
    with pytest.raises(PropositionError, match="重複"):
        lh.pooled([first, first], "baseline")


def test_cli_takes_cycle_dir_without_aoi_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("AOI_ROOT", raising=False)
    monkeypatch.delenv("AOI_RESEARCH_CYCLE", raising=False)
    cycle = tmp_path / "cycle"
    cycle.mkdir()
    with pytest.raises(FileNotFoundError, match="analysis.toml"):
        lh.main(["--cycle-dir", str(cycle)])
    with pytest.raises(SystemExit, match="--cycle-dir"):
        lh.main([])


def test_committed_backtest_replays_frozen_models_without_research_masks(monkeypatch):
    import aoi_paths
    cycle = aoi_paths.resolve_cycle_path(aoi_paths.resolve_aoi_root(anchor=Path(__file__)))
    path = cycle / "outputs/exploratory/littlebit-holdout/holdout.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    df, spec, _ = la.load_cycle(cycle)
    for name, digest in report["inputs_sha256"].items():
        assert hashlib.sha256((cycle / name).read_bytes()).hexdigest() == digest
    def forbidden(*args, **kwargs):
        pytest.fail("saved models must replay without fit or sample-mask attachment")
    monkeypatch.setattr(littlebit_prune, "littlebit", forbidden)
    monkeypatch.setattr(la, "attach_loadings", forbidden)
    for fold in [report["primary"], *report["walk_forward"]["folds"]]:
        train, test = lh._split(df, fold["train_years"], fold["test_years"])
        model = fold["fitted"]["model"]
        assert model["training_rows"] == train.height
        assert max(fold["train_years"]) < min(fold["test_years"])
        for frame, field in [(train, "training"), (test, "evaluation")]:
            projected = lh.transform(frame, model)
            for key, rule in fold["fitted"]["selected"].items():
                expected = None if rule is None else rule["training"] if field == "training" else fold["evaluation"][key]
                assert (None if rule is None else lh.score(projected, spec["target"], rule)) == expected
    assert report["independent_confirmation"] is False
    assert report["posthoc"] is True
    assert lh.render_markdown(report) == path.with_suffix(".md").read_text(encoding="utf-8")
