import math
from itertools import product

import polars as pl
import pytest

from pythdragoras import apply_exclusions, binom_tail, cumulative_test, rank_test, run_tail, summary_markdown


@pytest.mark.parametrize("n,r,p", [(1, 1, 0.5), (3, 2, 0.5), (5, 3, 0.3), (6, 2, 0.5), (8, 4, 0.7), (4, 0, 0.5)])
def test_run_tail_matches_brute_force(n, r, p):
    def longest(seq):
        best = cur = 0
        for s in seq:
            cur = cur + 1 if s else 0
            best = max(best, cur)
        return best

    brute = sum(p ** sum(s) * (1 - p) ** (n - sum(s)) for s in product([0, 1], repeat=n) if longest(s) >= r)
    assert run_tail(n, r, p) == pytest.approx(brute)


def test_known_values():
    assert run_tail(14, 14, 0.5) == pytest.approx(0.5**14)  # 14年連続で下位半分
    assert binom_tail(14, 14, 0.5) == pytest.approx(0.5**14)
    assert binom_tail(14, 0, 0.5) == pytest.approx(1.0)
    assert binom_tail(4, 3, 0.5) == pytest.approx(5 / 16)


def test_exclusion_requires_reason_and_is_recorded():
    st = pl.DataFrame({"season": [2019, 2020, 2020], "team": ["d", "d", "g"]})
    with pytest.raises(ValueError):
        apply_exclusions(st, [{"season": 2020}])
    with pytest.raises(ValueError):
        apply_exclusions(st, [{"season": 2020, "reason": "  "}])
    kept, rec = apply_exclusions(st, [{"season": 2020, "reason": "短縮シーズン"}])
    assert kept["season"].to_list() == [2019]
    assert rec == [{"season": 2020, "reason": "短縮シーズン", "rows_removed": 2}]


def _ranks(team_ranks):
    rows = []
    for team, ranks in team_ranks.items():
        for i, r in enumerate(ranks):
            rows.append({"season": 2012 + i, "team": team, "team_name": team, "rank": r,
                         "rank_tie": False, "league_size": 6})
    return pl.DataFrame(rows)


def test_rank_test_counts_streaks_and_compares_with_all_teams():
    st = _ranks({"d": [2, 4, 5, 6, 3, 5], "g": [1, 1, 2, 4, 1, 1], "t": [6, 6, 1, 1, 2, 2]})
    res = {r["team"]: r for r in rank_test(st)}
    d = res["d"]
    assert (d["bottom_seasons"], d["longest_bottom_run"], d["p_bottom_null"]) == (4, 3, 0.5)
    assert d["null_p_at_least_k"] == pytest.approx(binom_tail(6, 4, 0.5))
    assert d["null_p_run_at_least"] == pytest.approx(run_tail(6, 3, 0.5))
    assert d["empirical_share_at_least_k"] == pytest.approx(1 / 3)  # 3球団中、下位4シーズン以上は中日だけ
    assert res["t"]["longest_bottom_run"] == 2 and res["g"]["bottom_seasons"] == 1


def test_cumulative_z():
    st = pl.DataFrame({"season": [2024, 2025], "team": ["d", "d"], "team_name": ["中日", "中日"],
                       "W": [60, 60], "L": [80, 80], "pythag_fixed": [0.5, 0.5], "pythag_var": [0.5, 0.5]})
    ct = cumulative_test(st).filter(pl.col("model") == "fixed").row(0, named=True)
    assert ct["excess_wins"] == pytest.approx(-20)
    assert ct["z"] == pytest.approx(-20 / math.sqrt(70))


def test_summary_renders():
    st = _ranks({"d": [4, 5, 6]})
    md = summary_markdown("d", [{"season": 2020, "reason": "r", "rows_removed": 6}],
                          pl.DataFrame({"model": ["fixed"], "team": ["d"], "team_name": ["d"], "seasons": [3],
                                        "excess_wins": [-3.0], "z": [-1.0], "p_two_sided": [0.3]}),
                          rank_test(st), [{"id": "H1", "statement": "s", "status": "untested"}])
    assert isinstance(md, str) and md  # 表示は作れること（中身の言葉は判定に使わない）


def test_next_seeds_cli_renders_from_propositions_jsonl(tmp_path):
    import json
    import next_seeds as ns

    props_jsonl = tmp_path / "propositions.jsonl"
    res = ns.reading({"id": "P1", "key": "[all] rd>0 => rank<=3", "forms": [
        {"form": "original", "code": 1, "rate": 0.9, "n": 10, "counterexamples": []},
        {"form": "contrapositive", "code": 1, "rate": 0.9, "n": 10, "counterexamples": []},
        {"form": "converse", "code": 3, "rate": 0.4, "n": 10, "counterexamples": []},
        {"form": "inverse", "code": 3, "rate": 0.4, "n": 10, "counterexamples": []},
    ], "judgement": {"code": 1}, "meta": {"sha256": "sha123"}})
    props_jsonl.write_text(json.dumps({"id": "P1", "key": "[all] rd>0 => rank<=3", "forms": [
        {"form": "original", "code": 1, "rate": 0.9, "n": 10, "counterexamples": []},
        {"form": "contrapositive", "code": 1, "rate": 0.9, "n": 10, "counterexamples": []},
        {"form": "converse", "code": 3, "rate": 0.4, "n": 10, "counterexamples": []},
        {"form": "inverse", "code": 3, "rate": 0.4, "n": 10, "counterexamples": []},
    ], "judgement": {"code": 1}, "meta": {"sha256": "sha123"}}) + "\n", encoding="utf-8")

    out_md = tmp_path / "next_seeds.md"
    out_jsonl = tmp_path / "next_seeds.jsonl"
    code = ns.main(["render-md", str(props_jsonl), "--out", str(out_md), "--jsonl", str(out_jsonl)])
    assert code == 0
    assert out_md.exists()
    assert out_jsonl.exists()
    assert "sha123" in out_md.read_text(encoding="utf-8")
    assert '"reading": 1' in out_jsonl.read_text(encoding="utf-8")


@pytest.mark.parametrize("n,r,p_bb,p_tb,pi_b", [(4, 2, 0.7, 0.3, 0.5), (5, 3, 0.66, 0.34, 0.5), (6, 1, 0.9, 0.1, 0.2), (3, 3, 0.5, 0.5, 0.5)])
def test_markov_run_tail_matches_brute_force(n, r, p_bb, p_tb, pi_b):
    from pythdragoras import markov_run_tail

    def prob(seq):
        pr = pi_b if seq[0] else 1 - pi_b
        for a, b in zip(seq, seq[1:]):
            pb = p_bb if a else p_tb
            pr *= pb if b else 1 - pb
        return pr

    def longest(seq):
        best = cur = 0
        for s in seq:
            cur = cur + 1 if s else 0
            best = max(best, cur)
        return best

    brute = sum(prob(s) for s in product([0, 1], repeat=n) if longest(s) >= r)
    assert markov_run_tail(n, r, p_bb, p_tb, pi_b) == pytest.approx(brute)


def test_markov_with_no_persistence_equals_independent():
    from pythdragoras import markov_run_tail
    assert markov_run_tail(14, 5, 0.5, 0.5, 0.5) == pytest.approx(run_tail(14, 5, 0.5))


def test_transitions_counts():
    from pythdragoras import transitions
    st = _ranks({"d": [4, 5, 6, 2], "g": [1, 2, 4, 1]})
    t = transitions(st)
    assert (t["bb"], t["bt"], t["tb"], t["tt"]) == (2, 2, 1, 1)
    assert t["p_bb"] == pytest.approx(0.5) and t["p_tb"] == pytest.approx(0.5)


def test_reproduces_external_report_markov_figure():
    # 外部レポート（Grok, 2026-10-03）: P(B→B)=0.657 のマルコフ連鎖で「14年連続B」は 0.21%
    from pythdragoras import markov_run_tail
    assert markov_run_tail(14, 14, 0.657, 0.343, 0.5) == pytest.approx(0.0021, abs=0.0001)


def test_persistence_pairs_only_adjacent_seasons_and_reports_missing():
    from pythdragoras import persistence
    st = pl.DataFrame({"team": ["a"] * 4 + ["b"] * 4 + ["c"] * 3,
                       "season": [2012, 2013, 2014, 2016] * 2 + [2012, 2013, 2014],
                       "x": [1.0, 2.0, 3.0, 9.0, 2.0, 1.0, 0.0, 9.0, 5.0, 5.5, 6.0]})
    p = {r["column"]: r for r in persistence(st, ["x", "nope"])}
    # 組: a(12→13,13→14) b(12→13,13→14) c(12→13,13→14) = 6組。2014→2016 は隣り合わないので使わない
    assert p["x"]["n"] == 6
    xs, ys = [1, 2, 2, 1, 5, 5.5], [2, 3, 1, 0, 5.5, 6]
    mx, my = sum(xs) / 6, sum(ys) / 6
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    assert p["x"]["r"] == pytest.approx(r) and p["x"]["ci"][0] < r < p["x"]["ci"][1]
    assert p["nope"]["note"] and p["nope"]["r"] is None


def test_persistence_too_few_pairs():
    from pythdragoras import persistence
    st = pl.DataFrame({"team": ["a", "a"], "season": [2012, 2013], "x": [1.0, 2.0]})
    assert persistence(st, ["x"])[0]["r"] is None


def test_allocation_cumulative_sums_and_z():
    from pythdragoras import allocation_cumulative
    st = pl.DataFrame({"team": ["d", "d", "g"], "team_name": ["中日", "中日", "巨人"],
                       "alloc_net": [-2.0, -1.0, 3.0], "alloc_var": [4.0, 5.0, 9.0],
                       "alloc_net_strat": [-1.0, -1.0, 2.0], "alloc_var_strat": [4.0, 4.0, 4.0]})
    a = {(r["baseline"], r["team"]): r for r in allocation_cumulative(st).iter_rows(named=True)}
    assert a[("全試合", "d")]["alloc_net"] == pytest.approx(-3.0)
    assert a[("全試合", "d")]["z"] == pytest.approx(-3.0 / 3.0)
    assert a[("ホーム/ビジター別", "d")]["z"] == pytest.approx(-2.0 / math.sqrt(8.0))
    assert allocation_cumulative(pl.DataFrame({"team": ["x"]})).height == 0


def test_rank_expectation_poisson_binomial():
    from pythdragoras import rank_expectation
    st = pl.DataFrame({"team": ["d"] * 3 + ["g"] * 2, "team_name": ["中日"] * 3 + ["巨人"] * 2,
                       "season": [2013, 2014, 2015, 2013, 2014], "upper_half": [False, False, True, True, True],
                       "sim_p_upper": [0.5, 0.5, 0.2, 0.9, 0.5]})
    r = {x["team"]: x for x in rank_expectation(st)}
    # d: X = 2つの 0.5 と 1つの 0.2。P(X ≤ 1) = 1 − P(X=2) − P(X=3) = 1 − (0.25·0.8 + 2·0.25·0.2) − 0.25·0.2
    assert r["d"]["expected"] == pytest.approx(1.2) and r["d"]["observed"] == 1
    assert r["d"]["p_le_obs"] == pytest.approx(1 - (0.2 + 0.1) - 0.05)
    assert r["g"]["p_ge_obs"] == pytest.approx(0.45) and r["g"]["p_le_obs"] == pytest.approx(1.0)
    assert rank_expectation(st.drop("sim_p_upper")) == []


def test_persistence_by_team_separates_teams():
    from pythdragoras import persistence_by_team
    rows = []
    for t, sign in (("d", 1), ("g", -1)):
        for i, y in enumerate(range(2013, 2021)):
            rows.append({"team": t, "team_name": t.upper(), "season": y, "x": float(i) * sign})
    r = {x["team"]: x for x in persistence_by_team(pl.DataFrame(rows), ["x"])}
    assert r["d"]["r"] == pytest.approx(1.0) and r["g"]["r"] == pytest.approx(1.0) and r["d"]["n"] == 7


def test_shape_expectation_compares_observed_peaks_with_the_constant_strength_baseline():
    from pythdragoras import TRAJ_FLAGS, shape_expectation
    rows = [("d", 2014, False, True, 0.5), ("d", 2015, False, True, 0.5), ("g", 2014, True, False, 0.2)]
    st = pl.DataFrame([{"team": t, "team_name": t.upper(), "season": s, "upper_half": u,
                        **{f: v for f in TRAJ_FLAGS}, **{f"{f}_base": b for f in TRAJ_FLAGS}} for t, s, u, v, b in rows])
    got = {(r["group"], r["flag"]): r for r in shape_expectation(st)}
    d = got[("D・B クラス", "traj_rank_peak")]
    # 2年とも山があり、力が一定なら山ができる割合は 0.5 ずつ: 期待 1.0、P(2回以上) = 0.25
    assert (d["units"], d["observed"], d["expected"], d["p_ge_obs"], d["p_le_obs"]) == (2, 2, 1.0, 0.25, 1.0)
    whole = got[("全体", "traj_wl_valley")]
    assert (whole["units"], whole["observed"], whole["expected"]) == (3, 2, 1.2)
    assert shape_expectation(st.drop("traj_wl_peak_base")) == []   # 列がなければ何も出さない（落とさずに失敗させない）


def test_trajectory_sensitivity_lines_up_each_start_and_group():
    from pythdragoras import SENS_COLS, trajectory_sensitivity

    def row(team, up, rank, settle, pct, decay, limit, peak, m):
        return {"season": 2024, "team": team, "team_name": team.upper(), "rank": rank, "upper_half": up, "traj_start_games": m,
                "traj_end": "season",
                "wave_settle_x": settle, "wave_settle_pct": pct, "wave_decay": decay, "wave_limit_rank": limit,
                "traj_rank_peak": peak, "traj_rank_peak_base": 0.5, "traj_wl_peak": peak, "traj_wl_peak_base": 0.5}
    st = pl.DataFrame([row("d", False, 5, 0.3, 0.2, 3.0, 5.2, True, 10), row("g", False, 4, None, 0.9, 0.0, None, False, 10),
                       row("t", True, 1, 0.4, 0.6, 2.0, 1.4, True, 10)], schema_overrides={"wave_limit_rank": pl.Float64,
                                                                                           "wave_settle_x": pl.Float64})
    units = st.with_columns(traj_start_games=pl.lit(20), wave_settle_x=pl.lit(0.5))
    units = pl.concat([units, st.with_columns(traj_end=pl.lit("decided"), wave_settle_x=pl.lit(0.7))], how="vertical_relaxed")
    got = {(r["min_games"], r["end"], r["group"]): r for r in trajectory_sensitivity(st, units, "d")}
    d10 = got[(10, "season", "注目チームの B クラス")]
    assert (d10["units"], d10["settle_median"], d10["settle_never"], d10["early_share"]) == (1, 0.3, 0, 1.0)
    assert (d10["converge_share"], d10["near_final_share"]) == (1.0, 1.0)
    assert d10["rank_peak"] == {"observed": 1, "expected": 0.5, "p_ge_obs": 0.5}
    g10 = got[(10, "season", "他の B クラス")]
    assert (g10["settle_median"], g10["settle_never"], g10["converge_share"], g10["near_final_share"]) == (None, 1, 0.0, None)
    assert got[(20, "season", "注目チームの B クラス")]["settle_median"] == 0.5
    assert got[(10, "decided", "注目チームの B クラス")]["settle_median"] == 0.7
    assert trajectory_sensitivity(st.drop(SENS_COLS[-1]), units, "d") == []
