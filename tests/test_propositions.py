"""命題エンジン。docs/propositions.md の規則をそのままテストにする。"""

import math
from itertools import product

import polars as pl
import pytest

import propositions as pr


def table(rows):
    """rows: (team, season, rd, rank, rank_pythag)"""
    return pl.DataFrame([{"team": t, "team_name": t.upper(), "season": s, "league": "C", "rd": rd,
                          "rank": rk, "rank_pythag": rp, "rank_gap": rk - rp}
                         for t, s, rd, rk, rp in rows])


P1 = {"id": "P1", "statement": "得失点差がプラスならAクラス", "strength": "usually", "min_n": 1,
      "if": [{"col": "rd", "op": ">", "value": 0}], "then": [{"col": "rank", "op": "<=", "value": 3}],
      "surprise": "rank_gap", "context": ["rank_pythag"], "links": ["H3"]}


# ---------- 読み込み ----------

@pytest.mark.parametrize("bad,msg", [
    ({"id": "X", "statement": "s", "then": [{"col": "a", "op": ">", "value": 1}]}, "strength"),
    ({"id": "X", "statement": "s", "strength": "sometimes", "then": [{"col": "a", "op": ">", "value": 1}]}, "strength"),
    ({"id": "X", "statement": "s", "strength": "usually", "then": [{"col": "a", "op": "~", "value": 1}]}, "op"),
    ({"id": "X", "statement": "s", "strength": "usually", "then": [{"col": "a", "op": ">"}]}, "col, op, value"),
    ({"id": "X", "statement": "s", "strength": "usually", "then": []}, "then"),
])
def test_validate_rejects_bad_propositions(bad, msg):
    with pytest.raises(pr.PropositionError, match=msg):
        pr.validate(bad)


def test_load_returns_sha_and_rejects_duplicate_ids(tmp_path):
    body = '[[proposition]]\nid="A"\nstatement="s"\nstrength="always"\nthen=[{col="x",op=">",value=0}]\n'
    f = tmp_path / "p.toml"
    f.write_text(body, encoding="utf-8")
    props, sha = pr.load(f)
    assert props[0]["id"] == "A" and len(sha) == 64
    f.write_text(body * 2, encoding="utf-8")
    with pytest.raises(pr.PropositionError, match="一意"):
        pr.load(f)


# ---------- 統計 ----------

def test_wilson_known_values():
    assert pr.wilson(0, 0) == (0.0, 1.0)
    lo, hi = pr.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert pr.wilson(10, 10)[1] == 1.0 and pr.wilson(0, 10)[0] == 0.0


@pytest.mark.parametrize("a,b,c,d", [(3, 1, 1, 3), (5, 0, 0, 5), (0, 4, 4, 0), (2, 3, 4, 1), (0, 0, 3, 3), (6, 2, 3, 9)])
def test_fisher_matches_enumeration(a, b, c, d):
    r1, c1, n = a + b, a + c, a + b + c + d
    if r1 == 0 or c1 == 0:
        assert pr.fisher_greater(a, b, c, d) == 1.0
        return
    total = math.comb(n, c1)
    expect = sum(math.comb(r1, i) * math.comb(n - r1, c1 - i) for i in range(0, min(r1, c1) + 1) if i >= a) / total
    assert pr.fisher_greater(a, b, c, d) == pytest.approx(expect)
    assert pr.fisher_greater(3, 1, 1, 3) == pytest.approx(17 / 70)


# ---------- 判定表（docs の上から順） ----------

@pytest.mark.parametrize("args,expected", [
    (("statistical", 0.75, 5, 5, (0.9, 1.0), 0.0, 10, 0.05), "Inconclusive"),   # 1: n < min_n
    (("universal", None, 10, 10, (0.7, 1.0), 0.0, 10, 0.05), "Supported"),      # 2
    (("universal", None, 10, 9, (0.6, 0.99), 0.0, 10, 0.05), "Rejected"),       # 3: 反例1件で棄却
    (("statistical", 0.75, 50, 45, (0.75, 0.95), 0.0, 10, 0.05), "Supported"),  # 4: 下限ちょうど
    (("statistical", 0.75, 50, 38, (0.62, 0.75), 0.0, 10, 0.05), "Inconclusive"),  # 5: 上限ちょうど
    (("statistical", 0.75, 50, 30, (0.46, 0.73), 0.01, 10, 0.05), "Refined"),   # 6
    (("statistical", 0.75, 50, 30, (0.46, 0.73), 0.20, 10, 0.05), "Rejected"),  # 7
    (("statistical", 0.75, 50, 30, (0.46, 0.73), 0.05, 10, 0.05), "Rejected"),  # p == alpha は有意でない
])
def test_verdict_table(args, expected):
    assert pr.verdict(*args) == expected


# ---------- 評価 ----------

ROWS = [("d", 2019, 19, 5, 2), ("d", 2021, -30, 5, 5), ("g", 2019, 50, 1, 1), ("g", 2021, 40, 2, 1),
        ("t", 2019, -5, 3, 4), ("t", 2021, 10, 4, 2), ("c", 2019, -20, 6, 6), ("c", 2021, 5, 1, 3)]


def test_four_forms_counts_and_counterexamples():
    r = pr.evaluate(P1, table(ROWS), focus="d")
    f = {x["form"]: x for x in r["forms"]}
    # rd>0: d19 g19 g21 t21 c21 の5件、そのうち rank<=3 は g19 g21 c21 の3件
    assert (f["original"]["n"], f["original"]["hold"]) == (5, 3)
    assert [c["unit"] for c in f["original"]["counterexamples"]] == ["d-2019", "t-2021"]  # rank_gap 3 > 2 の順
    assert f["original"]["counterexamples"][0]["focus"] is True
    assert {c["unit"] for c in f["contrapositive"]["counterexamples"]} == {"d-2019", "t-2021"}
    # 逆: rank<=3 は g19 g21 t19 c21 の4件。rd>0 でないのは t19
    assert (f["converse"]["n"], f["converse"]["hold"]) == (4, 3)
    assert [c["unit"] for c in f["converse"]["counterexamples"]] == ["t-2019"]
    assert {c["unit"] for c in f["inverse"]["counterexamples"]} == {"t-2019"}
    assert r["objection"] is True
    c = f["original"]["counterexamples"][0]
    assert c["values"] == {"rd": 19, "rank": 5} and c["context"] == {"rank_pythag": 2} and c["surprise"] == 3
    assert "rd > 0" in c["question"] and "rank <= 3" in c["question"] and c["links"] == ["H3"]


def test_contrapositive_invariant_holds_on_random_tables():
    import random
    rnd = random.Random(0)
    for _ in range(50):
        rows = [(t, s, rnd.randint(-30, 30), rnd.randint(1, 6), rnd.randint(1, 6))
                for t, s in product("abcdef", range(2012, 2016))]
        r = pr.evaluate(P1, table(rows))  # 不一致なら evaluate が AssertionError を出す
        f = {x["form"]: x for x in r["forms"]}
        assert f["original"]["n"] - f["original"]["hold"] == len(f["original"]["counterexamples"])


def test_empty_if_means_every_unit_and_has_no_converse():
    p = {**P1, "if": []}
    r = pr.evaluate(p, table(ROWS))
    assert [f["form"] for f in r["forms"]] == ["original", "contrapositive"]
    assert r["forms"][0]["n"] == len(ROWS)


def test_missing_column_is_an_error():
    with pytest.raises(pr.PropositionError, match="存在しない列"):
        pr.evaluate({**P1, "then": [{"col": "nope", "op": "<=", "value": 3}]}, table(ROWS))
    with pytest.raises(pr.PropositionError, match="surprise"):
        pr.evaluate({**P1, "surprise": "nope"}, table(ROWS))


def test_null_values_are_counted_not_dropped():
    st = table(ROWS).with_columns(rd=pl.when(pl.col("team") == "c").then(None).otherwise(pl.col("rd")))
    f = pr.evaluate(P1, st)["forms"][0]
    assert f["undetermined"] == 2 and f["n"] == 4


def test_scope_filter():
    r = pr.evaluate({**P1, "scope": {"seasons": "2021"}}, table(ROWS))
    assert r["units"] == 4
    assert pr.evaluate({**P1, "scope": {"team": "d"}}, table(ROWS))["units"] == 2


def test_excluded_units_keep_their_counterexamples():
    excluded = table([("d", 2020, -60, 3, 6), ("g", 2020, 30, 5, 1)])
    r = pr.evaluate(P1, table(ROWS), excluded, focus="d")
    assert [c["unit"] for c in r["excluded_counterexamples"]] == ["g-2020"]
    # 統計には入らない
    assert r["forms"][0]["n"] == 5


def test_result_carries_objection_and_excluded_structure():
    excluded = table([("g", 2020, 30, 5, 1)])
    r = pr.evaluate(P1, table(ROWS), excluded, focus="d")
    assert r["objection"] is True
    assert [c["unit"] for c in r["excluded_counterexamples"]] == ["g-2020"]
    assert r["judgement"]["code"] != 0


def test_every_exit_code_has_both_languages_and_render_smoke():
    for code, (ja, en) in pr.EXIT.items():
        assert ja.strip() and en.strip() and pr.label(code, "ja") == ja and pr.label(code, "en") == en
    r = pr.evaluate(P1, table(ROWS), table([("g", 2020, 30, 5, 1)]), focus="d")
    for lang in ("ja", "en"):  # 表示は落ちずに作れること、判定のコードの言葉が入ること
        md = pr.render([r], {"sha256": "x"}, lang=lang)
        assert pr.label(r["judgement"]["code"], lang) in md


# ---------- 外部の主張 ----------

def test_check_claims_statuses():
    st = table(ROWS).with_columns(wins_vs_pythag=pl.Series([1.5, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, 1.0]))
    claims = [
        {"id": "C1", "statement": "d の合計", "kind": "sum", "column": "wins_vs_pythag", "value": -0.5, "tolerance": 0.01,
         "scope": {"team": "d"}},
        {"id": "C2", "statement": "外れ", "kind": "sum", "column": "wins_vs_pythag", "value": 99, "tolerance": 1,
         "scope": {"team": "d"}},
        {"id": "C3", "statement": "範囲外", "kind": "sum", "column": "wins_vs_pythag", "value": 1, "scope": {"seasons": "1990"}},
        {"id": "C4", "statement": "運が悪かったわけではない", "kind": "interpretation"},
        {"id": "C5", "statement": "列なし", "kind": "mean", "column": "nope", "value": 1},
        {"id": "C6", "statement": "件数", "kind": "count", "filter": [{"col": "rank", "op": "<=", "value": 3}], "value": 4},
    ]
    got = {c["id"]: c["status"] for c in pr.check_claims(claims, st)}
    assert got == {"C1": "reproduced", "C2": "not-reproduced", "C3": "not-measurable",
                   "C4": "interpretation", "C5": "not-measurable", "C6": "reproduced"}


def test_cycle1_files_reference_real_columns():
    import tomllib
    from pathlib import Path

    from sakanalytics import add_persistence, batting_join, inning_decomposition, season_table, to_team_games

    import aoi_paths
    root = aoi_paths.resolve_cycle_path(aoi_paths.resolve_aoi_root(anchor=Path(__file__)))
    teams = tomllib.loads((root / "analysis.toml").read_text(encoding="utf-8"))["teams"]
    games = pl.DataFrame([{"key": "1", "date": "2024-04-01", "home": "d", "away": "g", "hs": 3, "as": 2},
                          {"key": "2", "date": "2024-04-02", "home": "t", "away": "d", "hs": 1, "as": 0}])
    st = season_table(to_team_games(games, teams))
    # パイプラインと同じく --innings の列も作る（実際の CSV と同じ venue・window・率の列をそろえる）
    innings = pl.DataFrame([{"season": 2024, "team": "x", "venue": v, "window": w, "games": 0, "innings": 2, "runs": 0,
                             "scoring_innings": 1, "single_run_inning_rate": 0.5, "big_inning_rate": 0.0}
                            for v in ("all", "home", "away") for w in ("all", "first6")],
                           schema_overrides={"season": pl.Int32})
    bat = pl.DataFrame([{"season": 2024, "team": "x", **{k: 1 for k in ("g", "r", "pa", "ab", "h", "b2", "b3", "hr", "tb", "bb",
                                                                          "hbp", "sf", "so", "ibb")}, "avg": 1.0, "slg": 1.0, "obp": 0.75}])
    cols = set(add_persistence(batting_join(inning_decomposition(st, innings), bat)).columns)
    props, _ = pr.load(root / "propositions.toml")
    for p in props:
        for c in [*p.get("if", []), *p["then"], *(p.get("scope") or {}).get("where", [])]:
            assert c["col"] in cols, (p["id"], c["col"])
        for c in [p.get("surprise"), *p.get("context", [])]:
            assert c is None or c in cols, (p["id"], c)
    claims = tomllib.loads((root / "claims.toml").read_text(encoding="utf-8"))["claim"]
    for c in claims:
        for f in c.get("filter", []):
            assert f["col"] in cols, (c["id"], f["col"])
        assert c.get("column") is None or c["column"] in cols, c["id"]
        for x in [c.get("num"), *c.get("den", [])]:
            assert x is None or x in cols, (c["id"], x)
        assert c["kind"] in ("sum", "mean", "count", "ratio", "interpretation")


def test_claim_premise_not_met_is_not_reproduced_but_missing_data_is_not_measurable():
    st = table(ROWS).with_columns(wins_vs_pythag=pl.lit(0.0))
    claims = [
        {"id": "A", "statement": "2019 d はピタゴラス2位", "kind": "count", "scope": {"team": "d", "seasons": "2019"},
         "filter": [{"col": "rank_pythag", "op": "==", "value": 9}], "value": 1},
        {"id": "B", "statement": "条件つき合計", "kind": "sum", "column": "wins_vs_pythag", "scope": {"team": "d"},
         "filter": [{"col": "rd", "op": "==", "value": -60}], "value": 9.4},
        {"id": "C", "statement": "範囲外", "kind": "count", "scope": {"seasons": "1958"}, "value": 1},
    ]
    got = {c["id"]: c for c in pr.check_claims(claims, st)}
    assert got["A"]["status"] == "not-reproduced" and got["A"]["computed"] == 0
    assert got["B"]["status"] == "not-reproduced" and "条件" in got["B"]["note"]
    assert got["C"]["status"] == "not-measurable"


def test_claim_table_escapes_pipes():
    md = pr.render_claims([{"id": "X", "statement": "P(B | A)=0.6", "source": "s", "date": "d", "status": "interpretation"}])
    assert "P(B \\| A)" in md


@pytest.mark.parametrize("skip,reason,ok", [
    (["converse", "inverse"], "主語の選択", True),
    (["converse", "inverse"], "", False),
    (["converse"], "x", False),
    (["original", "contrapositive"], "x", False),
])
def test_skip_forms_rules(skip, reason, ok):
    p = {**P1, "skip_forms": skip, "skip_reason": reason}
    if ok:
        pr.validate(p)
        r = pr.evaluate(p, table(ROWS))
        assert [f["form"] for f in r["forms"]] == ["original", "contrapositive"]
        assert r["skipped_forms"] == ["converse", "inverse"] and r["skip_reason"] == "主語の選択"
    else:
        with pytest.raises(pr.PropositionError):
            pr.validate(p)


# ---------- 系譜・きっかけ以外での判定・台帳 ----------

P1A = {**P1, "id": "P1a", "statement": "得失点差プラスかつピタゴラス3位以内ならAクラス",
       "if": [{"col": "rd", "op": ">", "value": 0}, {"col": "rank_pythag", "op": "<=", "value": 3}],
       "parent": "P1", "change": "d-2019 を見てピタゴラス順位の条件を足した", "motivated_by": ["d-2019"]}


def _toml(props):
    import json as _j

    def cond(cs):
        return "[" + ", ".join(f'{{col="{c["col"]}", op="{c["op"]}", value={_j.dumps(c["value"])}}}' for c in cs) + "]"
    out = []
    for p in props:
        out.append("[[proposition]]")
        for k in ("id", "statement", "strength", "parent", "change"):
            if p.get(k):
                out.append(f'{k} = "{p[k]}"')
        out.append(f"if = {cond(p.get('if', []))}")
        out.append(f"then = {cond(p['then'])}")
        if p.get("motivated_by"):
            out.append(f"motivated_by = {_j.dumps(p['motivated_by'])}")
    return "\n".join(out) + "\n"


def test_parent_must_come_first_and_needs_change(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text(_toml([P1, P1A]), encoding="utf-8")
    assert [p["id"] for p in pr.load(f)[0]] == ["P1", "P1a"]
    f.write_text(_toml([P1A, P1]), encoding="utf-8")
    with pytest.raises(pr.PropositionError, match="前の方"):
        pr.load(f)
    with pytest.raises(pr.PropositionError, match="change"):
        pr.validate({**P1A, "change": ""})
    pr.validate({**P1, "motivated_by": ["d-2019"]})  # 作り直しでない命題も、思いついたときに見た単位を書ける
    with pytest.raises(pr.PropositionError, match="team-season"):
        pr.validate({**P1A, "motivated_by": ["2019"]})


def test_held_out_also_applies_to_a_new_proposition_built_from_seen_units():
    p = {**P1, "motivated_by": ["d-2019"]}
    r = pr.evaluate(p, table(ROWS), focus="d")
    assert r["held_out"]["excluded_units"] == ["d-2019"] and r["parent"] is None
    assert r["held_out"]["n"] == r["forms"][0]["n"] - 1  # d-2019 は rd > 0 で前件に当たる


def test_held_out_excludes_the_motivating_units():
    r = pr.evaluate(P1A, table(ROWS), focus="d")
    orig = r["forms"][0]
    h = r["held_out"]
    # rd>0 かつ rank_pythag<=3: d19 g19 g21 t21 c21 → 5件、判例は d19（5位）と t21（4位）
    assert (orig["n"], [c["unit"] for c in orig["counterexamples"]]) == (5, ["d-2019", "t-2021"])
    # きっかけ d-2019 を除くと 4件、判例は t-2021 だけ残る = 作り直しは他の単位ではまだ異議を受けている
    assert (h["n"], h["hold"], h["counterexamples"]) == (4, 3, ["t-2021"])
    assert h["excluded_units"] == ["d-2019"]
    assert (r["parent"], r["change"]) == ("P1", P1A["change"])
    assert r["judgement"]["form"] == "original"  # 元の命題で既に判例がある


def test_definition_sha_tracks_meaning_not_annotations():
    base = pr.definition_sha(P1)
    assert pr.definition_sha({**P1, "links": ["H9"], "context": []}) == base
    assert pr.definition_sha({**P1, "then": [{"col": "rank", "op": "<=", "value": 2}]}) != base
    assert pr.definition_sha({**P1, "strength": "almost_always"}) != base


def test_ledger_appends_once_per_data_version_and_counts_streak(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with_cx = [pr.evaluate(P1, table(ROWS))]
    clean = [pr.evaluate(P1, table([r for r in ROWS if r[:2] not in {("d", 2019), ("t", 2021)}]))]
    assert clean[0]["forms"][0]["counterexamples"] == []
    pr.update_ledger(path, with_cx, "data-v1", None)
    pr.update_ledger(path, with_cx, "data-v1", None)          # 同じデータは二度書かない
    pr.update_ledger(path, clean, "data-v2", None)
    rows = pr.update_ledger(path, clean, "data-v3", "abc")
    s = pr.ledger_summary(rows, "P1")
    # 逆には t-2019 の判例が残り続けるので「どれかの形」は3回とも異議あり
    assert s == {"evaluations": 3, "objections": 1, "objections_any_form": 3, "no_objection_streak": 2, "definitions": 1}


def test_ledger_flags_redefinition_under_the_same_id(tmp_path):
    path = tmp_path / "ledger.jsonl"
    pr.update_ledger(path, [pr.evaluate(P1, table(ROWS))], "v1", None)
    changed = {**P1, "strength": "more_often_than_not"}  # 結果を見てから基準を弱めた
    rows = pr.update_ledger(path, [pr.evaluate(changed, table(ROWS))], "v1", None)
    assert pr.ledger_summary(rows, "P1")["definitions"] == 2
    assert pr.judge(pr.evaluate(changed, table(ROWS)), 2)["code"] == 6


# ---------- 終了コード ----------

@pytest.mark.parametrize("verdict_,cx,und,code", [
    ("Supported", 0, 0, 0), ("Supported", 0, 2, 5), ("Supported", 3, 0, 1), ("Refined", 3, 0, 2),
    ("Rejected", 3, 0, 3), ("Inconclusive", 0, 0, 4), ("Inconclusive", 3, 0, 4),
])
def test_form_code(verdict_, cx, und, code):
    assert pr.form_code(verdict_, cx, und) == code


def test_all_hold_but_too_few_units_is_inconclusive_not_zero():
    few = [(t, s, 10 if i < 3 else -10, i + 1, i + 1) for s in (2012, 2013) for i, t in enumerate("abcdef")]
    j = pr.evaluate({**P1, "min_n": 1}, table(few))["judgement"]
    assert (j["code"], j["form"]) == (4, "original")  # 6件全部成立でも「概ね」は証明できない


def test_form_code_rejects_impossible_combination():
    with pytest.raises(AssertionError):
        pr.form_code("Rejected", 0, 0)


def _clean_rows():
    # rd>0 ⇔ rank<=3 が完全に成り立つ表（24単位、各形 n=12）。逆・裏まで判例なし。
    # 全件成立でも Wilson 下限が 0.75 に届くには n>=12 が要る（n/(n+z²) >= 0.75）
    return [(t, s, 10 if i < 3 else -10, i + 1, i + 1) for s in range(2012, 2016) for i, t in enumerate("abcdef")]


def test_stages_confirmed_only_after_converse_and_inverse():
    r = pr.evaluate({**P1, "min_n": 1}, table(_clean_rows()))
    j = r["judgement"]
    assert (j["code"], j["stage"], j["form"]) == (0, "confirmed", None)
    sk = pr.evaluate({**P1, "min_n": 1, "skip_forms": ["converse", "inverse"], "skip_reason": "r"}, table(_clean_rows()))
    assert (sk["judgement"]["code"], sk["judgement"]["stage"]) == (0, "provisional")


def _many_clean_rows():
    return [(t, s, 10 if i < 3 else -10, i + 1, i + 1) for s in range(2000, 2016) for i, t in enumerate("abcdef")]


@pytest.mark.parametrize("rows,code", [
    (lambda: _clean_rows(), 4),        # 13件中12件: 判例はあるが件数が足りず保留
    (lambda: _many_clean_rows(), 1),   # 49件中48件: 「概ね」の範囲の例外
])
def test_stage_provisional_when_converse_objects(rows, code):
    data = rows() + [("g", 2016, -5, 2, 2)]  # Aクラスなのに得失点差マイナス = 逆の判例
    j = pr.evaluate({**P1, "min_n": 1}, table(data))["judgement"]
    assert (j["code"], j["stage"], j["form"]) == (code, "provisional", "converse")
    assert j["units"] == ["g-2016"]


def test_stage_none_when_original_objects():
    j = pr.evaluate(P1, table(ROWS))["judgement"]
    assert (j["stage"], j["form"]) == ("none", "original") and j["code"] in (1, 2, 3, 4)


def test_held_out_is_checked_right_after_original():
    rows = _clean_rows() + [("d", 2019, 19, 5, 2)]
    p = {**P1, "min_n": 1, "parent": "P0", "change": "c", "motivated_by": ["d-2019"]}
    j = pr.evaluate(p, table(rows))["judgement"]
    assert j["form"] == "original"  # 元の命題では d-2019 が判例
    r = pr.evaluate(p, table(rows))
    assert r["held_out"]["counterexamples"] == []  # きっかけを除けば判例なし


def _write_results(tmp_path, results):
    path = tmp_path / "propositions.jsonl"
    path.write_text("".join(__import__("json").dumps(r, ensure_ascii=False, default=str) + "\n" for r in results),
                    encoding="utf-8")
    return path


def test_judge_cli_exit_codes(tmp_path):
    clean = {**pr.evaluate({**P1, "min_n": 1}, table(_clean_rows())), "id": "OK"}
    bad = pr.evaluate(P1, table(ROWS))
    path = _write_results(tmp_path, [clean, bad])
    assert pr.main(["judge", str(path), "--id", "OK"]) == 0
    assert pr.main(["judge", str(path)]) == bad["judgement"]["code"] != 0
    assert pr.main(["judge", str(path), "--report-only"]) == 0     # 異議は失敗にしない
    assert pr.main(["judge", str(path), "--id", "NOPE"]) == 64       # 仕組みの不具合は失敗
    assert pr.main(["judge", str(tmp_path / "missing.jsonl")]) == 66
    assert pr.main(["judge", str(tmp_path / "missing.jsonl"), "--report-only"]) == 66


def test_judge_stops_at_first_nonzero_unless_keep_going(tmp_path):
    bad = pr.evaluate(P1, table(ROWS))
    clean = {**pr.evaluate({**P1, "min_n": 1}, table(_clean_rows())), "id": "OK"}
    lines = []
    assert pr.run_judge([bad, clean], None, None, False, "ja", out=lines.append) != 0
    assert len(lines) == 1  # && でつないだときと同じく、最初の失敗で止まる
    lines.clear()
    pr.run_judge([bad, clean], None, None, True, "en", out=lines.append)
    assert [line.split("]")[0] for line in lines][1] == "[0"  # 2件目は OK（コードだけを見る）
    assert len(lines) == 2


def test_judge_flags_redefinition_from_ledger(tmp_path):
    clean = {**pr.evaluate({**P1, "min_n": 1}, table(_clean_rows())), "id": "OK"}
    path = _write_results(tmp_path, [clean])
    ledger = [{"id": "OK", "definition_sha256": d, "data_sha256": "x", "objection": False, "forms": {}} for d in ("a", "b")]
    (tmp_path / "ledger.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in ledger))
    assert pr.main(["judge", str(path)]) == 6


def test_pipeline_cli_maps_system_errors(tmp_path):
    from season import main as pd_main
    season = tmp_path / "season.jsonl"
    pl.DataFrame({"season": [2024], "team": ["d"], "team_name": ["中日"], "league": ["C"], "W": [1], "L": [1],
                  "rank": [1], "rank_tie": [False], "league_size": [1], "pythag_fixed": [0.5], "pythag_var": [0.5]}
                 ).write_ndjson(season)
    cfg = tmp_path / "a.toml"
    cfg.write_text("[focus]\nteam='d'\n", encoding="utf-8")
    props = tmp_path / "p.toml"
    props.write_text('[[proposition]]\nid="X"\nstatement="s"\nstrength="usually"\nthen=[{col="nope",op=">",value=0}]\n',
                     encoding="utf-8")
    assert pd_main(["--season", str(season), "--config", str(cfg), "--propositions", str(props), "--outdir", str(tmp_path / "o")]) == 65
    props.write_text('[[proposition]]\nid="X"\nstatement="s"\nstrength="sometimes"\nthen=[{col="W",op=">",value=0}]\n',
                     encoding="utf-8")
    assert pd_main(["--season", str(season), "--config", str(cfg), "--propositions", str(props), "--outdir", str(tmp_path / "o")]) == 64


def test_ratio_claims_and_uncomputable_values():
    st = table(ROWS).with_columns(w_1=pl.Series([3, 1, 2, 0, 1, 2, 0, 1]), l_1=pl.Series([1, 3, 0, 2, 1, 2, 0, 1]))
    claims = [
        {"id": "R1", "statement": "d 通算の1点差勝率", "kind": "ratio", "num": "w_1", "den": ["w_1", "l_1"],
         "scope": {"team": "d"}, "value": 0.5, "tolerance": 0.001},   # (3+1)/(3+1+1+3)
        {"id": "R2", "statement": "分母0", "kind": "ratio", "num": "w_1", "den": ["w_1", "l_1"],
         "scope": {"team": "c", "seasons": "2019"}, "value": 0.5},
        {"id": "R3", "statement": "列なし", "kind": "ratio", "num": "nope", "den": ["l_1"], "value": 0.5},
    ]
    got = {c["id"]: c for c in pr.check_claims(claims, st)}
    assert (got["R1"]["status"], got["R1"]["computed"]) == ("reproduced", 0.5)
    assert got["R2"]["status"] == "not-measurable" and got["R3"]["status"] == "not-measurable"


@pytest.mark.parametrize("field", ["falsifier", "note"])
def test_falsifier_and_note_must_be_text_and_do_not_change_definition(field):
    with pytest.raises(pr.PropositionError):
        pr.validate({**P1, field: "  "})
    pr.validate({**P1, field: "x"})
    assert pr.definition_sha({**P1, field: "x"}) == pr.definition_sha(P1)  # 注記は定義（事前登録）を変えない
    assert pr.evaluate({**P1, field: "x"}, table(ROWS))[field] == "x"


# ---------- 範囲の条件（scope.where） ----------

def _where_table():
    # 列 low: 範囲の条件。low=True の中では rd>0 ⇒ rank<=3 が1件だけ破れる。範囲の外には判例を置く
    rows = [("a", 2020, 5, 1, 1, True), ("b", 2020, 3, 2, 2, True), ("c", 2020, 4, 5, 2, True),
            ("d", 2020, -2, 6, 6, True), ("e", 2020, 9, 6, 1, False), ("f", 2020, 1, 4, 3, None)]
    return pl.DataFrame([{"team": t, "team_name": t.upper(), "season": s, "league": "C", "rd": rd, "rank": rk,
                          "rank_pythag": rp, "rank_gap": rk - rp, "low": low} for t, s, rd, rk, rp, low in rows])


def test_where_narrows_units_and_keeps_converse_inside_the_scope():
    p = {**P1, "scope": {"where": [{"col": "low", "op": "==", "value": True}]}}
    r = pr.evaluate(p, _where_table())
    by = {f["form"]: f for f in r["forms"]}
    assert r["units"] == 4                                    # e（範囲外）と f（値が空）は入らない
    assert [c["unit"] for c in by["original"]["counterexamples"]] == ["c-2020"]   # e-2020 は範囲外なので判例にならない
    assert (by["converse"]["n"], by["converse"]["hold"]) == (2, 2)                # 逆も範囲の中だけで数える


def test_where_with_missing_value_is_undetermined_not_silently_outside():
    p = {**P1, "scope": {"where": [{"col": "low", "op": "==", "value": True}]}}
    r = pr.evaluate(p, _where_table())
    assert {f["undetermined"] for f in r["forms"]} == {1}


def test_where_condition_is_validated_and_changes_the_definition():
    bad = {**P1, "scope": {"where": [{"col": "low", "op": "~", "value": True}]}}
    with pytest.raises(pr.PropositionError, match="op"):
        pr.validate(bad)
    with pytest.raises(pr.PropositionError, match="scope"):
        pr.validate({**P1, "scope": {"filter": []}})
    with pytest.raises(pr.DataError, match="存在しない列"):
        pr.evaluate({**P1, "scope": {"where": [{"col": "nope", "op": "==", "value": 1}]}}, _where_table())
    assert pr.definition_sha({**P1, "scope": {"where": [{"col": "low", "op": "==", "value": True}]}}) != pr.definition_sha(P1)


# ---------- 識別子（同じ問いの重複を止める） ----------

def _blocks(*blocks):
    return "\n".join("[[proposition]]\n" + b for b in blocks)


A = 'id="A"\nstatement="s"\nstrength="usually"\nif=[{col="x",op=">",value=0},{col="y",op="<",value=1}]\nthen=[{col="z",op="==",value=true}]\n'


def test_same_question_under_another_number_is_stopped(tmp_path):
    # 文言・強さ・条件の並び順が違っても、範囲・条件・結論が同じなら同じ問い
    b = 'id="B"\nstatement="別の言い方"\nstrength="always"\nif=[{col="y",op="<",value=1},{col="x",op=">",value=0}]\nthen=[{col="z",op="==",value=true}]\n'
    f = tmp_path / "p.toml"
    f.write_text(_blocks(A, b), encoding="utf-8")
    with pytest.raises(pr.PropositionError, match="A と同じ問い"):
        pr.load(f)


def test_restatement_with_parent_may_share_the_question(tmp_path):
    b = 'id="B"\nparent="A"\nchange="強さを下げた"\nstatement="s"\nstrength="more_often_than_not"\nif=[{col="x",op=">",value=0},{col="y",op="<",value=1}]\nthen=[{col="z",op="==",value=true}]\n'
    c = 'id="C"\nparent="B"\nchange="さらに注記を変えた"\nstatement="s"\nstrength="more_often_than_not"\nif=[{col="x",op=">",value=0},{col="y",op="<",value=1}]\nthen=[{col="z",op="==",value=true}]\n'
    f = tmp_path / "p.toml"
    f.write_text(_blocks(A, b, c), encoding="utf-8")
    props, _ = pr.load(f)  # 祖先（A）と同じ問いでも通る
    assert [p["id"] for p in props] == ["A", "B", "C"]


def test_identity_key_is_readable_and_scope_matters():
    a = {"if": [{"col": "y", "op": "<", "value": 1}, {"col": "x", "op": ">", "value": 0}],
         "then": [{"col": "z", "op": "==", "value": True}], "scope": {"team": "d", "where": [{"col": "rank_rf", "op": ">=", "value": 5}]}}
    i = pr.identity(a)
    assert i["key"] == '[team=d, where:rank_rf>=5] x>0 & y<1 => z==true'
    assert pr.identity({**a, "scope": {"team": "g"}})["signature"] != i["signature"]
    sib = {**a, "if": [{"col": "w", "op": ">", "value": 0}]}
    assert pr.identity(sib)["family"] == i["family"] and pr.identity(sib)["signature"] != i["signature"]


def test_render_lists_siblings():
    st = table([("a", 2020, 5, 1, 1), ("b", 2020, -3, 5, 4), ("c", 2020, 2, 2, 3)])
    p2 = {**P1, "id": "P2", "if": [{"col": "rank_pythag", "op": "<=", "value": 3}]}
    md = pr.render([pr.evaluate(P1, st), pr.evaluate(p2, st)], {})
    assert md.count("兄弟（範囲と結論が同じ、条件が違う）") == 2


# ---------- 「または」（if_any） ----------

def test_if_any_is_or_of_and_groups_and_all_forms_follow():
    # rows: (team, season, rd, rank, rank_pythag)。前件 = rd > 0 または rank_pythag <= 2
    st = table([("a", 2020, 5, 1, 4), ("b", 2020, -3, 2, 2), ("c", 2020, -1, 5, 5), ("d", 2020, 2, 6, 6), ("e", 2020, -9, 3, 6)])
    p = {**P1, "if": [], "if_any": [[{"col": "rd", "op": ">", "value": 0}], [{"col": "rank_pythag", "op": "<=", "value": 2}]]}
    r = pr.evaluate(p, st)
    by = {f["form"]: f for f in r["forms"]}
    assert (by["original"]["n"], by["original"]["hold"]) == (3, 2)                 # a, b, d が前件。d は 6位
    assert [c["unit"] for c in by["original"]["counterexamples"]] == ["d-2020"]
    assert [c["unit"] for c in by["contrapositive"]["counterexamples"]] == ["d-2020"]
    assert [c["unit"] for c in by["converse"]["counterexamples"]] == ["e-2020"]     # 3位なのに前件を満たさない
    assert "または" in by["original"]["counterexamples"][0]["question"]


def test_if_any_combines_with_if_by_and_and_appears_in_identity():
    p = {**P1, "if": [{"col": "rank", "op": "<=", "value": 6}],
         "if_any": [[{"col": "rd", "op": ">", "value": 0}, {"col": "rank_gap", "op": "<", "value": 1}], [{"col": "rank_pythag", "op": "<=", "value": 2}]]}
    assert pr.identity(p)["key"] == "[all] rank<=6 & ((rank_gap<1 & rd>0) | (rank_pythag<=2)) => rank<=3"
    assert pr.identity(p)["signature"] != pr.identity({**p, "if_any": p["if_any"][:1]})["signature"]
    assert pr.definition_sha(p) != pr.definition_sha({**p, "if_any": []})
    with pytest.raises(pr.PropositionError, match="if_any"):
        pr.validate({**P1, "if_any": [[]]})


def test_render_index_lists_every_proposition_with_form_codes():
    st = table([("a", 2020, 5, 1, 1), ("b", 2020, -3, 5, 4), ("c", 2020, 2, 2, 3), ("d", 2020, 4, 4, 2)])
    p2 = {**P1, "id": "P2", "if": [{"col": "rank_pythag", "op": "<=", "value": 3}], "parent": "P1", "change": "条件を変えた"}
    rs = [pr.evaluate(P1, st), pr.evaluate(p2, st)]
    for r in rs:
        r["judgement"] = pr.judge(r)
    md = pr.render_index(rs, {"sha256": "x"})
    rows = [l for l in md.splitlines() if l.startswith("| P")]
    assert [l.split("|")[1].strip() for l in rows] == ["P1", "P2"]
    p1 = {f["form"]: f for f in rs[0]["forms"]}
    assert f"{p1['original']['code']}・{p1['original']['rate']:.2f}（{p1['original']['n']}）" in rows[0]
    assert rows[1].rstrip(" |").endswith("P1")                     # 親
    assert "総合の終了コードの内訳" in md


def test_confirm_units_reports_each_new_unit_in_all_four_cells():
    # rd > 0 ⇒ rank <= 3。a: 成立、b: 判例（元・対偶）、c: 逆の判例（逆・裏）、e: どちらも当たらない。2020 は確かめに使わない
    st = table([("a", 2020, 5, 1, 1), ("b", 2020, -3, 5, 4), ("a", 2026, 4, 2, 1), ("b", 2026, 3, 5, 2),
                ("c", 2026, -1, 3, 5), ("e", 2026, -2, 6, 6)])
    # 年の範囲は作ったときのデータの範囲なので外す（外さないと 2026 が必ず範囲外になる）
    p = {**P1, "scope": {"seasons": "2013-2025"}}
    c = pr.confirm_units(p, st, [2026])
    assert c["units"] == [{"unit": "a-2026", "status": "成立"}, {"unit": "b-2026", "status": "判例"},
                          {"unit": "c-2026", "status": "逆の判例"}, {"unit": "e-2026", "status": "どちらも当たらない"}]
    units = ["a-2026", "b-2026", "c-2026", "e-2026"]
    md = pr.render_confirmation([c], [2026], "2026-10-04", units, detail=["P1"])
    cells = [x.strip() for x in [l for l in md.splitlines() if l.startswith("| P1 ")][0].split("|")]
    assert cells[3:9] == ["4", "1", "b-2026", "c-2026", "1", "0"]
    assert "| c-2026 | 逆の判例 |" in md
    assert "元の命題・対偶の判例が出たもの 1 件、逆・裏の判例が出たもの 1 件" in md
    assert pr.confirm_units(P1, st, [2027])["units"] == []
    assert "まだない" in pr.render_confirmation([pr.confirm_units(P1, st, [2027])], [2027], "-", [])


def test_confirm_units_keeps_team_and_where_scope_and_reports_unknown_values():
    st = table([("a", 2026, 4, 2, 1), ("b", 2026, 3, 5, 2), ("c", 2026, -1, 3, 5)])
    st = st.with_columns(w=pl.Series([1.0, None, 0.0]))
    p = {**P1, "scope": {"seasons": "2013-2025", "where": [{"col": "w", "op": ">", "value": 0.5}]}}
    c = pr.confirm_units(p, st, [2026])
    assert c["units"] == [{"unit": "a-2026", "status": "成立"}, {"unit": "b-2026", "status": "判定不能"}]
    d = pr.confirm_units({**P1, "if": [], "scope": {"team": "c"}}, st, [2026])
    assert d["units"] == [{"unit": "c-2026", "status": "成立"}] and not d["has_if"]
    md = pr.render_confirmation([d], [2026], "-", ["a-2026", "b-2026", "c-2026"], detail=["P1"])
    assert "（前件なし）" in md and "| a-2026 | 範囲外 |" in md


def test_cycle1_confirmation_setting_names_real_propositions():
    import tomllib
    from pathlib import Path

    import aoi_paths
    root = aoi_paths.resolve_cycle_path(aoi_paths.resolve_aoi_root(anchor=Path(__file__)))
    confirm = tomllib.loads((root / "analysis.toml").read_text(encoding="utf-8")).get("confirm")
    if confirm is None:
        pytest.skip("確かめの設定がない")
    props, _ = pr.load(root / "propositions.toml")
    assert set(confirm.get("detail", [])) <= {p["id"] for p in props}
    assert (root / "research" / "R18-confirmation-2026.md").exists()  # confirmation.md が指す手順の記録


def test_sigma_stats_measure_the_distance_from_the_strength_threshold():
    # 13単位中10で成立、基準 0.5: z = (10 − 6.5) ÷ √(13 × 0.25) = 1.94σ、P(X ≥ 10) = (286 + 78 + 13 + 1) ÷ 2¹³
    s = pr.sigma_stats(10, 13, 0.5)
    assert s["sigma"] == pytest.approx(3.5 / math.sqrt(3.25)) and s["sigma_band"] == 1
    assert s["p_above"] == pytest.approx(378 / 8192) and s["p_below"] == pytest.approx(1 - (78 + 13 + 1) / 8192)
    assert pr.sigma_stats(2, 13, 0.75)["sigma_band"] == -3          # 0.15 は概ね（0.75）から3σ 以上下
    assert pr.sigma_stats(5, 0, 0.5)["sigma"] is None and pr.sigma_stats(5, 5, None)["sigma"] is None
    r = pr.evaluate(P1, table(ROWS), focus="d")
    f = r["forms"][0]
    assert f["sigma"] == pytest.approx((f["hold"] - f["n"] * 0.75) / math.sqrt(f["n"] * 0.75 * 0.25))
