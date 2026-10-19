"""next_seeds.py（#48 で消えた propositions.py の next サブコマンド）。#48 の前の test_propositions.py のテストを移した。"""

import pytest
import polars as pl

import next_seeds as ns
import propositions as pr


def table(rows):
    """rows: (team, season, rd, rank, rank_pythag)"""
    return pl.DataFrame([{"team": t, "team_name": t.upper(), "season": s, "league": "C", "rd": rd,
                          "rank": rk, "rank_pythag": rp, "rank_gap": rk - rp}
                         for t, s, rd, rk, rp in rows])


P1 = {"id": "P1", "statement": "得失点差がプラスならAクラス", "strength": "usually", "min_n": 1,
      "if": [{"col": "rd", "op": ">", "value": 0}], "then": [{"col": "rank", "op": "<=", "value": 3}],
      "surprise": "rank_gap", "context": ["rank_pythag"], "links": ["H3"]}


# ---------- 4つの形を並べて読む（次の命題の種） ----------

def _result(orig, conv, cx_o=(), cx_c=()):
    """形の終了コードと判例だけを持つ、読みのための結果。conv が None なら逆・裏を省略した命題。"""
    def cx(units):
        return [{"unit": u, "team": u.split("-")[0], "season": int(u.split("-")[1]), "focus": u.startswith("d-")}
                for u in units]
    forms = [{"form": "original", "code": orig, "rate": 0.9, "n": 20, "counterexamples": cx(cx_o)},
             {"form": "contrapositive", "code": orig, "rate": 0.8, "n": 15, "counterexamples": cx(cx_o)}]
    if conv is not None:
        forms += [{"form": "converse", "code": conv, "rate": 0.6, "n": 18, "counterexamples": cx(cx_c)},
                  {"form": "inverse", "code": conv, "rate": 0.5, "n": 12, "counterexamples": cx(cx_c)}]
    return {"id": "X", "key": "[all] a>0 => b>0", "signature": "s", "forms": forms, "judgement": {"code": orig}}


@pytest.mark.parametrize("orig,conv,expected", [
    (0, 0, 0), (1, 1, 0),                # 両方の向きで支持
    (1, 5, 5), (5, 1, 5), (5, 5, 5), (5, 3, 5),  # 未判定を含む
    (0, 3, 1), (1, 2, 1),                 # 十分条件だけ
    (3, 1, 2), (2, 0, 2),                 # 必要条件だけ
    (2, 2, 3), (2, 3, 3), (3, 2, 3),      # 結びつきはあるが主張より弱い
    (3, 3, 4),                            # どちらも不成立
    (4, 1, 5), (1, 4, 5), (4, 4, 5),      # 判断保留を含む
    (0, None, 6), (3, None, 6),           # 逆・裏を省略
])
def test_reading_from_original_and_converse_codes(orig, conv, expected):
    assert ns.reading(_result(orig, conv))["reading"] == expected


def test_reading_every_code_has_a_label():
    for code, (ja, en) in ns.READING.items():
        assert ja and en


def test_reading_seeds_split_narrow_and_route_and_show_clusters():
    r = ns.reading(_result(1, 2, cx_o=["d-2014"], cx_c=["d-2019", "d-2021", "g-2019"]))
    assert r["narrow"]["units"] == ["d-2014"] and r["narrow"]["focus"] == ["d-2014"]
    assert r["route"]["n"] == 3 and r["route"]["top_team"] == ("d", 2) and r["route"]["top_season"] == (2019, 2)
    assert r["route"]["focus"] == ["d-2019", "d-2021"]
    assert ns.reading(_result(0, None))["route"] is None


def test_reading_agrees_with_real_evaluation():
    """十分だが必要ではない関係を実際に評価して、読みが 1 になり、逆の判例が道筋の種になる。"""
    # rd>0 の単位は必ず A。A のうち rd<=0 の単位が多い（A に至る別の道筋がある）
    rows = [("a", 1960 + i, 10, 1, 1) for i in range(40)] + [("b", 1960 + i, -10, 2, 4) for i in range(40)] \
        + [("c", 1960 + i, -10, 5, 5) for i in range(40)]
    p = {**P1, "min_n": 10, "strength": "almost_always"}
    r = pr.evaluate(p, table(rows))
    x = ns.reading(r)
    assert x["original"] == 0 and x["converse"] in (2, 3) and x["reading"] == 1
    assert x["narrow"]["n"] == 0 and x["route"]["n"] == 40 and x["route"]["top_team"] == ("b", 40)


def test_render_next_groups_by_reading_and_marks_focus():
    md = ns.render_next([ns.reading(_result(1, 3, cx_c=["d-2014", "g-2015"])), ns.reading(_result(0, 0))],
                        {"sha256": "abc", "code_version": "v"})
    assert "## 1 " in md and "## 0 " in md and "**d-2014**" in md and "abc" in md


def test_cli_next_prints_readings_and_never_fails_on_objections(tmp_path, capsys):
    f = tmp_path / "propositions.jsonl"
    import json
    f.write_text(json.dumps(_result(3, 3, cx_o=["d-2019"])) + "\n", encoding="utf-8")
    assert ns.main(["next", str(f)]) == 0
    assert capsys.readouterr().out.startswith("[4] X")
    assert ns.main(["next", str(f), "--id", "Nope"]) == 64
    assert ns.main(["next", str(tmp_path / "none.jsonl")]) == 66


def test_cli_next_outputs_markdown_json_and_limit(tmp_path):
    f = tmp_path / "propositions.jsonl"
    md_out = tmp_path / "out" / "next_seeds.md"
    json_out = tmp_path / "out" / "next_seeds.jsonl"
    import json

    cx_many = [f"d-{2000 + i}" for i in range(12)]
    res = {**_result(1, 3, cx_c=cx_many), "meta": {"sha256": "hash123", "code_version": "v1.0"}}
    f.write_text(json.dumps(res) + "\n", encoding="utf-8")

    code = ns.main(["next", str(f), "--markdown", str(md_out), "--json", str(json_out), "--limit", "3"])
    assert code == 0

    assert md_out.exists()
    md_text = md_out.read_text(encoding="utf-8")
    assert "# 次の命題の種（自動生成）" in md_text
    assert "hash123" in md_text
    assert "ほか9" in md_text  # 12 items with limit 3 => 3 shown + 9 more

    assert json_out.exists()
    json_lines = [json.loads(line) for line in json_out.read_text(encoding="utf-8").splitlines()]
    assert len(json_lines) == 1
    assert json_lines[0]["id"] == "X"
    assert json_lines[0]["reading"] == 1


def test_reading_keeps_all_four_forms_and_render_shows_them():
    x = ns.reading(_result(1, 3, cx_c=["d-2014"]))
    assert set(x["forms"]) == {"original", "contrapositive", "converse", "inverse"}
    assert x["forms"]["inverse"] == (3, 0.5, 12)
    md = ns.render_next([x, ns.reading(_result(0, None))], {"sha256": "s"})
    assert "| 元の命題 | 対偶 | 逆 | 裏 |" in md and "3・0.50（12）" in md and "省略: " in md
