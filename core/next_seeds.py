"""PythDRagoras: 元の命題と逆の判定を並べて読み、次の命題の種を表示する（判定ではない）。

#48（大文字フォルダの削除）で消えた `PythDRagoras/logic/propositions.py` の `next` サブコマンドと、
その読み（READING・reading・render_next）を、そのままの中身でここに戻した。

propositions.py には戻していない。propositions.py の SHA-256 は LittleBit holdout の来歴
（holdout.provenance.json の source_divergence）と、seed 評価（seed-evaluation.json の
predicate_code_sha256）に記録されていて、書き換えると記録の基準を動かすことになるため。
読みは判定ではなく、終了コードも変えないので、別のファイルに分けても判定の結果は変わらない。

    python pythdragoras/next_seeds.py next cycles/c001-chunichi/outputs/propositions.jsonl [--id P1 ...] [--reading 1 2]
        [--markdown next_seeds.md] [--jsonl next_seeds.jsonl] [--limit 8]
    python pythdragoras/next_seeds.py render-md cycles/c001-chunichi/outputs/propositions.jsonl [--out next_seeds.md] [--jsonl next_seeds.jsonl]
"""
from __future__ import annotations

import json
from pathlib import Path

from propositions import _cell, label


# ---------- 4つの形を並べて読む（次の命題の種） ----------

# 元の命題（A ⇒ B、十分の向き）と逆（B ⇒ A、必要の向き）の判定の組から決まる読み。判定ではない。
# 判定とテストは番号で比べ、言葉は表示だけに使う（終了コードと同じ扱い）。
READING = {
    0: ("両方の向きで支持（同値に近い）", "both directions supported (close to equivalence)"),
    1: ("十分条件だが必要条件ではない", "sufficient, not necessary"),
    2: ("必要条件だが十分条件ではない", "necessary, not sufficient"),
    3: ("どちらの向きも主張より弱い結びつき", "a relationship weaker than claimed in both directions"),
    4: ("どちらの向きも不成立", "neither direction holds"),
    5: ("判断保留・未判定を含む（不足を確認して読み直す）", "inconclusive or undetermined in at least one direction"),
    6: ("一方向だけ（逆・裏は省略）", "one direction only (converse and inverse skipped)"),
}
_SUPPORTED = (0, 1)  # 形の終了コードのうち、その向きが支持されたもの


def reading(r: dict) -> dict:
    """元の命題と逆の形の終了コードから、読み（READING の番号）と次の問いの種を作る。

    - 元の命題の判例（A かつ B でない）: A を満たしたのに B に至らなかった単位。足りない条件（かつ）や、範囲（where）の候補
    - 逆の判例（B かつ A でない）: A を通らずに B に至った単位。別の道筋（または）の候補
    種は材料であって判定ではない。種から作る命題は新しい id で事前登録し、きっかけの単位を motivated_by に書く。
    """
    by = {f["form"]: f for f in r["forms"]}
    o, c = by["original"]["code"], by.get("converse", {}).get("code")
    if c is None:
        code = 6
    elif any(v in (4, 5) for v in (o, c)):
        code = 5
    elif o in _SUPPORTED and c in _SUPPORTED:
        code = 0
    elif o in _SUPPORTED:
        code = 1
    elif c in _SUPPORTED:
        code = 2
    elif 2 in (o, c):
        code = 3
    else:
        code = 4
    rates = {f: (by[f]["code"], by[f]["rate"], by[f]["n"]) if f in by else None
             for f in ("original", "contrapositive", "converse", "inverse")}  # 4つの形すべて（漏れなく並べる、R41）
    return {"id": r["id"], "key": r.get("key"), "signature": r.get("signature"), "reading": code,
            "original": o, "converse": c, "exit": r["judgement"]["code"], "forms": rates,
            "skip_reason": r.get("skip_reason"),
            "narrow": _seed(by["original"]["counterexamples"]),
            "route": _seed(by["converse"]["counterexamples"]) if c is not None else None}


def _seed(cx: list[dict]) -> dict:
    """判例を、種として読める形にまとめる。判例が1つの球団・1つの年に集まっているかも示す（分けるかは人が決める）。"""
    from collections import Counter

    units = [x["unit"] for x in cx]
    teams, seasons = Counter(x["team"] for x in cx), Counter(x["season"] for x in cx)
    top_team, top_season = teams.most_common(1), seasons.most_common(1)
    return {"n": len(units), "units": units, "focus": [x["unit"] for x in cx if x.get("focus")],
            "top_team": top_team[0] if top_team else None, "top_season": top_season[0] if top_season else None}


def render_next(readings: list[dict], meta: dict, limit: int = 8) -> str:
    """読みの一覧（自動生成）。人が次の命題を考えるための表。"""
    lines = ["# 次の命題の種（自動生成）", "",
             f"命題ファイル SHA-256: `{meta.get('sha256', '-')}` / コード: `{meta.get('code_version') or '測定なし'}`", "",
             "元の命題（A ⇒ B）と逆（B ⇒ A）の判定を並べて読む。読みは判定ではなく、終了コードも変えない。",
             "- **絞る**: 元の命題の判例（A なのに B でない）。足りない条件（かつ）や、範囲（where）の候補",
             "- **道筋**: 逆の判例（B なのに A でない）。A を通らずに B に至った単位。別の道筋（または）の候補",
             "- 判例が1つの球団・1つの年に集まるかを並べる。分けるかどうかは人が決める（docs/propositions.md の Split）",
             "- 種から作る命題は、新しい id で事前登録し、きっかけの単位を `motivated_by` に書く（きっかけ以外で読む）", ""]
    from collections import Counter

    counts = Counter(x["reading"] for x in readings)
    lines += ["| 読み | 件数 |", "|---|---|"]
    lines += [f"| {k} {READING[k][0]} | {counts[k]} |" for k in sorted(counts)]

    def units(s):
        if not s or not s["n"]:
            return "—"
        focus = set(s["focus"])
        shown = [f"**{u}**" if u in focus else u for u in s["units"][:limit]]
        more = f" ほか{s['n'] - limit}" if s["n"] > limit else ""
        team, season = s["top_team"], s["top_season"]
        where = f"（最多: {team[0]} {team[1]}件、{season[0]}年 {season[1]}件）" if s["n"] > 1 else ""
        return f"{s['n']}件: " + ", ".join(shown) + more + where

    def form(x, f):
        v = (x.get("forms") or {}).get(f)
        if v is None:
            return "—"
        code, rate, n = v
        return f"{code}・{'-' if rate is None else format(rate, '.2f')}（{n}）"

    for k in sorted(counts):
        lines += ["", f"## {k} {READING[k][0]}", "",
                  "| id | 識別子 | 元の命題 | 対偶 | 逆 | 裏 | 総合 | 絞る（元の判例） | 道筋（逆の判例） |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for x in readings:
            if x["reading"] != k:
                continue
            route = units(x["route"]) if x["route"] is not None else ("省略: " + (x.get("skip_reason") or "条件なしの命題"))
            lines.append(f"| {x['id']} | `{_cell(x.get('key') or '-')}` | {form(x, 'original')} | {form(x, 'contrapositive')} "
                         f"| {form(x, 'converse')} | {form(x, 'inverse')} | {x['exit']} "
                         f"| {_cell(units(x['narrow']))} | {_cell(route)} |")
    lines += ["", "太字は焦点の球団。形のセルは「終了コード・成立率（単位数）」。終了コードは 0 異議なし、1 例外あり、2 主張が強すぎる、3 不成立、4 判断保留、5 判定できない単位。",
              "逆・裏が「—」の命題は、条件が空か、主語を選ぶだけの条件で省略したもの（道筋の欄に理由）。条件を範囲（where）に書いた命題は、条件を「もし」に移すと逆・裏が作られる（R41）"]
    return "\n".join(lines) + "\n"


def render_claims(claims: list[dict]) -> str:
    if not claims:
        return ""
    lines = ["# 外部の主張との照合（自動生成）", "",
             "外部の主張は観測ではない。再現できても観測にはならず、再現できなければそれ自体が調べるべき矛盾になる。", "",
             "| id | 主張 | 出どころ | 状態 | 主張値 | 計算値 |", "|---|---|---|---|---|---|"]
    ja = {"reproduced": "再現", "not-reproduced": "**再現せず**", "not-measurable": "測定不能", "interpretation": "解釈（数では確かめない）"}
    for c in claims:
        lines.append(f"| {c['id']} | {_cell(c['statement'])} | {c.get('source')} {c.get('date') or ''} | {ja[c['status']]} "
                     f"| {_v(c.get('claimed'))} | {_v(c.get('computed'))} |")
    return "\n".join(lines) + "\n"


def _cell(x) -> str:
    return str(x).replace("|", "\\|")


def _v(v):
    """表示用。率など1未満の値は小数3桁、勝利数などは符号つき小数2桁。"""
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1 else f"{v:+.2f}"
    return "-" if v is None else str(v)


def _positive_int(value: str) -> int:
    import argparse

    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"1 以上の整数を指定する: {value}")
    return n


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonl(readings: list[dict]) -> str:
    return "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in readings)


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="元の命題と逆の判定を並べて読み、次の命題の種を表示する（判定ではない）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    nx = sub.add_parser("next", help="元の命題と逆の判定を並べて読み、次の命題の種を表示する（判定ではない）")
    nx.add_argument("results", type=Path, help="pythdragoras.py が書いた propositions.jsonl")
    nx.add_argument("--id", nargs="+", dest="ids")
    nx.add_argument("--reading", type=int, nargs="+", choices=sorted(READING), help="この読みの番号だけを表示する")
    nx.add_argument("--markdown", "-m", type=Path, help="読みのMarkdownレポートを出力するファイルパス")
    nx.add_argument("--jsonl", type=Path, help="読みの結果をJSONLとして出力するファイルパス")
    nx.add_argument("--limit", type=_positive_int, default=8, help="Markdownで表示する判例単位の件数上限（既定: 8）")
    rmd = sub.add_parser("render-md", help="propositions.jsonl の読みをすべて Markdown（next_seeds.md）にする")
    rmd.add_argument("results", type=Path, help="pythdragoras.py が書いた propositions.jsonl")
    rmd.add_argument("--out", type=Path, help="Markdown の出力先ファイルパス（未指定なら標準出力）")
    rmd.add_argument("--jsonl", type=Path, help="読みの結果をJSONLとして出力するファイルパス")
    rmd.add_argument("--limit", type=_positive_int, default=8, help="Markdownで表示する判例単位の件数上限（既定: 8）")


    rp = sub.add_parser("report", help="propositions.jsonl から next_seeds.md（および --claims があれば claims.md）を生成する")
    rp.add_argument("results", type=Path, help="pythdragoras.py が書いた propositions.jsonl")
    rp.add_argument("--claims", type=Path, help="pythdragoras.py が書いた claims.json")
    output = rp.add_mutually_exclusive_group()
    output.add_argument("--outdir", type=Path, help="出力ディレクトリ（next_seeds.md と claims.md を書き出す）")
    output.add_argument("--out", type=Path, help="出力ファイル（next_seeds.md の書き出し先）")

    args = ap.parse_args(argv)

    if not args.results.exists():
        print(f"[66] {label(66)}: {args.results}", file=sys.stderr)
        return 66
    results = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = results[0].get("meta", {}) if results else {}

    if args.cmd == "render-md":
        readings = [reading(r) for r in results]
        md = render_next(readings, meta, limit=args.limit)
        if args.out:
            _write(args.out, md)
        else:
            print(md, end="")
        if args.jsonl:
            _write(args.jsonl, _jsonl(readings))
        return 0  # 読みは判定ではないので、失敗にしない

    if args.cmd == "next":
        known = {r["id"] for r in results}
        if missing := [i for i in args.ids or [] if i not in known]:
            print(f"[64] {label(64)}（そんな id はない: {', '.join(missing)}）", file=sys.stderr)
            return 64
        filtered = []
        for x in map(reading, results):
            if (args.ids and x["id"] not in args.ids) or (args.reading and x["reading"] not in args.reading):
                continue
            filtered.append(x)
            route = "—" if x["route"] is None else x["route"]["n"]
            print(f"[{x['reading']}] {x['id']} {READING[x['reading']][0]}  元 {x['original']} 逆 {x['converse'] if x['converse'] is not None else '—'}"
                  f"  絞る {x['narrow']['n']} 道筋 {route}  {x['key']}")

        if args.markdown:
            _write(args.markdown, render_next(filtered, meta, limit=args.limit))
        if args.jsonl:
            _write(args.jsonl, _jsonl(filtered))

        return 0  # 読みは判定ではないので、失敗にしない

    if args.cmd == "report":
        if args.claims and not args.claims.exists():
            print(f"[66] {label(66)}: {args.claims}", file=sys.stderr)
            return 66
        # Refuse output/input aliases before writing either report.
        output_paths = []
        if args.outdir:
            output_paths.append(args.outdir / "next_seeds.md")
            if args.claims:
                output_paths.append(args.outdir / "claims.md")
        elif args.out:
            output_paths.append(args.out)
            if args.claims:
                output_paths.append(args.out.parent / "claims.md")
        resolved_outputs = [path.resolve() for path in output_paths]
        inputs = {args.results.resolve()}
        if args.claims:
            inputs.add(args.claims.resolve())
        if len(set(resolved_outputs)) != len(resolved_outputs) or inputs.intersection(resolved_outputs):
            ap.error("report outputs must be distinct from each other and input files")
        readings = [reading(r) for r in results]
        md_next = render_next(readings, meta)

        md_claims = ""
        if args.claims:
            claims_data = json.loads(args.claims.read_text(encoding="utf-8"))
            md_claims = render_claims(claims_data)

        if args.outdir:
            args.outdir.mkdir(parents=True, exist_ok=True)
            (args.outdir / "next_seeds.md").write_text(md_next, encoding="utf-8")
            if args.claims:
                (args.outdir / "claims.md").write_text(md_claims, encoding="utf-8")
        elif args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(md_next, encoding="utf-8")
            if args.claims:
                claims_out = args.out.parent / "claims.md"
                claims_out.write_text(md_claims, encoding="utf-8")
        else:
            sys.stdout.write(md_next)
            if md_claims:
                sys.stdout.write("\n" + md_claims)

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
