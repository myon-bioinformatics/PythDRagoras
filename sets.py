"""命題の集合（R48）。登録した命題の条件を集合として名前で呼び、∩・∪・¬ の式を命題として判定する。

    python pythdragoras/sets.py --season cycles/c001-chunichi/outputs/season.jsonl \
        --config cycles/c001-chunichi/analysis.toml --propositions cycles/c001-chunichi/propositions.toml \
        --sets cycles/c001-chunichi/sets.toml --outdir cycles/c001-chunichi/outputs [--report-only]

集合の定義（TOML）:

    [[set]]
    id = "OFF_SHORT"          # 式の中で使う名前（英大文字・数字・_）
    from = "P147"             # 登録済みの命題の id。その命題の条件（part）を集合にする
    part = "if"               # "if"（前件、既定）・"then"（後件）・"where"（範囲の条件、team を除く）

    [[expr]]
    id = "E1"
    expr = "OFF_SHORT & ~DEF_CLEAR"   # & かつ、| または、~ でない、( ) まとまり
    then = [{ col = "upper_half", op = "==", value = false }]   # 式に入る単位について言うこと（既定は B クラス）
    strength = "usually"
    statement = "..."

式は「かつ」をつないだ組を「または」で並べた形（if_any）に直して、propositions.py の命題として判定する。
4つの形・終了コード・判例は命題と同じ規則（docs/propositions.md）。さらに、焦点の球団の「後件に当たる単位」のうち
式に入るものの数（覆い）を出す。元の命題の形の終了コードが 0（判例なし。対偶も同じ判例）で、焦点の後件の単位を
すべて覆う式を「通る」と書く。逆（後件なら式に入る）は求めない（後件に至る道筋は1つとは限らない）。
"""

import json
import re
import sys
import tomllib
from pathlib import Path

import polars as pl

NEG = {"<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "=="}
MAX_GROUPS = 64  # 「または」の組が増えすぎた式は、読める命題にならないので止める
TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|[&|~()])")


class SetError(ValueError):
    """集合・式の定義の誤り（終了コード 64）。"""


# ---------- 集合 ----------

def set_dnf(s: dict, props: dict[str, dict]) -> list[list[dict]]:
    """集合を「かつ」の組の並び（または）にする。"""
    p = props.get(s.get("from", ""))
    if p is None:
        raise SetError(f"{s.get('id')}: from の命題がない: {s.get('from')}")
    part = s.get("part", "if")
    if part == "then":
        return [list(p["then"])]
    if part == "where":  # 範囲の条件（team の条件は除く。集合は全単位の上で作る）
        conds = [c for c in (p.get("scope") or {}).get("where", []) if c["col"] != "team"]
        if not conds:
            raise SetError(f"{s['id']}: {p['id']} の範囲に（team 以外の）条件がない")
        return [conds]
    if part != "if":
        raise SetError(f"{s['id']}: part は if・then・where のどれか")
    base, any_ = list(p.get("if", [])), p.get("if_any", [])
    groups = [base + list(g) for g in any_] if any_ else [base]
    if not any(groups):
        raise SetError(f"{s['id']}: {p['id']} の前件が空（すべての単位）なので集合にしない")
    return groups


# ---------- 式 ----------

def tokens(expr: str) -> list[str]:
    out, pos = [], 0
    end = len(expr.rstrip())
    while pos < end:
        m = TOKEN.match(expr, pos)
        if not m:
            raise SetError(f"式を読めない: {expr[pos:]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def parse(expr: str):
    """式を木にする。優先順位は ~ > & > |。"""
    toks, i = tokens(expr), 0

    def peek():
        return toks[i] if i < len(toks) else None

    def take(t=None):
        nonlocal i
        if i >= len(toks) or (t and toks[i] != t):
            raise SetError(f"式の形が違う（{t or '項'} が要る）: {expr}")
        i += 1
        return toks[i - 1]

    def unary():
        if peek() == "~":
            take("~")
            return ("not", unary())
        if peek() == "(":
            take("(")
            node = disj()
            take(")")
            return node
        name = take()
        if name in "&|~()":
            raise SetError(f"式の形が違う: {expr}")
        return ("set", name)

    def conj():
        node = unary()
        while peek() == "&":
            take("&")
            node = ("and", node, unary())
        return node

    def disj():
        node = conj()
        while peek() == "|":
            take("|")
            node = ("or", node, conj())
        return node

    node = disj()
    if i != len(toks):
        raise SetError(f"式の終わりに余りがある: {expr}")
    return node


def _key(c: dict) -> tuple:
    return (c["col"], c["op"], json.dumps(c["value"]))


def _clean(groups: list[list[dict]]) -> list[list[dict]]:
    """組の中の重なりを除き、同じ列の == と != が同じ値でぶつかる組（空集合）を落とす。"""
    out, seen = [], set()
    for g in groups:
        uniq = list({_key(c): c for c in g}.values())
        keys = {_key(c) for c in uniq}
        if any((c["col"], NEG[c["op"]], json.dumps(c["value"])) in keys for c in uniq if c["op"] in ("==", "!=")):
            continue
        sig = tuple(sorted(keys))
        if sig not in seen:
            seen.add(sig)
            out.append(uniq)
    return out


def dnf(node, sets: dict[str, list[list[dict]]]) -> list[list[dict]]:
    kind = node[0]
    if kind == "set":
        if node[1] not in sets:
            raise SetError(f"定義のない集合: {node[1]}")
        return [list(g) for g in sets[node[1]]]
    if kind == "or":
        return _clean(dnf(node[1], sets) + dnf(node[2], sets))
    if kind == "and":
        left, right = dnf(node[1], sets), dnf(node[2], sets)
        return _clean([a + b for a in left for b in right])
    # not: ¬(g1 ∨ g2 ∨ …) = ¬g1 ∧ ¬g2 ∧ …、¬(c1 ∧ c2) = ¬c1 ∨ ¬c2
    out = [[]]
    for g in dnf(node[1], sets):
        neg = [[{**c, "op": NEG[c["op"]]}] for c in g]
        out = _clean([a + b for a in out for b in neg])
        if len(out) > MAX_GROUPS:
            raise SetError(f"「または」の組が {MAX_GROUPS} を超えた（読める式にならない）")
    return out


def compile_expr(e: dict, sets: dict[str, list[list[dict]]]) -> dict:
    """式を propositions.py の命題の形にする。"""
    groups = dnf(parse(e["expr"]), sets)
    if not groups:
        raise SetError(f"{e['id']}: 空集合の式")
    if len(groups) > MAX_GROUPS:
        raise SetError(f"{e['id']}: 「または」の組が {MAX_GROUPS} を超えた")
    p = {"id": e["id"], "statement": e.get("statement", e["expr"]), "strength": e.get("strength", "usually"),
         "then": e.get("then", [{"col": "upper_half", "op": "==", "value": False}]),
         "note": f"集合の式: {e['expr']}" + (f"。{e['note']}" if e.get("note") else "")}
    for k in ("scope", "surprise", "context", "links", "falsifier", "min_n"):
        if k in e:
            p[k] = e[k]
    if len(groups) == 1:
        p["if"] = groups[0]
    else:
        p["if_any"] = groups
    return p


def load_sets(path: Path, props: list[dict]) -> tuple[dict, list[dict], str]:
    import hashlib

    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    by_id = {p["id"]: p for p in props}
    sets = {}
    for s in data.get("set", []):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s.get("id", "")):
            raise SetError(f"集合の id は英字・数字・_: {s.get('id')}")
        if s["id"] in sets:
            raise SetError(f"集合の id が重なる: {s['id']}")
        sets[s["id"]] = set_dnf(s, by_id)
    exprs = data.get("expr", [])
    ids = [e.get("id") for e in exprs]
    if len(ids) != len(set(ids)) or set(ids) & set(by_id):
        raise SetError("式の id は一意で、命題の id と重ならないこと")
    return sets, exprs, hashlib.sha256(raw).hexdigest()


# ---------- 覆い ----------

def coverage(p: dict, st: pl.DataFrame, focus: str | None) -> dict:
    """焦点の球団の、後件に当たる単位のうち、式に入るもの。"""
    import propositions as pr

    if not focus:
        return {"focus": None, "n": 0, "covered": 0, "missed": []}
    df = pr.scope_filter(st, p.get("scope")).filter(pl.col("team") == focus)
    flags = df.select("season", x=pr._antecedent(p, df.columns), y=pr._cond(p["then"], df.columns))
    target = flags.filter(pl.col("y").fill_null(False))
    inside = target.filter(pl.col("x").fill_null(False))
    missed = sorted(set(target["season"].to_list()) - set(inside["season"].to_list()))
    return {"focus": focus, "n": target.height, "covered": inside.height, "missed": [f"{focus}-{s}" for s in missed]}


def render(results: list[dict], meta: dict) -> str:
    lines = ["# 命題の集合（自動生成）", "",
             f"集合と式のファイル SHA-256: `{meta['sha256']}` / 命題ファイル: `{meta['props_sha256']}`", "",
             "登録した命題の条件を集合にし、式（& かつ、| または、~ でない）を命題として判定した（R48）。",
             "形のセルは「終了コード・成立率（単位数）」。覆いは、焦点の球団の後件に当たる単位のうち、式に入る単位の数。",
             "**通る** = 元の命題の形が終了コード 0（判例なし）で、焦点の後件の単位をすべて覆う。逆は求めない（後件に至る道筋は1つとは限らない）。",
             "通るは評価データ内での条件充足。事後構成と明示した式は過去の結果を参照して作成・選択しており、独立確認なし。計算前の式の登録だけでは独立確認にならない。",
             "1つ通っても正解とは書かない（別々の命題から組んだ式が複数通ることを求める）。", "",
             "| 式 | 集合の式 | 組の数 | 元の命題 | 対偶 | 逆 | 裏 | 総合 | 焦点の覆い | 通る | 判例（元） |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        by = {f["form"]: f for f in r["forms"]}

        def cell(f):
            x = by.get(f)
            return "—" if x is None else f"{x['code']}・{'-' if x['rate'] is None else format(x['rate'], '.2f')}（{x['n']}）"
        cov = r["coverage"]
        cx = [c["unit"] for c in by["original"]["counterexamples"]]
        stage = "（事後構成・独立確認なし）" if r.get("posthoc") else ""
        lines.append(f"| {r['id']}{stage} | `{r['expr']}` | {r['groups']} | {cell('original')} | {cell('contrapositive')} | {cell('converse')} "
                     f"| {cell('inverse')} | {r['judgement']['code']} | {cov['covered']}/{cov['n']} | {'**通る**' if r['passes'] else ''} "
                     f"| {', '.join(cx[:6])}{' ほか' if len(cx) > 6 else ''} |")
    missed = [(r["id"], r["coverage"]["missed"]) for r in results if r["coverage"]["missed"]]
    if missed:
        lines += ["", "覆えなかった焦点の単位:", ""] + [f"- {i}: {', '.join(m)}" for i, m in missed]
    ledger = stoppers(results)
    if ledger:
        lines += ["", "## 式を止めた単位（例外の台帳、R50）", "",
                  "どの式の判例になったか（判例）・どの式で覆えなかったか（覆えない）を、単位ごとに数える。",
                  "多くの式を止める単位ほど、今の集合（命題の条件）では表しきれていない。", "",
                  "| 単位 | 判例になった式 | 覆えなかった式 |", "|---|---|---|"]
        for u, v in ledger:
            lines.append(f"| {u} | {len(v['counterexample'])}（{', '.join(v['counterexample']) or '—'}） "
                         f"| {len(v['missed'])}（{', '.join(v['missed']) or '—'}） |")
    return "\n".join(lines) + "\n"


def stoppers(results: list[dict]) -> list[tuple[str, dict]]:
    """式を止めた単位の台帳（R50）。判例になった式・覆えなかった式を単位ごとに集め、止めた数の多い順に並べる。"""
    book: dict[str, dict] = {}
    for r in results:
        original = next(f for f in r["forms"] if f["form"] == "original")
        for c in original["counterexamples"]:
            book.setdefault(c["unit"], {"counterexample": [], "missed": []})["counterexample"].append(r["id"])
        for u in r["coverage"]["missed"]:
            book.setdefault(u, {"counterexample": [], "missed": []})["missed"].append(r["id"])
    return sorted(book.items(), key=lambda kv: (-(len(kv[1]["counterexample"]) + len(kv[1]["missed"])), kv[0]))


def junit(results: list[dict], meta: dict) -> str:
    """式を JUnit 形式のテスト報告にする（R49）。通らなかった式は failure として、止まった理由を残す。

    止まった理由（判例・覆えなかった焦点の単位）は、足りる指標・足りない指標を見つける材料なので、
    CI の成果物としても積み上げる。
    """
    import xml.etree.ElementTree as ET

    failures = sum(not r["passes"] for r in results)
    suite = ET.Element("testsuite", name="pythdragoras.sets", tests=str(len(results)), failures=str(failures),
                       errors="0", skipped="0")
    props = ET.SubElement(suite, "properties")
    ET.SubElement(props, "property", name="sets_sha256", value=meta["sha256"])
    ET.SubElement(props, "property", name="propositions_sha256", value=meta["props_sha256"])
    for r in results:
        case = ET.SubElement(suite, "testcase", classname="sets", name=f"{r['id']}: {r['expr']}")
        if r["passes"]:
            continue
        original = next(f for f in r["forms"] if f["form"] == "original")
        cx = [c["unit"] for c in original["counterexamples"]]
        cov = r["coverage"]
        reasons = []
        if cx:
            reasons.append(f"判例 {len(cx)}件: {', '.join(cx)}")
        elif original["code"] != 0:
            reasons.append(f"元の命題の終了コード {original['code']}（判例なし、判断保留など）")
        if cov["covered"] < cov["n"]:
            reasons.append(f"覆えなかった焦点の単位 {cov['n'] - cov['covered']}件: {', '.join(cov['missed'])}")
        msg = "；".join(reasons) or "通らない"
        fail = ET.SubElement(case, "failure", message=msg, type=f"exit{original['code']}")
        fail.text = (f"元の命題 {original['hold']}/{original['n']}、総合の終了コード {r['judgement']['code']}、"
                     f"覆い {cov['covered']}/{cov['n']}")
    ET.indent(suite)
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def main(argv=None) -> int:
    import argparse

    import propositions as pr
    from pythdragoras import apply_exclusions

    ap = argparse.ArgumentParser(description="命題の条件を集合にし、集合の式を命題として判定する（R48）")
    ap.add_argument("--season", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--propositions", type=Path, required=True)
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--report-only", action="store_true", help="異議（1〜6）では失敗にしない。仕組みの不具合（64 以上）は失敗にする")
    args = ap.parse_args(argv)

    for f in (args.season, args.config, args.propositions, args.sets):
        if not f.exists():
            print(f"[66] {pr.label(66)}: {f}", file=sys.stderr)
            return 66
    with args.config.open("rb") as f:
        cfg = tomllib.load(f)
    focus = cfg.get("focus", {}).get("team")
    try:
        props, props_sha = pr.load(args.propositions)
        sets, exprs, sha = load_sets(args.sets, props)
        compiled = [compile_expr(e, sets) for e in exprs]
        for p in compiled:
            pr.validate(p)
    except (SetError, pr.PropositionError) as e:
        print(f"[64] {e}", file=sys.stderr)
        return 64
    full = pl.read_ndjson(args.season, infer_schema_length=None)
    included, exclusions = apply_exclusions(full, cfg.get("exclude", []))
    excluded = full.filter(pl.col("season").is_in([int(x["season"]) for x in exclusions]))
    try:
        results = []
        for e, p in zip(exprs, compiled):
            r = pr.evaluate(p, included, excluded, focus)
            cov = coverage(p, included, focus)
            original = next(f for f in r["forms"] if f["form"] == "original")
            r |= {"posthoc": bool(e.get("posthoc", False)), "expr": e["expr"], "groups": len(p.get("if_any") or [p["if"]]), "coverage": cov,
                  "passes": original["code"] == 0 and cov["n"] > 0 and cov["covered"] == cov["n"]}
            results.append(r)
    except pr.DataError as e:
        print(f"[65] {e}", file=sys.stderr)
        return 65
    meta = {"sha256": sha, "props_sha256": props_sha}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "sets.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in results),
                                            encoding="utf-8")
    (args.outdir / "sets.md").write_text(render(results, meta), encoding="utf-8")
    (args.outdir / "sets.junit.xml").write_text(junit(results, meta), encoding="utf-8")
    first = 0
    for r in results:
        code = r["judgement"]["code"]
        print(f"[{code}] {r['id']} {pr.label(code)} 覆い {r['coverage']['covered']}/{r['coverage']['n']}"
              f"{' 通る' if r['passes'] else ''}  {r['expr']}")
        first = first or code
    return 0 if args.report_only else first


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent)]   # pythdragoras/ とリポジトリルート
    raise SystemExit(main())
