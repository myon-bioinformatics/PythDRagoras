"""条件の組み合わせ（「または」でつないだ「かつ」の組）を総当たりで探し、目的の状態をどれだけ言い当てるかで並べる。

探索は命題を作るための材料で、命題の判定ではない。見つかった式は、それを見た後に作った命題として登録し、
作るきっかけになったデータとは別のデータ（翌年以降など）で確かめる（docs/propositions.md の Restating）。

  候補の条件は探索の前にファイルに書く（候補を後から足して探し直すと、それ自体が選び直しになる）
  式の形: (c1 かつ c2) または (c3 かつ c4)。組の数・組の中の条件の数の上限は引数で決める
  評価: 前件 ⇒ 目的（成立率 = precision）と 目的 ⇒ 前件（逆の成立率 = recall）の両方を数え、Matthews 相関係数で並べる。
        同点なら条件の少ない式を上にし、それでも同じなら式の文字列で並べる

数え方は2つ。結果が一致する範囲（「または」が2組まで、かつ単位数が浮動小数の交わり数で厳密なとき）は
numpy のビット行列を既定にする。それ以外は、これまでどおり Python 整数のビット集合を使う。

使い方:
  python pythdragoras/rule_search.py --season season.jsonl --config analysis.toml --candidates rule_candidates.toml \\
      --out outputs/rules.jsonl [--top 20] [--backend bitmatrix|bitset]

潜在パターンから条件を足す経路は littlebit_approx（探索の記録。成否の判定はこのファイルのまま）。
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import itertools
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from propositions import OPS, PropositionError, _cond, text

# 交わりの数は単位数そのもの以下。float64 の仮数で厳密なのは 2^53 未満。
# MCC の分母は (tp+fp)(tp+fn)(tn+fp)(tn+fn) で、各因子は単位数以下。n^4 が 2^53 未満なら積も厳密。
# 8192^4 = 2^52。これを超える単位数はビット集合のままにする（結果がずれないように）。
_EXACT_UNITS = 8192
# ビット行列は組どうしの T×T 行列（交わり・tp・fp など）を同時に持つ。候補が多いと T が増えて
# メモリが T^2 で増えるので、見積りがこの上限を超える入力は既定ではビット集合に戻す（明示指定なら止める）。
_MATRIX_BYTES = 512 * 1024 * 1024


def _mask(df: pl.DataFrame, conds: list[dict]) -> int:
    """条件を満たす単位をビットで表す（i 番目の単位が満たせば i ビット目が 1）。"""
    flags = df.select(x=_cond(conds, df.columns))["x"].to_list()
    return sum(1 << i for i, f in enumerate(flags) if f)


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


# 判定に使う鍵。これ以外は出どころの記録で、マスクにも並べる鍵にも入れない。
CONDITION_KEYS = {"col", "op", "value"}
PROVENANCE_KEYS = {
    "source", "rank", "seed", "stage", "pattern", "posthoc", "exploratory",
    "registration", "bundle", "sign", "rule", "n_loading",
}


def _check(target: list[dict], candidates: list[dict]) -> None:
    allowed = CONDITION_KEYS | PROVENANCE_KEYS
    for c in [*target, *candidates]:
        if not isinstance(c, dict) or not CONDITION_KEYS <= set(c) <= allowed or c["op"] not in OPS:
            raise PropositionError(f"条件は {{col, op, value}}（記録用のキーは {sorted(PROVENANCE_KEYS)}）: {c}")
        if not isinstance(c["col"], str) or not c["col"]:
            raise PropositionError(f"列名が空: {c}")


def _logic_key(groups: list[list[dict]]) -> str:
    """同点の順序は col, op, value だけを見る。出どころの記録では順序を変えない。"""
    slim = []
    for g in groups:
        row = []
        for c in g:
            if set(c) == CONDITION_KEYS:
                row.append(c)
            else:
                row.append({k: c[k] for k in ("col", "op", "value")})
        slim.append(row)
    return json.dumps(slim, ensure_ascii=False)


def _build_terms(df: pl.DataFrame, candidates: list[dict], max_conds: int) -> list[tuple[list[dict], int]]:
    """同じ列を2度使わない「かつ」の組。並びは combinations の順（ビット集合版と同じ）。"""
    n = df.height
    cand = [(c, _mask(df, [c])) for c in candidates]
    full = (1 << n) - 1
    terms = []
    for k in range(1, max_conds + 1):
        for combo in itertools.combinations(cand, k):
            if len({c["col"] for c, _ in combo}) < k:
                continue
            m = full
            for _, cm in combo:
                m &= cm
            terms.append(([c for c, _ in combo], m))
    return terms


def _mask_bits(mask: int, n: int) -> np.ndarray:
    """Python 整数のビット i を、長さ n の bool 配列の i 番目にする。"""
    if n <= 0:
        return np.zeros(0, dtype=bool)
    width = (n + 7) // 8
    raw = np.frombuffer(mask.to_bytes(width, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


def _masks_to_matrix(masks: list[int], n: int) -> np.ndarray:
    if n <= 0 or not masks:
        return np.zeros((len(masks), max(n, 0)), dtype=bool)
    width = (n + 7) // 8
    buf = bytearray()
    for m in masks:
        buf += m.to_bytes(width, "little")
    raw = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(len(masks), width)
    # bool にしてから数える。uint8 のまま sum すると 255 を超えた単位数で溢れる。
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :n].astype(bool)


def _record(groups: list[list[dict]], tp: int, fp: int, n_pos: int, n_neg: int) -> tuple[tuple, dict]:
    tp, fp = int(tp), int(fp)
    fn, tn = n_pos - tp, n_neg - fp
    mcc = _mcc(tp, fp, fn, tn)
    conds = sum(len(g) for g in groups)
    key = (-round(mcc, 12), conds, _logic_key(groups))
    row = {"if_any": groups, "conditions": conds, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "mcc": mcc,
           "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
    return key, row


def _search_bitset(df: pl.DataFrame, target: list[dict], candidates: list[dict], max_terms: int = 2,
                   max_conds: int = 2, top: int = 20) -> list[dict]:
    """Python 整数のビット集合で総当たりする（従来の数え方）。"""
    _check(target, candidates)
    n = df.height
    full = (1 << n) - 1
    y = _mask(df, target)
    terms = _build_terms(df, candidates, max_conds)
    # 上位 top 件だけを、並べる鍵（MCC の高い順 → 条件の少ない順 → 式の文字列）の順に持つ。
    # 組の数を増やすと式の数は百万を超えるので、全部をためてから並べない（並び順は全部を並べたときと同じ）
    best: list[tuple[tuple, dict]] = []
    for k in range(1, max_terms + 1):
        for combo in itertools.combinations(terms, k):
            m = 0
            for _, tm in combo:
                m |= tm
            tp, fp = (m & y).bit_count(), (m & ~y & full).bit_count()
            fn, tn = (~m & y & full).bit_count(), n - (m | y).bit_count()
            mcc, conds = _mcc(tp, fp, fn, tn), sum(len(g) for g, _ in combo)
            if len(best) == top and (-round(mcc, 12), conds) > best[-1][0][:2]:
                continue
            groups = [g for g, _ in combo]
            key = (-round(mcc, 12), conds, _logic_key(groups))
            if len(best) == top and key >= best[-1][0]:
                continue
            bisect.insort(best, (key, {"if_any": groups, "conditions": conds, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                       "mcc": mcc, "precision": tp / (tp + fp) if tp + fp else None,
                                       "recall": tp / (tp + fn) if tp + fn else None}), key=lambda t: t[0])
            del best[top:]
    return [r for _, r in best]


def _mcc_vec(tp, fp, n_pos: int, n_neg: int):
    fn, tn = n_pos - tp, n_neg - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        prod = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        d = np.sqrt(np.where(prod > 0, prod, 0.0))
        return np.where(d > 0, (tp * tn - fp * fn) / d, 0.0)


def _exact_counts(T: np.ndarray, y: np.ndarray):
    """組どうしの OR の tp・fp。交わりの数は 0/1 行列の積（単位数が _EXACT_UNITS 以下なら float64 で厳密）。"""
    Ty, Tn = T & y, T & ~y
    a_tp = Ty.sum(1).astype(np.int64)
    a_fp = Tn.sum(1).astype(np.int64)
    Iy = Ty.astype(np.float64) @ Ty.T.astype(np.float64)
    In = Tn.astype(np.float64) @ Tn.T.astype(np.float64)
    tp = a_tp[:, None] + a_tp[None, :] - Iy.astype(np.int64)
    fp = a_fp[:, None] + a_fp[None, :] - In.astype(np.int64)
    return a_tp, a_fp, tp, fp


def _keep_mcc(mcc: np.ndarray, kth: float) -> np.ndarray:
    """丸めた MCC が、上位を決める値以上の組を残す。同点は条件数と式の文字列で並ぶので、境界の同点は全部残す。"""
    return np.nonzero(-np.round(mcc, 12) <= kth)[0]


def _search_bitmatrix(df: pl.DataFrame, target: list[dict], candidates: list[dict], max_terms: int = 2,
                      max_conds: int = 2, top: int = 20) -> list[dict]:
    """ビット行列で交わりを数え、並べる鍵はビット集合版と同じにする。max_terms は 1 または 2。"""
    _check(target, candidates)
    if not 1 <= max_terms <= 2:
        raise PropositionError("ビット行列は max_terms が 1 または 2 のときだけ")
    if df.height > _EXACT_UNITS:
        raise PropositionError(f"単位数 {df.height} はビット行列で交わり数を厳密に表せる範囲（{_EXACT_UNITS}）を超える")
    n = df.height
    y_int = _mask(df, target)
    n_pos, n_neg = y_int.bit_count(), n - y_int.bit_count()
    terms = _build_terms(df, candidates, max_conds)
    if not terms or top <= 0:
        return []
    groups = [g for g, _ in terms]
    T = _masks_to_matrix([m for _, m in terms], n)
    y = _mask_bits(y_int, n)
    if max_terms == 2 and len(terms) >= 2:
        a_tp, a_fp, tp, fp = _exact_counts(T, y)
    else:
        a_tp = (T & y).sum(1).astype(np.int64)
        a_fp = (T & ~y).sum(1).astype(np.int64)
        tp = fp = None
    m1 = _mcc_vec(a_tp.astype(np.float64), a_fp.astype(np.float64), n_pos, n_neg)
    iu = np.triu_indices(len(terms), 1) if tp is not None else (np.array([], np.int64), np.array([], np.int64))
    if iu[0].size:
        pair_tp, pair_fp = tp[iu], fp[iu]
        m2 = _mcc_vec(pair_tp.astype(np.float64), pair_fp.astype(np.float64), n_pos, n_neg)
    else:
        pair_tp = pair_fp = m2 = np.zeros(0, np.float64)
    allm = np.concatenate([m1, m2])
    if allm.size > top:
        kth = float(np.partition(-np.round(allm, 12), top - 1)[top - 1])
        keep1, keep2 = _keep_mcc(m1, kth), _keep_mcc(m2, kth)
    else:
        keep1, keep2 = np.arange(m1.size), np.arange(m2.size)
    # MCC の境界で全候補が同点になる場合も、Python の結果オブジェクトを全件保持しない。
    # 鍵は従来どおりで、同点も条件数・式の文字列まで比較して上位 top 件を選ぶ。
    singles = (_record([groups[i]], a_tp[i], a_fp[i], n_pos, n_neg) for i in keep1)
    pairs = (_record([groups[iu[0][k]], groups[iu[1][k]]], pair_tp[k], pair_fp[k], n_pos, n_neg)
             for k in keep2)
    rows = heapq.nsmallest(top, itertools.chain(singles, pairs), key=lambda t: t[0])
    return [r for _, r in rows]


def _bitmatrix_exact(max_terms: int, n_units: int) -> bool:
    return max_terms <= 2 and n_units <= _EXACT_UNITS


def _count_terms(candidates: list[dict], max_conds: int) -> int:
    """_build_terms が作る組の数（マスクは作らない）。"""
    cols = [c["col"] for c in candidates]
    return sum(1 for k in range(1, max_conds + 1) for combo in itertools.combinations(cols, k)
               if len(set(combo)) == k)


def _matrix_bytes(n_terms: int, n_units: int, max_terms: int) -> int:
    """ビット行列版が同時に持つ量の保守的な見積り（bytes）。
    T×n: bool 行列 3 枚 + float64 2 枚。max_terms=2 なら T×T: float64/int64 を 6 枚（Iy, In, tp, fp と一時配列）
    + 上三角の組 P=T(T-1)/2 についての index 2 本・tp/fp/MCC など 6 本（いずれも 8 bytes）。"""
    t, n = n_terms, max(n_units, 0)
    b = t * n * (3 + 2 * 8)
    if max_terms == 2 and t >= 2:
        pairs = t * (t - 1) // 2
        b += 6 * t * t * 8 + 8 * pairs * 8
    return b


def search(df: pl.DataFrame, target: list[dict], candidates: list[dict], max_terms: int = 2,
           max_conds: int = 2, top: int = 20, *, backend: str | None = None, prune: str | None = None,
           rank: int = 8, stages: int = 2) -> list[dict]:
    """df の各単位について、候補の条件からできる式を総当たりで評価する。値が空の単位は呼び出す側で除いておく。

    backend を省略したときは、ビット行列の結果がビット集合版と一致する範囲だけビット行列を使う。
    それ以外（max_terms が 3 以上、単位数が厳密に数える範囲を超える、または組の数が多くて
    行列の見積りメモリが _MATRIX_BYTES を超える）はビット集合のまま。top は 1 以上（それ以外は ValueError）。

    prune="littlebit" は実験用の保証付き枝刈りで、既定ではオフ。確認済み入力では厳密計算と一致するが、
    float64 の丸め誤差まで含む任意入力での形式的保証ではない。
    今の単位数では厳密計算より遅い（littlebit_prune のモジュール文書）。prune と backend は同時に指定できない
    （ValueError）。prune も m×m の行列を持つので、見積りが _MATRIX_BYTES を超えると PropositionError で止める。

    条件辞書は col, op, value でマスクを作る。PROVENANCE_KEYS の記録が付いていても、数え方と並べる鍵は同じである。
    """
    if backend not in (None, "bitmatrix", "bitset"):
        raise PropositionError(f"backend は bitmatrix / bitset / 省略: {backend}")
    if prune is not None and prune != "littlebit":
        raise PropositionError(f"prune は littlebit / 省略: {prune}")
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ValueError(f"top は 1 以上の整数: {top!r}")
    if prune is not None and backend is not None:
        raise ValueError(f"prune と backend は同時に指定できない: prune={prune}, backend={backend}")
    _check(target, candidates)
    if prune == "littlebit":
        need = _matrix_bytes(_count_terms(candidates, max_conds), df.height, 2)
        if need > _MATRIX_BYTES:
            raise PropositionError(
                f"littlebit の見積りメモリ {need} bytes が上限 {_MATRIX_BYTES} bytes を超える（候補を減らすか prune なし）")
        from littlebit_prune import search_guaranteed
        return search_guaranteed(df, target, candidates, max_terms, max_conds, top, rank=rank, stages=stages)
    use_matrix = _bitmatrix_exact(max_terms, df.height)
    if backend == "bitmatrix" and not use_matrix:
        raise PropositionError(
            f"ビット行列は max_terms<=2 かつ単位数<={_EXACT_UNITS} のときだけ（今 max_terms={max_terms}, 単位数={df.height}）")
    if use_matrix and backend != "bitset":
        need = _matrix_bytes(_count_terms(candidates, max_conds), df.height, max_terms)
        if need > _MATRIX_BYTES:
            if backend == "bitmatrix":
                raise PropositionError(
                    f"ビット行列の見積りメモリ {need} bytes が上限 {_MATRIX_BYTES} bytes を超える（候補を減らすか bitset）")
            use_matrix = False
    if backend == "bitset" or not use_matrix:
        return _search_bitset(df, target, candidates, max_terms, max_conds, top)
    return _search_bitmatrix(df, target, candidates, max_terms, max_conds, top)


def rule_text(r: dict) -> str:
    return " または ".join(f"[{text(g)}]" for g in r["if_any"])


def main(argv=None) -> int:
    from pythdragoras import apply_exclusions

    ap = argparse.ArgumentParser(description="条件の組み合わせを総当たりで探す（命題を作るための材料）")
    ap.add_argument("--season", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="分析設定（除外を適用する）")
    ap.add_argument("--candidates", type=Path, required=True, help="目的と候補の条件（TOML）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--backend", choices=("bitmatrix", "bitset"), default=None,
                    help="省略時は、結果が一致する範囲だけ bitmatrix。max_terms>=3 などは bitset")
    ap.add_argument("--prune", choices=("littlebit",), default=None,
                    help="実験用。littlebit は保証付き枝刈り（既定ではオフ。今の規模では厳密計算より遅い）")
    ap.add_argument("--rank", type=int, default=8, help="--prune littlebit の潜在次元")
    ap.add_argument("--stages", type=int, default=2, help="--prune littlebit の残差の段数")
    args = ap.parse_args(argv)

    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    spec = tomllib.loads(args.candidates.read_text(encoding="utf-8"))
    st = pl.read_ndjson(args.season)
    st, _ = apply_exclusions(st, cfg.get("exclude", []))
    cols = sorted({c["col"] for c in [*spec["target"], *spec["candidate"]]})
    usable = st.drop_nulls(cols)  # 値が空の単位は探索に入れない（件数は出力に残す）
    rules = search(usable, spec["target"], spec["candidate"], int(spec.get("max_terms", 2)),
                   int(spec.get("max_conds", 2)), args.top, backend=args.backend, prune=args.prune,
                   rank=args.rank, stages=args.stages)
    meta = {"units": usable.height, "dropped_for_missing_values": st.height - usable.height,
            "target": text(spec["target"]), "candidates": len(spec["candidate"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps({**meta, "rank": i + 1, **r}, ensure_ascii=False) + "\n"
                                for i, r in enumerate(rules)), encoding="utf-8")
    print(f"units={meta['units']} dropped={meta['dropped_for_missing_values']} target: {meta['target']}")
    for i, r in enumerate(rules[:10], 1):
        print(f"{i:2d}. mcc={r['mcc']:.3f} 成立率={r['precision']:.2f} 逆={r['recall']:.2f} 条件={r['conditions']}  {rule_text(r)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
