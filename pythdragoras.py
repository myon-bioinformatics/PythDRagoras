"""PythDRagoras — 本当にその説明で十分か？ 汎用版（標準ライブラリだけ・この 1 ファイルで動く）

題材を問わない計算だけを置く。シーズン表の検証（残差・順位の偏り・命題の実行）は npb/season.py。

  🧮 classification_metrics(tp, fp, fn, tn)  混同行列の指標（core/propositions.py と同じ定義）
  🎲 binom_tail / run_tail / markov_run_tail  回数・連続の珍しさ
  ✂️ apply_exclusions(rows, exclude, key)     理由つきの除外。何を・なぜ除いたかを返す
  🧱 littlebit(M, rank, stages)               低ランク × 符号の二値化 × スケール（交互最小二乗）
  📏 overlap_bounds(X, rank, stages)          行どうしの内積の近似と、Cauchy–Schwarz で示せる誤差幅
  🔝 mcc_upper(...)                           tp・fp が区間にあるときの MCC の上限
  🔢 to_bits / or_counts                      0/1 の列を整数のビット列にして、OR の tp・fp を popcount で数える

  python pythdragoras.py metrics 40 10 5 45
  python pythdragoras.py littlebit matrix.json --rank 2

LittleBit は発想と計算手法だけを借りる（公式実装の踏襲ではない）。行列は list の list、数百行・列までを想定
（純 Python の片側 Jacobi SVD）。numpy 版は core/littlebit_prune.py で、近似の結果が一致することをテストで確かめる。
"""
from __future__ import annotations

import math
from functools import lru_cache

_EPS = math.ulp(1.0)


class PropositionError(ValueError):
    """命題ファイルの誤り（終了コード 64）。"""

class DataError(PropositionError):
    """データの誤り。命題が参照する列がない、など（終了コード 65）。"""

def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Diagnostic classification axes; never change a proposition's verdict."""
    if any(type(v) is not int or v < 0 for v in (tp, fp, fn, tn)):
        raise DataError("confusion counts must be nonnegative integers")
    n = tp + fp + fn + tn
    divide = lambda a, b: a / b if b else None
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    values = dict(precision=divide(tp, tp + fp), recall=recall,
                  f1=divide(2 * tp, 2 * tp + fp + fn), accuracy=divide(tp + tn, n),
                  balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                  mcc=divide(tp * tn - fp * fn, denominator))
    reasons = dict(precision="no_positive_predictions", recall="no_positive_targets",
                   f1="no_positive_predictions_or_targets", accuracy="no_determined_units",
                   balanced_accuracy="missing_target_class", mcc="constant_prediction_or_target")
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, **values,
                none_reasons={key: reasons[key] if value is None else None for key, value in values.items()})


def binom_tail(n: int, k: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p)。"""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def run_tail(n: int, r: int, p: float) -> float:
    """独立な n 回の試行で、確率 p の事象が r 回以上連続する確率。"""
    if r <= 0:
        return 1.0

    @lru_cache(maxsize=None)
    def no_run(i: int, cur: int) -> float:  # 残り i 回で、現在 cur 連続中のとき、r 連続に達しない確率
        if cur >= r:
            return 0.0
        if i == 0:
            return 1.0
        return p * no_run(i - 1, cur + 1) + (1 - p) * no_run(i - 1, 0)

    return 1 - no_run(n, 0)


def markov_run_tail(n: int, r: int, p_bb: float, p_tb: float, pi_b: float) -> float:
    """2状態マルコフ連鎖（下位/上位）で、n シーズン中に下位が r 回以上連続する確率。

    p_bb = P(翌年も下位 | 今年下位), p_tb = P(翌年下位 | 今年上位), pi_b = 最初の年に下位の確率。
    戦力の持ち越し（前年の順位が翌年に影響すること）を入れた帰無仮説。
    """
    if r <= 0:
        return 1.0
    if n == 0:
        return 0.0

    @lru_cache(maxsize=None)
    def no_run(i: int, cur: int, last_bottom: bool) -> float:  # 残り i 年、現在 cur 連続、直前が下位か
        if cur >= r:
            return 0.0
        if i == 0:
            return 1.0
        pb = p_bb if last_bottom else p_tb
        return pb * no_run(i - 1, cur + 1, True) + (1 - pb) * no_run(i - 1, 0, False)

    first = pi_b * no_run(n - 1, 1, True) + (1 - pi_b) * no_run(n - 1, 0, False)
    return 1 - first


def apply_exclusions(table, exclude: list[dict], key: str = "season"):
    """exclude = [{key: 値, "reason": "..."}]。理由のない除外は受け付けない。

    table は dict の list か、polars の DataFrame（列の演算だけ使い、polars は import しない）。残りと、除いた記録を返す。
    """
    for e in exclude:
        if not str(e.get("reason", "")).strip():
            raise ValueError(f"除外には理由が必要: {e}")
    values = [int(e[key]) for e in exclude]
    if hasattr(table, "filter"):
        column = table[key]
        record = [{**e, "rows_removed": int((column == int(e[key])).sum())} for e in exclude]
        return table.filter(~column.is_in(values)), record
    rows = list(table)
    record = [{**e, "rows_removed": sum(1 for r in rows if r.get(key) == int(e[key]))} for e in exclude]
    return [r for r in rows if r.get(key) not in values], record


# ── 🧱 LittleBit（行列は list の list） ─────────────────────────────────────────────

def _matrix(M) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in M]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("行列は空でない長方形（list の list）にする")
    if not all(math.isfinite(v) for r in rows for v in r):
        raise ValueError("行列の値は有限の数にする")
    return rows


def _t(A):
    return [list(col) for col in zip(*A)]


def _mul(A, B):
    Bt = _t(B)
    return [[math.fsum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def svd(A) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """薄い SVD（片側 Jacobi）。U (m×k)、S (k、降順)、Vt (k×n)、k = min(m, n)。U の各列は絶対値最大の成分を正にする。"""
    A = _matrix(A)
    transposed = len(A) < len(A[0])
    W = [row[:] for row in (_t(A) if transposed else A)]
    m, n = len(W), len(W[0])
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(60):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                app = math.fsum(W[i][p] * W[i][p] for i in range(m))
                aqq = math.fsum(W[i][q] * W[i][q] for i in range(m))
                apq = math.fsum(W[i][p] * W[i][q] for i in range(m))
                if abs(apq) <= _EPS * math.sqrt(app * aqq) or apq == 0.0:
                    continue
                rotated = True
                tau = (aqq - app) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for R in (W, V):
                    for row in R:
                        a, b = row[p], row[q]
                        row[p], row[q] = c * a - s * b, s * a + c * b
        if not rotated:
            break
    norms = [math.sqrt(math.fsum(W[i][j] * W[i][j] for i in range(m))) for j in range(n)]
    order = sorted(range(n), key=lambda j: (-norms[j], j))
    S = [norms[j] for j in order]
    U = [[W[i][j] / norms[j] if norms[j] > 0.0 else 0.0 for j in order] for i in range(m)]
    Vt = [[V[i][j] for i in range(n)] for j in order]
    for k in range(n):
        if U and max((U[i][k] for i in range(m)), key=abs) < 0.0:
            for i in range(m):
                U[i][k] = -U[i][k]
            Vt[k] = [-v for v in Vt[k]]
    if transposed:
        return _t(Vt), S, _t(U)
    return U, S, Vt


def lstsq(A, b) -> list[float]:
    """最小二乗の解（最小ノルム）。numpy.linalg.lstsq(rcond=None) と同じく、ε·max(m, n)·σ_max 以下の特異値は 0 とみなす。"""
    U, S, Vt = svd(A)
    cutoff = _EPS * max(len(U), len(Vt[0])) * (S[0] if S else 0.0)
    coef = [math.fsum(U[i][k] * b[i] for i in range(len(U))) / s if s > cutoff else 0.0 for k, s in enumerate(S)]
    return [math.fsum(Vt[k][j] * coef[k] for k in range(len(S))) for j in range(len(Vt[0]))]


def _sign(x: float) -> float:
    return -1.0 if x < 0.0 else 1.0  # 0 は +1（numpy 版と同じ）


def littlebit(M, rank: int, stages: int = 2, als_iters: int = 6) -> list[dict]:
    """M ≈ Σ_段 diag(h) sign(U) diag(l) sign(V)^T diag(g)。段ごとに h・u（±1）・l・v（±1）・g を返す。"""
    R = _matrix(M)
    if rank < 1 or stages < 1 or als_iters < 0:
        raise ValueError("rank と stages は 1 以上、als_iters は 0 以上にする")
    m, n = len(R), len(R[0])
    mean = lambda xs: math.fsum(xs) / len(xs)
    parts = []
    for _ in range(stages):
        U, S, Vt = svd(R)
        r = min(rank, len(S))
        if r <= 0 or S[0] == 0.0:
            break
        Us = [[_sign(U[i][k]) for k in range(r)] for i in range(m)]
        Vs = [[_sign(Vt[k][j]) for k in range(r)] for j in range(n)]
        h = [mean([abs(U[i][k]) for k in range(r)]) for i in range(m)]
        g = [mean([abs(Vt[k][j]) for k in range(r)]) for j in range(n)]
        h = [x / (mean(h) or 1.0) for x in h]
        g = [x / (mean(g) or 1.0) for x in g]
        l = [S[k] * mean([abs(U[i][k]) for i in range(m)]) * mean([abs(Vt[k][j]) for j in range(n)])
             / ((mean(h) or 1.0) * (mean(g) or 1.0)) for k in range(r)]
        for _it in range(als_iters):
            C = [[math.fsum(Us[i][k] * l[k] * Vs[j][k] for k in range(r)) for j in range(n)] for i in range(m)]
            h = [math.fsum(R[i][j] * C[i][j] * g[j] for j in range(n))
                 / max(math.fsum((C[i][j] * g[j]) ** 2 for j in range(n)), 1e-12) for i in range(m)]
            g = [math.fsum(R[i][j] * C[i][j] * h[i] for i in range(m))
                 / max(math.fsum((C[i][j] * h[i]) ** 2 for i in range(m)), 1e-12) for j in range(n)]
            Hu = [[h[i] * Us[i][k] for k in range(r)] for i in range(m)]
            Gv = [[g[j] * Vs[j][k] for k in range(r)] for j in range(n)]
            HH, GG = _mul(_t(Hu), Hu), _mul(_t(Gv), Gv)
            A = [[HH[a][b] * GG[a][b] for b in range(r)] for a in range(r)]  # 因子の内積。m×n の外積を作らない
            RGv = _mul(R, Gv)
            b = [math.fsum(Hu[i][k] * RGv[i][k] for i in range(m)) for k in range(r)]
            l = lstsq(A, b)
        part = {"h": h, "u": Us, "l": l, "v": Vs, "g": g}
        parts.append(part)
        approx = reconstruct([part], m, n)
        R = [[R[i][j] - approx[i][j] for j in range(n)] for i in range(m)]
    return parts


def reconstruct(parts: list[dict], m: int | None = None, n: int | None = None) -> list[list[float]]:
    """littlebit の段を足し合わせた近似 Â。"""
    if not parts:
        return [[0.0] * (n or 0) for _ in range(m or 0)]
    m, n = len(parts[0]["h"]), len(parts[0]["g"])
    out = [[0.0] * n for _ in range(m)]
    for p in parts:
        r = len(p["l"])
        for i in range(m):
            for j in range(n):
                out[i][j] += p["h"][i] * p["g"][j] * math.fsum(p["u"][i][k] * p["l"][k] * p["v"][j][k] for k in range(r))
    return out


def overlap_bounds(X, rank: int, stages: int = 2) -> tuple[list[list[float]], list[list[float]]]:
    """行どうしの内積 x_a·x_b の近似 â_a·â_b と、|x_a·x_b − â_a·â_b| ≤ ||r_a||·||x_b|| + ||â_a||·||r_b|| の幅（対称）。"""
    X = _matrix(X)
    m, n = len(X), len(X[0])
    Ahat = reconstruct(littlebit(X, rank, stages), m, n)
    approx = _mul(Ahat, _t(Ahat))
    norm = lambda row: math.sqrt(math.fsum(v * v for v in row))
    residual = [norm([X[i][j] - Ahat[i][j] for j in range(n)]) for i in range(m)]
    row_norm, approx_norm = [norm(r) for r in X], [norm(r) for r in Ahat]
    err = [[residual[a] * row_norm[b] + approx_norm[a] * residual[b] for b in range(m)] for a in range(m)]
    return approx, [[max(err[a][b], err[b][a]) for b in range(m)] for a in range(m)]


def mcc_upper(tp_lo: float, tp_hi: float, fp_lo: float, fp_hi: float, n_pos: int, n_neg: int) -> float:
    """tp ∈ [tp_lo, tp_hi]、fp ∈ [fp_lo, fp_hi] のときの MCC の上限。

    分子 tp·N − fp·P は tp の上端と fp の下端で最大。分母の s(n−s) は凹なので s = tp+fp の区間の両端で最小。
    分子の最大が 0 以下なら上限は 0、分母が 0 になりうるなら上限は ∞（捨てない側に倒す）。
    """
    n = n_pos + n_neg
    num = tp_hi * n_neg - fp_lo * n_pos
    if num <= 0:
        return 0.0
    s_lo, s_hi = min(max(tp_lo + fp_lo, 0), n), min(max(tp_hi + fp_hi, 0), n)
    q = min(s_lo * (n - s_lo), s_hi * (n - s_hi))
    return num / math.sqrt(n_pos * n_neg * q) if q > 0 and n_pos * n_neg > 0 else math.inf


def to_bits(values) -> int:
    """0/1（真偽）の列を整数のビット列にする。i 番目の値が i ビット目。"""
    return sum(1 << i for i, v in enumerate(values) if v)


def or_counts(a: int, b: int, target: int, n: int) -> tuple[int, int]:
    """条件 a または b が成り立つ単位のうち、target が真の数（tp）と偽の数（fp）。n は単位の数。"""
    hit = a | b
    return (hit & target).bit_count(), (hit & ~target & ((1 << n) - 1)).bit_count()


def main(argv=None):
    import argparse
    import json
    import sys
    from pathlib import Path
    argv = list(sys.argv[1:] if argv is None else argv)
    # Retain the initial draft's positional interface.
    if argv and argv[0] not in ("metrics", "bayes", "expected", "pythagorean", "littlebit", "-h", "--help"):
        argv.insert(0, "metrics")
    parser = argparse.ArgumentParser(description="Calculate classification metrics and expectations.")
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("metrics", help="classification metrics from confusion counts")
    metrics.add_argument("counts", nargs="*", type=int, metavar="COUNT")
    for name in ("tp", "fp", "fn", "tn"):
        metrics.add_argument("--" + name, type=int)
    bayes = sub.add_parser("bayes", help="posterior probability given a positive result")
    for name in ("prior", "sensitivity", "specificity"):
        bayes.add_argument("--" + name, type=float, required=True)
    expected = sub.add_parser("expected", help="expected value from value/probability pairs")
    expected.add_argument("pairs", type=float, nargs="+", metavar="NUMBER")
    pyth = sub.add_parser("pythagorean", help="Pythagorean expectation")
    pyth.add_argument("--for", dest="scored", type=float, required=True)
    pyth.add_argument("--against", type=float, required=True)
    pyth.add_argument("--exponent", type=float, default=2.0)
    lb = sub.add_parser("littlebit", help="🧱 LittleBit factors of a JSON matrix (list of rows)")
    lb.add_argument("matrix", nargs="?", type=Path, help="JSON file; standard input when omitted")
    lb.add_argument("--rank", type=int, default=2)
    lb.add_argument("--stages", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "metrics":
        flags = [args.tp, args.fp, args.fn, args.tn]
        if args.counts and any(x is not None for x in flags):
            parser.error("use four positional counts or all four named counts")
        counts = args.counts if args.counts else flags
        if len(counts) != 4 or any(x is None for x in counts):
            parser.error("provide TP FP FN TN counts")
        try:
            result = classification_metrics(*counts)
        except DataError as exc:
            parser.error(str(exc))
    elif args.command == "bayes":
        prior, sensitivity, specificity = args.prior, args.sensitivity, args.specificity
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (prior, sensitivity, specificity)):
            parser.error("probabilities must be finite and between 0 and 1")
        # Test mathematical zero before multiplication: valid tiny probabilities
        # can otherwise underflow and be mistaken for an impossible result.
        true_possible = prior > 0 and sensitivity > 0
        false_possible = prior < 1 and specificity < 1
        if true_possible and false_possible:
            log_true = math.log(prior) + math.log(sensitivity)
            log_false = math.log1p(-prior) + math.log1p(-specificity)
            scale = max(log_true, log_false)
            true_weight = math.exp(log_true - scale)
            false_weight = math.exp(log_false - scale)
            posterior = true_weight / (true_weight + false_weight)
        else:
            posterior = float(true_possible) if true_possible or false_possible else None
        result = {"posterior": posterior,
                  "none_reason": "positive_result_has_zero_probability" if posterior is None else None}
    elif args.command == "expected":
        pairs = args.pairs
        if len(pairs) % 2 or any(not math.isfinite(x) for x in pairs):
            parser.error("provide finite value/probability pairs")
        values, probabilities = pairs[::2], pairs[1::2]
        if any(not 0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            parser.error("probabilities must be between 0 and 1 and sum to 1")
        result = {"expected_value": math.fsum(v * p for v, p in zip(values, probabilities))}
    elif args.command == "littlebit":
        try:
            text = args.matrix.read_text(encoding="utf-8") if args.matrix else sys.stdin.read()
            M = _matrix(json.loads(text))
            parts = littlebit(M, args.rank, args.stages)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f"littlebit: {exc}")
        approx = reconstruct(parts, len(M), len(M[0]))
        total = math.sqrt(math.fsum(v * v for row in M for v in row))
        diff = math.sqrt(math.fsum((x - y) ** 2 for row, arow in zip(M, approx) for x, y in zip(row, arow)))
        result = {"shape": [len(M), len(M[0])], "rank": args.rank, "stages": len(parts),
                  "relative_error": diff / total if total else 0.0,
                  "parts": [{**p, "u": [[int(x) for x in row] for row in p["u"]],
                             "v": [[int(x) for x in row] for row in p["v"]]} for p in parts]}
    else:
        a, b, exponent = args.scored, args.against, args.exponent
        if any(not math.isfinite(x) for x in (a, b, exponent)) or min(a, b) < 0 or exponent <= 0:
            parser.error("scores must be nonnegative finite numbers and exponent must be positive")
        if a == b == 0:
            result = {"expectation": None, "none_reason": "both_scores_zero"}
        else:
            scale = max(a, b)
            x, y = (a / scale) ** exponent, (b / scale) ** exponent
            result = {"expectation": x / (x + y), "none_reason": None, "exponent": exponent}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
