"""実験用。search の既定では使わない。

低ランク分解、符号での二値化、行・列・潜在次元のスケール（交互最小二乗）、残差の段で
組の交わりを近似し、Cauchy–Schwarz で誤差の上限を出す。MCC の上限が、すでに厳密に数えた
候補の top 件目より低い組だけ捨て、残った組はビット列の popcount で数え直す。
実数演算では、捨てた組が上位に入らない範囲だけを捨てる。実装は float64 と固定の許容誤差を使うため、
任意の入力・BLAS に対する丸め誤差込みの形式的保証ではない。確認済み入力では厳密計算と一致する。

観測（試作、2026-10-12、共有の機械で 1 回）: 今の規模では厳密計算より遅い。
156 単位・80 条件で約 7 秒対 2.6 秒、合成 5,000 単位・45 条件で約 47〜54 秒対 0.37 秒。
解釈: 単位が 156 しかないと交わりは 156 次元の内積で済み、近似を作るコストの方が重い。
速くなるのは単位がとても多く、近似を一度作ってから何度も使う場合に限られそうだが、この範囲では未確認。
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from propositions import PropositionError
from rule_search import _build_terms, _mask, _mask_bits, _masks_to_matrix, _mcc_vec, _record


def littlebit(M: np.ndarray, rank: int, stages: int = 2, als_iters: int = 6):
    """M ≈ Σ_stage diag(h) sign(U) diag(l) sign(V)^T diag(g)。各段の因子を返す。"""
    R = np.array(M, dtype=np.float64, copy=True)
    parts = []
    for _ in range(stages):
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        r = min(rank, int(S.shape[0]))
        if r <= 0:
            break
        Us, Vs = np.sign(U[:, :r]), np.sign(Vt[:r].T)
        Us[Us == 0] = 1
        Vs[Vs == 0] = 1
        h = np.abs(U[:, :r]).mean(1)
        g = np.abs(Vt[:r].T).mean(1)
        h = h / (h.mean() if h.mean() else 1.0)
        g = g / (g.mean() if g.mean() else 1.0)
        # Â_ij = h_i g_j Σ_k l_k Us_ik Vs_jk
        l = S[:r] * np.abs(U[:, :r]).mean(0) * np.abs(Vt[:r].T).mean(0) / ((h.mean() or 1.0) * (g.mean() or 1.0))
        for _it in range(als_iters):
            C = (Us * l) @ Vs.T
            Bh = C * g[None, :]
            h = (R * Bh).sum(1) / np.maximum((Bh * Bh).sum(1), 1e-12)
            Bg = C * h[:, None]
            g = (R * Bg).sum(0) / np.maximum((Bg * Bg).sum(0), 1e-12)
            Hu, Gv = h[:, None] * Us, g[:, None] * Vs
            A = (Hu.T @ Hu) * (Gv.T @ Gv)  # 因子の内積。m×n の外積を作らない
            b = np.einsum("ik,ij,jk->k", Hu, R, Gv, optimize=True)
            l = np.linalg.lstsq(A, b, rcond=None)[0]
        F = h[:, None] * Us * l
        parts.append((F, Vs, g))
        R = R - F @ (Vs * g[:, None]).T
    return parts


def _gram(parts):
    """近似 Â の Â Â^T を、潜在次元 r の積で計算する。"""
    F = np.hstack([p[0] for p in parts])
    W = np.hstack([p[1] * p[2][:, None] for p in parts])
    return F, W.T @ W


def _dense(parts):
    return sum(F @ (Vs * g[:, None]).T for F, Vs, g in parts)


def overlap_bounds(X: np.ndarray, rank: int, stages: int):
    """行どうしの内積の近似と、Cauchy–Schwarz で証明できる誤差幅。"""
    parts = littlebit(X, rank, stages)
    if not parts:
        m = X.shape[0]
        return np.zeros((m, m)), np.zeros((m, m))
    F, G = _gram(parts)
    approx = F @ G @ F.T
    Ahat = _dense(parts)
    residual = np.linalg.norm(X - Ahat, axis=1)
    row_norm = np.sqrt(np.maximum(X.sum(1), 0.0))
    approx_norm = np.linalg.norm(Ahat, axis=1)
    # |x_a·x_b − â_a·â_b| ≤ ||r_a||·||x_b|| + ||â_a||·||r_b||
    err = residual[:, None] * row_norm[None, :] + approx_norm[:, None] * residual[None, :]
    err = np.maximum(err, err.T)
    return approx, err


def mcc_upper(tp_lo, tp_hi, fp_lo, fp_hi, n_pos: int, n_neg: int):
    """tp・fp がそれぞれの区間にあるときの MCC の上限。

    分子 tp·N − fp·P は tp の上端と fp の下端で最大。分母の s(n−s) は凹なので、
    s=tp+fp の区間の両端で最小になる。分子の最大が 0 以下なら、箱の中の MCC は 0 以下。
    """
    n = n_pos + n_neg
    num = tp_hi * n_neg - fp_lo * n_pos
    s_lo = np.clip(tp_lo + fp_lo, 0, n)
    s_hi = np.clip(tp_hi + fp_hi, 0, n)
    q = np.minimum(s_lo * (n - s_lo), s_hi * (n - s_hi))
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = np.where(q > 0, num / np.sqrt(n_pos * n_neg * np.where(q > 0, q, 1.0)), np.inf)
    return np.where(num <= 0, 0.0, ub)


def _count_interval(a, b, approx, err, cap: int):
    """交わり I ∈ [Î−e, Î+e] ∩ [0, min(|a|,|b|)] から、和集合の大きさ |a|+|b|−I の下端と上端。"""
    feasible = np.minimum(np.minimum(a, b), cap)
    i_hi = np.clip(np.floor(approx + err + 1e-9), 0, feasible)
    i_lo = np.clip(np.ceil(approx - err - 1e-9), 0, feasible)
    # 数値誤差で区間が裏返ったときは、ありうる範囲を全部残す（捨てすぎない）
    bad = i_lo > i_hi
    i_lo = np.where(bad, 0, i_lo)
    i_hi = np.where(bad, feasible, i_hi)
    lo = np.maximum(a + b - i_hi, np.maximum(a, b))
    hi = np.minimum(a + b - i_lo, cap)
    bad = lo > hi
    lo = np.where(bad, np.maximum(a, b), lo)
    hi = np.where(bad, np.minimum(a + b, cap), hi)
    return lo, hi


def _exact_pairs(T: np.ndarray, y: np.ndarray, I, J, chunk: int = 200_000):
    """生き残った組だけ、詰めたビット列の OR と popcount で厳密に数える。"""
    Py = np.packbits(T & y, axis=1)
    Pn = np.packbits(T & ~y, axis=1)
    tp = np.empty(len(I), np.int64)
    fp = np.empty(len(I), np.int64)
    for s in range(0, len(I), chunk):
        i, j = I[s:s + chunk], J[s:s + chunk]
        tp[s:s + chunk] = np.bitwise_count(Py[i] | Py[j]).sum(1)
        fp[s:s + chunk] = np.bitwise_count(Pn[i] | Pn[j]).sum(1)
    return tp, fp


def search_guaranteed(df, target, candidates, max_terms: int = 2, max_conds: int = 2, top: int = 20, *,
                      rank: int = 8, stages: int = 2, seed_mult: int = 5):
    """保証付き枝刈り。max_terms=2 のみ。結果はビット集合版と同じ上位 top 件。"""
    if max_terms != 2:
        raise PropositionError("保証付き枝刈りは max_terms=2 のみ")
    if rank < 1 or stages < 1:
        raise PropositionError(f"rank と stages は 1 以上: rank={rank}, stages={stages}")
    if top <= 0:
        return []
    n = df.height
    y_int = _mask(df, target)
    n_pos, n_neg = y_int.bit_count(), n - y_int.bit_count()
    terms = _build_terms(df, candidates, max_conds)
    if not terms:
        return []
    if n == 0:
        # 単位が無いと近似も枝刈りも意味がない。厳密計算（全 0 の行）と同じ結果を返す
        from rule_search import _search_bitset
        return _search_bitset(df, target, candidates, max_terms, max_conds, top)
    groups = [g for g, _ in terms]
    T = _masks_to_matrix([m for _, m in terms], n)
    y = _mask_bits(y_int, n)
    Ty, Tn = T & y, T & ~y
    a_tp = Ty.sum(1).astype(np.int64)
    a_fp = Tn.sum(1).astype(np.int64)
    m1 = _mcc_vec(a_tp.astype(np.float64), a_fp.astype(np.float64), n_pos, n_neg)
    iu = np.triu_indices(len(terms), 1)
    if iu[0].size == 0:
        rows = [_record([groups[i]], a_tp[i], a_fp[i], n_pos, n_neg) for i in range(len(groups))]
        rows.sort(key=lambda t: t[0])
        return [r for _, r in rows[:top]]

    by, ey = overlap_bounds(Ty.astype(np.float64), rank, stages)
    bn, en = overlap_bounds(Tn.astype(np.float64), rank, stages)
    Iy, Ey = by[iu], ey[iu]
    In, En = bn[iu], en[iu]
    tp_hat = a_tp[iu[0]] + a_tp[iu[1]] - Iy
    fp_hat = a_fp[iu[0]] + a_fp[iu[1]] - In
    approx_m = _mcc_vec(tp_hat, fp_hat, n_pos, n_neg)
    # 近似の上位を厳密に数え、確定した MCC の top 件目を捨てる基準の下端にする
    # （部分集合の top 件目は、全体の top 件目より高くならない）
    n_pairs = int(approx_m.size)
    seed_n = min(n_pairs, top * seed_mult)
    seed = np.argpartition(-np.nan_to_num(approx_m, nan=-np.inf), seed_n - 1)[:seed_n]
    s_tp, s_fp = _exact_pairs(T, y, iu[0][seed], iu[1][seed])
    known = np.concatenate([m1, _mcc_vec(s_tp.astype(np.float64), s_fp.astype(np.float64), n_pos, n_neg)])
    thr = float(np.sort(known)[::-1][min(top, known.size) - 1])
    lo_y, hi_y = _count_interval(a_tp[iu[0]], a_tp[iu[1]], Iy, Ey, n_pos)
    lo_n, hi_n = _count_interval(a_fp[iu[0]], a_fp[iu[1]], In, En, n_neg)
    ub = mcc_upper(lo_y, hi_y, lo_n, hi_n, n_pos, n_neg)
    # 上限が有限でない組は捨てない（数値誤差で上限が壊れたときに取りこぼさない）
    keep = np.nonzero(~np.isfinite(ub) | (ub + 1e-9 >= thr))[0]
    k_tp, k_fp = _exact_pairs(T, y, iu[0][keep], iu[1][keep])
    m2 = _mcc_vec(k_tp.astype(np.float64), k_fp.astype(np.float64), n_pos, n_neg)
    allm = np.concatenate([m1, m2])
    if allm.size > top:
        kth = float(np.partition(-np.round(allm, 12), top - 1)[top - 1])
        keep1 = np.nonzero(-np.round(m1, 12) <= kth)[0]
        keep2 = np.nonzero(-np.round(m2, 12) <= kth)[0]
    else:
        keep1, keep2 = np.arange(m1.size), np.arange(m2.size)
    # 境界が全同点でも結果オブジェクトを全件保持しない（厳密 bitmatrix と同じ順位キー）。
    singles = (_record([groups[i]], a_tp[i], a_fp[i], n_pos, n_neg) for i in keep1)
    pairs = (_record([groups[iu[0][keep[k]]], groups[iu[1][keep[k]]]], k_tp[k], k_fp[k], n_pos, n_neg)
             for k in keep2)
    rows = heapq.nsmallest(top, itertools.chain(singles, pairs), key=lambda t: t[0])
    return [r for _, r in rows]
