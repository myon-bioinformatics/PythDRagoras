"""候補条件の 0/1 行列を LittleBit 式に分解し、潜在パターンを新しい条件候補にする。

探索の材料を足す経路である。成否（成立・反例）は rule_search.search のビット演算で数える。
因子の計算は littlebit_prune.littlebit を使う（低ランク、符号による二値、行・列・潜在のスケール、残差の段）。
公式の LittleBit カーネルは使わない。目的の列（B クラスなど）は因子に渡さない。

規則（結果を見て動かさない）:

  行列 X は単位 × 候補条件。列 j は search と同じ _mask。
  littlebit の各段・各潜在軸 k について、単位側の因子を F[:, k] とする。
  littlebit の定義では F = h * Us * l（行スケール h、符号二値 Us、潜在スケール l）。
  列側の重みは w_j = s * g_j * Vs_jk（列スケール g、符号二値 Vs、軸の符号 s）。
  s は +1 または -1。np.sign(F[:, k]) の和が正なら +1、負なら -1。
  和が 0（正と負が同数）のときだけ seed を使う。seed から作った乱数の最初の 1 ビットが
  0 なら最初の非ゼロ成分が正、1 なら負になる向きを s とする。同点以外では seed は記録するだけで軸を変えない。
  単位 i が潜在パターン k に乗るのは、s * F[i, k] > 0 のとき。0 ちょうどは乗らない。
  これは因子の符号であり、.040 / .060 のような境界を刻む手続きではない。

  次の軸は条件にしない（人が書いた候補は消さない）:
    zero-factor     F の列が全部 0
    constant-mask   乗る単位が 0 か全員
    duplicate-mask  すでに足したパターンと同じ 0/1
    rank-shortfall  要求した rank に対して SVD の成分が足りない
    no-units / no-candidates  行列が空

  残差は X - Xhat の行ごとの二乗和。平均より大きい単位を、判例を探す手がかりにする。
  平均ちょうどは手がかりにしない。目的の列では分けない。

出どころ: source=littlebit-approx、rank、seed、束ねた元の条件と重み、posthoc=true。
registration は registration_audit.classify の結果（この作り方では posthoc）。
exploratory は探索であることの印で、登録区分を増やすものではない。

既定の rank=8, stages=2 は littlebit 枝刈りのコマンド既定と同じである。seed=0。
この並びの MCC を見てから選んだ値ではない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

import littlebit_prune
import registration_audit
import rule_search
from propositions import PropositionError, text
from rule_search import rule_text, search

# littlebit() の既定。呼び出す側と記録を同じ数にする。
ALS_ITERS = 6
RULE_ID = "canonical-sign-of-F"
SOURCE = "littlebit-approx"
DEFAULT_RANK = 8
DEFAULT_STAGES = 2
DEFAULT_SEED = 0
SEARCH_FIELDS = (
    "col", "op", "value", "source", "rank", "seed", "stage", "pattern", "posthoc",
    "exploratory", "registration", "bundle", "sign", "rule", "n_loading",
)


def _positive_int(name: str, value: object, *, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PropositionError(f"{name} は整数: {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        raise PropositionError(f"{name} は {'0 以上' if allow_zero else '1 以上'}: {value}")
    return value


def pattern_sign(values: np.ndarray, seed: int) -> int:
    """軸の符号 s。詳細はモジュール冒頭。戻りは +1 または -1。"""
    balance = float(np.sign(values).sum())
    if balance > 0:
        return 1
    if balance < 0:
        return -1
    nz = np.flatnonzero(values != 0)
    if nz.size == 0:
        return 1
    bit = int(np.random.Generator(np.random.PCG64(seed)).integers(0, 2))
    first = int(np.sign(values[int(nz[0])]))
    return first if bit == 0 else -first


def indicator_matrix(df: pl.DataFrame, candidates: list[dict]) -> np.ndarray:
    """単位 × 候補条件の bool 行列。列は search のマスクと同じ。"""
    rule_search._check([], candidates)
    masks = [rule_search._mask(df, [c]) for c in candidates]
    return rule_search._masks_to_matrix(masks, df.height).T.copy()


def unit_labels(df: pl.DataFrame) -> list[str]:
    """球団年。team と season が無い表は row-番号。重複は止める。"""
    labels = []
    has_id = "team" in df.columns and "season" in df.columns
    for i in range(df.height):
        if has_id:
            labels.append(f"{df['team'][i]}-{int(df['season'][i])}")
        else:
            labels.append(f"row-{i}")
    if len(labels) != len(set(labels)):
        raise PropositionError("球団年の標識が重複している")
    return labels


def _logic_condition(cand: dict) -> dict:
    return {"col": cand["col"], "op": cand["op"], "value": cand["value"]}


def _cell(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def derive(df: pl.DataFrame, candidates: list[dict], *, rank: int = DEFAULT_RANK,
           stages: int = DEFAULT_STAGES, seed: int = DEFAULT_SEED) -> dict:
    """潜在パターン由来の条件と、残差の手がかりを作る。target は受け取らない。"""
    rank = _positive_int("rank", rank, allow_zero=False)
    stages = _positive_int("stages", stages, allow_zero=False)
    seed = _positive_int("seed", seed, allow_zero=True)
    rule_search._check([], candidates)
    n, p = df.height, len(candidates)
    requested = rank * stages
    labels = unit_labels(df) if n else []
    drops: list[dict] = []
    if n == 0 or p == 0:
        reason = "no-units" if n == 0 else "no-candidates"
        drops = [{"stage": s, "pattern": k, "reason": reason} for s in range(stages) for k in range(rank)]
        report = _report(rank, stages, seed, n, p, [], drops, 0, np.zeros((n, p)), np.zeros((n, p)),
                         labels, df, candidates)
        return {"frame": df, "candidates": [], "report": report}

    matrix = indicator_matrix(df, candidates)
    parts = littlebit_prune.littlebit(matrix, rank, stages, als_iters=ALS_ITERS)
    seen: dict[bytes, str] = {}
    search_cands: list[dict] = []
    report_cands: list[dict] = []
    series: list[pl.Series] = []
    sign_ties = 0
    for stage in range(stages):
        if stage >= len(parts):
            drops.extend({"stage": stage, "pattern": k, "reason": "rank-shortfall"} for k in range(rank))
            continue
        factor, signs_v, col_scale = parts[stage][0], parts[stage][1], parts[stage][2]
        available = int(factor.shape[1])
        drops.extend({"stage": stage, "pattern": k, "reason": "rank-shortfall"}
                     for k in range(available, rank))
        for k in range(available):
            column = factor[:, k]
            if not np.any(column):
                drops.append({"stage": stage, "pattern": k, "reason": "zero-factor", "n_loading": 0})
                continue
            if float(np.sign(column).sum()) == 0.0:
                sign_ties += 1
            sign = pattern_sign(column, seed)
            loads = (sign * column) > 0
            n_load = int(loads.sum())
            if n_load == 0 or n_load == n:
                drops.append({"stage": stage, "pattern": k, "reason": "constant-mask", "n_loading": n_load})
                continue
            key = np.ascontiguousarray(loads).tobytes()
            col = f"littlebit_s{stage}_k{k}"
            if key in seen:
                drops.append({"stage": stage, "pattern": k, "reason": "duplicate-mask",
                              "same_as": seen[key], "n_loading": n_load})
                continue
            if col in df.columns:
                raise PropositionError(f"列がすでにある: {col}")
            seen[key] = col
            weights = sign * col_scale * signs_v[:, k]
            bundle = []
            for j, cand in enumerate(candidates):
                weight = float(weights[j])
                bundle.append({**_logic_condition(cand), "weight": 0.0 if weight == 0.0 else weight})
            record = {
                "col": col, "op": "==", "value": True, "source": SOURCE, "rank": rank, "seed": seed,
                "stage": stage, "pattern": k, "posthoc": True, "exploratory": True, "rule": RULE_ID,
                "sign": sign, "n_loading": n_load, "bundle": bundle,
            }
            record["registration"] = registration_audit.classify(record).status
            search_cands.append(record)
            report_cands.append({**record, "loading_units": [labels[i] for i, flag in enumerate(loads.tolist()) if flag]})
            series.append(pl.Series(col, loads.tolist()))
    frame = df.with_columns(series) if series else df
    approx = _reconstruct(parts, n, p)
    report = _report(rank, stages, seed, n, p, report_cands, drops, sign_ties, matrix, approx, labels, df, candidates)
    return {"frame": frame, "candidates": search_cands, "report": report}


def _reconstruct(parts, n: int, p: int) -> np.ndarray:
    acc = np.zeros((n, p), dtype=np.float64)
    for part in parts:
        factor, signs_v, col_scale = part[0], part[1], part[2]
        acc += factor @ (signs_v * col_scale[:, None]).T
    return acc


def _report(rank, stages, seed, n, p, derived, drops, sign_ties, matrix, approx, labels, df, candidates) -> dict:
    resid = matrix.astype(np.float64) - approx
    energy = np.sum(resid * resid, axis=1) if n else np.zeros(0, dtype=np.float64)
    mean_energy = float(energy.mean()) if n else 0.0
    if mean_energy == 0.0:
        mean_energy = 0.0
    units = []
    for i in range(n):
        order = np.argsort(-np.abs(resid[i]))[:min(3, p)]
        worst = []
        for j in order:
            j = int(j)
            gap = float(resid[i, j])
            worst.append({**_logic_condition(candidates[j]), "residual": 0.0 if gap == 0.0 else gap})
        satisfied = [_logic_condition(c) for j, c in enumerate(candidates) if matrix[i, j]]
        value = float(energy[i])
        item = {
            "unit": labels[i], "row": i,
            "residual_energy": 0.0 if value == 0.0 else value,
            "lead": bool(value > mean_energy),
            "n_satisfied": len(satisfied), "satisfied": satisfied,
            "largest_abs_residual": worst,
        }
        if "team" in df.columns:
            item["team"] = _cell(df["team"][i])
        if "season" in df.columns:
            item["season"] = int(df["season"][i])
        if "team_name" in df.columns:
            item["team_name"] = _cell(df["team_name"][i])
        units.append(item)
    units.sort(key=lambda row: (-row["residual_energy"], row["row"]))
    added = len(derived)
    return {
        "rank": rank, "stages": stages, "seed": seed, "als_iters": ALS_ITERS,
        "patterns_requested": rank * stages, "candidates_added": added,
        "candidates_dropped": len(drops), "candidates_before": p, "candidates_after": p + added,
        "sign_ties": sign_ties, "rule": RULE_ID, "seed_scope": "sign-tie-only",
        "matrix_shape": [n, p], "drop_reasons": drops, "derived_candidates": derived,
        "residual_mean_energy": mean_energy, "residual_units": units,
        "residual_leads": [row for row in units if row["lead"]],
    }


def as_search_candidate(record: dict) -> dict:
    """保存した派生候補から、search に渡す辞書を作る。"""
    return {key: record[key] for key in SEARCH_FIELDS}


def attach_loadings(df: pl.DataFrame, derived: list[dict]) -> pl.DataFrame:
    """保存した loading_units を列にして、因子を再計算せずに判定できる表を作る。"""
    labels = unit_labels(df)
    index = {label: i for i, label in enumerate(labels)}
    series = []
    for record in derived:
        flags = [False] * df.height
        for label in record["loading_units"]:
            if label not in index:
                raise PropositionError(f"載っている単位が表にない: {label}")
            flags[index[label]] = True
        if record["col"] in df.columns or any(s.name == record["col"] for s in series):
            raise PropositionError(f"列がすでにある: {record['col']}")
        series.append(pl.Series(record["col"], flags))
    return df.with_columns(series) if series else df


def top_view(rows: list[dict], derived_cols: set[str]) -> list[dict]:
    """上位の式。成立は tp、反例は fp（前件が真で目的が偽）。"""
    view = []
    for rank, row in enumerate(rows, 1):
        cols = [cond["col"] for group in row["if_any"] for cond in group]
        view.append({
            "rank": rank, "formula": rule_text(row), "mcc": row["mcc"],
            "holds": row["tp"], "counterexamples": row["fp"], "fn": row["fn"], "tn": row["tn"],
            "precision": row["precision"], "recall": row["recall"], "conditions": row["conditions"],
            "derived_columns": [col for col in cols if col in derived_cols],
        })
    return view


def _same_ranking(left: list[dict], right: list[dict], derived_cols: set[str]) -> bool:
    return top_view(left, derived_cols) == top_view(right, derived_cols)


def classification_sets(df: pl.DataFrame, target: list[dict], row: dict) -> dict:
    """Exact, bounded confusion sets identified by unit, rather than counts alone."""
    labels = unit_labels(df)
    truth = rule_search._mask(df, target)
    prediction = 0
    for group in row["if_any"]:
        prediction |= rule_search._mask(df, group)
    full = (1 << df.height) - 1
    masks = {"tp": prediction & truth, "fp": prediction & ~truth & full,
             "fn": ~prediction & truth & full, "tn": ~prediction & ~truth & full}
    return {key: sorted(label for i, label in enumerate(labels) if mask & (1 << i))
            for key, mask in masks.items()}


def unit_comparison(df: pl.DataFrame, target: list[dict], baseline: list[dict],
                    augmented: list[dict]) -> dict:
    """Compare every retained rule to the best baseline on the discovery sample."""
    reference = classification_sets(df, target, baseline[0]) if baseline else None
    baseline_sets = [classification_sets(df, target, row) for row in baseline]
    rows = []
    for rank, row in enumerate(augmented, 1):
        units = classification_sets(df, target, row)
        entry = {"rank": rank, "formula": rule_text(row), "sets": units,
                 "matching_baseline_top_ranks": [i for i, sets in enumerate(baseline_sets, 1) if sets == units]}
        if reference is not None:
            overlap = {}
            for key in reference:
                left, right = set(reference[key]), set(units[key])
                overlap[key] = {"common": sorted(left & right),
                                "baseline_only": sorted(left - right),
                                "rule_only": sorted(right - left)}
            entry.update({"overlap": overlap,
                          "same_predictions": units == reference,
                          "corrected_fp": sorted(set(reference["fp"]) & set(units["tn"])),
                          "corrected_fn": sorted(set(reference["fn"]) & set(units["tp"])),
                          "introduced_fp": sorted(set(reference["tn"]) & set(units["fp"])),
                          "introduced_fn": sorted(set(reference["tp"]) & set(units["fn"]))})
        rows.append(entry)
    return {"scope": "same discovery sample; not independent validation",
            "independent_validation_performed": False,
            "unit_labels": sorted(unit_labels(df)),
            "baseline_best": reference, "augmented_top": rows}


def compare(df: pl.DataFrame, target: list[dict], candidates: list[dict], *,
            rank: int = DEFAULT_RANK, stages: int = DEFAULT_STAGES, seed: int = DEFAULT_SEED,
            top: int = 20, max_terms: int = 2, max_conds: int = 2, meta: dict | None = None) -> dict:
    """既存候補だけの search と、派生候補を足した search を並べる。判定は両方とも search。"""
    derived = derive(df, candidates, rank=rank, stages=stages, seed=seed)
    extra = derived["candidates"]
    info = derived["report"]
    kw = {"max_terms": max_terms, "max_conds": max_conds, "top": top}
    baseline = search(df, target, candidates, **kw)
    baseline_bits = search(df, target, candidates, backend="bitset", **kw)
    if not _same_ranking(baseline, baseline_bits, set()):
        raise PropositionError("既存候補の判定がビット集合と一致しない")
    both_cands = [*candidates, *extra]
    both = search(derived["frame"], target, both_cands, **kw)
    both_bits = search(derived["frame"], target, both_cands, backend="bitset", **kw)
    derived_cols = {cand["col"] for cand in extra}
    if not _same_ranking(both, both_bits, derived_cols):
        raise PropositionError("派生候補を足した判定がビット集合と一致しない")
    with_top = top_view(both, derived_cols)
    by_col = {cand["col"]: cand for cand in info["derived_candidates"]}
    entered = []
    seen: set[str] = set()
    for row in with_top:
        for col in row["derived_columns"]:
            if col not in seen:
                seen.add(col)
                entered.append(by_col[col])
    data = {
        "units": df.height, "target": text(target), "candidates_before": len(candidates),
        "max_terms": max_terms, "max_conds": max_conds,
    }
    if meta:
        data["dropped_for_missing_values"] = meta["dropped_for_missing_values"]
        data["rows_before_exclusion"] = meta["rows_before_exclusion"]
        data["exclusions"] = meta["exclusions"]
    return {
        "record_kind": "exploratory", "exploratory": True, "posthoc": True, "registration": "posthoc",
        "confirmation_file": "outputs/confirmation.md", "issue": 22,
        "data": data,
        "approximation": {
            "method": "littlebit_prune.littlebit", "judging": "rule_search.search",
            "rank": info["rank"], "stages": info["stages"], "seed": info["seed"],
            "seed_scope": info["seed_scope"], "als_iters": info["als_iters"], "rule": info["rule"],
            "top": top, "patterns_requested": info["patterns_requested"],
            "candidates_added": info["candidates_added"], "candidates_dropped": info["candidates_dropped"],
            "candidates_before": info["candidates_before"], "candidates_after": info["candidates_after"],
            "sign_ties": info["sign_ties"], "matrix_shape": info["matrix_shape"],
            "matrix": "units x candidate-conditions", "backends_agree": True,
            "drop_reasons": info["drop_reasons"],
        },
        "baseline": {"top": top_view(baseline, set())},
        "with_derived": {"top": with_top},
        "unit_comparison": unit_comparison(derived["frame"], target, baseline, both),
        "derived_in_top": entered,
        "derived_candidates": info["derived_candidates"],
        "residual": {
            "rule": "residual-energy-above-mean",
            "mean_energy": info["residual_mean_energy"],
            "n_units": df.height, "n_leads": len(info["residual_leads"]),
            "description": "残差エネルギーは条件行列と再構成の差の二乗和。平均より大きい単位を手がかりにする。目的の列では分けない。",
        },
        "residual_units": info["residual_units"],
    }


def render_markdown(report: dict) -> str:
    """観測と解釈を節で分ける。数は観測、読み方は解釈。"""
    approx = report["approximation"]
    data = report["data"]
    lines = [
        "# littlebit-approx の探索",
        "",
        "このファイルは探索の記録である。確かめの記録は `confirmation.md` に置く。",
        "ファイルの登録区分は posthoc である。exploratory は探索であることの印であり、登録区分の第三の値ではない。",
        "事前登録の証拠はこのファイルに付けていない。",
        "",
        "## 観測",
        "",
        f"- 目的: `{data['target']}`",
        f"- 単位数: {data['units']}",
        f"- 人が書いた候補: {approx['candidates_before']}",
        f"- 近似: {approx['method']}、判定: {approx['judging']}",
        f"- rank={approx['rank']}、stages={approx['stages']}、seed={approx['seed']}（{approx['seed_scope']}）、als_iters={approx['als_iters']}",
        f"- 規則: `{approx['rule']}`。行列の形: {approx['matrix_shape'][0]} × {approx['matrix_shape'][1]}（単位 × 候補条件）",
        f"- 要求した潜在パターン: {approx['patterns_requested']}",
        f"- 足した候補: {approx['candidates_added']}、落とした潜在パターン: {approx['candidates_dropped']}",
        f"- 符号が同数だった軸: {approx['sign_ties']}",
        f"- ビット集合と既定の判定が一致: {approx['backends_agree']}",
        f"- 残差の平均エネルギー: {report['residual']['mean_energy']}",
        f"- 残差の手がかり: {report['residual']['n_leads']} / {report['residual']['n_units']}",
        "",
        "落とした理由の内訳:",
        "",
    ]
    reasons: dict[str, int] = {}
    for item in approx["drop_reasons"]:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    if reasons:
        for reason in sorted(reasons):
            lines.append(f"- {reason}: {reasons[reason]}")
    else:
        lines.append("- （なし）")
    lines += ["", "成立は前件が真かつ目的が真の単位数（tp）。反例は前件が真かつ目的が偽の単位数（fp）。", ""]
    lines += _formula_table("既存の候補だけ", report["baseline"]["top"])
    lines += _formula_table("既存の候補に派生候補を足したもの", report["with_derived"]["top"])
    comparison = report["unit_comparison"]
    lines += ["", "### 球団年ごとの判定集合", "",
              "既存最良式との比較。全単位のID、TP/FP/FN/TN集合、共通部分と両方向の差分は comparison.json の `unit_comparison` に保存する。",
              "FP修正は既存FP→TN、FN修正は既存FN→TP。新規誤りと合わせて評価する。", "",
              "| 順位 | 判定集合一致 | FP修正 | FN修正 | 新規FP | 新規FN |",
              "|---:|---|---|---|---|---|"]
    for row in comparison["augmented_top"]:
        cells = [str(row["rank"]), str(row.get("same_predictions"))]
        cells += [", ".join(row.get(key, [])) or "—" for key in
                  ("corrected_fp", "corrected_fn", "introduced_fp", "introduced_fn")]
        lines.append("| " + " | ".join(cells) + " |")
    matched = sum(bool(row["matching_baseline_top_ranks"]) for row in comparison["augmented_top"])
    lines += ["", f"上位{len(comparison['augmented_top'])}式中{matched}式は既存上位式のいずれかと判定集合が一致する。",
              "この比較は探索に使った同じ標本での記述であり、独立検証ではない。目的列を因子分解に渡さなくても、式の選択は目的列で採点される。",
              "独立検証は未実施。式・変換・設定を固定し、未使用の球団年に適用して評価する必要がある。保存済みloading_unitsは標本内のマスクであり、新しい球団年への外挿器ではない。",

              "`matching_baseline_top_ranks` は既存上位式との集合一致を示す。一致がなくても既存候補全体で表現不能という証明ではない。",
              "同一判定の式は追加の識別能力を示さない。誤りの修正があっても新規誤りと相殺される場合は改善と扱わない。", ""]
    lines += ["", "### 上位に入った派生候補", ""]
    if report["derived_in_top"]:
        for cand in report["derived_in_top"]:
            lines.append(_candidate_block(cand))
    else:
        lines.append("上位に入った派生候補は 0 件。")
        lines.append("")
    lines += ["### 足した派生候補", ""]
    if report["derived_candidates"]:
        lines.append("| 列 | 段 | 軸 | 乗った単位 | 登録 |")
        lines.append("|---|---:|---:|---:|---|")
        for cand in report["derived_candidates"]:
            lines.append(f"| `{cand['col']}` | {cand['stage']} | {cand['pattern']} | {cand['n_loading']} | {cand['registration']} |")
    else:
        lines.append("足した派生候補は 0 件。")
    lines += ["", "### 残差の手がかり", ""]
    leads = [row for row in report["residual_units"] if row["lead"]]
    if leads:
        lines.append("| 単位 | 球団年の残差エネルギー | 満たした候補の数 |")
        lines.append("|---|---:|---:|")
        for row in leads:
            lines.append(f"| {row['unit']} | {row['residual_energy']:.6f} | {row['n_satisfied']} |")
    else:
        lines.append("平均より残差が大きい単位は 0 件。")
    lines += [
        "",
        "単位ごとの残差エネルギー全体は comparison.json の `residual_units` にある。",
        "",
        "## 解釈",
        "",
        "この節は、上の数をどう読むかである。次の年に成り立つという主張は、ここには書かない。",
        "",
        "上位の式は、この表の上での Matthews 相関係数の順序である。因子の近似値そのものは、成立の証拠にしていない。成立と反例は、保存した 0/1 を既存のビット演算で数えた数である。",
        "",
        "残差の手がかりは、候補条件の行列を因子が再構成しきれなかった球団年である。目的の真偽で選んだ一覧ではない。その単位を判例だと決めた一覧でもない。あとで判例を探すときの順番として残してある。",
        "",
        "rank、stages、seed は、枝刈り側のコマンド既定（rank 8、stages 2）と seed 0 に固定してある。この並びを見たあとで動かした値ではない。符号の同点以外では、seed は軸の向きを変えない。",
        "",
        "ここに並んだ式を命題ファイルへ移す作業は、この記録の外である。確かめは confirmation.md の手続きで行う。",
        "",
    ]
    return "\n".join(lines)


def _formula_table(title: str, rows: list[dict]) -> list[str]:
    lines = ["", f"### {title}", "", "| 順位 | MCC | 成立 | 反例 | 条件数 | 入った派生列 | 式 |", "|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        derived = ", ".join(f"`{col}`" for col in row["derived_columns"]) or "—"
        formula = row["formula"].replace("|", "\\|")
        lines.append(
            f"| {row['rank']} | {row['mcc']:.12f} | {row['holds']} | {row['counterexamples']} | "
            f"{row['conditions']} | {derived} | {formula} |"
        )
    if not rows:
        lines.append("| — | — | — | — | — | — | — |")
    return lines


def _candidate_block(cand: dict) -> str:
    weights = sorted(cand["bundle"], key=lambda item: (-abs(item["weight"]), item["col"], item["op"], str(item["value"])))
    lines = [
        f"- `{cand['col']}` stage={cand['stage']} pattern={cand['pattern']} sign={cand['sign']} "
        f"n_loading={cand['n_loading']} source={cand['source']} rank={cand['rank']} seed={cand['seed']} "
        f"registration={cand['registration']} exploratory={cand['exploratory']} rule={cand['rule']}",
    ]
    for item in weights:
        lines.append(f"  - weight {item['weight']:.8g}: {item['col']} {item['op']} {item['value']}")
    return "\n".join(lines) + "\n"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number == 0.0:
            return 0.0
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_comparison(report: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_clean(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (dest / "comparison.json").write_text(payload, encoding="utf-8")
    (dest / "comparison.md").write_text(render_markdown(report), encoding="utf-8")


def load_cycle(cycle: Path) -> tuple[pl.DataFrame, dict, dict]:
    """search と同じ単位（除外と、目的・候補の欠損行を除いた表）。因子には target を渡さない。"""
    from pythdragoras import apply_exclusions

    cfg = tomllib.loads((cycle / "analysis.toml").read_text(encoding="utf-8"))
    spec = tomllib.loads((cycle / "rule_candidates.toml").read_text(encoding="utf-8"))
    table = pl.read_ndjson(cycle / "outputs" / "season.jsonl")
    filtered, removed = apply_exclusions(table, cfg.get("exclude", []))
    cols = sorted({cond["col"] for cond in [*spec["target"], *spec["candidate"]]})
    usable = filtered.drop_nulls(cols)
    meta = {
        "units": usable.height,
        "rows_before_exclusion": table.height,
        "dropped_for_missing_values": filtered.height - usable.height,
        "exclusions": removed,
    }
    return usable, spec, meta


def run_cycle(cycle: Path, *, rank: int = DEFAULT_RANK, stages: int = DEFAULT_STAGES,
              seed: int = DEFAULT_SEED, top: int = 20) -> dict:
    usable, spec, meta = load_cycle(cycle)
    return compare(
        usable, spec["target"], spec["candidate"], rank=rank, stages=stages, seed=seed, top=top,
        max_terms=int(spec.get("max_terms", 2)), max_conds=int(spec.get("max_conds", 2)), meta=meta,
    )


def resolve_cycle_dir(root: Path | str | None = None, cycle_dir: Path | str | None = None) -> Path:
    """サイクルのフォルダを返す。aoi_paths は探さない。

    ``cycle_dir`` があればそのパス。なければ ``root``（なければ環境変数 AOI_ROOT）の
    ``cycles/<AOI_RESEARCH_CYCLE>``。
    """
    if cycle_dir is not None:
        return Path(cycle_dir).expanduser().resolve()
    root_value = root if root is not None else os.environ.get("AOI_ROOT")
    cycle_name = os.environ.get("AOI_RESEARCH_CYCLE")
    if not root_value or not cycle_name:
        raise SystemExit(
            "サイクルのフォルダを --cycle-dir で渡すか、--root（または AOI_ROOT）と AOI_RESEARCH_CYCLE を渡す"
        )
    if cycle_name in {".", ".."} or "/" in cycle_name or "\\" in cycle_name:
        raise SystemExit(f"AOI_RESEARCH_CYCLE が不正: {cycle_name}")
    return (Path(root_value).expanduser().resolve() / "cycles" / cycle_name).resolve()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="潜在パターンを条件候補にして、既存候補だけの探索と並べる（探索。確かめではない）")
    ap.add_argument("--root", type=Path, default=None,
                    help="Aoi のルート。省略時は AOI_ROOT。--cycle-dir があるときは不要")
    ap.add_argument("--cycle-dir", type=Path, default=None,
                    help="サイクルのフォルダ。省略時は <root>/cycles/<AOI_RESEARCH_CYCLE>")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--stages", type=int, default=DEFAULT_STAGES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)
    cycle = resolve_cycle_dir(args.root, args.cycle_dir)
    dest = args.out or (cycle / "outputs" / "exploratory" / "littlebit-approx")
    report = run_cycle(cycle, rank=args.rank, stages=args.stages, seed=args.seed, top=args.top)
    write_comparison(report, dest)
    base = report["baseline"]["top"][0] if report["baseline"]["top"] else None
    alt = report["with_derived"]["top"][0] if report["with_derived"]["top"] else None
    entered = [cand["col"] for cand in report["derived_in_top"]]
    print(
        f"units={report['data']['units']} added={report['approximation']['candidates_added']} "
        f"dropped={report['approximation']['candidates_dropped']} "
        f"baseline_mcc={None if base is None else base['mcc']} "
        f"with_derived_mcc={None if alt is None else alt['mcc']} "
        f"derived_in_top={entered} leads={report['residual']['n_leads']} out={dest}"
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
