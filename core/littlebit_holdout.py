"""Train-only LittleBit decoders, frozen row projection, and temporal backtests.

This is a new inductive extension, not an extrapolation of saved loading_units.
Historical backtests remain posthoc: these years were inspected in earlier research.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import polars as pl

import littlebit_approx as la
import littlebit_prune
from propositions import PropositionError
import rule_search as rs

METHOD = "littlebit-frozen-projection/1"
RCOND = 1e-12
ZERO_TOL = 1e-10
VARIANTS = ("baseline", "augmented", "derived")


def _matrix(df: pl.DataFrame, candidates: list[dict]) -> np.ndarray:
    cols = sorted({c["col"] for c in candidates})
    if not candidates or set(cols) - set(df.columns):
        raise PropositionError("候補条件の入力列が必要")
    if df.select(cols).null_count().sum_horizontal().item():
        raise PropositionError("候補条件の欠損値は呼出し側で除外する")
    return la.indicator_matrix(df, candidates).astype(np.float64)


def _project(residual: np.ndarray, stage: dict) -> tuple[np.ndarray, np.ndarray]:
    encoder = np.asarray(stage["encoder"], dtype=np.float64)
    decoder = np.asarray(stage["decoder"], dtype=np.float64)
    # Row-wise operations keep the calculation independent of evaluation batches.
    scores = np.asarray([row @ encoder for row in residual]).reshape(len(residual), encoder.shape[1])
    reconstructed = np.asarray([row @ decoder.T for row in scores]).reshape(residual.shape)
    return scores, residual - reconstructed


def fit(df: pl.DataFrame, candidates: list[dict], *, rank: int = la.DEFAULT_RANK,
        stages: int = la.DEFAULT_STAGES, seed: int = la.DEFAULT_SEED) -> dict:
    """Fit feature-only decoders and select projected masks on training rows only."""
    rank = la._positive_int("rank", rank, allow_zero=False)
    stages = la._positive_int("stages", stages, allow_zero=False)
    seed = la._positive_int("seed", seed, allow_zero=True)
    if df.height == 0:
        raise PropositionError("学習行が必要")
    matrix = _matrix(df, candidates)
    parts = littlebit_prune.littlebit(matrix, rank, stages, als_iters=la.ALS_ITERS)
    model = {"method": METHOD, "rank": rank, "stages": stages, "seed": seed,
             "als_iters": la.ALS_ITERS, "rcond": RCOND, "zero_tol": ZERO_TOL,
             "training_rows": df.height, "input_candidates": deepcopy(candidates),
             "projection_stages": [], "candidates": [], "dropped": []}
    residual = matrix.copy()
    seen = {}
    for s, (_, signs_v, scale) in enumerate(parts):
        decoder = signs_v * scale[:, None]
        stage = {"decoder": decoder.tolist(),
                 "encoder": np.linalg.pinv(decoder.T, rcond=RCOND).tolist()}
        scores, residual = _project(residual, stage)
        model["projection_stages"].append(stage)
        for k in range(rank):
            if k >= scores.shape[1]:
                model["dropped"].append({"stage": s, "pattern": k, "reason": "rank-shortfall"})
                continue
            column = scores[:, k].copy()
            column[np.abs(column) <= ZERO_TOL] = 0.0
            sign = la.pattern_sign(column, seed)
            flags = sign * column > ZERO_TOL
            key = flags.tobytes()
            reason = "constant-mask" if flags.all() or not flags.any() else (
                "duplicate-mask" if key in seen else None)
            if reason:
                model["dropped"].append({"stage": s, "pattern": k, "reason": reason})
                continue
            col = f"littlebit_project_s{s}_k{k}"
            seen[key] = col
            model["candidates"].append({
                "col": col, "op": "==", "value": True, "source": METHOD,
                "stage": s, "pattern": k, "sign": sign, "rank": rank, "seed": seed,
                "rule": "sign-of-frozen-projection", "posthoc": True,
                "exploratory": True, "registration": "posthoc",
                "n_loading": int(flags.sum()),
            })
    return model


def transform(df: pl.DataFrame, model: dict) -> pl.DataFrame:
    """Apply frozen operators without factorization, target, unit IDs, or refitting."""
    if model.get("method") != METHOD:
        raise PropositionError("固定射影モデルが必要（loading_units は外挿器ではない）")
    residual = _matrix(df, model["input_candidates"])
    columns = []
    for s, stage in enumerate(model["projection_stages"]):
        scores, residual = _project(residual, stage)
        for candidate in model["candidates"]:
            if candidate["stage"] != s:
                continue
            col = candidate["col"]
            if col in df.columns:
                raise PropositionError(f"列がすでにある: {col}")
            flags = candidate["sign"] * scores[:, candidate["pattern"]] > model["zero_tol"]
            columns.append(pl.Series(col, flags.tolist(), dtype=pl.Boolean))
    return df.with_columns(columns)


def best_derived(df: pl.DataFrame, target: list[dict], candidates: list[dict],
                 derived_cols: set[str], *, max_terms: int, max_conds: int) -> dict | None:
    """Exhaustive constrained search; do not require a derived rule in the top 20."""
    if not derived_cols:
        return None
    terms = rs._build_terms(df, candidates, max_conds)
    marked = [any(c["col"] in derived_cols for c in group) for group, _ in terms]
    truth = rs._mask(df, target)
    n_pos, n_neg = truth.bit_count(), df.height - truth.bit_count()
    best = None
    for size in range(1, max_terms + 1):
        for indices in itertools.combinations(range(len(terms)), size):
            if not any(marked[i] for i in indices):
                continue
            mask = 0
            for i in indices:
                mask |= terms[i][1]
            tp, fp = (mask & truth).bit_count(), (mask & ~truth).bit_count()
            count = sum(len(terms[i][0]) for i in indices)
            if best and (-round(rs._mcc(tp, fp, n_pos - tp, n_neg - fp), 12), count) > best[0][:2]:
                continue
            item = rs._record([terms[i][0] for i in indices], tp, fp, n_pos, n_neg)
            if best is None or item[0] < best[0]:
                best = item
    return None if best is None else best[1]


def score(df: pl.DataFrame, target: list[dict], rule: dict) -> dict:
    sets = la.classification_sets(df, target, rule)
    tp, fp, fn, tn = (len(sets[key]) for key in ("tp", "fp", "fn", "tn"))
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {"units": df.height, "holds": tp, "antecedent": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "mcc": rs._mcc(tp, fp, fn, tn), "mcc_defined": bool(denominator), "sets": sets}


def fit_rules(train: pl.DataFrame, target: list[dict], candidates: list[dict], *,
              rank: int, stages: int, seed: int, max_terms: int, max_conds: int) -> dict:
    """The entire fitted package is complete before evaluation data is accepted."""
    rs._check(target, candidates)
    if {c["col"] for c in target} & {c["col"] for c in candidates}:
        raise PropositionError("目的列を因子分解の候補列へ含めない")
    if max_terms not in (1, 2) or max_conds not in (1, 2):
        raise PropositionError("年別検証は max_terms/max_conds が 1 または 2")
    model = fit(train.select(sorted({c["col"] for c in candidates})), candidates,
                rank=rank, stages=stages, seed=seed)
    projected = transform(train, model)
    extra = model["candidates"]
    cols = {c["col"] for c in extra}
    kw = {"max_terms": max_terms, "max_conds": max_conds, "top": 1}
    baseline = rs.search(train, target, candidates, **kw)[0]
    augmented = rs.search(projected, target, candidates + extra, **kw)[0]
    derived = (augmented if any(c["col"] in cols for g in augmented["if_any"] for c in g)
               else best_derived(projected, target, candidates + extra, cols,
                                 max_terms=max_terms, max_conds=max_conds))
    selected = {"baseline": baseline, "augmented": augmented, "derived": derived}
    return {"model": model, "target": deepcopy(target),
            "max_terms": max_terms, "max_conds": max_conds,
            "selected": {key: None if rule is None else {
                "if_any": deepcopy(rule["if_any"]), "formula": rs.rule_text(rule),
                "training": score(projected, target, rule)} for key, rule in selected.items()}}


def _split(df: pl.DataFrame, train_years: list[int], test_years: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    if ("season" not in df.columns or "team" not in df.columns
            or df["season"].null_count() or df["team"].null_count()):
        raise PropositionError("欠損のない season と team が必要")
    la.unit_labels(df)
    if not train_years or not test_years or max(train_years) >= min(test_years):
        raise PropositionError("評価年は学習年の後にし、重複させない")
    available = set(df["season"].to_list())
    if not set(train_years + test_years) <= available:
        raise PropositionError("指定年が入力に存在しない")
    return tuple(df.filter(pl.col("season").is_in(years)).sort("season", "team")
                 for years in (train_years, test_years))


def backtest(df: pl.DataFrame, target: list[dict], candidates: list[dict], *,
             train_years: list[int], test_years: list[int], rank: int = la.DEFAULT_RANK,
             stages: int = la.DEFAULT_STAGES, seed: int = la.DEFAULT_SEED,
             max_terms: int = 2, max_conds: int = 2) -> dict:
    train, test = _split(df, train_years, test_years)
    fitted = fit_rules(train, target, candidates, rank=rank, stages=stages, seed=seed,
                       max_terms=max_terms, max_conds=max_conds)
    # No test data reaches fit, sign orientation, deduplication, or rule selection.
    projected = transform(test, fitted["model"])
    evaluation = {key: None if rule is None else score(projected, target, rule)
                  for key, rule in fitted["selected"].items()}
    by_year = {str(year): {key: None if rule is None else score(
        projected.filter(pl.col("season") == year), target, rule)
        for key, rule in fitted["selected"].items()} for year in test_years}
    return {"train_years": train_years, "test_years": test_years,
            "training_units": la.unit_labels(train), "evaluation_units": la.unit_labels(test),
            "fitted": fitted, "evaluation": evaluation, "by_year": by_year}


def pooled(folds: list[dict], variant: str) -> dict | None:
    rows = [fold["evaluation"][variant] for fold in folds]
    if any(row is None for row in rows):
        return None  # No silently omitted folds if a training set has no derived mask.
    sets = {key: [unit for row in rows for unit in row["sets"][key]] for key in ("tp", "fp", "fn", "tn")}
    labels = [unit for group in sets.values() for unit in group]
    if len(labels) != len(set(labels)):
        raise PropositionError("評価単位を重複集計しない")
    tp, fp, fn, tn = (len(sets[key]) for key in ("tp", "fp", "fn", "tn"))
    return {"units": len(labels), "holds": tp, "antecedent": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "mcc": rs._mcc(tp, fp, fn, tn),
            "mcc_defined": bool((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
            "sets": {key: sorted(value) for key, value in sets.items()}}


def run_cycle(cycle: Path, *, train_through: int = 2022, min_train_years: int = 5) -> dict:
    df, spec, meta = la.load_cycle(cycle)
    years = sorted(set(df["season"].to_list()))
    la._positive_int("min_train_years", min_train_years, allow_zero=False)
    kw = {"max_terms": int(spec.get("max_terms", 2)), "max_conds": int(spec.get("max_conds", 2))}
    primary = backtest(df, spec["target"], spec["candidate"],
                       train_years=[y for y in years if y <= train_through],
                       test_years=[y for y in years if y > train_through], **kw)
    if min_train_years >= len(years):
        raise PropositionError("walk-forward の評価年が必要")
    folds = [backtest(df, spec["target"], spec["candidate"],
                      train_years=years[:i], test_years=[years[i]], **kw)
             for i in range(min_train_years, len(years))]
    hashes = {path: hashlib.sha256((cycle / path).read_bytes()).hexdigest()
              for path in ("outputs/season.jsonl", "rule_candidates.toml", "analysis.toml")}
    return {"record_kind": "temporal-backtest", "posthoc": True, "exploratory": True,
            "independent_confirmation": False, "method": METHOD,
            "settings": {"rank": la.DEFAULT_RANK, "stages": la.DEFAULT_STAGES,
                         "seed": la.DEFAULT_SEED, "als_iters": la.ALS_ITERS,
                         "rcond": RCOND, "zero_tol": ZERO_TOL, **kw,
                         "train_through": train_through, "min_train_years": min_train_years},
            "inputs_sha256": hashes, "cohort": meta,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__},
            "primary": primary, "walk_forward": {"folds": folds,
                "pooled": {key: pooled(folds, key) for key in VARIANTS}}}


def render_markdown(report: dict) -> str:
    primary = report["primary"]
    lines = ["# LittleBit 年別バックテスト", "", "記録: 2026-10-15", "",
             "探索記録とは別の、学習・評価を年で分離した計算である。既に閲覧した過去年を再分割しており、",
             "未閲覧データでの独立確認・事前登録・将来予測の検証ではない。命題・合格条件は変更しない。", "",
             f"- 学習年: {primary['train_years']}", f"- 評価年: {primary['test_years']}",
             f"- 固定設定: `{json.dumps(report['settings'], ensure_ascii=False, sort_keys=True)}`", "",
             "## 未使用年（当該fitから除外）の結果", "",
             "成立=TP、反例=FP。augmented は追加候補込みの最良式、derived は少なくとも一つ派生列を含む最良式。",
             "全て学習年のMCCだけで選び、評価年で再探索しない。", "",
             "| 評価 | 式の選択 | 単位 | MCC | 成立TP | FP | FN | TN |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    def row(label, key, result):
        if result is None:
            return f"| {label} | {key} | — | — | — | — | — | — |"
        mcc = f"{result['mcc']:.12f}" if result["mcc_defined"] else "未定義（採点規約0）"
        return f"| {label} | {key} | {result['units']} | {mcc} | {result['tp']} | {result['fp']} | {result['fn']} | {result['tn']} |"
    for key in VARIANTS:
        lines.append(row(f"{min(primary['test_years'])}–{max(primary['test_years'])} block", key, primary["evaluation"][key]))
    for year, results in primary["by_year"].items():
        for key in VARIANTS:
            lines.append(row(year, key, results[key]))
    for key in VARIANTS:
        lines.append(row("walk-forward pooled", key, report["walk_forward"]["pooled"][key]))
    baseline = primary["evaluation"]["baseline"]
    derived = primary["evaluation"]["derived"]
    same = derived is not None and baseline["sets"] == derived["sets"]
    matching_folds = sum(fold["evaluation"]["derived"] is not None and
                         fold["evaluation"]["baseline"]["sets"] == fold["evaluation"]["derived"]["sets"]
                         for fold in report["walk_forward"]["folds"])
    lines += ["", f"block評価でbaselineとderivedの判定集合が完全一致: {same}。",
              f"walk-forwardでも全TP/FP/FN/TN集合が一致した年: {matching_folds}/{len(report['walk_forward']['folds'])}。",
              "改善の有無は件数だけでなく球団年の集合で比較する。今回の同点は新しい誤り修正を示さない。",
              "手法全体の失敗を意味せず、固定した候補生成方式・設定における追加識別力の観測である。"]
    lines += ["", "## 学習側で固定した式", ""]
    for key in VARIANTS:
        rule = primary["fitted"]["selected"][key]
        lines.append(f"- {key}: 生成なし" if rule is None else
                     f"- {key}: `{rule['formula']}`（学習MCC {rule['training']['mcc']:.12f}）")
    lines += ["", "## Walk-forward（その年より前の年だけで再学習）", "",
              "| 評価年 | 式の選択 | 単位 | MCC | 成立TP | FP | FN | TN |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for fold in report["walk_forward"]["folds"]:
        for key in VARIANTS:
            lines.append(row(str(fold["test_years"][0]), key, fold["evaluation"][key]))
    lines += ["", "## 技術的な意味と制限", "",
              "元のderiveは行因子Fの符号を保存するだけで、新しい行のh・Usを求める規則を持たない。",
              "bundleは列重みであり、それ単独でFの符号を再現する分類器ではない。", "",
              "今回の拡張では学習年だけでLittleBit分解し、各段のW=Vs*列scaleを固定する。",
              "R0=X、Zs=Rs @ pinv(Ws.T)、Rs+1=Rs−Zs @ Ws.Tとし、学習側で固定した軸の向きsについて",
              "s*Zs[:,k] > 1e-10を条件にする。pinvのrcond=1e-12、学習側も評価側も同じtransformを使う。",
              "学習側だけで符号の向き・定数/重複の除外・式選択を確定する。保存モデルはdecoder/encoder/条件を持ち、",
              "loading_unitsを持たない。評価行のIDや他の評価行に依存せず適用できる。", "",
              "これは符号制約付きFの再現ではなく、新しい固定最小二乗射影による候補生成である。",
              "全標本由来littlebit_s0_k0/k4等の式をそのまま独立検証した結果とは解釈しない。",
              "baselineも学習側で選び直す。元の全標本最良式のMCC 0.820782681668は探索値である。", "",
              "特徴量は保存済みseason.jsonlのシーズン終了時集計を使う（sim_p_upperは同年同リーグ、",
              "alloc_z_stratは同年球団の試合群から計算）。年をまたぐfitの漏洩を避けるが、開幕前/シーズン途中の予測ではない。",
              "候補閾値自体も過去の探索を経ており、未閲覧年での強い独立検証には、このモデル・式・入力生成設定を先に固定する必要がある。", "",
              "pooled MCCは各年の予測を合計して再計算し、年別MCCの平均ではない。blockとwalk-forwardは評価年が重なるので合算しない。",
              "全学習/評価ID、式、混同行列の集合、凍結モデル、入力ハッシュはholdout.jsonに保存する。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None,
                    help="Aoi のルート。省略時は AOI_ROOT。--cycle-dir があるときは不要")
    ap.add_argument("--cycle-dir", type=Path, default=None,
                    help="サイクルのフォルダ。省略時は <root>/cycles/<AOI_RESEARCH_CYCLE>")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--train-through", type=int, default=2022)
    ap.add_argument("--min-train-years", type=int, default=5)
    args = ap.parse_args(argv)
    cycle = la.resolve_cycle_dir(args.root, args.cycle_dir)
    report = run_cycle(cycle, train_through=args.train_through, min_train_years=args.min_train_years)
    out = args.out or cycle / "outputs/exploratory/littlebit-holdout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (out / "holdout.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in result.items() if k != "sets"} if result else None
                      for key, result in report["walk_forward"]["pooled"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
