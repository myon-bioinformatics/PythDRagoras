"""One-coordinate score counterfactuals with exact incremental sufficient statistics.

This is an exploratory sensitivity calculation, not an NNUE network or causal model.
Only final scores change; opponents' scores in the same game change consistently.
No standings, inning sequence, or downstream game decisions are simulated.
"""
from __future__ import annotations

import argparse, json, tomllib
from dataclasses import dataclass; from datetime import date; from pathlib import Path


@dataclass
class ScoreState:
    RF: int = 0
    RA: int = 0
    W: int = 0
    L: int = 0
    T: int = 0

    def update(self, rf: int, ra: int, sign: int = 1) -> None:
        self.RF += sign * rf
        self.RA += sign * ra
        self.W += sign * (rf > ra)
        self.L += sign * (rf < ra)
        self.T += sign * (rf == ra)

    def metrics(self) -> dict:
        decided = self.W + self.L
        total = self.RF ** 1.83 + self.RA ** 1.83
        p = self.RF ** 1.83 / total if total else None
        return {**vars(self), "wpct": self.W / decided if decided else None,
                "pythag_fixed": p,
                "expected_wins": decided * p if p is not None else None}


def _delta(now: dict, before: dict) -> dict:
    return {k: now[k] - before[k] if now[k] is not None and before[k] is not None else None
            for k in now}


def team_games(games: list[dict], team: str) -> list[dict]:
    """Validate the complete observation input before selecting a team."""
    seen, out = set(), []
    for g in games:
        key = g["key"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"duplicate/invalid game key: {key!r}")
        seen.add(key)
        day = date.fromisoformat(g["date"])
        if not isinstance(g["home"], str) or not isinstance(g["away"], str) or g["home"] == g["away"]:
            raise ValueError(f"invalid teams: {key}")
        for col in ("hs", "as"):
            if type(g[col]) is not int or g[col] < 0:
                raise ValueError(f"nonnegative integer score required: {key}/{col}")
        if team in (g["home"], g["away"]):
            home = team == g["home"]
            out.append({"key": key, "date": day.isoformat(), "season": day.year,
                        "rf": g["hs"] if home else g["as"],
                        "ra": g["as"] if home else g["hs"]})
    if not out:
        raise ValueError(f"no observations for team: {team}")
    return sorted(out, key=lambda g: (g["date"], g["key"]))


def sweep(games: list[dict], axis: str, max_step: int = 3) -> dict:
    """Increase RF or decrease RA by 0..max_step runs per game.

    Each step starts from the preceding state; remove the old game contribution,
    add the new one. RA has an explicit zero floor. Every game remains in sample.
    Timeline records the cumulative intervention over chronological games, holding
    all not-yet-edited games at their preceding-step values (not a forecast).
    """
    if axis not in ("RF", "RA") or type(max_step) is not int or max_step < 1:
        raise ValueError("axis must be RF/RA and max_step a positive integer")
    state = ScoreState()
    scores = [(g["rf"], g["ra"]) for g in games]
    for rf, ra in scores:
        state.update(rf, ra)
    baseline = state.metrics()
    curve = [{"step": 0, "changed_runs": 0, "metrics": baseline,
              "delta": _delta(baseline, baseline)}]
    timeline, budget = [], 0
    for step in range(1, max_step + 1):
        for i, g in enumerate(games):
            old_rf, old_ra = scores[i]
            rf = g["rf"] + step if axis == "RF" else g["rf"]
            ra = max(0, g["ra"] - step) if axis == "RA" else g["ra"]
            if (rf, ra) == scores[i]:
                continue
            before = state.metrics()
            state.update(old_rf, old_ra, -1)
            state.update(rf, ra)
            scores[i] = (rf, ra)
            budget += abs(rf - old_rf) + abs(ra - old_ra)
            now = state.metrics()
            timeline.append({"step": step, "key": g["key"], "date": g["date"],
                             "old": [old_rf, old_ra], "new": [rf, ra],
                             "changed_runs": budget, "metrics": now,
                             "local_delta": _delta(now, before), "delta": _delta(now, baseline)})
        now = state.metrics()
        curve.append({"step": step, "changed_runs": budget, "metrics": now,
                      "delta": _delta(now, baseline)})
    return {"axis": axis, "baseline": baseline, "curve": curve, "timeline": timeline}


def analyze(games: list[dict], team: str, *, max_step: int = 3,
            excluded: list[dict] = (), holdout: list[int] = (), min_seasons: int = 3) -> dict:
    if type(min_seasons) is not int or min_seasons < 1:
        raise ValueError("min_seasons must be positive")
    if any(not str(e.get("reason", "")).strip() for e in excluded):
        raise ValueError("exclusions require reasons")
    rows = team_games(games, team)
    excluded_years = {e["season"] for e in excluded}
    seasons = sorted({g["season"] for g in rows} - excluded_years)
    results, candidates = [], []
    for axis in ("RF", "RA"):
        training = []
        for season in seasons:
            result = sweep([g for g in rows if g["season"] == season], axis, max_step)
            result.update(season=season, role="holdout" if season in holdout else "exploration")
            results.append(result)
            if season not in holdout:
                training.append(result)
        # Compare at the preregistered first step; larger steps are sensitivity only.
        changed = sum(r["curve"][1]["changed_runs"] for r in training)
        net = sum(r["curve"][1]["delta"]["W"] - r["curve"][1]["delta"]["L"] for r in training)
        positive = sum(r["curve"][1]["delta"]["W"] - r["curve"][1]["delta"]["L"] > 0 for r in training)
        eligible = len(training) >= min_seasons and positive == len(training) and changed > 0
        candidates.append({"axis": axis, "step": 1, "seasons": len(training),
                           "positive_seasons": positive, "changed_runs": changed,
                           "net_wins_delta": net, "net_wins_per_changed_run": net / changed if changed else None,
                           "status": "exploratory_candidate" if eligible else "insufficient_evidence",
                           "proposition": f"{team}: {axis}を各試合1点改善する反実仮想では勝ち越し数が増える"})
    candidates.sort(key=lambda c: (c["status"] == "exploratory_candidate",
                                   c["net_wins_per_changed_run"] or 0), reverse=True)
    return {"schema": "aoi-score-sensitivity/1", "team": team,
            "policy": {"max_step": max_step, "min_seasons": min_seasons,
                       "selection_step": 1, "excluded": list(excluded), "holdout": list(holdout),
                       "RA_floor": 0, "ranking": "net wins delta / actual changed runs",
                       "candidate_rule": "at least min_seasons exploration years; positive net wins delta in every year",
                       "interpretation": "探索用の得失点感度。実行可能性・因果効果・順位・Aクラス確率は示さない。候補の支持判定には独立検証が必要。"},
            "results": results, "candidates": candidates}


def markdown(report: dict) -> str:
    lines = [f"# 試合ごとの得失点感度分析 ({report['team']})", "",
             report["policy"]["interpretation"], "",
             "| 軸 | ステップ | 探索年数 | 全年勝ち越し増加 | 変更点数 | 勝ち越し増加 | 変更1点あたりの勝ち越し | 状態 | 命題 |",
             "|---|---:|---:|---|---:|---:|---:|---|---|"]
    def fmt(v, prec=3):
        return "未定義" if v is None else f"{v:.{prec}f}"

    for c in report["candidates"]:
        pos = f"{c['positive_seasons']}/{c['seasons']}"
        net = f"{c['net_wins_delta']:+d}" if c["net_wins_delta"] is not None else "—"
        rate = fmt(c["net_wins_per_changed_run"]) if c["net_wins_per_changed_run"] is not None else "—"
        lines.append(f"| {c['axis']} | {c['step']} | {c['seasons']} | {pos} | {c['changed_runs']} | {net} | {rate} | {c['status']} | {c['proposition']} |")

    lines += ["", "## シーズン別得失点変更曲線", ""]
    for r in report["results"]:
        lines += [f"### {r['season']}年 ({r['axis']}軸, {r['role']})", "",
                  "| ステップ | 変更点数 | W | L | T | 勝率 | fixed期待勝率 | 期待勝利数 | 勝ち越し差分 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for p in r["curve"]:
            m = p["metrics"]
            d = p["delta"]
            wpct = fmt(m["wpct"])
            pythag = fmt(m["pythag_fixed"])
            exp_w = fmt(m["expected_wins"], 2)
            net_delta = f"{d['W'] - d['L']:+d}" if d["W"] is not None and d["L"] is not None else "—"
            lines.append(f"| {p['step']} | {p['changed_runs']} | {m['W']} | {m['L']} | {m['T']} | {wpct} | {pythag} | {exp_w} | {net_delta} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--team")
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--min-seasons", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    team = args.team or config["focus"]["team"]
    if team not in config["teams"]:
        parser.error(f"undefined team: {team}")
    games = [json.loads(line) for line in args.games.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = analyze(games, team, max_step=args.max_step, min_seasons=args.min_seasons,
                     excluded=config.get("exclude", []), holdout=config.get("confirm", {}).get("seasons", []))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
