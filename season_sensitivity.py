"""Season-first exploration: custom linear indices, parameter slices and year blocks.

Pure Python 3.11+. No learned model, causal intervention, or significance verdict.
A score is x + weight*y; score >= threshold predicts the configured Boolean outcome.
Only threshold crossings update the confusion counts. Weights and rules are explicit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tomllib


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_grid(values, name):
    if not values or not all(finite(v) for v in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f'{name}: finite, strictly increasing grid required')


def prepare(rows):
    """Preserve inputs; derive rate and same-season peer-centred scores from raw RF/RA/G."""
    rows = deepcopy(rows)
    seen, groups = set(), defaultdict(list)
    for r in rows:
        key = (r['season'], r['team'])
        if type(key[0]) is not int or not isinstance(key[1], str) or key in seen:
            raise ValueError(f'duplicate/invalid team-season: {key}')
        seen.add(key)
        groups[(r['season'], r['league'])].append(r)
        for col in ('RF', 'RA'):
            if finite(r.get(col)) and finite(r.get('G')) and r['G'] > 0:
                r[col.lower() + '_per_g'] = r[col] / r['G']
    for group in groups.values():
        for r in group:
            for col, name, sign in [('rf_per_g', 'rf_vs_peers', 1), ('ra_per_g', 'ra_saved_vs_peers', -1)]:
                others = [p.get(col) for p in group if p is not r]
                r[name] = sign * (r[col] - sum(others) / len(others)) if finite(r.get(col)) and others and all(finite(v) for v in others) else None
    return rows


def counts_metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    yes, no = tp + fp, fn + tn
    p_yes, p_no = (tp / yes if yes else None), (fn / no if no else None)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n': n,
            'mcc': (tp*tn-fp*fn)/math.sqrt(den) if den else None,
            'p_outcome_given_high': p_yes, 'p_outcome_given_low': p_no,
            'probability_gap': p_yes-p_no if p_yes is not None and p_no is not None else None}


def threshold_curve(scores, outcomes, thresholds):
    """Exact sweep: start with every row high, remove rows as the threshold advances."""
    validate_grid(thresholds, 'thresholds')
    if len(scores) != len(outcomes) or not all(finite(s) for s in scores) or not all(type(y) is bool for y in outcomes):
        raise ValueError('finite scores and aligned Boolean outcomes required')
    ordered = sorted(zip(scores, outcomes))
    tp, fp, fn, tn = sum(outcomes), len(outcomes)-sum(outcomes), 0, 0
    result, cursor = [], 0
    for threshold in thresholds:
        changed = 0
        while cursor < len(ordered) and ordered[cursor][0] < threshold:
            _, y = ordered[cursor]
            tp -= y
            fn += y
            fp -= not y
            tn += not y
            cursor += 1
            changed += 1
        result.append({'threshold': threshold, 'crossed_units': changed, **counts_metrics(tp, fp, fn, tn)})
    return result


def shape(points, x='threshold', y='mcc'):
    """Finite differences/trapezoids on sampled values; never a smooth derivative claim."""
    slopes, area, span = [], 0.0, 0.0
    for left, right in zip(points, points[1:]):
        dx = right[x]-left[x]
        valid = left[y] is not None and right[y] is not None
        slopes.append({'left': left[x], 'right': right[x],
                       'slope': (right[y]-left[y])/dx if valid else None})
        if valid:
            area += dx*(left[y]+right[y])/2
            span += dx
    bends = []
    for a, b in zip(slopes, slopes[1:]):
        if a['slope'] is not None and b['slope'] is not None:
            dx = (b['left']+b['right']-a['left']-a['right'])/2
            bends.append({'at': a['right'], 'slope_change_per_unit': (b['slope']-a['slope'])/dx})
    valid = [p for p in points if p[y] is not None]
    best = max((p[y] for p in valid), default=None)
    # Consecutive equal maxima are represented as intervals, including one-point peaks.
    plateaus = []
    for i, p in enumerate(points):
        if best is not None and p[y] == best:
            if i and points[i-1][y] == best:
                plateaus[-1]['end'] = p[x]
            else:
                plateaus.append({'start': p[x], 'end': p[x]})
    extrema = []
    for a, b, c in zip(points, points[1:], points[2:]):
        if all(p[y] is not None for p in (a, b, c)):
            if b[y] > a[y] and b[y] > c[y]:
                extrema.append({'at': b[x], 'kind': 'sampled_max'})
            if b[y] < a[y] and b[y] < c[y]:
                extrema.append({'at': b[x], 'kind': 'sampled_min'})
    return {'finite_slopes': slopes, 'slope_changes': bends, 'sampled_extrema': extrema,
            'max_intervals': plateaus, 'trapezoid_area': area if span else None,
            'covered_span': span, 'mean_over_grid': area/span if span else None}


def evaluate(rows, spec, weight, threshold, outcome):
    tp = fp = fn = tn = 0
    details = []
    for r in rows:
        score = r[spec['x']] + weight*r[spec['y']]
        high, actual = score >= threshold, r[outcome]
        tp += high and actual
        fp += high and not actual
        fn += not high and actual
        tn += not high and not actual
        details.append({'season': r['season'], 'team': r['team'], 'score': score,
                        'high': high, 'outcome': actual, 'matches': high == actual})
    return {**counts_metrics(tp, fp, fn, tn), 'units': details}


def surface(rows, spec, outcome):
    result = []
    for weight in spec['weights']:
        scores = [r[spec['x']]+weight*r[spec['y']] for r in rows]
        points = threshold_curve(scores, [r[outcome] for r in rows], spec['thresholds'])
        for point in points:
            point['weight'] = weight
        result.extend(points)
    return result


def select(points, baseline_weight, baseline_threshold, min_support):
    eligible = [p for p in points if p['mcc'] is not None and p['tp']+p['fp'] >= min_support and p['fn']+p['tn'] >= min_support]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p['mcc'], -abs(p['weight']-baseline_weight),
                                       -abs(p['threshold']-baseline_threshold), -p['weight'], -p['threshold']))


def analyze(rows, config):
    search = config['search']
    outcome = search.get('outcome', 'upper_half')
    for name in ('min_support', 'min_years'):
        if type(search[name]) is not int or search[name] < 1:
            raise ValueError(f'{name} must be positive')
    if not finite(search['min_mcc']) or not finite(search['min_year_fraction']) or not 0 <= search['min_year_fraction'] <= 1:
        raise ValueError('finite candidate criteria required')
    if any(not str(e.get('reason', '')).strip() for e in config.get('exclude', [])):
        raise ValueError('exclusion reason required')
    specs = config['index']
    if len({s['id'] for s in specs}) != len(specs):
        raise ValueError('duplicate index id')
    for spec in specs:
        validate_grid(spec['weights'], 'weights')
        validate_grid(spec['thresholds'], 'thresholds')
        if spec['baseline_weight'] not in spec['weights'] or spec['baseline_threshold'] not in spec['thresholds']:
            raise ValueError('baseline must be included in grid')
    excluded = {e['season'] for e in config.get('exclude', [])}
    holdout = set(search['holdout'])
    all_rows = prepare(rows)
    scoped = [r for r in all_rows if r['league'] == search['league'] and r['season'] >= search['start'] and r['season'] <= search['end'] and r['season'] not in excluded]
    if not scoped:
        raise ValueError('no season rows in scope')
    expected = set(search.get('expected_teams', []))
    if expected:
        for year in {r['season'] for r in scoped}:
            present = {r['team'] for r in scoped if r['season'] == year}
            if present != expected:
                raise ValueError(f'incomplete league in {year}: expected {sorted(expected)}, got {sorted(present)}')
    if any(type(r.get(outcome)) is not bool for r in scoped):
        raise ValueError(f'{outcome}: Boolean outcome required for every row in scope')
    if any(r.get('rank_tie') for r in scoped):
        raise ValueError('rank ties need an explicit outcome policy before exploration')
    results = []
    for spec in specs:
        missing = [r for r in scoped if not all(finite(r.get(c)) for c in (spec['x'], spec['y']))]
        available = [r for r in scoped if r not in missing]
        train = [r for r in available if r['season'] not in holdout]
        test = [r for r in available if r['season'] in holdout]
        points = surface(train, spec, outcome)
        baseline = next(p for p in points if p['weight'] == spec['baseline_weight'] and p['threshold'] == spec['baseline_threshold'])
        for point in points:
            point['delta_from_baseline'] = {
                key: point[key]-baseline[key] if point[key] is not None and baseline[key] is not None else None
                for key in ('mcc', 'probability_gap')}
        best = select(points, spec['baseline_weight'], spec['baseline_threshold'], search['min_support'])
        record = {'id': spec['id'], 'definition': dict(spec),
                  'missing_units': [{'season': r['season'], 'team': r['team']} for r in missing],
                  'surface': points, 'candidate': None, 'holdout': None, 'focus_trajectory': [], 'leave_one_year_out': []}
        for weight in spec['weights']:
            curve = [p for p in points if p['weight'] == weight]
            record.setdefault('threshold_shapes', []).append({'weight': weight, **shape(curve)})
        # Weight slice holds threshold fixed at the declared baseline (one coordinate at a time).
        record['weight_shape'] = shape([p for p in points if p['threshold'] == spec['baseline_threshold']], 'weight')
        if best is not None:
            fixed_by_year = []
            years = sorted({r['season'] for r in train})
            for year in years:
                kept = [r for r in train if r['season'] != year]
                selected = select(surface(kept, spec, outcome), spec['baseline_weight'], spec['baseline_threshold'], search['min_support'])
                held = [r for r in train if r['season'] == year]
                tested = evaluate(held, spec, selected['weight'], selected['threshold'], outcome) if selected and len(years)-1 >= search['min_years'] else None
                record['leave_one_year_out'].append({'season': year,
                                                     'selected': {'weight': selected['weight'], 'threshold': selected['threshold']} if tested else None,
                                                     'test': tested})
                fixed_by_year.append({'season': year, **evaluate(held, spec, best['weight'], best['threshold'], outcome)})
            comparable = [r for r in fixed_by_year if r['mcc'] is not None]
            positive = sum(r['mcc'] > 0 for r in comparable)
            supported = (len(years) >= search['min_years'] and best['mcc'] >= search['min_mcc'] and
                         len(comparable) >= search['min_years'] and positive/len(comparable) >= search['min_year_fraction'])
            wi, ti = spec['weights'].index(best['weight']), spec['thresholds'].index(best['threshold'])
            neighbour_coords = {(spec['weights'][j], best['threshold']) for j in (wi-1, wi+1) if 0 <= j < len(spec['weights'])}
            neighbour_coords |= {(best['weight'], spec['thresholds'][j]) for j in (ti-1, ti+1) if 0 <= j < len(spec['thresholds'])}
            neighbours = [{k: p[k] for k in ('weight', 'threshold', 'mcc', 'probability_gap')}
                          for p in points if (p['weight'], p['threshold']) in neighbour_coords]
            record['candidate'] = {'weight': best['weight'], 'threshold': best['threshold'],
                                   'exploration': best, 'baseline': baseline,
                                   'mcc_delta': best['mcc']-baseline['mcc'] if baseline['mcc'] is not None else None,
                                   'years': len(years), 'comparable_years': len(comparable), 'positive_years': positive,
                                   'neighbours': neighbours,
                                   'status': 'exploratory_candidate' if supported else 'insufficient_stability',
                                   'by_year': fixed_by_year}
            record['holdout'] = evaluate(test, spec, best['weight'], best['threshold'], outcome)
            record['holdout']['missing_years'] = sorted(holdout-{r['season'] for r in test})
            focus = evaluate([r for r in available if r['team'] == search['team']], spec, best['weight'], best['threshold'], outcome)['units']
            focus.sort(key=lambda r: r['season'])
            for i, r in enumerate(focus):
                prev = focus[i-1] if i else None
                adjacent = prev is not None and r['season'] == prev['season']+1
                r['year_difference'] = r['score']-prev['score'] if adjacent else None
                r['area_from_previous_year'] = (r['score']+prev['score'])/2 if adjacent else None
                r['role'] = 'holdout' if r['season'] in holdout else 'exploration'
            record['focus_trajectory'] = focus
            record['focus_counterexamples'] = [p for p in focus if not p['matches']]
            fold_scores = [p['test']['mcc'] for p in record['leave_one_year_out'] if p['test'] and p['test']['mcc'] is not None]
            record['year_block_summary'] = {'defined_years': len(fold_scores),
                                            'positive_years': sum(v > 0 for v in fold_scores),
                                            'mean_mcc': sum(fold_scores)/len(fold_scores) if fold_scores else None}
        results.append(record)
    return {'schema': 'aoi-season-sensitivity/1', 'config': config,
            'coverage': {'available_years': sorted({r['season'] for r in scoped}),
                         'missing_holdout_years': sorted(holdout-{r['season'] for r in scoped})},
            'results': results,
            'interpretation': '探索的な関連。有限差分はグリッド依存、面積は設定区間依存。因果・有意性・CS出場確率を示さない。2026の結果は既知のため完全な盲検ではない。'}


def markdown(report):
    lines = ['# 年単位の指標感度探索', '', report['interpretation'], '',
             '入力年: '+', '.join(map(str, report['coverage']['available_years'])), '',
             '| 指標 | 重み | 閾値 | 探索MCC | 基準からの差 | 正の関連の年 | 別年MCC | 状態 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(v):
        return '未定義' if v is None else f'{v:.4f}'
    for r in report['results']:
        c = r['candidate']
        if c is None:
            lines.append(f"| {r['id']} | — | — | — | — | — | — | 候補なし |")
            continue
        lines.append(f"| {r['id']} | {c['weight']:g} | {c['threshold']:g} | {fmt(c['exploration']['mcc'])} | {fmt(c['mcc_delta'])} | {c['positive_years']}/{c['comparable_years']} | {fmt(r['holdout']['mcc'])} | {c['status']} |")
    for r in report['results']:
        if r['candidate'] is None:
            continue
        lines += ['', f"## {r['id']}", '', f"式: `{r['definition']['x']} + weight × {r['definition']['y']}`。以上側が上位半分を予測。", '',
                  f"欠損単位: {len(r['missing_units'])}。別年の欠損: {r['holdout']['missing_years']}。", '',
                  '| 年 | 中日の指標値 | 前年差 | 条件成立 | 上位半分 | 用途 |', '|---|---:|---:|---|---|---|']
        for p in r['focus_trajectory']:
            lines.append(f"| {p['season']} | {fmt(p['score'])} | {fmt(p['year_difference'])} | {p['high']} | {p['outcome']} | {p['role']} |")
        h = r['holdout']
        exceptions = ', '.join(f"{u['season']}/{u['team']}" for u in h['units'] if not u['matches'])
        lines += ['', f"別年: P(上位半分 | 条件成立)={fmt(h['p_outcome_given_high'])}、条件不成立={fmt(h['p_outcome_given_low'])}。", '',
                  '別年の反例: '+(exceptions or ('なし' if h['units'] else '未検証'))]
        lines += ['', '### 選択点の近傍（片方だけ変更）', '',
                  '| 重み | 閾値 | MCC | 条件付き確率の差 |', '|---|---|---|---|']
        for n in r['candidate']['neighbours']:
            lines.append(f"| {n['weight']:g} | {n['threshold']:g} | {fmt(n['mcc'])} | {fmt(n['probability_gap'])} |")
        lines += ['', '### 重みを固定した閾値の推移', '',
                  '| 閾値 | MCC | 基準との差 | 条件成立側の確率 | 条件不成立側の確率 |', '|---|---|---|---|---|']
        for n in r['surface']:
            if n['weight'] == r['candidate']['weight']:
                lines.append(f"| {n['threshold']:g} | {fmt(n['mcc'])} | {fmt(n['delta_from_baseline']['mcc'])} | {fmt(n['p_outcome_given_high'])} | {fmt(n['p_outcome_given_low'])} |")
        lines += ['', '### 年全体を外して選び直す検証', '',
                  '| 外した年 | 重み | 閾値 | 外した年のMCC |', '|---|---|---|---|']
        for fold in r['leave_one_year_out']:
            selected = fold['selected']
            lines.append(f"| {fold['season']} | {fmt(selected['weight']) if selected else '未選択'} | {fmt(selected['threshold']) if selected else '未選択'} | {fmt(fold['test']['mcc']) if fold['test'] else '未実施'} |")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=Path, required=True)
    p.add_argument('--additional-season', type=Path, action='append', default=[])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--markdown', type=Path)
    args = p.parse_args(argv)
    sources = [args.season, *args.additional_season]
    rows = [json.loads(line) for path in sources for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    report = analyze(rows, tomllib.loads(args.config.read_text(encoding='utf-8')))
    report['sources'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources]
    report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding='utf-8')


if __name__ == '__main__':
    main()
