import json, math, random, subprocess, sys
from copy import deepcopy; from pathlib import Path

import pytest
from season_sensitivity import analyze, counts_metrics, prepare, shape, threshold_curve


def config():
    return {'search': dict(team='d', league='C', start=2013, end=2026, holdout=[2026],
                           min_support=2, min_years=3, min_mcc=0.3, min_year_fraction=0.75),
            'index': [dict(id='toy', x='x', y='y', weights=[0., 1., 2.], thresholds=[-2., -1., 0., 1., 2.],
                           baseline_weight=1., baseline_threshold=0.)]}


def rows():
    return [dict(season=year, team=team, league='C', x=x, y=y, upper_half=upper)
            for year in [2013, 2014, 2015, 2016, 2026]
            for team, x, y, upper in [('d', -1., -1., False), ('a', 1., 0., True), ('b', -2., 0., False), ('c', 2., 1., True)]]


def full_counts(scores, outcomes, threshold):
    high = [s >= threshold for s in scores]
    return (sum(h and y for h, y in zip(high, outcomes)), sum(h and not y for h, y in zip(high, outcomes)),
            sum(not h and y for h, y in zip(high, outcomes)), sum(not h and not y for h, y in zip(high, outcomes)))


def test_incremental_threshold_counts_match_full_scan_with_ties_and_negative_values():
    rng = random.Random(1729)
    scores = [rng.randrange(-10, 11)/4 for _ in range(100)]
    outcomes = [bool(rng.randrange(2)) for _ in scores]
    thresholds = [i/8 for i in range(-30, 31)]
    for point in threshold_curve(scores, outcomes, thresholds):
        expected = full_counts(scores, outcomes, point['threshold'])
        assert tuple(point[k] for k in ['tp', 'fp', 'fn', 'tn']) == expected
        tp, fp, fn, tn = expected
        d = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        assert point['mcc'] == ((tp*tn-fp*fn)/d if d else None)


def test_finite_differences_extrema_and_integral_have_known_values():
    points = [dict(threshold=x, mcc=y) for x, y in [(0., 0.), (1., 1.), (2., 0.), (4., 0.)]]
    s = shape(points)
    assert [p['slope'] for p in s['finite_slopes']] == [1., -1., 0.]
    assert s['slope_changes'][0]['slope_change_per_unit'] == -2
    assert s['sampled_extrema'] == [dict(at=1., kind='sampled_max')]
    assert s['trapezoid_area'] == 1.
    assert s['mean_over_grid'] == .25
    plateau = shape([dict(threshold=x, mcc=1.) for x in [0., 1., 2.]])
    assert plateau['max_intervals'] == [dict(start=0., end=2.)]
    assert plateau['sampled_extrema'] == []


def test_input_is_preserved_and_peer_rates_do_not_mix_seasons():
    data = [dict(season=y, team=t, league='C', RF=rf, RA=ra, G=10) for y in [2025, 2026] for t,rf,ra in [('d',30,20),('a',20,40)]]
    original = deepcopy(data)
    result = prepare(data)
    assert data == original
    assert result[0]['rf_vs_peers'] == 1.
    assert result[0]['ra_saved_vs_peers'] == 2.
    assert prepare([data[0]])[0]['rf_vs_peers'] is None


def test_holdout_does_not_change_selection_or_surface_or_year_folds():
    data, cfg = rows(), config()
    original = deepcopy(data)
    a = analyze(data, cfg)['results'][0]
    for r in data:
        if r['season'] == 2026:
            r.update(x=100., y=-100., upper_half=not r['upper_half'])
    b = analyze(data, cfg)['results'][0]
    assert a['candidate'] == b['candidate']
    assert a['surface'] == b['surface']
    assert a['leave_one_year_out'] == b['leave_one_year_out']
    assert a['holdout'] != b['holdout']
    # Baseline case remains perfectly separated without any mutation in analyze.
    assert analyze(original, cfg)['results'][0]['candidate']['exploration']['mcc'] == 1.


def test_year_block_excludes_every_team_of_that_year():
    data = rows()
    a = analyze(data, config())['results'][0]
    for r in data:
        if r['season'] == 2014:
            r.update(x=100., y=100., upper_half=not r['upper_half'])
    b = analyze(data, config())['results'][0]
    fold_a = next(f for f in a['leave_one_year_out'] if f['season'] == 2014)
    fold_b = next(f for f in b['leave_one_year_out'] if f['season'] == 2014)
    assert fold_a['selected'] == fold_b['selected']
    assert fold_a['test'] != fold_b['test']


def test_missing_years_and_metric_are_reported_without_bridging_gaps():
    data = [r for r in rows() if r['season'] != 2014]
    for r in data:
        if r['season'] == 2026:
            r.pop('y')
    result = analyze(data, config())['results'][0]
    assert result['holdout']['missing_years'] == [2026]
    assert len(result['missing_units']) == 4
    assert next(p for p in result['focus_trajectory'] if p['season'] == 2015)['year_difference'] is None


@pytest.mark.parametrize('change', ['duplicate', 'nonboolean', 'tie', 'no_reason', 'bad_grid', 'bad_baseline'])
def test_rejects_invalid_definitions_and_inputs(change):
    data, cfg = rows(), config()
    if change == 'duplicate': data.append(dict(data[0]))
    if change == 'nonboolean': data[0]['upper_half'] = 1
    if change == 'tie': data[0]['rank_tie'] = True
    if change == 'no_reason': cfg['exclude'] = [dict(season=2013)]
    if change == 'bad_grid': cfg['index'][0]['weights'] = [0., float('nan')]
    if change == 'bad_baseline': cfg['index'][0]['baseline_threshold'] = .123
    with pytest.raises(ValueError): analyze(data, cfg)


def test_constant_outcome_is_undefined_not_zero_evidence():
    assert counts_metrics(0, 0, 0, 5)['mcc'] is None
    data = rows()
    for r in data: r['upper_half'] = False
    assert analyze(data, config())['results'][0]['candidate'] is None


def test_cli_without_site_packages(tmp_path):
    source, cfg, out, md = [tmp_path/name for name in ['season.jsonl','config.toml','out.json','out.md']]
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows()), encoding='utf-8')
    cfg.write_text('''[search]
team="d"
league="C"
start=2013
end=2026
holdout=[2026]
min_support=2
min_years=3
min_mcc=0.3
min_year_fraction=0.75
[[index]]
id="toy"
x="x"
y="y"
weights=[0.0,1.0]
thresholds=[-1.0,0.0,1.0]
baseline_weight=1.0
baseline_threshold=0.0
''', encoding='utf-8')
    subprocess.run([sys.executable,'-S',str(Path(__file__).parents[1]/'npb'/'season_sensitivity.py'), '--season',str(source),'--config',str(cfg),'--out',str(out),'--markdown',str(md)],check=True)
    result = json.loads(out.read_text())
    assert result['sources'][0]['sha256']
    assert '中日の指標値' in md.read_text()
    # Keep machine-readable stdout clean in both mixed-output directions.
    if Path('/dev/stdout').exists():
        command = [sys.executable, '-S', str(Path(__file__).parents[1]/'npb'/'season_sensitivity.py'),
                   '--season', str(source), '--config', str(cfg)]
        mixed_md = subprocess.run(command + ['--out', str(out), '--markdown', '/dev/stdout'],
                                  check=True, capture_output=True, text=True)
        assert mixed_md.stdout == md.read_text(encoding='utf-8')
        assert out.resolve().as_uri() in mixed_md.stderr
        mixed_json = subprocess.run(command + ['--out', '/dev/stdout', '--markdown', str(md)],
                                    check=True, capture_output=True, text=True)
        assert json.loads(mixed_json.stdout) == result
        assert md.resolve().as_uri() in mixed_json.stderr
        # /dev/stdout may resolve to a regular file after shell redirection.
        redirected = tmp_path / 'redirected.json'
        with redirected.open('w', encoding='utf-8') as stream:
            redirected_run = subprocess.run(command + ['--out', '/dev/stdout', '--markdown', str(md)],
                                            check=True, stdout=stream, stderr=subprocess.PIPE, text=True)
        assert json.loads(redirected.read_text(encoding='utf-8')) == result
        assert md.resolve().as_uri() in redirected_run.stderr
        assert redirected.resolve().as_uri() not in redirected_run.stderr
        regular = subprocess.run(command + ['--out', str(out), '--markdown', str(md)],
                                 check=True, capture_output=True, text=True)
        assert out.resolve().as_uri() in regular.stdout
        assert md.resolve().as_uri() in regular.stdout
    # --out・--markdown が無ければ Markdown を標準出力に
    shown = subprocess.run([sys.executable,'-S',str(Path(__file__).parents[1]/'npb'/'season_sensitivity.py'), '--season',str(source),'--config',str(cfg)],
                           check=True, capture_output=True, text=True).stdout
    assert shown == md.read_text(encoding='utf-8')


def test_incomplete_league_is_not_silently_used_as_peer_reference():
    cfg = config()
    cfg['search']['expected_teams'] = ['d', 'a', 'b', 'c']
    with pytest.raises(ValueError, match='incomplete league'):
        analyze([r for r in rows() if not (r['season'] == 2014 and r['team'] == 'd')], cfg)


def test_2026_reference_rates_match_unrounded_counts_and_league_peer_means():
    import aoi_paths
    cycle = aoi_paths.resolve_cycle_path(aoi_paths.resolve_aoi_root(anchor=Path(__file__)))
    path = cycle / 'references/season-2026-central-summary.jsonl'
    data = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(data) == 6
    assert all(r['G'] == r['W']+r['L']+r['T'] == 143 for r in data)
    for r in data:
        b = r['batting_counts']
        assert r['bat_obp'] == (b['h']+b['bb']+b['hbp'])/(b['ab']+b['bb']+b['hbp']+b['sf'])
        assert r['bat_iso'] == (b['tb']-b['h'])/b['ab']
        for col in ['obp','iso']:
            assert r['bat_d_'+col] == pytest.approx(r['bat_'+col]-sum(p['bat_'+col] for p in data if p is not r)/5)
