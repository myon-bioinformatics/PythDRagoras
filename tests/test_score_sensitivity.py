import json, random, subprocess, sys
from pathlib import Path
import pytest
from score_sensitivity import analyze, sweep, team_games


def oracle(scores):
    w, l = sum(a > b for a,b in scores), sum(a < b for a,b in scores)
    rf, ra = sum(a for a,_ in scores), sum(b for _,b in scores)
    p = rf**1.83/(rf**1.83+ra**1.83) if rf+ra else None
    return dict(RF=rf, RA=ra, W=w, L=l, T=len(scores)-w-l, wpct=w/(w+l) if w+l else None,
                pythag_fixed=p, expected_wins=(w+l)*p if p is not None else None)


@pytest.mark.parametrize('axis', ['RF','RA'])
def test_every_increment_matches_full_recomputation(axis):
    rng = random.Random(71)
    games = [dict(key=str(i),date='2025-04-01',rf=rng.randrange(8),ra=rng.randrange(8)) for i in range(80)]
    result = sweep(games,axis,4)
    scores = [(g['rf'],g['ra']) for g in games]
    assert result['baseline'] == oracle(scores)
    budget = 0
    for edit in result['timeline']:
        i = int(edit['key'])
        before = oracle(scores)
        assert list(scores[i]) == edit['old']
        budget += sum(abs(a-b) for a,b in zip(edit['old'],edit['new']))
        scores[i] = tuple(edit['new'])
        expected = oracle(scores)
        assert edit['metrics'] == pytest.approx(expected)
        assert edit['changed_runs'] == budget
        for k,v in before.items():
            assert edit['local_delta'][k] == (expected[k]-v if v is not None and expected[k] is not None else None)
    assert result['curve'][-1]['metrics'] == pytest.approx(oracle(scores))


def game(year=2025,key='g',hs=0,home='d',away='t'):
    return dict(key=key,date=f'{year}-04-01',home=home,away=away,hs=hs,**{'as':1})


def test_win_draw_loss_zero_floor():
    rows = team_games([game()],'d')
    result = sweep(rows,'RF')
    assert [(p['metrics']['W'],p['metrics']['T'],p['metrics']['L']) for p in result['curve']] == [(0,0,1),(0,1,0),(1,0,0),(1,0,0)]
    assert result['curve'][1]['delta']['wpct'] is None
    assert [p['changed_runs'] for p in sweep(rows,'RA')['curve']] == [0,1,1,1]
    assert sweep([dict(key='z',date='2025-01-01',rf=0,ra=0)],'RF')['baseline']['pythag_fixed'] is None


def test_orientation_exclusions_and_holdout():
    games = [game(y,str(y),home='t',away='d',hs=2) for y in range(2020,2026)]
    opts = dict(excluded=[dict(season=2020,reason='short')],holdout=[2025])
    result = analyze(games,'d',**opts)
    assert {r['season'] for r in result['results']} == set(range(2021,2026))
    assert all(r['baseline']['RF'] == 1 for r in result['results'])
    assert all(c['seasons']==4 and c['status']=='exploratory_candidate' for c in result['candidates'])
    changed = [dict(g,hs=100) if g['date'].startswith('2025') else g for g in games]
    assert analyze(changed,'d',**opts)['candidates'] == result['candidates']


@pytest.mark.parametrize('games', [[game(),game()],[game(hs=-1)],[game(hs=True)],[game(hs=1.5)],[dict(game(),date='bad')]])
def test_invalid_input(games):
    with pytest.raises(ValueError):
        analyze(games,'d')


def test_empty_exclusion_and_sample_size():
    with pytest.raises(ValueError):
        analyze([game()],'x')
    with pytest.raises(ValueError):
        analyze([game()],'d',excluded=[dict(season=2025)])
    assert all(c['status']=='insufficient_evidence' for c in analyze([game()],'d')['candidates'])


def test_cli(tmp_path):
    source = tmp_path/'games.jsonl'
    source.write_text(json.dumps(game())+'\n',encoding='utf-8')
    config = tmp_path/'analysis.toml'
    config.write_text('[focus]\nteam="d"\n[teams]\nd={name="中日"}\n',encoding='utf-8')
    out = tmp_path/'out.json'
    md_out = tmp_path/'out.md'
    subprocess.run([sys.executable,str(Path(__file__).parents[1]/'npb'/'score_sensitivity.py'),'--games',str(source),'--config',str(config),'--out',str(out),'--markdown',str(md_out)],check=True)
    assert json.loads(out.read_text())['schema']=='aoi-score-sensitivity/1'
    assert md_out.exists()
    content = md_out.read_text(encoding='utf-8')
    assert '# 試合ごとの得失点感度分析 (d)' in content
    assert '## シーズン別得失点変更曲線' in content
