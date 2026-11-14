"""Common diagnostics never replace the fixed proposition judgement."""
import json, math
from copy import deepcopy; from pathlib import Path

import polars as pl
import pytest
import propositions as pr

import aoi_paths

ROOT = aoi_paths.resolve_aoi_root(anchor=Path(__file__))
CYCLE = aoi_paths.resolve_cycle_path(ROOT)


def prop(pid='X'):
    return dict(id=pid, statement='x implies y', strength='usually', min_n=1,
                **{'if': [dict(col='x', op='>', value=0)], 'then': [dict(col='y', op='>', value=0)]})


def result(pid='X'):
    st = pl.DataFrame(dict(x=[1,1,0,0], y=[1,0,1,0], team=list('abcd'), season=[2025]*4, team_name=list('abcd')))
    return pr.evaluate(prop(pid), st)


def group(definition):
    condition = dict(definition_sha256=definition, years=[2023,2024,2025,2026], loss=0.)
    counts = [dict(tp=0,fp=2,fn=2,tn=0), dict(tp=1,fp=1,fn=1,tn=1), dict(tp=2,fp=0,fn=0,tn=2)]
    return dict(condition=condition, expected_seeds=[0,1,2], runs=[dict(seed=s,condition=deepcopy(condition),counts=c)
                                                               for s,c in enumerate(counts)])


def evidence(rows):
    return dict(schema='aoi-proposition-seed-evaluation/1', synthetic=True, real_sample_increment=0,
                propositions={r['id']:dict(definition_sha256=r['definition_sha256'],groups=[group(r['definition_sha256'])]) for r in rows})


def test_diagnostics_all_forms_and_no_seed_are_uniform():
    r = result()
    assert r['metrics']['precision'] == r['metrics']['recall'] == r['metrics']['f1'] == .5
    assert r['metrics']['mcc'] == 0
    assert all('metrics' in f for f in r['forms'])
    assert r['seed_sensitivity']['groups'] is None and r['seed_sensitivity']['seed_count'] is None
    assert 'None' in pr.render_index([r], {})


def test_perfect_sufficient_condition_does_not_require_converse():
    m = pr.classification_metrics(16,0,56,72)
    assert m['precision']==1 and m['recall']==pytest.approx(16/72)
    assert m['mcc']==pytest.approx(.3535533905932738)


def test_zero_denominators_remain_none_with_reasons():
    empty=pr.classification_metrics(0,0,0,0)
    assert all(empty[k] is None and empty['none_reasons'][k] for k in pr.EVALUATION_METRICS)
    all_true=pr.classification_metrics(10,3,0,0)
    assert all_true['recall']==1 and all_true['mcc'] is None
    assert all_true['none_reasons']['recall'] is None


@pytest.mark.parametrize('bad',[-1,True,.5,None])
def test_invalid_counts_fail_instead_of_becoming_none(bad):
    with pytest.raises(pr.DataError):pr.classification_metrics(bad,0,0,0)


def test_seed_spread_recomputed_from_counts_and_signs():
    g=pr.summarize_seed_group(group('fixed'),'fixed')
    m=g['metrics']['mcc']
    assert m['median']==0 and m['range']==2 and m['population_sd']==pytest.approx(math.sqrt(2/3))
    assert g['mcc_sign_flip'] is True and g['mcc_signs']['positive']==1


def test_invariant_single_seed_and_undefined_are_different():
    g=group('fixed');g['runs']=[g['runs'][1]];g['expected_seeds']=[1]
    m=pr.summarize_seed_group(g,'fixed')['metrics']['mcc']
    assert m['status']=='single_seed' and m['range'] is None and m['population_sd'] is None
    g['runs'].append({**deepcopy(g['runs'][0]),'seed':2});g['expected_seeds'].append(2)
    assert pr.summarize_seed_group(g,'fixed')['metrics']['mcc']['status']=='invariant_in_tested_seeds'
    for run in g['runs']:run['counts']=dict(tp=1,fp=1,fn=0,tn=0)
    assert pr.summarize_seed_group(g,'fixed')['metrics']['mcc']['status']=='undefined'


@pytest.mark.parametrize('change',['missing','duplicate','condition','definition','negative-count'])
def test_bad_seed_evidence_is_atomic(change):
    rows=[result('X'),result('Y')];original=deepcopy(rows);body=evidence(rows)
    g=body['propositions']['Y']['groups'][0]
    if change=='missing':g['runs'].pop()
    elif change=='duplicate':g['runs'].append(deepcopy(g['runs'][0]))
    elif change=='condition':g['runs'][0]['condition']['loss']=.4
    elif change=='definition':body['propositions']['Y']['definition_sha256']='different'
    else:g['runs'][0]['counts']['fp']=-1
    with pytest.raises(pr.DataError):pr.attach_seed_evaluation(rows,body)
    assert rows==original


def test_unavailable_fields_and_fixed_verdict_preserved():
    rows=[result('X'),result('Y')];before=[r['judgement'] for r in rows]
    body=evidence(rows)
    body['propositions']['Y'].update(groups=None,reason='inning_data_unavailable')
    pr.attach_seed_evaluation(rows,body)
    assert rows[0]['seed_sensitivity']['status']=='evaluated'
    assert rows[1]['seed_sensitivity']['reason']=='inning_data_unavailable'
    assert [r['judgement'] for r in rows]==before


def test_all_114_existing_verdicts_and_counterexamples_unchanged():
    props,_=pr.load(CYCLE/'propositions.toml')
    saved={r['id']:r for r in map(json.loads,(CYCLE/'outputs/propositions.jsonl').read_text().splitlines())}
    st=pl.read_ndjson(CYCLE/'outputs/season.jsonl');included=st.filter(pl.col('season')!=2020)
    excluded=st.filter(pl.col('season')==2020)
    for p in props:
        r=pr.evaluate(p,included,excluded,'d');old=saved[p['id']]
        assert r['judgement']==old['judgement']
        assert r['definition_sha256']==old['definition_sha256']
        assert r['metrics']['tp']==r['forms'][0]['hold']
        for a,b in zip(r['forms'],old['forms']):
            for key,value in a.items():
                if key=='metrics':
                    continue
                # Float summation changed in Python 3.12. Keep the verdict,
                # integer counts and counterexamples exact across 3.11–3.15.
                if isinstance(value,float) or key=='ci':
                    assert value==pytest.approx(b[key],rel=1e-12,abs=1e-12)
                else:
                    assert value==b[key]


def test_saved_seed_registry_matches_every_registered_definition():
    props,_=pr.load(CYCLE/'propositions.toml')
    st=pl.read_ndjson(CYCLE/'outputs/season.jsonl').filter(pl.col('season')!=2020)
    rows=[pr.evaluate(p,st,focus='d') for p in props]
    body=json.loads((CYCLE/'outputs/exploratory/proposition-evaluation/seed-evaluation.json').read_text())
    pr.attach_seed_evaluation(rows,body)
    assert len(rows)==114
    assert sum(r['seed_sensitivity']['status']=='evaluated' for r in rows)==13
    assert next(r for r in rows if r['id']=='P62')['seed_sensitivity']['groups'] is None
    p15=next(r for r in rows if r['id']=='P15')
    loss=[g for g in p15['seed_sensitivity']['groups'] if g['condition']['dataset']=='skill-budget']
    assert len(loss)==2 and all(g['seed_count']==10 for g in loss)


def test_cli_attaches_diagnostics_to_all_propositions(tmp_path, capsys):
    from season import main
    assert main(['--season',str(CYCLE/'outputs/season.jsonl'),'--config',str(CYCLE/'analysis.toml'),
                 '--propositions',str(CYCLE/'propositions.toml'),'--seed-evaluation',
                 str(CYCLE/'outputs/exploratory/proposition-evaluation/seed-evaluation.json'),
                 '--outdir',str(tmp_path)])==0
    capsys.readouterr()
    rows=list(map(json.loads,(tmp_path/'propositions.jsonl').read_text().splitlines()))
    assert len(rows)==114 and all('metrics' in r and 'seed_sensitivity' in r for r in rows)
    assert sum(r['seed_sensitivity']['groups'] is None for r in rows)==101
    md=(tmp_path/'index.md').read_text()
    assert md.count('| **P')==114 and 'Precision' in md and 'None' in md
