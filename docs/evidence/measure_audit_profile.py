"""Opt-in, local six-environment comparison; never changes CI or downloads Python.

Usage: python docs/evidence/measure_audit_profile.py --python /path/to/python --aoi /path/to/Aoi
Requires uv and network access to the lock's registry. Writes raw logs under build/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TESTS = ['tests/test_registration_audit.py', 'tests/test_score_sensitivity.py',
         'tests/test_season_sensitivity.py', 'tests/test_littlebit_survey.py']
PLUGIN = '''import json, os
from pathlib import Path
phases = []
ids = []
def pytest_collection_finish(session):
    ids.extend(item.nodeid for item in session.items)
def pytest_runtest_logreport(report):
    phases.append([report.nodeid, report.when, report.outcome])
def pytest_sessionfinish(session, exitstatus):
    config = session.config
    Path(os.environ['RECEIPT']).write_text(json.dumps(dict(
        ids=ids, phases=phases, exitstatus=int(exitstatus),
        rootdir=str(config.rootpath), config=str(config.inipath),
        plugins=sorted([dist.project_name, dist.version] for _, dist in config.pluginmanager.list_plugin_distinfo())
    ), indent=2))
'''
INVENTORY = '''import importlib.metadata, json
print(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))
'''
def run(cmd, env, log):
    start = time.perf_counter()
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - start
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{cmd!r} failed: see {log}')
    return elapsed

def size(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--aoi', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    base = root/'build/audit-comparison'
    base.mkdir(parents=True, exist_ok=False)
    (base/'profile_receipt.py').write_text(PLUGIN)
    result = dict(source=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  aoi_source=subprocess.check_output(['git','-C',args.aoi,'rev-parse','HEAD'], text=True).strip(),
                  python=subprocess.check_output([args.python,'--version'], text=True).strip(),
                  uv=subprocess.check_output(['uv','--version'], text=True).strip(),
                  lock_sha256=hashlib.sha256((root/'uv.lock').read_bytes()).hexdigest(), samples=[])
    (base/'games.jsonl').write_text(json.dumps(dict(key='g',date='2025-04-01',home='d',away='t',hs=0,**{'as':1}))+'\n')
    (base/'analysis.toml').write_text('[focus]\nteam="d"\n[teams]\nd={name="中日"}\n')
    expected = None
    hashes = None
    try:
        for i, mode in enumerate(['full','nano','nano','full','full','nano']):
            sample = base/f'{i+1}-{mode}'
            sample.mkdir()
            env = dict(os.environ, AOI_ROOT=str(Path(args.aoi).resolve()), UV_PYTHON_DOWNLOADS='never',
                       UV_PROJECT_ENVIRONMENT=str(sample/'venv'), UV_CACHE_DIR=str(sample/'cache'),
                       PYTHONPATH=str(base), RECEIPT=str(sample/'receipt.json'))
            cmd = ['uv','sync','--locked','--python',args.python]
            if mode == 'nano': cmd += ['--only-group','audit-test']
            install = run(cmd, env, sample/'install.log')
            py = str(sample/'venv/bin/python')
            inventory = json.loads(subprocess.check_output([py,'-c',INVENTORY],env=env,text=True))
            venv_bytes, cache_bytes = size(sample/'venv'), size(sample/'cache')
            test = run([py,'-m','pytest','-p','profile_receipt',*TESTS,'-q'],env,sample/'tests.log')
            receipt = json.loads((sample/'receipt.json').read_text())
            if receipt['exitstatus'] != 0 or any(p[2] != 'passed' for p in receipt['phases']):
                raise RuntimeError('Non-passing test phase')
            if expected is None: expected = receipt
            if receipt != expected: raise RuntimeError('Collection, outcomes, configuration or plugins differ')
            run([py,'score_sensitivity.py','--games',str(base/'games.jsonl'),'--config',str(base/'analysis.toml'),
                 '--out',str(sample/'score.json'),'--markdown',str(sample/'score.md')],env,sample/'outputs.log')
            output_hashes = {name:hashlib.sha256((sample/name).read_bytes()).hexdigest() for name in ['score.json','score.md']}
            if hashes is None: hashes = output_hashes
            if hashes != output_hashes: raise RuntimeError('Output byte hashes differ')
            run(['uv','pip','check','--python',py,'--offline'],env,sample/'pip-check.log')
            result['samples'].append(dict(mode=mode,install_seconds=install,test_seconds=test,inventory=inventory,
                venv_regular_file_bytes=venv_bytes,cache_regular_file_bytes=cache_bytes,
                collected=len(receipt['ids']),phases=receipt['phases'],nodeids=receipt['ids'],plugins=receipt['plugins'],
                output_sha256=output_hashes))
            print(mode, install, test, venv_bytes, flush=True)
        result['complete'] = True
    finally:
        (base/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__ == '__main__': main()
