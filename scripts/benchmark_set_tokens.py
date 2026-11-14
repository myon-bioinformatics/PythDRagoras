"""python -S -m scripts.benchmark_set_tokens; isolates tokenizer from Polars."""
import ast, json, platform, re, statistics, subprocess, time
from pathlib import Path

BASELINE = '9fb998c6db1be112cacda5d5963b9b751fd3c4e4'


def load(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tokens')
    namespace = dict(TOKEN=re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*|[&|~()])'), SetError=ValueError)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<tokenizer>', 'exec'), namespace)
    return namespace['tokens']


def main():
    old = load(subprocess.check_output(['git', 'show', BASELINE + ':sets.py']))
    new = load(Path('core/sets.py').read_text())
    rows = []
    for count in (10, 10000):
        expression = 'A &' * count + ' B   '
        expected = old(expression)
        if new(expression) != expected:
            raise SystemExit('token mismatch')
        samples = [[], []]
        for repeat in range(9):
            for i in ((0, 1) if repeat % 2 else (1, 0)):
                start = time.perf_counter_ns()
                result = (old, new)[i](expression)
                samples[i].append(time.perf_counter_ns() - start)
                if result != expected:
                    raise SystemExit('token mismatch')
        rows.append(dict(terms=count, before_ns=samples[0], after_ns=samples[1],
                         median_before_ns=statistics.median(samples[0]),
                         median_after_ns=statistics.median(samples[1])))
    print(json.dumps(dict(baseline=BASELINE, python=platform.python_version(),
                         platform=platform.platform(), repetitions=9, results=rows), indent=2))


if __name__ == '__main__':
    main()
