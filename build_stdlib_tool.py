"""Build a single-file classification metrics CLI from the canonical implementation."""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYPROJECT = '''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pythdragoras-stdlib"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.scripts]
pythdragoras = "pythdragoras:main"
pythdragoras-metrics = "pythdragoras:main"

[tool.setuptools]
py-modules = ["pythdragoras"]
'''
CLI = '''
def main(argv=None):
    import argparse
    import json
    import math
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    # Retain the initial draft's positional interface.
    if argv and argv[0] not in ("metrics", "bayes", "expected", "pythagorean", "-h", "--help"):
        argv.insert(0, "metrics")
    parser = argparse.ArgumentParser(description="Calculate classification metrics and expectations.")
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("metrics", help="classification metrics from confusion counts")
    metrics.add_argument("counts", nargs="*", type=int, metavar="COUNT")
    for name in ("tp", "fp", "fn", "tn"):
        metrics.add_argument("--" + name, type=int)
    bayes = sub.add_parser("bayes", help="posterior probability given a positive result")
    for name in ("prior", "sensitivity", "specificity"):
        bayes.add_argument("--" + name, type=float, required=True)
    expected = sub.add_parser("expected", help="expected value from value/probability pairs")
    expected.add_argument("pairs", type=float, nargs="+", metavar="NUMBER")
    pyth = sub.add_parser("pythagorean", help="Pythagorean expectation")
    pyth.add_argument("--for", dest="scored", type=float, required=True)
    pyth.add_argument("--against", type=float, required=True)
    pyth.add_argument("--exponent", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.command == "metrics":
        flags = [args.tp, args.fp, args.fn, args.tn]
        if args.counts and any(x is not None for x in flags):
            parser.error("use four positional counts or all four named counts")
        counts = args.counts if args.counts else flags
        if len(counts) != 4 or any(x is None for x in counts):
            parser.error("provide TP FP FN TN counts")
        try:
            result = classification_metrics(*counts)
        except DataError as exc:
            parser.error(str(exc))
    elif args.command == "bayes":
        prior, sensitivity, specificity = args.prior, args.sensitivity, args.specificity
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (prior, sensitivity, specificity)):
            parser.error("probabilities must be finite and between 0 and 1")
        positive = prior * sensitivity + (1 - prior) * (1 - specificity)
        result = {"posterior": prior * sensitivity / positive if positive else None,
                  "none_reason": None if positive else "positive_result_has_zero_probability"}
    elif args.command == "expected":
        pairs = args.pairs
        if len(pairs) % 2 or any(not math.isfinite(x) for x in pairs):
            parser.error("provide finite value/probability pairs")
        values, probabilities = pairs[::2], pairs[1::2]
        if any(not 0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            parser.error("probabilities must be between 0 and 1 and sum to 1")
        result = {"expected_value": math.fsum(v * p for v, p in zip(values, probabilities))}
    else:
        a, b, exponent = args.scored, args.against, args.exponent
        if any(not math.isfinite(x) for x in (a, b, exponent)) or min(a, b) < 0 or exponent <= 0:
            parser.error("scores must be nonnegative finite numbers and exponent must be positive")
        if a == b == 0:
            result = {"expectation": None, "none_reason": "both_scores_zero"}
        else:
            scale = max(a, b)
            x, y = (a / scale) ** exponent, (b / scale) ** exponent
            result = {"expectation": x / (x + y), "none_reason": None, "exponent": exponent}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new directory for the generated uv tool project")
    args = parser.parse_args()
    source = (ROOT / "propositions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = ("PropositionError", "DataError", "classification_metrics")
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names}
    if len(definitions) != len(names):
        raise ValueError("canonical proposition definitions changed")
    generated = '"""Generated stdlib classification metrics; edit propositions.py instead."""\nimport math\n\n'
    generated += "\n\n".join(ast.get_source_segment(source, definitions[name]) for name in names) + "\n" + CLI
    ast.parse(generated)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "pythdragoras.py").write_text(generated, encoding="utf-8")
    (args.out / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")


if __name__ == "__main__":
    main()
