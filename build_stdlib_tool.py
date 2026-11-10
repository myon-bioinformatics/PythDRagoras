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
pythdragoras-metrics = "pythdragoras:main"

[tool.setuptools]
py-modules = ["pythdragoras"]
'''
CLI = '''
def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Calculate classification metrics from TP FP FN TN counts.")
    for name in ("tp", "fp", "fn", "tn"):
        parser.add_argument(name, type=int)
    args = parser.parse_args(argv)
    try:
        result = classification_metrics(args.tp, args.fp, args.fn, args.tn)
    except DataError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
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
