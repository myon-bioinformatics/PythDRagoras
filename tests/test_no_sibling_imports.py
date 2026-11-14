"""PythDRagoras は Aoi の兄弟サブプロジェクト（sakanalytics など）と Aoi の scripts/ を import しない。

使ってよいのは、標準ライブラリ、pyproject の依存（polars / numpy）と pytest、このリポジトリの
モジュール（Aoi の aoi_paths.py と同じ中身の aoi_paths.py を含む）だけ。既知の例外は KNOWN_EXCEPTIONS に (ファイル, 対象) で書く。
例外が使われなくなったら、このテストが落ちるので消す。Aoi の pythdragoras/ でも同じ結果になる。
"""
import ast, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".test-deps", "build", "reports", "__pycache__"}
THIRD_PARTY = {"polars", "numpy", "pytest"}
# Aoi の pyproject の pythonpath にある兄弟と、Aoi の scripts/
SIBLING_DIRS = ("blueprobe", "queryu", "sakanalytics", "nagoyaction", "dragowing", "scripts")
# Aoi のルートの aoi_paths.py（AOI_ROOT から読む。このリポジトリには置かない）。
# テストのデータの場所だけが読む。littlebit の CLI は --cycle-dir か AOI_ROOT を受け取り、配置は探さない
AOI_PROVIDED = {"aoi_paths"}
KNOWN_EXCEPTIONS = {
    # 実データの列名と sakanalytics の出力を突き合わせる
    ("tests/test_propositions.py", "import:sakanalytics"),
    # 兄弟の import 経路は AOI_ROOT の Aoi の aoi.py（environment()）に聞く
    ("conftest.py", "import:aoi"),
    # Aoi の scripts/compare_c001_replay.py を AOI_ROOT から読む
    ("tests/test_c001_replay_compare.py", "import:compare_c001_replay"),
    ("tests/test_c001_replay_compare.py", "path:scripts"),
    # 親の共通 uv tool ビルダー（Aoi から配布）。"[project.scripts]" の文字列で、Aoi の scripts/ ではない
    ("vendor/uv_tool_build.py", "path:scripts"),
    ("tests/test_standalone_metrics.py", "path:scripts"),
    # Aoi の scripts/check_c001_exploratory_candidates.py を importlib で読む
    ("tests/test_c001_exploratory_candidates.py", "path:scripts/check_c001_exploratory_candidates.py"),
}


def _sources():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        # このファイルは検出用の名前を文字列で持つので、自分は対象にしない
        if not SKIP_DIRS.intersection(rel.parts[:-1]) and path != Path(__file__).resolve():
            yield rel.as_posix(), path


def _local_modules():
    return {p.stem for d in (".", "core", "npb", "tests") for p in (ROOT / d).glob("*.py")}


def _uses(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield "import:" + alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield "import:" + node.module.split(".")[0]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip("/")
            if any(value == d or value.startswith(d + "/") for d in SIBLING_DIRS):
                yield "path:" + value


def find_uses():
    allowed = set(sys.stdlib_module_names) | THIRD_PARTY | AOI_PROVIDED | _local_modules()
    found = set()
    for rel, path in _sources():
        for use in _uses(ast.parse(path.read_text(encoding="utf-8"), filename=rel)):
            if use.startswith("path:") or use[len("import:"):] not in allowed:
                found.add((rel, use))
    return found


def test_no_sibling_subproject_imports():
    unexpected = find_uses() - KNOWN_EXCEPTIONS
    assert not unexpected, f"Aoi の兄弟サブプロジェクトか scripts/ を使っている: {sorted(unexpected)}"


def test_known_exceptions_are_still_used():
    stale = KNOWN_EXCEPTIONS - find_uses()
    assert not stale, f"使われなくなった例外を KNOWN_EXCEPTIONS から消す: {sorted(stale)}"


def test_aoi_layout_shim_is_gone():
    assert not (ROOT / "aoi_layout.py").exists()
    for name in ("littlebit_approx.py", "littlebit_holdout.py"):
        tree = ast.parse((ROOT / "core" / name).read_text(encoding="utf-8"), filename=name)
        imported = [alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        imported += [node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        assert "aoi_layout" not in imported


def test_detector_catches_imports_and_paths():
    code = ("import sakanalytics.x\nfrom queryu import a\nfrom . import b\nimport json\n"
            "p = 'scripts/x.py'\nq = '/dragowing/'\nr = 'cycles/c001'\ns = 'scriptsx'\n")
    uses = set(_uses(ast.parse(code)))
    assert {"import:sakanalytics", "import:queryu", "import:json", "path:scripts/x.py", "path:dragowing"} == uses
