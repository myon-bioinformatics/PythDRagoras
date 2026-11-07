"""PythDRagoras 単独リポジトリ用。サイクルデータ・scripts/・sakanalytics は Aoi に残す。

Aoi の場所は AOI_ROOT だけで渡す。sakanalytics への import 経路はここで作り、子プロセスにも PYTHONPATH で渡す。
"""
import os, sys
from pathlib import Path

import pytest


def _export_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.append(text)
    # CLI のテストは子プロセスで動くので、同じ import 経路を環境変数にも入れる
    current = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if text not in current:
        os.environ["PYTHONPATH"] = os.pathsep.join([*current, text])


def _root() -> Path:
    # Aoi の場所は AOI_ROOT で受け取る（Aoi の中では Aoi の conftest.py が入れる）。ここでは推定しない
    value = os.environ.get("AOI_ROOT")
    here = Path(__file__).resolve().parent
    if not value and (here.parent / "aoi_paths.py").is_file():
        # Aoi に subtree として埋め込まれ、子のディレクトリだけで pytest を動かしたとき（rootdir が子になる）
        value = str(here.parent)
    if not value:
        raise pytest.UsageError("AOI_ROOT=<Aoi のルート> を渡す（.github/workflows/ci.yml と同じ）")
    root = Path(value).expanduser().resolve()
    # テストのデータの場所（既定のサイクル）は Aoi の aoi_paths.py に聞く。このリポジトリには置かない
    if str(root) not in sys.path:
        sys.path.append(str(root))
    _export_path(root / "sakanalytics")
    return root


_root()


def pytest_configure(config):
    root = _root()
    if not (root / "cycles").is_dir():
        raise pytest.UsageError(
            f"Aoi のデータが見つからない（{root} に cycles/ がない）。"
            "Aoi を checkout して AOI_ROOT=<Aoi のルート> を渡す（.github/workflows/ci.yml と同じ）"
        )


def pytest_report_header(config):
    return f"AOI_ROOT={_root()}"


@pytest.fixture(scope="session")
def aoi_root() -> Path:
    return _root()
