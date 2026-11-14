import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StandaloneMetricsTest(unittest.TestCase):
    def test_generated_script_metrics_and_input_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "tool"
            target.mkdir()
            shutil.copy2(ROOT / "pythdragoras.py", target / "pythdragoras.py")  # 汎用版はこの 1 ファイルだけで動く
            script = target / "pythdragoras.py"
            result = subprocess.run([sys.executable, "-I", str(script), "16", "0", "56", "72"],
                                    capture_output=True, text=True, cwd=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["n"], 144)
            self.assertEqual(data["precision"], 1)
            self.assertAlmostEqual(data["recall"], 16 / 72)
            empty = subprocess.run([sys.executable, "-I", str(script), "0", "0", "0", "0"],
                                   capture_output=True, text=True, cwd=temp, check=True)
            self.assertIsNone(json.loads(empty.stdout)["precision"])
            invalid = subprocess.run([sys.executable, "-I", str(script), "-1", "0", "0", "0"],
                                     capture_output=True, text=True, cwd=temp)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("nonnegative", invalid.stderr)


def _random_matrix(rng, m, n):
    # 決め打ちの刻みにしない: A×10^-N の A と N を乱数で選ぶ
    return [[rng.uniform(-9.99, 9.99) * 10.0 ** -rng.randint(0, 4) for _ in range(n)] for _ in range(m)]


class GenericLittleBitTest(unittest.TestCase):
    """汎用版（標準ライブラリだけ）の LittleBit。numpy 版（core/littlebit_prune.py）と同じ近似になる。"""

    def test_reconstruction_matches_the_numpy_version(self):
        import random
        import numpy as np
        import pythdragoras as generic
        import littlebit_prune
        rng = random.Random(20261114)
        for m, n, rank, stages in ((7, 5, 2, 2), (5, 9, 3, 1), (12, 6, 2, 3)):
            M = _random_matrix(rng, m, n)
            mine = np.array(generic.reconstruct(generic.littlebit(M, rank, stages), m, n))
            theirs = littlebit_prune._dense(littlebit_prune.littlebit(np.array(M), rank, stages))
            scale = max(1.0, float(np.abs(np.array(M)).max()))
            self.assertLess(float(np.abs(mine - theirs).max()) / scale, 1e-6, (m, n, rank, stages))

    def test_svd_and_lstsq_match_numpy(self):
        import random
        import numpy as np
        import pythdragoras as generic
        rng = random.Random(7)
        for m, n in ((6, 4), (3, 8), (5, 5)):
            M = _random_matrix(rng, m, n)
            U, S, Vt = generic.svd(M)
            self.assertTrue(np.allclose(S, np.linalg.svd(np.array(M), compute_uv=False), rtol=1e-9, atol=1e-12))
            self.assertTrue(np.allclose(np.array(U) @ np.diag(S) @ np.array(Vt), M, atol=1e-9))
            b = [rng.uniform(-1, 1) for _ in range(m)]
            self.assertTrue(np.allclose(generic.lstsq(M, b), np.linalg.lstsq(np.array(M), b, rcond=None)[0], atol=1e-9))

    def test_overlap_bounds_contain_the_true_inner_products(self):
        import random
        import pythdragoras as generic
        rng = random.Random(3)
        X = [[rng.random() < 0.4 for _ in range(11)] for _ in range(9)]
        approx, err = generic.overlap_bounds(X, rank=2, stages=2)
        for a in range(9):
            for b in range(9):
                true = sum(x and y for x, y in zip(X[a], X[b]))
                self.assertLessEqual(abs(true - approx[a][b]), err[a][b] + 1e-9)

    def test_mcc_upper_bounds_every_point_in_the_box(self):
        import pythdragoras as generic
        n_pos, n_neg = 30, 50
        ub = generic.mcc_upper(10, 14, 3, 6, n_pos, n_neg)
        for tp in range(10, 15):
            for fp in range(3, 7):
                mcc = generic.classification_metrics(tp, fp, n_pos - tp, n_neg - fp)["mcc"]
                self.assertLessEqual(mcc, ub + 1e-12)
        self.assertEqual(generic.mcc_upper(0, 1, 20, 25, n_pos, n_neg), 0.0)

    def test_bits_and_or_counts(self):
        import pythdragoras as generic
        a, b, y = generic.to_bits([1, 0, 0, 1, 0]), generic.to_bits([0, 1, 0, 1, 0]), generic.to_bits([1, 1, 1, 0, 0])
        self.assertEqual(generic.or_counts(a, b, y, 5), (2, 1))

    def test_cli_and_uv_tool_without_site_packages(self):
        import tomllib
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "pythdragoras.py"
            shutil.copy2(ROOT / "pythdragoras.py", script)
            run = subprocess.run([sys.executable, "-I", str(script), "littlebit", "--rank", "1"], input="[[1, 2], [2, 4.5]]",
                                 capture_output=True, text=True, cwd=temp)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(run.stdout)
            self.assertEqual((data["shape"], len(data["parts"][0]["l"])), ([2, 2], 1))
            bad = subprocess.run([sys.executable, "-I", str(script), "littlebit"], input="[[1, 2], [3]]",
                                 capture_output=True, text=True, cwd=temp)
            self.assertEqual(bad.returncode, 2)
            out = Path(temp) / "tool"
            subprocess.run([sys.executable, "-S", str(ROOT / "vendor" / "uv_tool_build.py"), str(ROOT / "pythdragoras.py"),
                            "--out", str(out)], check=True, capture_output=True)
            project = tomllib.loads((out / "pyproject.toml").read_text(encoding="utf-8"))["project"]
            self.assertEqual((project["dependencies"], project["scripts"]), ([], {"pythdragoras": "pythdragoras:main"}))


def test_generic_metrics_are_the_canonical_definitions():
    import ast
    names = ("PropositionError", "DataError", "classification_metrics")

    def defs(path):
        source = path.read_text(encoding="utf-8")
        return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if getattr(n, "name", None) in names}

    assert defs(ROOT / "pythdragoras.py") == defs(ROOT / "core" / "propositions.py")


def test_apply_exclusions_on_plain_rows_and_on_a_polars_frame():
    import polars as pl
    import pythdragoras as generic
    rows = [{"season": 2019, "x": 1}, {"season": 2020, "x": 2}, {"season": 2020, "x": 3}, {"season": 2021, "x": 4}]
    exclude = [{"season": 2020, "reason": "短縮シーズン"}]
    kept, record = generic.apply_exclusions(rows, exclude)
    assert [r["x"] for r in kept] == [1, 4] and record[0]["rows_removed"] == 2
    frame, frame_record = generic.apply_exclusions(pl.DataFrame(rows), exclude)
    assert frame["x"].to_list() == [1, 4] and frame_record == record
    try:
        generic.apply_exclusions(rows, [{"season": 2020}])
    except ValueError:
        pass
    else:
        raise AssertionError("an exclusion without a reason must be refused")
