import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class StandaloneExamples(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        target = Path(self.temp.name) / "tool"
        target.mkdir()
        shutil.copy2(ROOT / "pythdragoras.py", target / "pythdragoras.py")  # 汎用版はこの 1 ファイルだけで動く
        self.script = target / "pythdragoras.py"

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", str(self.script), *args], cwd=self.temp.name, text=True, capture_output=True)

    def call(self, *args):
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_examples(self):

        data = self.call("metrics", "--tp", "80", "--fp", "10", "--fn", "20", "--tn", "90")
        self.assertEqual(data["accuracy"], .85)
        self.assertAlmostEqual(data["f1"], 160 / 190)
        self.assertEqual(self.call("metrics", "0", "0", "0", "0")["none_reasons"]["precision"], "no_positive_predictions")
        self.assertAlmostEqual(self.call("bayes", "--prior", ".01", "--sensitivity", ".9", "--specificity", ".95")["posterior"], .153846153846)
        self.assertEqual(self.call("expected", "10", ".25", "20", ".75")["expected_value"], 17.5)
        self.assertEqual(self.run_cli("expected", "10", ".2").returncode, 2)
        self.assertAlmostEqual(self.call("pythagorean", "--for", "80", "--against", "60")["expectation"], .64)
        self.assertIsNone(self.call("pythagorean", "--for", "0", "--against", "0")["expectation"])
        self.assertEqual(self.run_cli("bayes", "--prior", "nan", "--sensitivity", ".9", "--specificity", ".95").returncode, 2)

    def test_bayes_tiny_positive_probabilities_are_not_impossible(self):
        for prior, sensitivity in (("1e-200", "1e-200"), ("5e-324", ".5")):
            with self.subTest(prior=prior, sensitivity=sensitivity):
                result = self.call("bayes", "--prior", prior, "--sensitivity", sensitivity,
                                   "--specificity", "1")
                self.assertEqual(result, {"posterior": 1.0, "none_reason": None})
        # Both causes are possible; avoid rounding the true-positive product
        # to zero before dividing by a small false-positive probability.
        result = self.call("bayes", "--prior", "5e-324", "--sensitivity", ".5",
                           "--specificity", "0.9999999999999999")
        self.assertAlmostEqual(result["posterior"] / 2.2250738585072014e-308, 1.0)
        self.assertIsNone(result["none_reason"])

    def test_bayes_boundary_probabilities_and_impossible_results(self):
        for prior, sensitivity, specificity, posterior in (
            (0, 1, 0, 0.0), (1, 1, 0, 1.0), (.5, 0, 0, 0.0),
            (.5, 1, 1, 1.0), (0, 1, 1, None), (1, 0, 0, None), (.5, 0, 1, None),
        ):
            with self.subTest(prior=prior, sensitivity=sensitivity, specificity=specificity):
                result = self.call("bayes", "--prior", str(prior), "--sensitivity", str(sensitivity),
                                   "--specificity", str(specificity))
                self.assertEqual(result, {"posterior": posterior, "none_reason":
                    "positive_result_has_zero_probability" if posterior is None else None})
