import json
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
        subprocess.run([sys.executable, str(ROOT / "build_stdlib_tool.py"), str(target)], check=True)
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
