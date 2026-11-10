import json
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
            subprocess.run([sys.executable, str(ROOT / "scripts/build_stdlib_tool.py"), str(target)], check=True)
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
