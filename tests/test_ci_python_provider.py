"""Offline guards for the CI interpreter preparation order."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _job(text, name):
    match = re.search(r"^  " + re.escape(name) + r":\n(.*?)(?=^  [\w-]+:|\Z)", text, re.M | re.S)
    assert match, name
    return match.group(1)


def test_python_setup_precedes_both_metadata_resolvers():
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    for name, version in (("test", "${{ matrix.python }}"), ("vendor-dispatch", "3.15")):
        job = _job(text, name)
        setup = job.index("uses: astral-sh/setup-uv@")
        python = re.search(r"\bpython3?\s+(?:-c|-\s|[A-Za-z_])", job)
        assert python is not None
        assert setup < python.start(), name
        preparation = job[setup:python.start()]
        assert 'python-version: "' + version + '"' in preparation
        assert "activate-environment: true" in preparation
        assert job.count("uses: astral-sh/setup-uv@") == 1
