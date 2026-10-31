"""Local rollout contract; adopted vendor locks are deliberately independent."""
import json
import re
from pathlib import Path

PARENT = "myon-bioinformatics/myon-bioinformatics"
PARENT_WORKFLOW_PIN = "9e3da7ae1846e9604ebc24bb5178f7de094a4e77"
PARENT_REPOSITORY_ID = "638801160"
LIVE_PARENT_REF = "${{ steps.parent.outputs.sha }}"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".github/workflows").is_dir())


def _workflow_texts():
    for path in sorted((ROOT / ".github/workflows").glob("*")):
        if path.suffix in (".yml", ".yaml"):
            yield path, path.read_text(encoding="utf-8")


def _checkout_ref(lines, index):
    for item in lines[index + 1:index + 12]:
        match = re.fullmatch(r"\s+ref:\s*(.+?)\s*", item)
        if match:
            return match.group(1).strip().strip("'\"")
    return None


def test_parent_workflow_references_use_reviewed_rollout_commit():
    pins = []
    live_refs = []
    for _path, text in _workflow_texts():
        # Capture mutable and shortened references too, rather than ignoring them.
        pins.extend(re.findall(r"uses:\s*['\"]?" + re.escape(PARENT) + r"/[^\s@'\"]+@([^\s'\"]+)", text))
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.fullmatch(r"\s+repository:\s*['\"]?" + re.escape(PARENT) + r"['\"]?\s*", line):
                ref = _checkout_ref(lines, i)
                if ref == LIVE_PARENT_REF:
                    live_refs.append(ref)
                else:
                    pins.append(ref)
    assert pins, "shared parent references must remain observable"
    assert set(pins) == {PARENT_WORKFLOW_PIN}, pins
    assert live_refs == [LIVE_PARENT_REF]


def test_vendor_dispatch_verifies_lock_on_resolved_parent_main():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "git ls-remote" in text
    assert PARENT_REPOSITORY_ID in text
    assert "pilot/vendor_dispatch.py" in text
    assert "--manifest vendor.lock.json" in text
    assert "--record" not in text
    assert 'node-version: "24"' in text
    assert 'python-version: "3.12"' in text
    assert "actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1 # v7.1.0" in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" in text
    for line in text.splitlines():
        if "vendor_dispatch.py" in line:
            assert "-S" not in line
            assert "--record" not in line
    lock = json.loads((ROOT / "vendor.lock.json").read_text(encoding="utf-8"))
    assert lock["schema"] == "vendor-lock/1"
    assert isinstance(lock["files"], list)
