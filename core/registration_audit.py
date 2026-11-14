"""Audit provenance labels without guessing preregistration from posthoc=false.

This module is deliberately stdlib-only.  It validates metadata contracts; it does
not rewrite research results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class Provenance:
    status: str
    evidence: str | None = None


def _has_text_evidence(value: object) -> bool:
    """Require a named record, not a truthy value coerced to text."""
    return isinstance(value, str) and bool(value.strip())


def classify(record: dict) -> Provenance:
    """Return only statuses supported by explicit machine-readable evidence."""
    if record.get("posthoc") is True:
        return Provenance("posthoc", "posthoc=true")
    prereg = record.get("preregistration")
    if isinstance(prereg, dict) and prereg.get("recorded_before_evaluation") is True and _has_text_evidence(prereg.get("evidence")):
        return Provenance("preregistered", prereg["evidence"])
    # Missing or false posthoc is deliberately not evidence of preregistration.
    return Provenance("unknown")


def audit_record(record: dict) -> list[str]:
    errors = []
    if record.get("posthoc") is True and record.get("preregistration"):
        errors.append("posthoc and preregistration cannot both be asserted")
    prereg = record.get("preregistration")
    if prereg is not None:
        if not isinstance(prereg, dict):
            errors.append("preregistration must be a table")
        else:
            if prereg.get("recorded_before_evaluation") is not True:
                errors.append("preregistration requires recorded_before_evaluation=true")
            if not _has_text_evidence(prereg.get("evidence")):
                errors.append("preregistration requires non-blank text evidence")
    return errors


def load_exprs(path: Path) -> list[dict]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return list(data.get("expr", []))


def narrative_candidates(path: Path) -> dict[str, str]:
    """Find comments claiming an expression family was written before evaluation.

    These are audit leads only.  A comment never upgrades a record automatically.
    """
    current = None
    leads: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") and ("計算前に書いた" in line or "判定はまだ" in line):
            current = line.lstrip("#").strip()
            continue
        m = re.fullmatch(r'id\s*=\s*"(E\d+)"', line)
        if m and current:
            leads[m.group(1)] = current
        if line.startswith("#") and current and not ("計算前に書いた" in line or "判定はまだ" in line):
            current = None
    return leads
