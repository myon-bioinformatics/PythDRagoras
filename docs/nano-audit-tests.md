# Optional audit and sensitivity test environment

`audit-test` is an opt-in, pytest-only dependency group for four complete existing
files: registration audit, score sensitivity, season sensitivity, and littlebit
survey. It is useful while editing those stdlib modules without installing the
numeric stack. It is **not** a replacement for the full suite, NumPy/Polars
oracles, Node tests, or CI. No tests, skips, plugins, conftest, workflows, or
runtime dependencies are changed.

## Use

From this repository, with an existing Aoi checkout (the normal conftest still
requires its cycles and source):

```sh
UV_PROJECT_ENVIRONMENT=build/audit-test/.venv uv sync --locked --only-group audit-test
AOI_ROOT=/absolute/path/to/Aoi UV_PROJECT_ENVIRONMENT=build/audit-test/.venv \
  uv run --locked --no-sync pytest -q \
  tests/test_registration_audit.py tests/test_score_sensitivity.py \
  tests/test_season_sensitivity.py tests/test_littlebit_survey.py
```

Keep `--no-sync` on subsequent `uv run` commands: an ordinary `uv run` may restore
the default project dependencies. Use the documented ordinary full-suite command
before submitting a change. The new group reuses the locked pytest; all ten
third-party lock records, versions, markers, sources, and hashes are unchanged.

## Local comparison (2026-11-04 UTC)

PythDRagoras source `9fb998c6db1be112cacda5d5963b9b751fd3c4e4`,
Aoi `157c67ce8b6f49008ad1063f1727f433d7ce4ab3`, Linux x86_64,
CPython 3.15.1, uv 0.13.5. The source above is the baseline before the two
metadata changes; the exact measured lock SHA is in the evidence JSON.

Six new environments and six separate initially empty uv caches, ordered
full/nano, nano/full, full/nano. All selected tests run in the same checkout;
plugin autoload remains enabled. Dependency preparation includes downloads and
installation, but Python acquisition and source cloning are outside the clock.

| Metric | Default full | audit-test |
| --- | ---: | ---: |
| Installed distributions | 8 | 5 |
| Environment regular-file bytes | 254,653,805 | 6,616,997 |
| Cache regular-file bytes | 254,627,152 | 6,589,076 |
| Install seconds, three samples | 7.511 / 8.262 / 4.603 | 6.703 / 3.008 / 3.842 |
| Install median seconds | 7.511 | 3.842 |
| Test process median seconds | 3.261 | 3.208 |

Environment logical regular-file length decreased by about 97.4%. These are not
physical disk or image sizes: hardlinks can share cache and environment bytes,
so their totals must not be added. Shared Python, stdlib, Aoi sources and OS
libraries remain installed and are excluded. No container isolation or CPU/RAM
limits were introduced.

Install time was lower in this local sample, but this is not a guaranteed
speedup or a CI-time claim. CDN/OS caches, network and host load were uncontrolled;
the first full install overlapped the end of a separate BlueProbe comparison.
All measured samples, including slower nano preparation, are retained. Test
process medians are close and do not establish meaningful execution speedup.

## Verification

- All six environments: 73 passed, no skip, 438 test calls total. Unique node
  IDs, all 219 setup/call/teardown outcomes per environment, rootdir/config and
  registered third-party plugin inventory match exactly.
- Real score-sensitivity CLI JSON and Markdown outputs are byte-identical in
  all six environments. Hashes are recorded. This is a deliberately bounded
  two-output fixture, not a claim that all research outputs were compared.
- `uv pip check --offline` succeeds in each environment. `uv lock --check
  --offline` succeeds. Initial offline lock preparation lacked NumPy metadata
  and failed; normal online lock resolution succeeded without changing any
  third-party records.
- Python 3.12.15 small environment also passes all 73 tests. The documented
  `uv run --locked --no-sync` command was additionally run offline on 3.15.1:
  73 passed, and the five-package inventory remained unchanged.
- Default full environment on Python 3.15.1: 478 passed / 71 skipped, exactly
  matching a clean baseline worktree. Of those skips, 69 need Node 26+ (local
  default Node is 24.22.1), one needs an original artifact ZIP, and one is an
  opt-in BLAS-dependent reconstruction. `HOLDOUT_SOURCE_PROVENANCE=report`
  matches the existing pull-request CI policy; strict provenance was not tested.
- NumPy/Polars are absent from the small environment. Collecting the existing
  `test_propositions.py` there fails with exit 2 and missing `polars`, rather
  than silently skipping a scientific oracle.
- The registration file has pre-existing early returns when a cycle sets.toml
  is absent. The actual pinned Aoi checkout used here has
  `cycles/c001-chunichi/sets.toml`; no new return or skip was introduced.

Main advanced during this work to `659f5c78187f228ba87320ebaf9e98ba72a7590b`
(the set-tokenizer change #37). The branch was fast-forwarded without conflicts.
Its two added tests and changed tokenizer do not overlap the four selected files
or dependency metadata. Revalidation on that source: 73 selected tests passed;
full suite and clean baseline both 480 passed / 71 skipped. Historical six-sample numbers remain pinned
to the earlier source and were not rewritten as newer measurements.

Raw samples, inventories, per-phase outcomes, node IDs and output hashes:
[`evidence/nano-audit-tests.json`](evidence/nano-audit-tests.json).
Reproducer: [`evidence/measure_audit_profile.py`](evidence/measure_audit_profile.py).
It refuses an existing output directory, retains logs and an incomplete summary
on failure, and uses explicitly supplied Python/Aoi paths. Subsequent hardening
sets `UV_PYTHON_DOWNLOADS=never`; the measured runs already used an existing
absolute Python executable, with no interpreter acquisition.

Other OSes, Python 3.11/3.13/3.14, whole-CI timing, physical disk savings and
repeated runtime throughput are not established by this experiment. CI and the
ordinary full dependency environment remain unchanged.
