# PythDRagoras

<p align="center">
  <img src="docs/brand/icon.png" alt="PythDRagoras icon" width="160" />
</p>

[Icon](docs/brand/icon.png) · [Favicon](docs/brand/favicon.png)

> Propositions, precedents, objections.
>
> Tests explanations as propositions with counterexamples, rule search and sensitivity analysis for NPB baseball outcomes. Split out from Aoi's pythdragoras/ (git subtree).

![PythDRagoras](https://github.com/myon-bioinformatics/Aoi/raw/main/docs/images/pythdragoras.webp)

*Concept image (a figure for sharing the idea; not the actual data or HTML structure)*

## Role in Aoi

**Propositions, proofs and coordinate decomposition.**

| Challenges | Approach | Outcomes & value |
|---|---|---|
| Explanations are often ambiguous | 1. State propositions and draw diagrams | Understanding through figures and text |
| Diagrams and concepts are hard to grasp | 2. Search for conditions | Try conditions by changing them |
| The verification procedure is unclear | 3. Run sensitivity analysis | Prompts for further thinking |

Topics: mathematics, visualization, education support

## Overview

PythDRagoras is a tool that writes explanations as "if A then B" propositions and always verifies them with counterexamples and objections laid out side by side. Its aim is less to produce answers than to find where explanations fall short.

- **Propositions and "objection" (異議あり)**: judged in four forms: the original proposition, contrapositive, converse, and inverse. Judgement criteria are declared before seeing results, and if there is even one counterexample, counterexamples are listed even when the verdict is "supported" ("Propositions and "objection"" below).
- **Condition search**: exhaustively searches combinations of candidate conditions and ranks them by how well they predict the target state (`rule_search.py`).
- **Sensitivity analysis**: examines changes when metrics, weights, and boundaries, or per-game runs scored and allowed, are varied (`season_sensitivity.py`, `score_sensitivity.py`).
- Exclusions require a reason, and what was excluded and why is kept in the results. External assertions are registered as claims, not observations.

The philosophy is in Aoi's [docs/PythDRagoras.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/PythDRagoras.md), and the proposition spec in [docs/propositions.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/propositions.md).
This repository was split out from the [Aoi](https://github.com/myon-bioinformatics/Aoi) subproject [`pythdragoras/`](https://github.com/myon-bioinformatics/Aoi/tree/main/pythdragoras) (see "Relationship with Aoi" below).

## Installation and usage

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The only runtime dependencies are polars and numpy (`pyproject.toml`; ortools is not used).

```bash
git clone https://github.com/myon-bioinformatics/PythDRagoras.git
cd PythDRagoras
uv sync
uv run python pythdragoras.py --help
```

Analysis inputs (cycle data `cycles/`, etc.) live in Aoi. Paths are resolved and passed in by the caller (Aoi's `aoi_paths.py`). The littlebit CLIs take `--cycle-dir`, or `--root` / `AOI_ROOT` and `AOI_RESEARCH_CYCLE`, and do not search for the layout. Actual command examples are in "Detailed spec and research usage" later on.

## Layout

| File | Contents |
|---|---|
| `pythdragoras.py` | CLI that verifies residuals, rank bias, and propositions from the season table (`apply_exclusions`, `cumulative_test`, `rank_test`, `summary_markdown`) |
| `propositions.py` | Propositions, counterexamples, objections (spec in [docs/propositions.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/propositions.md)) |
| `sets.py` | Names the conditions of registered propositions as sets, and judges ∩, ∪, ¬ expressions as propositions |
| `rule_search.py` | Exhaustively searches combinations of conditions ("or" of "and" groups) |
| `season_sensitivity.py` | Sensitivity search varying annual-stat metrics, weights, and boundaries (standard library only) |
| `score_sensitivity.py` | Sensitivity analysis varying per-game runs on one axis only (incremental updates) |
| `littlebit_approx.py`, `littlebit_prune.py`, `littlebit_holdout.py` | Search that turns latent patterns of a candidate 0/1 matrix into candidate conditions, and its verification (experimental) |
| `littlebit_survey.py` | Image region and centroid estimation without multiplication or floating point (experimental, single file) |
| `bitwise_surrogate.py` | Bitmask evaluator for candidate rules (`evaluate_fast`). Packs the conditions once and rechecks `rule_search` results and proposition antecedents with the same truth values (nulls stay undetermined). From a false-positive row it proposes a forbidden (NOT) condition as the next proposition to test; it never changes a verdict. The dropped repair and the measurements are in [docs/bitwise-surrogate-repair.md](docs/bitwise-surrogate-repair.md) |
| `registration_audit.py` | Audits provenance labels (pre-registered or posthoc) |
| `next_seeds.py` | Lays out judgements of the original proposition and its converse and shows seeds for next propositions (not a judgement) |

Main functions in `pythdragoras.py` and others:

| Function | Contents |
|---|---|
| `apply_exclusions` | Applies seasons excluded from analysis. Exclusions without a reason are rejected, and what was excluded and why is kept in the results |
| `cumulative_test` | Turns the cumulative deviation from expected winning percentage into a z value (both fixed and variable exponent) |
| `rank_test` | How unusual a bias toward low ranks is. Shows comparisons with the null hypothesis (each season independent and uniform) and with all teams' actual records side by side |
| `summary_markdown` | Verification record. Does not write causes. Lists the status of competing hypotheses |
| `propositions.py` | Propositions, counterexamples, objections (spec in [docs/propositions.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/propositions.md)) |
| `littlebit_approx.py` | Turns latent patterns of a candidate 0/1 matrix into candidate conditions (search; judgement is `rule_search.py`) |

## Tests

Cycle data (`cycles/`), `scripts/`, and SakAnalytics stay in Aoi. They are not copied; CI resolves Aoi's main at run time and checks it out. Locally, use main.

```bash
git clone https://github.com/myon-bioinformatics/Aoi.git .test-deps/aoi
git -C .test-deps/aoi checkout main
AOI_ROOT=$PWD/.test-deps/aoi PYTHONPATH=$PWD/.test-deps/aoi/sakanalytics \
  uv run pytest -q --junitxml=reports/junit.xml
```

CI (`.github/workflows/ci.yml`) does the same on Python 3.11–3.15. A ref can be given via `aoi_ref` in `workflow_dispatch`. The resolved SHA is left in the job summary.

- Tests receive Aoi only through `AOI_ROOT` and ask Aoi's `aoi_paths.py` for data locations. No `aoi_paths.py` is kept in this repository.
- If `AOI_ROOT` doesn't point to Aoi, pytest stops at the start and gives the reason.
- The only runtime dependencies are polars and numpy (ortools is not used).
- In addition, tests use Aoi at `AOI_ROOT`.
  - SakAnalytics: `tests/test_propositions.py:195` (`sakanalytics` on `PYTHONPATH`).
  - Aoi's `scripts/`: `tests/test_c001_replay_compare.py:7` (`compare_c001_replay`) and `tests/test_c001_exploratory_candidates.py:14` (`check_c001_exploratory_candidates.py`).
  - `tests/test_no_sibling_imports.py` confirms that no other Aoi sibling subprojects or `scripts/` are used (the three above are known exceptions).

## Relationship with Aoi

This repository was split out from the [Aoi](https://github.com/myon-bioinformatics/Aoi) subproject [`pythdragoras/`](https://github.com/myon-bioinformatics/Aoi/tree/main/pythdragoras).
The parent is Aoi (Aoi's parent is [myon-bioinformatics](https://github.com/myon-bioinformatics/myon-bioinformatics)). It is registered in Aoi's [`child-heads.json`](https://github.com/myon-bioinformatics/Aoi/blob/main/child-heads.json), and its tree equals Aoi's `pythdragoras/` at registration time (including `.github/`, `pyproject.toml`, `uv.lock`, `conftest.py`, `.gitignore`, and this README). It syncs with `git subtree` ("Syncing with Aoi" below; the full procedure is in Aoi's [docs/child-repositories.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/child-repositories.md)).
This README is the same file as Aoi's `pythdragoras/README.md`. Links to documents and data inside Aoi are written as GitHub URLs so they open from either place.

### Cycle paths

`aoi_paths.py` lives in only one place, Aoi's top level (not in this repository). `littlebit_approx.py` and `littlebit_holdout.py` do not guess the layout. The cycle folder is passed with `--cycle-dir`. When omitted, use `--root` or the `AOI_ROOT` environment variable plus the `AOI_RESEARCH_CYCLE` environment variable (`<root>/cycles/<name>`).

The SHA-256 of these 2 files is pinned in Aoi's `cycles/c001-chunichi/outputs/exploratory/littlebit-holdout/holdout.provenance.json`. The child's CI reads `source_sha256` and `source_divergence` from the checked-out Aoi as-is. On pull requests a source byte difference is only a warning; on push to main and on schedule it fails. It also fails when the flag is missing. The evidence's `evidence_sha256`, inputs, and `holdout.json` / `holdout.md` are not changed.

### Syncing with Aoi

This repository is a subtree of Aoi's `pythdragoras/`. No files are excluded, and the tree of Aoi's `pythdragoras/` equals this repository's tree. Inside Aoi, `pythdragoras/.github/` does not run (GitHub reads only the repository root's `.github/`), and `pythdragoras/conftest.py` checks `AOI_ROOT` the same way under Aoi's pytest.

- The history was split out from `pythdragoras/` in Aoi `85243d6d1285e3bf4667e1447308543c7d648ab7` (main, the merge commit of #51) with `git filter-branch --subdirectory-filter pythdragoras` (head `12522c890e64313005e2bb3cad07d90365d4cf60`, tree `873bd0d947a99da043e314cb11762189540c6b98`). This is because `git subtree split --prefix=pythdragoras` in git 2.47.3 stopped at 506 of Aoi's 585 commits and returned an old head from 10/4 ([gh_identity#74](https://github.com/myon-bioinformatics/gh_identity/issues/74)).
- This history is not connected to Aoi's history, so a join is added to Aoi once (Aoi PR #52). A squash commit with this repository's main commit in `git-subtree-split:` (tree equal to this repository) becomes the second parent of a merge that doesn't change Aoi's tree. After the join, `git subtree` starts from it, so it doesn't walk the range where #74 stops.
- The join must remain as the second parent. Aoi PRs containing a join are merged with a merge commit, not squashed (squashing drops the second parent and `git subtree pull` and `split` stop).
- The `git-subtree-split:` commit must be on this repository's main. When squash-merging a PR here changes main's sha, a join with the same tree is re-added in Aoi.

Child → Aoi (on a branch made from Aoi's main):

```bash
git subtree pull --prefix=pythdragoras https://github.com/myon-bioinformatics/PythDRagoras.git main --squash
test "$(git rev-parse HEAD:pythdragoras)" = "$(git rev-parse FETCH_HEAD^{tree})"   # when Aoi has no unsynced changes
```

Aoi → child (on Aoi's main):

```bash
git fetch https://github.com/myon-bioinformatics/PythDRagoras.git main   # bring the split's starting commit local
git subtree split --prefix=pythdragoras -b sync/pythdragoras
test "$(git rev-parse sync/pythdragoras^{tree})" = "$(git rev-parse HEAD:pythdragoras)"
git push https://github.com/myon-bioinformatics/PythDRagoras.git sync/pythdragoras:refs/heads/<new-branch>
```

- Open a PR to this repository from the pushed branch. After merging, do the "child → Aoi" pull in Aoi to reattach Aoi's starting point to this repository's main (squash merges don't keep the split commits on main).
- Until gh_identity#74 is fixed, gh_identity's `subtree-sync --write` is not used against Aoi. Use only the plan without `--write`, and check tree equality with the `test` above.
- Aoi's `scripts/check_child_heads.py --fail-on-diff` compares the tree of the sha registered in Aoi's `child-heads.json` with Aoi's `pythdragoras/`.

## Related subprojects

Each subproject of the parent [Aoi](https://github.com/myon-bioinformatics/Aoi) is its own repository. They are listed with their location in Aoi (subtree). The list and registration details are in Aoi's [docs/child-repositories.md](https://github.com/myon-bioinformatics/Aoi/blob/main/docs/child-repositories.md).

| Repository | Location in Aoi | Role |
|---|---|---|
| [BlueProbe](https://github.com/myon-bioinformatics/BlueProbe) | [`blueprobe/`](https://github.com/myon-bioinformatics/Aoi/tree/main/blueprobe) | Observation and extraction (reads saved HTML/CSS/DOM and records what is there) |
| [QueRyu](https://github.com/myon-bioinformatics/QueRyu) | [`queryu/`](https://github.com/myon-bioinformatics/Aoi/tree/main/queryu) | Fetching and building observation datasets (caches with provenance) |
| [SakAnalytics](https://github.com/myon-bioinformatics/SakAnalytics) | [`sakanalytics/`](https://github.com/myon-bioinformatics/Aoi/tree/main/sakanalytics) | Metrics and statistics from observation data (team × season metrics, outlook for remaining games) |
| **PythDRagoras** (this repository) | [`pythdragoras/`](https://github.com/myon-bioinformatics/Aoi/tree/main/pythdragoras) | Verifying explanations (propositions, counterexamples, objections) |
| [NagoyAction](https://github.com/myon-bioinformatics/NagoyAction) | [`nagoyaction/`](https://github.com/myon-bioinformatics/Aoi/tree/main/nagoyaction) | Pipeline execution and automation (the same definition locally and on GitHub Actions) |
| [DRAgoWing](https://github.com/myon-bioinformatics/DRAgoWing) | [`dragowing/`](https://github.com/myon-bioinformatics/Aoi/tree/main/dragowing) | Figures and reports (Plotly.js HTML, the four-forms table) |

## License

MIT License ([LICENSE](LICENSE)). Same contents as Aoi's `LICENSE`.

---

# Detailed spec and research usage (NPB)

The following covers usage in Aoi's research cycle (Chunichi and NPB data) and the detailed spec of each feature. Paths in the command examples (`pythdragoras/…`, `cycles/…`) are locations from Aoi's root.

## Propositions and "objection" (異議あり)

Explanations are written as "if A then B" propositions and judged in four forms: the original, contrapositive, converse, and inverse.
Judgement criteria are declared in words before seeing results (always (必ず) / almost (ほとんど) 0.90 / mostly (概ね) 0.75 / often (多くの場合) 0.50).

```text
Judgement (the first that applies, from the top)
  n < min_n                                    → suspended (判断保留)
  always and no counterexample / some          → supported (支持) / rejected (棄却)
  interval lower bound ≥ criterion             → supported
  interval straddles the criterion             → suspended
  interval upper bound < criterion, related    → revise (修正; the claim is too strong)
  interval upper bound < criterion, unrelated  → rejected
```

If there is even one counterexample, the counterexamples are listed as an **objection** (異議あり) even when the verdict is "supported". The counterexamples inside a supported explanation are exactly where the explanation falls short.
Counterexamples in excluded seasons are always shown as "counterexamples under exclusion", even though they are left out of the statistics.

External assertions (articles, reports, other AIs) are registered as claims, not observations, and checked for reproducibility in the pipeline.

## Turning latent patterns into candidate conditions

`littlebit_approx.py` turns human-written candidate condition masks into a unit × condition 0/1 matrix and decomposes it with the LittleBit formula (`littlebit_prune.littlebit`). Each latent axis becomes one condition, "does this unit follow the pattern", and is passed to `search`. Counting of holds and counterexamples remains `search`'s bit operations. The target column is not passed to the factors. Added conditions carry `source=littlebit-approx`, rank, seed, the original conditions and weights, and `posthoc=true`. The registration category follows `registration_audit.classify`, and this construction is posthoc.

The procedure and the rules for dropped axes and residual clues are in [docs/littlebit-approx.md](docs/littlebit-approx.md). The c001 rankings are under `cycles/c001-chunichi/outputs/exploratory/littlebit-approx/`. Confirmation records are in `confirmation.md`.

The null hypothesis treats each season as independent, so it ignores team strength carried over from the previous year and overestimates how unusual things are.
For that reason a comparison with all teams over the same period (an empirical baseline) is always shown alongside.

```bash
uv run python pythdragoras/pythdragoras.py --season cycles/c001-chunichi/outputs/season.jsonl \
  --config cycles/c001-chunichi/analysis.toml --hypotheses cycles/c001-chunichi/hypotheses.toml \
  --propositions cycles/c001-chunichi/propositions.toml --claims cycles/c001-chunichi/claims.toml \
  --outdir cycles/c001-chunichi/outputs
```

## Incremental search starting from seasons (season_sensitivity.py)

First vary the annual-stat metrics, weights, and boundaries, and examine changes and stability.
Once promising candidates or years where the explanation breaks are visible, move on to the per-game level in the next section.
This CLI runs on the standard library alone and does not need SciPy.

```bash
python -S pythdragoras/season_sensitivity.py \
  --season cycles/c001-chunichi/outputs/season.jsonl \
  --additional-season cycles/c001-chunichi/references/season-2026-central-summary.jsonl \
  --config cycles/c001-chunichi/season_sensitivity.toml \
  --out cycles/c001-chunichi/outputs/season_sensitivity.json \
  --markdown cycles/c001-chunichi/outputs/season_sensitivity.md
```

It sets `x + weight × y >= threshold` and computes sensitivity curves with one side fixed.
It outputs conditional probabilities, MCC, the difference from the baseline, finite-difference slopes, changes, extrema on the grid, the interval of the maximum, and trapezoidal area.
When moving the threshold, only team × years that cross the boundary update the confusion-matrix counts.
This is not treated as estimating derivatives or extrema of a smooth function. Formulas and candidate conditions are saved in TOML.

Because the same team appears across years, significance tests for independent samples are not done.
It keeps a check that reselects after leaving out all teams of one year, and a match with 2026 kept out of selection.
Missing columns, verification years, and team × years where metrics are missing are also listed, and Chunichi's year-by-year trend does not connect across missing years.
Shortfalls from the expected 6 teams, duplicate team × years, and undefined handling of tied ranks are errors.
Candidates are exploratory associations; causes, CS qualification probability, and support for propositions are not judged automatically.

The first results, definitions, and reading are in [the season-level search record](https://github.com/myon-bioinformatics/Aoi/blob/main/cycles/c001-chunichi/research/R27-season-sensitivity.md),
and the per-year table in [the measured results](https://github.com/myon-bioinformatics/Aoi/blob/main/cycles/c001-chunichi/outputs/season_sensitivity.md).
2026 is a retrospective match that includes known results, not fully blind.
After 2026 is merged into the regular season table, the auxiliary input is removed.

## Next stage: changing per-game runs (score_sensitivity.py)

Borrowing NNUE's idea of "update only what changed", it subtracts a game's contribution and adds the new one.
No neural net or trained weights are used. The target is the same raw game records as SakAnalytics.
RF, RA, W, L, and T are updated incrementally and only derived values are re-evaluated, so the update cost of each change does not depend on the number of games.
The whole search is proportional to target games × steps × 2 axes.

```bash
uv run python pythdragoras/score_sensitivity.py \
  --games data/observations/npb_calendar/games.jsonl \
  --config cycles/c001-chunichi/analysis.toml \
  --max-step 3 --out cycles/c001-chunichi/outputs/score_sensitivity.json
```

`--team` selects any configured team. The default is `[focus].team`.

| Output | Meaning |
|---|---|
| `baseline` | Runs scored/allowed, wins/losses, winning percentage, and expected winning percentage and expected wins with exponent 1.83, before changes |
| `curve` | Results of adding 0–3 runs scored to each game, or removing 0–3 runs allowed. Always only one axis changed |
| `delta` | Difference from the same season's baseline |
| `timeline` | Progress applying one game at a time in date/key order, the difference of that change, and the difference from baseline |
| `changed_runs` | Total runs actually changed. Runs allowed stop at 0 |
| `candidates` | Proposition candidates, in descending order of increase in games over .500 per changed run, from 1-run improvements to each game in search years |

The default candidate condition is "3 or more search years, and games over .500 increase in every search year".
It is set with `--min-seasons` and kept in the results along with the selection condition. The selection condition is not changed after seeing the 2- and 3-run results.
`[[exclude]]` is applied with reasons, and `[confirm].seasons` produce curves but are not used for candidate selection or ranking.
Candidates are not registered in `propositions.toml` automatically. Verification in independent years and human refinement of the proposition are needed.

For example, a 0–1 loss becomes a tie with RF +1 and a win with +2.
The relation between added runs and wins has steps, so both the smooth change in expected winning percentage and the actual change in wins/losses are kept.
If every game is a tie, winning percentage is undefined (null); if both teams' total runs are 0, expected winning percentage is null too.

This is a **sensitivity analysis that changes final scores**, not the causal effect of tactics or player ability that would actually improve scoring.
The opponent's runs in the same game are paired, but the standings of all teams are not recomputed.
Innings, extra innings, walk-offs, and subsequent managerial decisions are not reproduced, and A-class probability is not estimated.
`timeline` is the order changes are applied, not a forecast of future performance or actual mid-season records.
The input range is exactly the raw records the user prepared. Season completeness is not guaranteed automatically, so when including an unfinished year, explicitly exclude it or set it as a verification year.

Verification: for 80 random games, **each edit point** of both RF/RA axes and 4 steps is checked against an independent full recomputation.
Ties, 0 runs allowed, team orientation, duplicate/invalid raw records, exclusions, selection leakage from verification years, and the CLI are also tested.

## LittleBit 3-way survey (experimental, `littlebit_survey.py`)

An experiment measuring how well image regions and centroids can be estimated without multiplication or floating point. It is a single standard-library-only file and does not change existing processing such as `pythdragoras.py`.

- **Way1**: applies an 8-point FWHT (additions and subtractions only) to rows and columns of 8x8 patches (overlapping, shifted by 4 pixels). It measures the L1 energy of the AC components and turns the difference from the patch mean (DC >> 6) into ternary spins (+1 / 0 / -1, threshold 24). Patches with energy under 512, or with 0 valid spins, are skipped.
- **Way2**: perturbs valid-spin pixels with 8-bit rotation and brightness inversion, and makes pixels whose patch energy change ΔE (sum of 2) is 64 or more into candidates. Links are made when a same-sign candidate is a right/down neighbor at step 1/2/4. Candidates with fewer than 2 links are dropped as isolated noise.
- **Way3**: reconstructs regions from Way2's link history alone (without looking at the image). It scans rows and columns, entering the inside at + nodes and leaving at - nodes. Pixels inside on both row and column scans form the region, and the centroid is returned as an integer (numerator, denominator).
- **Core promise**: Way1–3 core functions (`CORE_FUNCTIONS`) use no `*`, `/`, `**`, or floating point. Indices are built by adding row offsets. Tests check the AST. Ratios become decimals only for display (`--report`). Random seeds are fixed.

```bash
python littlebit_survey.py --report
```

### Pass lines and their basis

Measured on 6 kinds of 64x64 synthetic images. Tests use fixed seeds (11–16), and pass lines were set with margin from the minimums measured over seeds 1–30.

| Scenario | Contents | Min IoU over seeds 1–30 | Max centroid error (px) | Pass line (IoU / skip rate / centroid error) |
|---|---|---:|---:|---|
| Dense Object | Filled ellipse | 0.986 | 0.18 | ≥ 0.95 / ≥ 0.70 / ≤ 0.5 |
| Thin Skeleton | 3 lines, 2 px wide | 0.935 | 0.43 | ≥ 0.90 / ≥ 0.65 / ≤ 1.0 |
| Hollow Ring | Hollow ring (hole left empty) | 0.996 | 0.03 | ≥ 0.95 / ≥ 0.35 / ≤ 0.5 |
| High Noise Dark | Dark background, ±8 jitter, 3% white-dot noise | 0.961 | 0.60 | ≥ 0.90 / — / ≤ 1.0 |
| Low Contrast | Ellipse with contrast 60 | 0.996 | 0.04 | ≥ 0.95 / ≥ 0.70 / ≤ 0.5 |
| Multi Blob | 2 separate ellipses | 0.986 | 0.36 | ≥ 0.95 / ≥ 0.65 / ≤ 1.0 |

- The skip rate is higher the more uniform a patch's interior. Rings have long edges, so it is low (0.44), and noisy images can hardly skip because of the white dots (0.06–0.18). So High Noise Dark has no skip-rate pass line.
- On High Noise Dark, removing the isolated-noise drop (2+ links) lowers IoU from 0.987 to 0.840. Tests also confirm that IoU is at least 0.05 higher with the drop.
- Shifting the reconstructed region by (3, 2) lowers IoU in every scenario (lines drop to around 0.07, filled shapes to about 0.6–0.74).
- All 6 scenarios are measurements on fixed synthetic images; performance on real images is not guaranteed.

## Node entry (Node 24+)

- `node littlebit_survey.cjs <args>` runs `littlebit_survey.py` (interpreter override: `LITTLEBIT_SURVEY_PYTHON`).
- `node season_sensitivity.cjs <args>` runs `season_sensitivity.py` (interpreter override: `SEASON_SENSITIVITY_PYTHON`).
- `node score_sensitivity.cjs <args>` runs `score_sensitivity.py` (interpreter override: `SCORE_SENSITIVITY_PYTHON`).

Each entry uses the vendored shared launcher `vendor/py_launch.cjs` (pinned in `vendor.lock.json`).
Arguments, stdin/stdout/stderr and the exit code are passed through unchanged; Python runs with `-S`, including interpreter discovery.
The launcher is copied from the merged [Aoi #117](https://github.com/myon-bioinformatics/Aoi/pull/117)
commit `bb5eb56f216eeff9135b7aa178197a79df152396`, not directly from Aoi's parent.
SIGINT and SIGTERM continue to be forwarded while the Python child is alive, even after an earlier signal was handled or ignored.
Launcher failures (no Python 3.9+, spawn error, signal) exit 2. Set `PY_LAUNCH_ERROR_FORMAT=json` for a single structured error on stderr; signal failures report the signal that actually terminated the child.

Regression tests cover every thin entry with acknowledged repeated signals, text/JSON errors, and a hostile `sitecustomize` hook. They also test `noSite` and `isolated` discovery independently without changing the entries' `noSite: true` configuration or the Node 24+ floor.
