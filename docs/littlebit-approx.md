# littlebit-approx

Builds latent patterns from candidate condition masks and adds them to PythDRagoras's candidate conditions. The shared identifier is `littlebit-approx`.

Holds and failures are judged by the bit operations of [core/rule_search.py](../rule_search.py). Factors are computed with `littlebit` in [core/littlebit_prune.py](../littlebit_prune.py) (low rank, binarization by sign, row, column, and latent scales, residual stages). The official LittleBit kernel is not used.

## Matrix

`indicator_matrix` is a unit × candidate condition 0/1 matrix. Column j is the same as the mask `search` already builds (`_mask`). The target column (e.g. B-class `upper_half`) is not put in this matrix. `derive` does not receive the target.

Aligning rows with `search` (exclusions, dropping rows whose target or candidates are empty) is done by the caller passing the same table as to `search`.

## Whether a unit follows a pattern

For each stage of `littlebit` and each latent axis k, let the unit-side factor be `F[:, k]`. By `littlebit`'s definition, this is the product of row scale h, sign binary Us, and latent scale l. The column side is sign binary Vs and column scale g.

The axis sign s is +1 or -1.

- If the sum of `np.sign(F[:, k])` is positive, s = +1
- If the sum is negative, s = -1
- Only when the sum is 0 (equal numbers of positive and negative) is the seed used. If the first bit of random numbers from the seed is 0, s is the orientation that makes the first nonzero component positive; if 1, negative

Unit i follows pattern k when `s * F[i, k] > 0`. Exactly 0 does not follow. This is the sign of the factor, not a procedure that cuts boundaries such as .040 or .060 to fit the data.

The weight of column j is `w_j = s * g_j * Vs[j, k]`. The reconstruction's contribution from that axis returns to the product of `(s * F[:, k])` and `w` (multiplying s on both sides gives s² = 1, the original product). Weights are kept in candidates as the arithmetic link to the bundled original conditions.

The column scale g is one per condition within that stage. Differences between latent axes show up in the signs `Vs[j, k]` and the unit-side `F[:, k]`. Within the same stage, the absolute values of weights match across axes, and the sign pattern separates the axes.

Defaults are rank=8, stages=2, seed=0, and `littlebit`'s default of 6 alternating least squares iterations. rank and stages are the same as the pruning command's defaults, not values chosen after seeing the search results' MCC. Except for ties, the seed does not change axes and is only recorded.

## Axes not made into conditions

Human-written candidates are never removed. Of the requested `rank * stages` axes, the following are not made into new conditions.

| Reason | Contents |
|---|---|
| `zero-factor` | The column of F is all 0 |
| `constant-mask` | No units follow, or all do |
| `duplicate-mask` | The same 0/1 as a pattern already added |
| `rank-shortfall` | SVD has fewer components than rank |
| `no-units` / `no-candidates` | The matrix is empty |

The sum of added and dropped equals the requested count. The settings (rank, stages, seed) and these two counts are kept in the comparison record.

## Provenance

Each derived candidate has the following.

- `source` = `littlebit-approx`
- `rank`, `seed`, `stage`, `pattern`
- `bundle`: the original conditions (`col`, `op`, `value`) and `weight`
- `posthoc` = true, `exploratory` = true
- `registration`: the result of `classify` in [core/registration_audit.py](../registration_audit.py)

This construction produces conditions after seeing results, so `classify` returns posthoc. `exploratory` marks it as exploration; it does not add a further value to posthoc / preregistered / unknown. Making it pre-registered requires evidence recorded before evaluation, as in [registration-metadata-audit.md](../../docs/registration-metadata-audit.md). No such evidence is attached here.

`search` builds masks from `col`, `op`, and `value` only, and its sort key looks only at those three. The records above don't change the counts of holds and counterexamples.

## Residuals

Residual energy is the squared difference between the condition matrix and the reconstruction, summed per unit. Units above the mean over all units become clues for finding counterexamples (`lead`). Exactly the mean is not a clue. It does not split by the truth of the target column.

A clue is a record that "the factors couldn't fully reconstruct the candidate pattern for this team-year". The list of units decided to be counterexamples is kept separately by the judgement in [propositions.md](../../docs/propositions.md).

## c001 rankings

For c001, the top results from passing only existing candidates to `search` and from passing them with the derived candidates are laid side by side and written to `cycles/c001-chunichi/outputs/exploratory/littlebit-approx/`. Observation and interpretation are in separate sections of [comparison.md](../../cycles/c001-chunichi/outputs/exploratory/littlebit-approx/comparison.md). The full machine-readable version is `comparison.json` in the same directory.

Confirmation records are in [confirmation.md](../../cycles/c001-chunichi/outputs/confirmation.md). The candidate file is [rule_candidates.toml](../../cycles/c001-chunichi/rule_candidates.toml). Regenerate with:

```bash
uv run python pythdragoras/core/littlebit_approx.py --cycle-dir cycles/c001-chunichi
```

## Fit/transform split by year

`derive`'s `loading_units` is the set of rows of the factorized table, not a predictor applied to unused years.
Column weights `bundle` alone don't determine `h` and `Us` for unknown rows, i.e. the sign of `F`.
Redoing SVD/ALS including unknown rows also moves the learned axes and conditions, so it isn't an independent evaluation.

[core/littlebit_holdout.py](../littlebit_holdout.py) implements **a different candidate generation method** that avoids this constraint.

1. Pass only the training years' candidate condition matrix to `littlebit`, and fix the per-stage column factors `W = Vs * g`.
2. At each stage, `Z = R @ pinv(W.T)`, and the next stage's input is `R - Z @ W.T`. The pseudo-inverse cutoff is `rcond=1e-12`.
3. Choose axis orientation from the training rows' `Z` only, and use `s * Z[:, k] > 1e-10` as the condition. Constant and duplicate exclusion is also done on the training side only.
4. Apply **the same transform to the training side too**, then choose formulas by the existing MCC and tie order; in evaluation years, factors, orientations, and formulas are not moved.

This projection does not reproduce the sign constraint of the original `F = h * Us * l`. So the column names are
`littlebit_project_s*_k*`, not to be confused with replications of the original `littlebit_s0_k0/k4`.
The saved encoder/decoder and formulas can be reloaded from JSON, and at evaluation time they don't depend on the target column, team-year IDs, or other evaluation rows.
Passing an old `loading_units` to `transform` is rejected.

```bash
uv run python pythdragoras/core/littlebit_holdout.py --cycle-dir cycles/c001-chunichi
```

The default block comparison is training 2012–2022 and evaluation 2023–2025. The auxiliary walk-forward starts from the first 5 years,
evaluates the next year, then extends the training period. It keeps the existing setting's 2020 exclusion, and saves all settings, input SHA-256, formulas, models, and
TP/FP/FN/TN ID sets in [holdout.json](../../cycles/c001-chunichi/outputs/exploratory/littlebit-holdout/holdout.json).
The comparison table is in [holdout.md](../../cycles/c001-chunichi/outputs/exploratory/littlebit-holdout/holdout.md).

The comparisons are the best formula from the training side's existing candidates (baseline), the best formula including added candidates (augmented),
and the best formula containing at least one derived column (derived). derived is searched exhaustively, not limited to formulas in the top 20.
Folds where no derived candidate can be made are recorded as no generation; the totals are not computed by silently dropping that fold.
When the MCC denominator is 0, the existing scoring convention gives 0, and the report also saves an undefined marker.

Because it re-splits years already viewed during the investigation, this is a **backtest on past years** that avoids leakage between fit and evaluation.
It is not independent confirmation on fully unseen data including candidate thresholds and methods, nor evidence of pre-registration.
The saved features are end-of-season aggregates and do not indicate prediction accuracy for future seasons or before opening day.
The existing search's best MCC, propositions, pass conditions, and `confirmation.md` are not changed.
