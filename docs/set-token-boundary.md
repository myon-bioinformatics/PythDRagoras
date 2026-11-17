# Cache the set-expression token boundary

`sets.tokens` computes the length after stripping trailing whitespace once per
call instead of once per token. Token values, whitespace handling and invalid
suffix diagnostics remain unchanged. No dependency, public API or CI changes.

Validation (in an Aoi checkout whose `pythdragoras/` is this tree; see the README): `uv run pytest -xq -c pyproject.toml pythdragoras/tests/test_sets.py pythdragoras/tests/test_set_token_boundary.py`
passed 16 tests on Linux/CPython 3.14.9. Added cases cover Unicode/trailing
whitespace, empty strings, 20,001 tokens and invalid suffix diagnostics.
The token-boundary cases now live in `tests/test_sets.py` (`TokenBoundaryTests`); the file above was merged into it.

`python -S -m scripts.benchmark_set_tokens` compares tokenizer functions extracted
with AST from the pinned baseline and current source, using identical regex and
exception namespaces. It deliberately excludes Polars import, parsing, evaluation,
and report generation. Nine alternating samples; CPython 3.12.14/Linux.

| Synthetic repeated terms | Baseline median | Current median |
| --- | ---: | ---: |
| 10 | 4,797 ns | 4,126 ns |
| 10,000 | 20,969,582 ns | 4,117,720 ns |

The long-input reduction is about 80%. Raw samples are in
`set-token-measurement.json`. Equality is checked explicitly, including under
optimized Python. These are isolated synthetic timings; no build-time, package
size or end-to-end speedup is claimed. Historical baseline availability is
required only for the manual benchmark, not the regression tests.
