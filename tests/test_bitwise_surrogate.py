"""The bitmask evaluator matches the existing one, and a NOT suggestion does not change the verdict.

Each randomized run draws a seed and logs it. Fidelities and microsecond
bands from the external proposal are not pass conditions.
"""

import ast, random, secrets, subprocess, sys, warnings
from pathlib import Path

import polars as pl
import pytest

import bitwise_surrogate as bs
import rule_search

ROOT = Path(__file__).resolve().parents[1]


def _seed():
    seed = secrets.randbits(63)
    warnings.warn("bitwise_surrogate seed=%d" % seed, UserWarning, stacklevel=2)
    return seed


def test_evaluate_fast_matches_the_set_reference():
    seed = _seed()
    rng = random.Random(seed)
    for _ in range(40):
        rule = bs.BitwiseSymbolicSurrogate(
            rng.randrange(1 << 12),
            tuple(rng.randrange(1, 1 << 12) for _ in range(rng.randrange(4))),
            rng.randrange(1 << 12),
        )
        for _ in range(30):
            x = rng.randrange(1 << 12)
            assert rule.evaluate_fast(x) is rule.evaluate_reference(x)


def test_rejects_bools_empty_clauses_and_negative_masks():
    with pytest.raises(ValueError):
        bs.BitwiseSymbolicSurrogate(True, (), 0)
    with pytest.raises(ValueError):
        bs.BitwiseSymbolicSurrogate(0, (0,), 0)
    with pytest.raises(ValueError):
        bs.BitwiseSymbolicSurrogate(-1, (), 0)
    rule = bs.BitwiseSymbolicSurrogate()
    with pytest.raises(ValueError):
        rule.evaluate_fast(-1)


def test_copy_does_not_alias():
    rule = bs.BitwiseSymbolicSurrogate(1, (2,), 4)
    copied = rule.copy()
    copied.m_and = 0
    copied.or_clauses.append(8)
    copied.m_not = 0
    assert rule.describe() == {"m_and_bits": [0], "or_clauses": [[1]], "m_not_bits": [2]}
    assert rule.evaluate_fast(1 | 2) is True
    assert copied.evaluate_fast(1 | 2) is False


def test_format_current_rule_names_and_or_and_not():
    rule = bs.BitwiseSymbolicSurrogate(bs.mask_of((0, 2)), (bs.mask_of((1, 3)),), bs.mask_of((4,)))
    assert bs.format_current_rule(rule) == "bit 0 AND bit 2 AND (bit 1 OR bit 3) AND NOT (bit 4)"
    names = ["a < 1", "b < 2", "c < 3", "d < 4", "e < 5"]
    assert bs.format_current_rule(rule, names) == "a < 1 AND c < 3 AND (b < 2 OR d < 4) AND NOT (e < 5)"
    assert bs.format_current_rule(bs.BitwiseSymbolicSurrogate()) == "(every row)"


def test_compile_bit_dnf_covers_the_fragment_and_rejects_a_general_dnf():
    assert bs.compile_bit_dnf([[0, 1]]).describe()["m_and_bits"] == [0, 1]
    factored = bs.compile_bit_dnf([[0, 1], [0, 2]])
    assert factored.describe() == {"m_and_bits": [0], "or_clauses": [[1, 2]], "m_not_bits": []}
    assert bs.compile_bit_dnf([[0], [0, 1]]).describe()["m_and_bits"] == [0]
    assert bs.compile_bit_dnf([[0], [1]]).describe()["or_clauses"] == [[0, 1]]
    assert bs.compile_bit_dnf([[0, 1], [2, 3]]) is None
    distilled = bs.distill_literals([(1, True), (4, False)])
    assert distilled.evaluate_fast(bs.mask_of((1,))) is True
    assert distilled.evaluate_fast(bs.mask_of((1, 4))) is False


def test_nulls_match_the_proposition_antecedent_and_a_not_literal():
    frame = pl.DataFrame({
        "a": [1, None, 1, 0, 1],
        "b": [1, 1, None, 0, 0],
    })
    first = {"col": "a", "op": ">=", "value": 1}
    second = {"col": "b", "op": ">=", "value": 1}
    conditions = [first, second]
    for groups in ([[first, second]], [[first], [second]], [[first]], [[second]], [[]]):
        got = bs.evaluate_antecedents(frame, [groups], conditions)[0]
        assert got == bs.polars_antecedent_flags(frame, groups)
    assert None in bs.evaluate_antecedents(frame, [[[first, second]]], conditions)[0]
    packed = bs.pack_feature_rows(frame, conditions)
    rule = bs.distill_literals([(0, True), (1, False)])
    expected = frame.select(x=(pl.col("a") >= 1) & ~(pl.col("b") >= 1))["x"].to_list()
    assert bs.antecedent_flags(packed, rule) == [None if flag is None else bool(flag) for flag in expected]
    # The forbidden bit being known-true is false even when the required bit is null.
    lone = pl.DataFrame({"a": [None], "b": [1]})
    lone_packed = bs.pack_feature_rows(lone, conditions)
    assert bs.antecedent_flags(lone_packed, rule) == [False]
    wide = pl.DataFrame({
        "a": [1, None, 0, 1],
        "b": [1, 1, 1, None],
        "c": [0, 1, None, 0],
        "d": [1, 0, 1, 1],
    })
    wide_conditions = [
        {"col": "a", "op": ">=", "value": 1},
        {"col": "b", "op": ">=", "value": 1},
        {"col": "c", "op": ">=", "value": 1},
        {"col": "d", "op": ">=", "value": 1},
    ]
    general = [
        [wide_conditions[0], wide_conditions[1]],
        [wide_conditions[2], wide_conditions[3]],
    ]
    assert bs.compile_search_rule(general, wide_conditions) is None
    assert bs.evaluate_antecedents(wide, [general], wide_conditions)[0] == bs.polars_antecedent_flags(wide, general)


def test_random_null_frames_match_the_proposition_antecedent():
    seed = _seed()
    rng = random.Random(seed)
    columns = ["a", "b", "c", "d"]
    conditions = [{"col": col, "op": ">=", "value": 1} for col in columns]
    for _ in range(25):
        height = rng.randrange(1, 12)
        frame = pl.DataFrame(
            {col: [rng.choice([0, 1, None]) for _ in range(height)] for col in columns},
            schema={col: pl.Int64 for col in columns},
        )
        groups = [
            rng.sample(conditions, rng.randrange(1, 4))
            for _ in range(rng.randrange(1, 4))
        ]
        got = bs.evaluate_antecedents(frame, [groups], conditions)[0]
        assert got == bs.polars_antecedent_flags(frame, groups), (seed, groups)


def test_compiled_rules_match_rule_search_and_the_proposition_antecedent():
    rows = [
        ("a", 2020, 40, 1, 2.0, True), ("b", 2020, 10, 3, -5.0, False), ("c", 2020, -20, 2, 4.0, True),
        ("d", 2020, -60, 5, 1.0, False), ("e", 2020, 5, 3, 0.5, True), ("f", 2020, -10, 6, -1.0, False),
        ("g", 2021, 30, 2, -6.0, False), ("h", 2021, -30, 5, 3.0, False), ("i", 2021, 60, 1, 0.0, True),
    ]
    frame = pl.DataFrame([
        {"team": team, "season": season, "rd": rd, "rank_ra": rank_ra, "wins_vs_pythag": wins, "upper_half": upper}
        for team, season, rd, rank_ra, wins, upper in rows
    ])
    target = [{"col": "upper_half", "op": "==", "value": False}]
    candidates = [
        {"col": "rd", "op": "<", "value": 0},
        {"col": "rank_ra", "op": ">=", "value": 4},
        {"col": "wins_vs_pythag", "op": "<", "value": -4},
    ]
    found = rule_search.search(frame, target, candidates, max_terms=2, max_conds=2, top=50)
    screened = bs.screen_search_rules(frame, found, candidates)
    assert any(flags is not None for flags in screened)
    outside = [
        [{"col": "rd", "op": "<", "value": 0}, {"col": "rank_ra", "op": ">=", "value": 4}],
        [{"col": "wins_vs_pythag", "op": "<", "value": -4}, {"col": "season", "op": "==", "value": 2020}],
    ]
    vocabulary = candidates + [{"col": "season", "op": "==", "value": 2020}]
    assert bs.compile_search_rule(outside, vocabulary) is None
    general = bs.evaluate_antecedents(frame, [outside], vocabulary)
    assert general == [bs.polars_antecedent_flags(frame, outside)]
    assert bs.format_search_rule(outside, vocabulary) == (
        "[rd < 0 AND rank_ra >= 4] OR [wins_vs_pythag < -4 AND season == 2020]"
    )
    for rule, flags in zip(found, screened):
        assert flags is not None
        assert flags == bs.polars_antecedent_flags(frame, rule["if_any"])
        counted = bs.counts(flags, bs.polars_antecedent_flags(frame, [target]))
        assert (counted["tp"], counted["fp"], counted["fn"], counted["tn"]) == (
            rule["tp"], rule["fp"], rule["fn"], rule["tn"])
        text = bs.format_search_rule(rule["if_any"], candidates)
        assert text
        compiled = bs.compile_search_rule(rule["if_any"], candidates)
        if compiled is not None:
            assert text == bs.format_current_rule(compiled, bs.condition_names(candidates))


def test_a_monotone_decoy_is_suggested_and_the_judged_rule_stays():
    seed = _seed()
    rng = random.Random(seed)
    required = (12, 19)
    banned = 25
    decoy = 30
    rule = bs.distill_literals([(bit, True) for bit in required] + [(banned, False)])
    original = rule.describe()

    def oracle(x):
        if (x & bs.mask_of(required)) != bs.mask_of(required):
            return False
        if x & bs.mask_of((banned, decoy)):
            return False
        return True

    noise = [bit for bit in rng.sample(range(40), 8) if bit not in required + (banned, decoy)]
    row = bs.mask_of(required + (decoy,) + tuple(noise))
    assert rule.evaluate_fast(row) is True
    assert oracle(row) is False
    names = ["bit %d" % bit for bit in range(40)]
    names[decoy] = "decoy feature"
    found = bs.suggest_not_proposition(rule, row, oracle, names)
    assert rule.describe() == original
    assert rule.evaluate_fast(row) is True
    assert found["supported"] is True
    assert found["bit"] == decoy
    assert found["suggestion"]["not_condition"] == "decoy feature"
    assert found["suggestion"]["text"].endswith("AND NOT (decoy feature)")
    for probe in found["probes"]:
        assert probe["oracle_true"] is (decoy in probe["left_bits"])
    suggested = bs.rule_from_describe(found["suggestion"]["rule"])
    for _ in range(80):
        x = rng.randrange(1 << 40)
        assert suggested.evaluate_fast(x) is oracle(x)
    assert rule.describe() == original


def test_a_false_positive_without_an_inhibitory_bit_is_not_applied():
    seed = _seed()
    rng = random.Random(seed)
    rule = bs.distill_literals([(0, True)])
    original = rule.describe()

    def oracle(x):
        return (x & bs.mask_of((0, 1))) == bs.mask_of((0, 1))

    extras = tuple(bit for bit in rng.sample(range(2, 30), 5))
    row = bs.mask_of((0,) + extras)
    assert rule.evaluate_fast(row) is True and oracle(row) is False
    found = bs.suggest_not_proposition(rule, row, oracle)
    assert found["supported"] is False
    assert found["suggestion"] is None
    assert found["reason"] == "bit_is_not_individually_inhibitory"
    assert rule.describe() == original
    assert rule.evaluate_fast(row) is True
    empty = bs.BitwiseSymbolicSurrogate()
    none = bs.suggest_not_proposition(empty, 0, lambda x: False)
    assert none["reason"] == "no_extra_bit" and none["suggestion"] is None
    assert empty.evaluate_fast(0) is True


def test_module_imports_under_python_s_and_stays_free_of_py310_syntax():
    completed = subprocess.run(
        [sys.executable, "-S", "-c", "import bitwise_surrogate"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    source = (ROOT / "bitwise_surrogate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Match) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Attribute) and node.attr == "bit_count" for node in ast.walk(tree))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top_level.append(node.module.split(".")[0])
    assert set(top_level) <= set(sys.stdlib_module_names)


def test_evaluate_fast_timing_is_a_positive_duration():
    seed = _seed()
    rng = random.Random(seed)
    rule = bs.BitwiseSymbolicSurrogate(
        rng.randrange(1 << 16),
        (rng.randrange(1, 1 << 16),),
        rng.randrange(1 << 16),
    )
    values = [rng.randrange(1 << 16) for _ in range(32)]

    def run():
        fast = rule.evaluate_fast
        for value in values:
            fast(value)

    timing = bs._time_call(run, min_seconds=0.05, min_calls=1)
    assert timing["calls"] >= 1
    assert timing["seconds"] > 0
    assert timing["seconds_per_call"] == timing["seconds"] / timing["calls"]


def test_real_rule_search_rules_match_and_the_recheck_is_timed(aoi_root):
    loaded = bs.load_search_frame(aoi_root)
    frame, target, candidates = loaded["frame"], loaded["target"], loaded["candidates"]
    found = rule_search.search(
        frame, target, candidates, loaded["max_terms"], loaded["max_conds"], 20,
    )
    screened = bs.screen_search_rules(frame, found, candidates)
    assert found and all(flags is not None for flags in screened)
    target_flags = bs.polars_antecedent_flags(frame, [target])
    for rule, flags in zip(found, screened):
        assert flags == bs.polars_antecedent_flags(frame, rule["if_any"])
        counted = bs.counts(flags, target_flags)
        assert counted["undetermined"] == 0
        assert (counted["tp"], counted["fp"], counted["fn"], counted["tn"]) == (
            rule["tp"], rule["fp"], rule["fn"], rule["tn"])
    singletons = [[[candidate]] for candidate in candidates]
    singleton_flags = bs.evaluate_antecedents(frame, singletons, candidates)
    assert all(flags is not None for flags in singleton_flags)
    for groups, flags in zip(singletons, singleton_flags):
        assert flags == bs.polars_antecedent_flags(frame, groups)
    report = bs.benchmark_rule_checks(
        frame, found, candidates, target,
        search_kwargs={"max_terms": loaded["max_terms"], "max_conds": loaded["max_conds"], "top": 20},
        min_seconds=0.0,
    )
    assert report["flags_equal"] is True
    assert report["counts_equal"] is True
    assert report["term_flags_equal"] is True
    assert report["conjunction_terms"] > report["searched_rules"]
    assert report["rows"] == frame.height
    for name in ("surrogate", "proposition_antecedent", "row_bitset", "rule_search",
                 "term_surrogate", "term_proposition", "term_row_bitset"):
        assert report[name]["calls"] >= 1
        assert report[name]["seconds"] > 0
