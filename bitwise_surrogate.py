"""Bitmask evaluator for candidate rules, and a NOT-literal suggestion.

A row is a Python int. Bit ``i`` is set when candidate condition ``i`` holds.
A rule in this fragment is

    evaluate_fast(X) = (X & M_AND) == M_AND
                       and (X & M_NOT) == 0
                       and every OR-clause mask shares a bit with X.

That is a conjunction of required conditions, forbidden conditions, and
positive disjunctions. It is not general DNF. ``[A AND B] OR [C AND D]``
does not fit; ``compile_search_rule`` returns None and the caller keeps
the existing evaluator.

``rule_search`` still enumerates OR-of-AND candidates with bitsets over
rows. That combination step is one integer OR per candidate. This module
does not replace it. It rechecks the rules that search (or a proposition)
already produced, when they fit the fragment: pack each candidate
condition once, then ``evaluate_fast``. The booleans match
``propositions._antecedent`` on the same rows, including nulls (null stays
undetermined, it is not turned into false). On the null-free frame
``rule_search`` actually scores, the true-positive and false-positive
counts match the search record.

``suggest_not_proposition`` does not edit the rule under test and does not
change its verdict. Given one row the rule accepts and an oracle that
rejects, it binary-searches extra bits. When one of those bits is
individually inhibitory, the function returns that NOT condition as a new
proposition to test. A bit that does not by itself remove the false
positive is not proposed.

The false-negative edit that relaxed a lost AND bit into an OR clause was
removed. Measured reasons are in ``docs/bitwise-surrogate-repair.md``.

Importing this module needs only the standard library and stays valid on
Python 3.9 (no ``match``, no ``int.bit_count``). Polars and the
proposition helpers are imported inside the functions that read a frame.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path


def _mask_value(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("%s must be a nonnegative int" % name)
    return value


def mask_of(bits):
    """Pack feature indexes into a bitmask. Order does not matter."""
    out = 0
    for bit in bits:
        _mask_value(bit, "bit")
        out |= 1 << bit
    return out


def iter_bits(mask):
    """Ascending feature indexes. Python ``int`` is unbounded."""
    _mask_value(mask, "mask")
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def bit_list(mask):
    return list(iter_bits(mask))


def _require_bool(value, name):
    if type(value) is not bool:
        raise ValueError("%s must be a bool" % name)
    return value


class BitwiseSymbolicSurrogate:
    """``M_AND``, OR-clause masks, and ``M_NOT`` as nonnegative Python ints."""

    def __init__(self, m_and=0, or_clauses=(), m_not=0):
        self.m_and = _mask_value(m_and, "m_and")
        self.m_not = _mask_value(m_not, "m_not")
        clauses = []
        for i, clause in enumerate(or_clauses):
            clause = _mask_value(clause, "or_clauses[%d]" % i)
            if clause == 0:
                raise ValueError("an OR clause mask must contain a feature")
            clauses.append(clause)
        self.or_clauses = clauses

    def copy(self):
        return BitwiseSymbolicSurrogate(self.m_and, tuple(self.or_clauses), self.m_not)

    def describe(self):
        return {
            "m_and_bits": bit_list(self.m_and),
            "or_clauses": [bit_list(clause) for clause in self.or_clauses],
            "m_not_bits": bit_list(self.m_not),
        }

    def evaluate_fast(self, x):
        """Two-valued predicate. ``X & ~M_AND`` is defined: Python ints are two's complement."""
        _mask_value(x, "x")
        if (x & self.m_and) != self.m_and:
            return False
        if x & self.m_not:
            return False
        for clause in self.or_clauses:
            if (x & clause) == 0:
                return False
        return True

    def evaluate_reference(self, x):
        """Same predicate, written with sets so tests can catch a bitmask slip."""
        _mask_value(x, "x")
        have = set(iter_bits(x))
        if not set(iter_bits(self.m_and)) <= have:
            return False
        if set(iter_bits(self.m_not)) & have:
            return False
        for clause in self.or_clauses:
            if not (set(iter_bits(clause)) & have):
                return False
        return True


def format_current_rule(rule, names=None):
    """Human-readable rule. ``names[i]`` is the text of bit ``i`` (``"col op value"``).

    Without names, bits are ``bit 0``. Connectives are AND, OR, and NOT.
    An empty rule holds on every row.
    """
    parts = []
    for bit in bit_list(rule.m_and):
        parts.append(_atom(bit, names))
    for clause in rule.or_clauses:
        bits = bit_list(clause)
        rendered = " OR ".join(_atom(bit, names) for bit in bits)
        parts.append(rendered if len(bits) == 1 else "(%s)" % rendered)
    for bit in bit_list(rule.m_not):
        parts.append("NOT (%s)" % _atom(bit, names))
    if not parts:
        return "(every row)"
    return " AND ".join(parts)


def _atom(bit, names):
    _mask_value(bit, "bit")
    if names is None:
        return "bit %d" % bit
    if bit >= len(names):
        raise ValueError("bit %d has no name" % bit)
    return names[bit]


def compile_bit_dnf(groups):
    """Compile an OR of positive ANDs into this fragment, or return None.

    Exact cases include one conjunction, a common prefix AND a single
    disjunction of the leftover literals, and a shorter group that subsumes
    the others. ``[A AND B] OR [C AND D]`` does not fit. Negated literals are
    not accepted here; use ``distill_literals`` for those.
    """
    if not groups or any(not group for group in groups):
        return None
    normalized = []
    for group in groups:
        seen = []
        for bit in group:
            _mask_value(bit, "bit")
            if bit not in seen:
                seen.append(bit)
        normalized.append(seen)
    common = set(normalized[0])
    for group in normalized[1:]:
        common &= set(group)
    remainders = [[bit for bit in group if bit not in common] for group in normalized]
    if any(len(rest) > 1 for rest in remainders):
        return None
    if any(not rest for rest in remainders):
        return BitwiseSymbolicSurrogate(mask_of(common), (), 0)
    if len(remainders) == 1:
        return BitwiseSymbolicSurrogate(mask_of(list(common) + remainders[0]), (), 0)
    return BitwiseSymbolicSurrogate(mask_of(common), (mask_of(rest[0] for rest in remainders),), 0)


def distill_literals(literals):
    """Build a conjunction. ``literals`` is ``(bit, polarity)`` with True meaning required."""
    m_and = 0
    m_not = 0
    for bit, positive in literals:
        _mask_value(bit, "bit")
        if type(positive) is not bool:
            raise ValueError("polarity must be a bool")
        if positive:
            m_and |= 1 << bit
        else:
            m_not |= 1 << bit
    return BitwiseSymbolicSurrogate(m_and, (), m_not)


def rule_from_describe(description):
    return BitwiseSymbolicSurrogate(
        mask_of(description["m_and_bits"]),
        tuple(mask_of(clause) for clause in description["or_clauses"]),
        mask_of(description["m_not_bits"]),
    )


def suggest_not_proposition(rule, x, oracle, names=None):
    """Propose one NOT condition as a new proposition. Do not edit ``rule``.

    ``rule`` must accept ``x`` and ``oracle(x)`` must be false. Extra bits
    (``X & ~M_AND``) are binary-searched from low index to high: if
    ``oracle(X ^ left_half)`` is true, the inhibitory bit is taken to lie in
    that half. The surviving bit is proposed only when flipping that bit
    alone makes the oracle true. Otherwise the suggestion is unsupported.
    The masks on ``rule`` are unchanged either way, so the verdict of the
    rule under test does not move.
    """
    _mask_value(x, "x")
    before = rule.describe()
    if not rule.evaluate_fast(x):
        return _suggestion(False, "rule_does_not_accept_row", [], None, before)
    if _require_bool(oracle(x), "oracle"):
        return _suggestion(False, "oracle_accepts_row", [], None, before)
    bits = bit_list(x & ~rule.m_and)
    probes = []
    if not bits:
        _assert_unchanged(rule, before)
        return _suggestion(False, "no_extra_bit", probes, None, before)
    lo, hi = 0, len(bits)
    while hi - lo > 1:
        mid = lo + (hi - lo) // 2
        left = bits[lo:mid]
        verdict = _require_bool(oracle(x ^ mask_of(left)), "oracle")
        probes.append({"left_bits": list(left), "oracle_true": verdict})
        if verdict:
            hi = mid
        else:
            lo = mid
    culprit = bits[lo]
    confirmed = _require_bool(oracle(x ^ (1 << culprit)), "oracle")
    probes.append({"left_bits": [culprit], "oracle_true": confirmed, "confirm_single_bit": True})
    _assert_unchanged(rule, before)
    if not confirmed:
        return _suggestion(False, "bit_is_not_individually_inhibitory", probes, culprit, before)
    suggested = rule.copy()
    suggested.m_not |= 1 << culprit
    _assert_unchanged(rule, before)
    return {
        "supported": True,
        "reason": "not_literal",
        "bit": culprit,
        "probes": probes,
        "judged_rule": before,
        "suggestion": {
            "text": format_current_rule(suggested, names),
            "not_condition": _atom(culprit, names),
            "rule": suggested.describe(),
        },
    }


def _suggestion(supported, reason, probes, bit, judged_rule):
    return {
        "supported": supported,
        "reason": reason,
        "bit": bit,
        "probes": probes,
        "judged_rule": judged_rule,
        "suggestion": None,
    }


def _assert_unchanged(rule, before):
    if rule.describe() != before:
        raise RuntimeError("suggest_not_proposition edited the rule under test")


def condition_key(cond):
    """Identity of a condition. Provenance keys are not part of it."""
    if not isinstance(cond, dict) or not {"col", "op", "value"} <= set(cond):
        raise ValueError("condition needs col, op, and value")
    value = cond["value"]
    if type(value) is bool:
        rendered = "true" if value else "false"
    elif type(value) is int:
        rendered = str(value)
    else:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return (cond["col"], cond["op"], rendered)


def condition_names(conditions):
    from propositions import text

    return [text([cond]) for cond in conditions]


def compile_search_rule(groups, conditions):
    """Compile ``rule_search`` groups (OR of ANDs of condition dicts), or return None."""
    if not isinstance(groups, list) or not groups:
        return None
    if groups == [[]]:
        return BitwiseSymbolicSurrogate()
    index = {}
    for bit, cond in enumerate(conditions):
        index.setdefault(condition_key(cond), bit)
    indexed = []
    for group in groups:
        if not group:
            return None
        bits = []
        for cond in group:
            bit = index.get(condition_key(cond))
            if bit is None:
                return None
            if bit not in bits:
                bits.append(bit)
        indexed.append(bits)
    return compile_bit_dnf(indexed)


class FeatureRows:
    """One feature mask per row. ``nulls`` marks conditions whose value is null."""

    def __init__(self, masks, nulls, names):
        if len(masks) != len(nulls):
            raise ValueError("masks and nulls differ in length")
        self.masks = list(masks)
        self.nulls = list(nulls)
        self.names = list(names)
        self.any_null = any(nulls)

    def __len__(self):
        return len(self.masks)


def pack_feature_rows(df, conditions):
    """Pack candidate-condition truth. Bit ``i`` is condition ``i``.

    One polars select. A null cell sets the null mask and does not set the
    feature bit. A true cell sets the feature bit.
    """
    from propositions import _cond

    names = condition_names(conditions)
    height = df.height
    if not conditions:
        return FeatureRows([0] * height, [0] * height, names)
    expressions = [_cond([cond], df.columns).alias("c%d" % i) for i, cond in enumerate(conditions)]
    table = df.select(expressions)
    columns = [table[name].to_list() for name in table.columns]
    masks = []
    nulls = []
    for row in range(height):
        mask = 0
        null = 0
        for bit, column in enumerate(columns):
            flag = column[row]
            if flag is None:
                null |= 1 << bit
                continue
            if type(flag) is not bool and type(flag).__name__ != "bool_":
                raise ValueError("condition flag must be bool or null, got %r" % (flag,))
            if flag:
                mask |= 1 << bit
        masks.append(mask)
        nulls.append(null)
    return FeatureRows(masks, nulls, names)


def evaluate_row(rule, x, nulls=0):
    """``evaluate_fast`` plus three-valued nulls.

    A known-false conjunct is false. A null conjunct with no known-false
    conjunct is undetermined (None). No nulls takes the ``evaluate_fast`` path.
    """
    _mask_value(x, "x")
    _mask_value(nulls, "nulls")
    if nulls == 0:
        return rule.evaluate_fast(x)
    and_false = bool(rule.m_and & ~x & ~nulls)
    not_false = bool(x & rule.m_not)
    or_false = False
    and_null = bool(rule.m_and & nulls)
    not_null = bool(rule.m_not & nulls)
    or_null = False
    for clause in rule.or_clauses:
        if x & clause:
            continue
        if clause & nulls:
            or_null = True
        else:
            or_false = True
    if and_false or not_false or or_false:
        return False
    if and_null or not_null or or_null:
        return None
    return True


def antecedent_flags(packed, rule):
    """Per-row antecedent. True, False, or None when a used condition is null."""
    if not packed.any_null:
        fast = rule.evaluate_fast
        return [fast(x) for x in packed.masks]
    return [evaluate_row(rule, x, nulls) for x, nulls in zip(packed.masks, packed.nulls)]


def evaluate_antecedents(df, group_lists, conditions):
    """Pack ``conditions`` once and evaluate each OR-of-AND.

    A rule that fits one surrogate is one ``evaluate_fast`` per row. A
    general OR of conjunctions, which does not fit that one mask, is the OR
    of one ``evaluate_fast`` per conjunction. Either way the flags match
    ``propositions._antecedent``. None is reserved for a group that is not a
    list of known conditions.
    """
    packed = pack_feature_rows(df, conditions)
    return [_flags_for_groups(packed, groups, conditions) for groups in group_lists]


def _flags_for_groups(packed, groups, conditions):
    compiled = compile_search_rule(groups, conditions)
    if compiled is not None:
        return antecedent_flags(packed, compiled)
    if not groups:
        return None
    columns = []
    for group in groups:
        compiled = compile_search_rule([group], conditions)
        if compiled is None:
            return None
        columns.append(antecedent_flags(packed, compiled))
    return _or_flags(columns)


def _or_flags(columns):
    """Three-valued OR. A true disjunct wins; otherwise a null stays null."""
    out = []
    width = len(columns[0])
    for index in range(width):
        decided = False
        for column in columns:
            if column[index]:
                out.append(True)
                decided = True
                break
        if decided:
            continue
        if any(column[index] is None for column in columns):
            out.append(None)
        else:
            out.append(False)
    return out


def format_search_rule(groups, conditions):
    """Readable text. One surrogate when the rule fits; otherwise OR of conjunctions."""
    names = condition_names(conditions)
    compiled = compile_search_rule(groups, conditions)
    if compiled is not None:
        return format_current_rule(compiled, names)
    parts = []
    for group in groups:
        compiled = compile_search_rule([group], conditions)
        if compiled is None:
            return None
        parts.append("[%s]" % format_current_rule(compiled, names))
    return " OR ".join(parts)


def screen_search_rules(df, rules, conditions):
    """Recheck ``rule_search.search`` results with ``evaluate_fast``.

    ``rules`` items carry ``if_any``. Each entry is the per-row antecedent.
    """
    return evaluate_antecedents(df, [rule["if_any"] for rule in rules], conditions)


def proposition_groups(proposition):
    """The antecedent as OR of ANDs, the shape ``_antecedent`` evaluates."""
    base = list(proposition.get("if") or [])
    any_groups = proposition.get("if_any") or []
    if any_groups:
        if base:
            return [base + list(group) for group in any_groups]
        return [list(group) for group in any_groups]
    if base:
        return [base]
    return [[]]


def evaluate_proposition_antecedent(df, proposition, conditions=None):
    """Flags for one proposition antecedent, or None when it does not fit.

    This does not run ``propositions.evaluate`` and does not emit a verdict.
    """
    groups = proposition_groups(proposition)
    if conditions is None:
        conditions = _vocabulary(groups)
    flags = evaluate_antecedents(df, [groups], conditions)
    return flags[0]


def _vocabulary(groups):
    ordered = []
    seen = set()
    for group in groups:
        for cond in group:
            key = condition_key(cond)
            if key not in seen:
                seen.add(key)
                ordered.append({k: cond[k] for k in ("col", "op", "value")})
    return ordered


def counts(flags, target_flags):
    """Confusion counts. A null antecedent or target is undetermined, not a negative."""
    if len(flags) != len(target_flags):
        raise ValueError("flags and target differ in length")
    tp = fp = fn = tn = undetermined = 0
    for antecedent, label in zip(flags, target_flags):
        if antecedent is None or label is None:
            undetermined += 1
            continue
        if antecedent and label:
            tp += 1
        elif antecedent and not label:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "undetermined": undetermined, "n": len(flags)}


def polars_antecedent_flags(df, groups):
    """Existing proposition antecedent, as True / False / None per row."""
    from propositions import _antecedent

    if groups == [[]]:
        return [True] * df.height
    if len(groups) == 1:
        proposition = {"if": groups[0]}
    else:
        proposition = {"if_any": groups}
    column = df.select(x=_antecedent(proposition, df.columns))["x"].to_list()
    out = []
    for flag in column:
        if flag is None:
            out.append(None)
        elif type(flag) is bool or type(flag).__name__ == "bool_":
            out.append(bool(flag))
        else:
            raise ValueError("antecedent flag must be bool or null, got %r" % (flag,))
    return out


def _time_call(fn, min_seconds=0.05, min_calls=1):
    fn()
    calls = 0
    start = time.perf_counter()
    while True:
        fn()
        calls += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds and calls >= min_calls:
            return {"calls": calls, "seconds": elapsed, "seconds_per_call": elapsed / calls}


def benchmark_rule_checks(df, rules, conditions, target, search_kwargs=None, min_seconds=0.05):
    """Time rechecking real ``rule_search`` rules.

    Surrogate: one feature-pack, then ``evaluate_fast`` (one call per row
    when the rule is a single mask, one call per conjunction otherwise).
    Proposition check: ``propositions._antecedent`` once per rule. Row
    bitset: the masks ``rule_search._mask`` builds, combined with AND/OR.
    The same three ways are timed on every conjunction term the search
    builds (``max_conds``). Also times ``rule_search.search`` itself.

    Equality is exact on the flags. Nothing here is a pass threshold for a
    claimed microsecond or a claimed fidelity.
    """
    import rule_search

    group_lists = [rule["if_any"] for rule in rules]
    compiled = [compile_search_rule(groups, conditions) for groups in group_lists]
    single_mask = sum(rule is not None for rule in compiled)
    if not group_lists:
        raise ValueError("no searched rules")

    def surrogate_flags():
        return evaluate_antecedents(df, group_lists, conditions)

    def proposition_flags():
        return [polars_antecedent_flags(df, groups) for groups in group_lists]

    def row_bitset_flags():
        row_masks = [rule_search._mask(df, [cond]) for cond in conditions]
        index = {}
        for bit, cond in enumerate(conditions):
            index.setdefault(condition_key(cond), bit)
        n = df.height
        full = (1 << n) - 1 if n else 0
        out = []
        for groups in group_lists:
            acc = 0
            for group in groups:
                mask = full
                for cond in group:
                    mask &= row_masks[index[condition_key(cond)]]
                acc |= mask
            out.append([bool(acc & (1 << i)) for i in range(n)])
        return out

    max_conds = int((search_kwargs or {}).get("max_conds", 2))
    term_groups = [[group] for group, _mask in rule_search._build_terms(df, conditions, max_conds)]

    def term_surrogate():
        return evaluate_antecedents(df, term_groups, conditions)

    def term_proposition():
        return [polars_antecedent_flags(df, groups) for groups in term_groups]

    def term_row_bitset():
        row_masks = [rule_search._mask(df, [cond]) for cond in conditions]
        index = {}
        for bit, cond in enumerate(conditions):
            index.setdefault(condition_key(cond), bit)
        n = df.height
        full = (1 << n) - 1 if n else 0
        out = []
        for groups in term_groups:
            mask = full
            for cond in groups[0]:
                mask &= row_masks[index[condition_key(cond)]]
            out.append([bool(mask & (1 << i)) for i in range(n)])
        return out

    surrogate = surrogate_flags()
    proposition = proposition_flags()
    row_bitset = row_bitset_flags()
    flags_equal = surrogate == proposition == row_bitset and all(flags is not None for flags in surrogate)
    target_flags = polars_antecedent_flags(df, [target])
    count_rows = []
    for flags, rule in zip(surrogate, rules):
        got = counts(flags, target_flags)
        recorded = (rule["tp"], rule["fp"], rule["fn"], rule["tn"])
        counted = (got["tp"], got["fp"], got["fn"], got["tn"])
        count_rows.append(recorded == counted and got["undetermined"] == 0)
    term_surrogate_flags = term_surrogate()
    term_equal = term_surrogate_flags == term_proposition() == term_row_bitset()
    kwargs = dict(search_kwargs or {})
    search_seconds = _time_call(
        lambda: rule_search.search(df, target, conditions, **kwargs),
        min_seconds=0.0,
        min_calls=3,
    )
    return {
        "rows": df.height,
        "searched_rules": len(rules),
        "single_mask_rules": single_mask,
        "conjunction_terms": len(term_groups),
        "flags_equal": flags_equal,
        "counts_equal": all(count_rows) and len(count_rows) == len(rules),
        "term_flags_equal": term_equal,
        "surrogate": _time_call(surrogate_flags, min_seconds=min_seconds),
        "proposition_antecedent": _time_call(proposition_flags, min_seconds=min_seconds),
        "row_bitset": _time_call(row_bitset_flags, min_seconds=min_seconds),
        "term_surrogate": _time_call(term_surrogate, min_seconds=min_seconds),
        "term_proposition": _time_call(term_proposition, min_seconds=min_seconds),
        "term_row_bitset": _time_call(term_row_bitset, min_seconds=min_seconds),
        "rule_search": search_seconds,
    }


def load_search_frame(root):
    """The frame ``rule_search``'s CLI scores: exclusions applied, nulls dropped."""
    import tomllib

    import polars as pl

    root = Path(root).expanduser().resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.append(root_text)
    import aoi_paths
    from pythdragoras import apply_exclusions

    cycle = aoi_paths.resolve_cycle_path(root)
    spec = tomllib.loads((cycle / "rule_candidates.toml").read_text(encoding="utf-8"))
    cfg = tomllib.loads((cycle / "analysis.toml").read_text(encoding="utf-8"))
    frame = pl.read_ndjson(cycle / "outputs" / "season.jsonl")
    frame, removed = apply_exclusions(frame, cfg.get("exclude", []))
    candidates = spec["candidate"]
    target = spec["target"]
    columns = sorted({item["col"] for item in [*target, *candidates]})
    before = frame.height
    usable = frame.drop_nulls(columns)
    return {
        "frame": usable,
        "target": target,
        "candidates": candidates,
        "max_terms": int(spec.get("max_terms", 2)),
        "max_conds": int(spec.get("max_conds", 2)),
        "dropped_null_rows": before - usable.height,
        "exclusions": removed,
        "cycle": str(cycle),
    }


def _format_seconds(timing):
    return "calls=%d seconds=%.6f per_call=%.6g" % (
        timing["calls"], timing["seconds"], timing["seconds_per_call"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recheck rule_search candidates with the bitmask evaluator and print timings.",
    )
    parser.add_argument("--aoi-root", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    import rule_search

    loaded = load_search_frame(args.aoi_root)
    rules = rule_search.search(
        loaded["frame"], loaded["target"], loaded["candidates"],
        loaded["max_terms"], loaded["max_conds"], args.top,
    )
    report = benchmark_rule_checks(
        loaded["frame"], rules, loaded["candidates"], loaded["target"],
        search_kwargs={
            "max_terms": loaded["max_terms"],
            "max_conds": loaded["max_conds"],
            "top": args.top,
        },
    )
    print("cycle=%s rows=%d dropped_nulls=%d candidates=%d" % (
        loaded["cycle"], report["rows"], loaded["dropped_null_rows"], len(loaded["candidates"])))
    print("searched=%d single_mask=%d conjunction_terms=%d flags_equal=%s counts_equal=%s term_flags_equal=%s" % (
        report["searched_rules"], report["single_mask_rules"], report["conjunction_terms"],
        report["flags_equal"], report["counts_equal"], report["term_flags_equal"]))
    for rule in rules[:5]:
        print("rule %s" % format_search_rule(rule["if_any"], loaded["candidates"]))
    print("surrogate_top %s" % _format_seconds(report["surrogate"]))
    print("proposition_antecedent_top %s" % _format_seconds(report["proposition_antecedent"]))
    print("row_bitset_top %s" % _format_seconds(report["row_bitset"]))
    print("surrogate_terms %s" % _format_seconds(report["term_surrogate"]))
    print("proposition_terms %s" % _format_seconds(report["term_proposition"]))
    print("row_bitset_terms %s" % _format_seconds(report["term_row_bitset"]))
    print("rule_search %s" % _format_seconds(report["rule_search"]))
    return 0 if report["flags_equal"] and report["counts_equal"] and report["term_flags_equal"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
