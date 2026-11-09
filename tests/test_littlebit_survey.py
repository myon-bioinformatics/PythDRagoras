"""littlebit_survey: 乗算・浮動小数なしのコア、FWHT の往復、シナリオごとの合格線（オフライン）。"""

import ast, inspect, random, subprocess, sys
from pathlib import Path

import pytest

import littlebit_survey as ls

# シナリオごとの合格線。seed 1〜30 で測った最小値（README の表）から余裕を取って決めた。
THRESHOLDS = {
    "Dense Object": {"iou": 0.95, "skip": 0.70, "com": 0.5},
    "Thin Skeleton": {"iou": 0.90, "skip": 0.65, "com": 1.0},
    "Hollow Ring": {"iou": 0.95, "skip": 0.35, "com": 0.5},
    "High Noise Dark": {"iou": 0.90, "skip": 0.0, "com": 1.0},
    "Low Contrast": {"iou": 0.95, "skip": 0.70, "com": 0.5},
    "Multi Blob": {"iou": 0.95, "skip": 0.65, "com": 1.0},
}


@pytest.fixture(scope="module")
def rows():
    return {row["name"]: row for row in ls.measure()}


def test_fwht_twice_returns_original_times_eight():
    rng = random.Random(0)
    for _ in range(50):
        v = [rng.randint(-300, 300) for _ in range(8)]
        assert ls.fwht8_add_only(ls.fwht8_add_only(v)) == [8 * x for x in v]
    assert ls.fwht8_add_only([1, 0, 0, 0, 0, 0, 0, 0]) == [1] * 8
    with pytest.raises(ValueError):
        ls.fwht8_add_only([1, 2, 3])


def test_2d_fwht_twice_returns_original_times_64():
    rng = random.Random(1)
    patch = [[rng.randint(0, 255) for _ in range(8)] for _ in range(8)]
    def transform(m):
        rows = [ls.fwht8_add_only(r) for r in m]
        cols = [ls.fwht8_add_only([rows[y][x] for y in range(8)]) for x in range(8)]
        return [[cols[x][y] for x in range(8)] for y in range(8)]
    assert transform(transform(patch)) == [[64 * v for v in r] for r in patch]
    energy, dc = ls.patch_energy(patch)
    assert dc == sum(map(sum, patch)) and energy >= 0
    assert ls.patch_energy([[7] * 8 for _ in range(8)])[0] == 0


def test_rotate_and_ternary():
    assert ls.rotate_8bit(0b1000_0001) == 0b0000_0011
    assert ls.rotate_8bit(0xFF) == 0xFF
    assert all(ls.rotate_8bit(ls.rotate_8bit(v, 3), 5) == v for v in range(256))
    assert [ls.fairyfuse_ternary_diff(v, 100, 24) for v in (130, 124, 100, 76, 70)] == [1, 0, 0, 0, -1]
    assert ls.row_offsets(5, 4) == [0, 5, 10, 15]


def _core_nodes():
    tree = ast.parse(Path(ls.__file__).read_text(encoding="utf-8"))
    wanted = set(ls.CORE_FUNCTIONS)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            found[node.name] = node
    assert set(found) == wanted
    return found


def test_core_has_no_multiplication_division_or_float():
    for name, node in _core_nodes().items():
        for sub in ast.walk(node):
            assert not isinstance(sub, (ast.Mult, ast.Div, ast.Pow, ast.MatMult)), (name, ast.dump(sub))
            if isinstance(sub, ast.Constant):
                assert not isinstance(sub.value, (float, complex)), (name, sub.value)
            if isinstance(sub, ast.Name):
                assert sub.id not in ("float", "math", "round"), (name, sub.id)


def test_core_results_are_integers(rows):
    for row in rows.values():
        r = row["result"]
        for pair in (r["skip"], r["iou"], r["iou_shifted"], *r["center"]):
            assert all(type(v) is int for v in pair)


def test_deterministic():
    a = ls.measure()
    b = ls.measure()
    for x, y in zip(a, b):
        assert x["result"]["region"] == y["result"]["region"] and x["result"]["iou"] == y["result"]["iou"]


def test_way3_uses_link_history_only():
    engine = ls.ThreeWaySurveyEngine()
    image, truth = ls.scenario_hollow_ring(random.Random(13))
    w2 = engine.way2(image, engine.way1(image)["patches"])
    assert set(inspect.signature(engine.way3).parameters) == {"links"}
    region = engine.way3(w2["links"])["region"]
    assert region == engine.run(image, truth)["region"]
    assert engine.way3([])["region"] == set() and engine.way3([])["center"] is None


@pytest.mark.parametrize("name", list(THRESHOLDS))
def test_scenario_thresholds(rows, name):
    row, limit = rows[name], THRESHOLDS[name]
    assert row["iou"] >= limit["iou"], row["iou"]
    assert row["skip_rate"] >= limit["skip"], row["skip_rate"]
    assert row["com_error"] is not None and row["com_error"] <= limit["com"], row["com_error"]


@pytest.mark.parametrize("name", list(THRESHOLDS))
def test_shift_reduces_iou(rows, name):
    r = rows[name]["result"]
    (i0, u0), (i1, u1) = r["iou"], r["iou_shifted"]
    assert i1 * u0 < i0 * u1


def test_hollow_ring_keeps_hole_empty(rows):
    hole = ls._ellipse(32, 32, 10, 10)
    assert len(rows["Hollow Ring"]["result"]["region"] & hole) == 0


def test_isolated_noise_rejection_matters():
    image, truth = ls.scenario_high_noise_dark(random.Random(14))
    with_rejection = ls.ThreeWaySurveyEngine().run(image, truth)
    without = ls.ThreeWaySurveyEngine(min_links=0).run(image, truth)
    assert with_rejection["rejected_isolated"] > 0
    assert ls.ratio(with_rejection["iou"]) > ls.ratio(without["iou"]) + 0.05


def test_report_cli():
    out = subprocess.run([sys.executable, str(Path(ls.__file__)), "--report"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0].startswith("| シナリオ |") and len(lines) == 2 + len(ls.SCENARIOS)
    assert all(name in out.stdout for name in THRESHOLDS)
