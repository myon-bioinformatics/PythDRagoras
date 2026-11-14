"""LittleBit 3-way survey（実験）: 乗算も浮動小数も使わずに、画像の領域と重心を推定できるかを測る。

標準ライブラリだけの単一ファイル。pythdragoras.py などの既存モジュールには触れない。

  Way1  8 点 FWHT（加算・減算のみ）を 8x8 パッチ（4 画素ずつずらして重ねる）の行と列に
        かけ、AC 成分の L1 エネルギーを測る。パッチ平均（DC >> 6）との差を三値
        （+1 / 0 / -1）のスピンにする。エネルギーが閾値未満、または有効スピンが 0 の
        パッチは飛ばす（skip）。
  Way2  有効スピンの画素を反実仮想的に揺らす（8 ビット回転、明暗の反転）。そのパッチの
        エネルギー変化 ΔE が閾値以上の画素だけを候補にし、刻み 1/2/4 の近傍に同じ符号の
        候補があればリンクを張る。リンクが 2 本未満の候補は孤立ノイズとして捨てる。
  Way3  Way2 のリンク履歴だけから領域を再構成する（画像は見ない）。行と列を走査し、
        + のノードで内側に入り、- のノードで外側に出る。行走査と列走査の両方で内側の
        画素を領域とする。重心は整数の分子と分母で返す。

コア（Way1〜3）の約束: `*` と浮動小数を使わない。添字は行オフセットを足し合わせて作る。
比は整数の (分子, 分母) のまま保ち、表示するときだけ小数にする。乱数は固定 seed。

  python littlebit_survey.py --report   シナリオごとの Markdown 表を出す
"""

from __future__ import annotations

import argparse, math, random, sys, time

WIDTH = 64
HEIGHT = 64
PATCH = 8
PATCH_STRIDE = 4
TERNARY_THRESHOLD = 24
ENERGY_THRESHOLD = 512
DELTA_E_THRESHOLD = 64
MIN_LINKS = 2
LINK_STRIDES = (1, 2, 4)
SHIFT = (3, 2)


# ---------------------------------------------------------------- core (no `*`, no float)

def fwht8_add_only(values):
    """8 点 Walsh-Hadamard 変換（自然順）。加算と減算だけ。2 回かけると 8 倍に戻る。"""
    if len(values) != 8:
        raise ValueError("fwht8 needs 8 values")
    a = list(values)
    h = 1
    while h < 8:
        i = 0
        while i < 8:
            j = i
            stop = i + h
            while j < stop:
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
                j += 1
            i = i + h + h
        h = h + h
    return a


def rotate_8bit(value, shift=1):
    """8 ビット値の左回転（シフトと論理演算のみ）。"""
    value = value & 0xFF
    shift = shift & 7
    return ((value << shift) | (value >> (8 - shift))) & 0xFF


def fairyfuse_ternary_diff(value, reference, threshold):
    """value と reference の差を三値にする: +1（明るい）/ 0 / -1（暗い）。"""
    diff = value - reference
    if diff > threshold:
        return 1
    if diff < -threshold:
        return -1
    return 0


def row_offsets(width, height):
    """各行の先頭添字。width を足し合わせて作る（乗算なし）。"""
    offsets = []
    acc = 0
    for _ in range(height):
        offsets.append(acc)
        acc += width
    return offsets


def zeros(n):
    return [0 for _ in range(n)]


class LittleBitTernaryPatch:
    """1 枚の 8x8 パッチ。行・列の FWHT エネルギーと三値スピンを持つ。"""

    def __init__(self, image, offsets, x0, y0, threshold=TERNARY_THRESHOLD):
        self.x0 = x0
        self.y0 = y0
        rows = []
        for dy in range(PATCH):
            base = offsets[y0 + dy] + x0
            rows.append([image[base + dx] for dx in range(PATCH)])
        self.rows = rows
        self.threshold = threshold
        self.energy, self.dc = patch_energy(rows)
        mean = self.dc >> 6
        self.mean = mean
        spins = []
        for dy in range(PATCH):
            for dx in range(PATCH):
                s = fairyfuse_ternary_diff(rows[dy][dx], mean, threshold)
                if s:
                    spins.append((x0 + dx, y0 + dy, s))
        self.spins = spins

    @property
    def active(self):
        return len(self.spins)

    def energy_with(self, dx, dy, value):
        """画素 (dx, dy) を value に置き換えたときのエネルギー（反実仮想）。"""
        rows = [list(r) for r in self.rows]
        rows[dy][dx] = value
        return patch_energy(rows)[0]


def patch_energy(rows):
    """行ごと・列ごとの FWHT の AC 成分の L1 和と、2 次元 DC（画素の総和）。"""
    energy = 0
    col_inputs = [zeros(PATCH) for _ in range(PATCH)]
    dc = 0
    for dy in range(PATCH):
        t = fwht8_add_only(rows[dy])
        dc += t[0]
        for k in range(1, PATCH):
            energy += abs(t[k])
        for dx in range(PATCH):
            col_inputs[dx][dy] = rows[dy][dx]
    for dx in range(PATCH):
        t = fwht8_add_only(col_inputs[dx])
        for k in range(1, PATCH):
            energy += abs(t[k])
    return energy, dc


def region_iou(region, truth):
    """IoU を (共通部分, 和集合) の整数の組で返す。"""
    inter = len(region & truth)
    union = len(region | truth)
    return inter, union


def shift_region(region, dx, dy, width=WIDTH, height=HEIGHT):
    out = set()
    for x, y in region:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height:
            out.add((nx, ny))
    return out


def center_of_mass(region):
    """重心を ((x の和, 個数), (y の和, 個数)) で返す。空なら None。"""
    if not region:
        return None
    sx = 0
    sy = 0
    for x, y in region:
        sx += x
        sy += y
    n = len(region)
    return (sx, n), (sy, n)


class ThreeWaySurveyEngine:
    """Way1〜3 を順に実行する。乗算と浮動小数を使わない。"""

    def __init__(self, width=WIDTH, height=HEIGHT, *, ternary_threshold=TERNARY_THRESHOLD,
                 energy_threshold=ENERGY_THRESHOLD, delta_e_threshold=DELTA_E_THRESHOLD,
                 min_links=MIN_LINKS, strides=LINK_STRIDES):
        self.width = width
        self.height = height
        self.offsets = row_offsets(width, height)
        self.ternary_threshold = ternary_threshold
        self.energy_threshold = energy_threshold
        self.delta_e_threshold = delta_e_threshold
        self.min_links = min_links
        self.strides = tuple(strides)

    def way1(self, image):
        """パッチを作り、飛ばすものと残すものに分ける。skip 率は (飛ばした数, 全数)。"""
        kept = []
        skipped = 0
        total = 0
        y0 = 0
        while y0 + PATCH <= self.height:
            x0 = 0
            while x0 + PATCH <= self.width:
                patch = LittleBitTernaryPatch(image, self.offsets, x0, y0, self.ternary_threshold)
                total += 1
                if patch.energy < self.energy_threshold or patch.active == 0:
                    skipped += 1
                else:
                    kept.append(patch)
                x0 += PATCH_STRIDE
            y0 += PATCH_STRIDE
        return {"patches": kept, "skip": (skipped, total)}

    def way2(self, image, patches):
        """反実仮想 ΔE で候補を選び、刻み 1/2/4 の同符号近傍でリンクする。"""
        votes = {}
        owner = {}
        for patch in patches:
            for x, y, s in patch.spins:
                key = (x, y)
                votes[key] = votes.get(key, 0) + s
                best = owner.get(key)
                if best is None or patch.energy > best.energy:
                    owner[key] = patch
        candidates = {}
        for key, vote in votes.items():
            if vote == 0:
                continue
            sign = 1 if vote > 0 else -1
            patch = owner[key]
            x, y = key
            dx = x - patch.x0
            dy = y - patch.y0
            value = patch.rows[dy][dx]
            delta_rot = abs(patch.energy_with(dx, dy, rotate_8bit(value)) - patch.energy)
            delta_flip = abs(patch.energy_with(dx, dy, 255 - value) - patch.energy)
            delta = delta_rot + delta_flip
            if delta >= self.delta_e_threshold:
                candidates[key] = (sign, delta)
        links = []
        degree = {}
        for (x, y), (sign, delta) in candidates.items():
            for k in self.strides:
                for q in ((x + k, y), (x, y + k)):
                    other = candidates.get(q)
                    if other is not None and other[0] == sign:
                        links.append({"a": (x, y), "b": q, "sign": sign, "stride": k,
                                      "delta_a": delta, "delta_b": other[1]})
                        degree[(x, y)] = degree.get((x, y), 0) + 1
                        degree[q] = degree.get(q, 0) + 1
        accepted = {key for key, d in degree.items() if d >= self.min_links}
        history = [link for link in links if link["a"] in accepted and link["b"] in accepted]
        rejected = len(candidates) - len(accepted)
        return {"links": history, "candidates": len(candidates), "rejected_isolated": rejected}

    def way3(self, links):
        """リンク履歴だけから領域と重心を作る（画像は見ない）。"""
        signs = {}
        for link in links:
            signs[link["a"]] = link["sign"]
            signs[link["b"]] = link["sign"]
        rows_inside = set()
        for y in range(self.height):
            inside = False
            for x in range(self.width):
                s = signs.get((x, y), 0)
                if s > 0:
                    inside = True
                elif s < 0:
                    inside = False
                if inside:
                    rows_inside.add((x, y))
        region = set()
        for x in range(self.width):
            inside = False
            for y in range(self.height):
                s = signs.get((x, y), 0)
                if s > 0:
                    inside = True
                elif s < 0:
                    inside = False
                if inside and (x, y) in rows_inside:
                    region.add((x, y))
        return {"region": region, "center": center_of_mass(region), "nodes": len(signs)}

    def run(self, image, truth, shift=SHIFT):
        w1 = self.way1(image)
        w2 = self.way2(image, w1["patches"])
        w3 = self.way3(w2["links"])
        region = w3["region"]
        return {"skip": w1["skip"], "candidates": w2["candidates"], "rejected_isolated": w2["rejected_isolated"],
                "links": len(w2["links"]), "region": region, "center": w3["center"],
                "iou": region_iou(region, truth),
                "iou_shifted": region_iou(shift_region(region, shift[0], shift[1], self.width, self.height), truth)}


CORE_FUNCTIONS = ("fwht8_add_only", "rotate_8bit", "fairyfuse_ternary_diff", "row_offsets", "zeros",
                  "patch_energy", "region_iou", "shift_region", "center_of_mass",
                  "LittleBitTernaryPatch", "ThreeWaySurveyEngine")


# ---------------------------------------------------------------- scenarios and report (display layer)

def _blank(value):
    return [value for _ in range(WIDTH * HEIGHT)]


def _paint(image, mask, value):
    for x, y in mask:
        image[y * WIDTH + x] = value


def _jitter(image, rng, amount):
    return [min(255, max(0, v + rng.randint(-amount, amount))) for v in image]


def _ellipse(cx, cy, rx, ry):
    return {(x, y) for y in range(HEIGHT) for x in range(WIDTH)
            if (x - cx) ** 2 * ry ** 2 + (y - cy) ** 2 * rx ** 2 <= rx ** 2 * ry ** 2}


def scenario_dense_object(rng):
    truth = _ellipse(30, 33, 17, 13)
    image = _blank(30)
    _paint(image, truth, 210)
    return _jitter(image, rng, 6), truth


def scenario_thin_skeleton(rng):
    truth = set()
    for x in range(10, 54):
        truth.update({(x, 20), (x, 21)})
    for y in range(20, 52):
        truth.update({(31, y), (32, y)})
    for i in range(0, 20):
        truth.update({(12 + i, 30 + i), (13 + i, 30 + i)})
    image = _blank(25)
    _paint(image, truth, 220)
    return _jitter(image, rng, 6), truth


def scenario_hollow_ring(rng):
    truth = _ellipse(32, 32, 22, 22) - _ellipse(32, 32, 13, 13)
    image = _blank(30)
    _paint(image, truth, 200)
    return _jitter(image, rng, 6), truth


def scenario_high_noise_dark(rng):
    truth = _ellipse(34, 30, 12, 10)
    image = _blank(12)
    _paint(image, truth, 150)
    image = _jitter(image, rng, 8)
    for i in range(len(image)):
        if rng.random() < 0.03:
            image[i] = 255
    return image, truth


def scenario_low_contrast(rng):
    truth = _ellipse(30, 30, 15, 11)
    image = _blank(100)
    _paint(image, truth, 160)
    return _jitter(image, rng, 6), truth


def scenario_multi_blob(rng):
    truth = _ellipse(18, 20, 9, 8) | _ellipse(44, 42, 10, 9)
    image = _blank(20)
    _paint(image, truth, 200)
    return _jitter(image, rng, 6), truth


SCENARIOS = (
    ("Dense Object", scenario_dense_object, 11),
    ("Thin Skeleton", scenario_thin_skeleton, 12),
    ("Hollow Ring", scenario_hollow_ring, 13),
    ("High Noise Dark", scenario_high_noise_dark, 14),
    ("Low Contrast", scenario_low_contrast, 15),
    ("Multi Blob", scenario_multi_blob, 16),
)


def ratio(pair):
    """表示用: (分子, 分母) を小数にする。"""
    num, den = pair
    return num / den if den else 0.0


def com_error(center, truth):
    """表示用: 推定重心と正解重心の距離（画素）。"""
    if center is None:
        return None
    t = center_of_mass(truth)
    return math.hypot(ratio(center[0]) - ratio(t[0]), ratio(center[1]) - ratio(t[1]))


def measure(engine=None):
    engine = engine or ThreeWaySurveyEngine()
    rows = []
    for name, factory, seed in SCENARIOS:
        image, truth = factory(random.Random(seed))
        start = time.perf_counter()
        result = engine.run(image, truth)
        elapsed = time.perf_counter() - start
        rows.append({"name": name, "seed": seed, "result": result, "truth": truth, "seconds": elapsed,
                     "skip_rate": ratio(result["skip"]), "iou": ratio(result["iou"]),
                     "iou_shifted": ratio(result["iou_shifted"]), "com_error": com_error(result["center"], truth)})
    return rows


def report(rows):
    lines = [f"| シナリオ | skip 率 | IoU | IoU（{SHIFT[0]},{SHIFT[1]} ずらし） | 重心誤差 (px) | 孤立として除外 | リンク | 実行時間 (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        r = row["result"]
        com = "—" if row["com_error"] is None else f"{row['com_error']:.2f}"
        lines.append(f"| {row['name']} | {row['skip_rate']:.3f} ({r['skip'][0]}/{r['skip'][1]}) | {row['iou']:.3f} | "
                     f"{row['iou_shifted']:.3f} | {com} | {r['rejected_isolated']} | {r['links']} | {row['seconds']:.2f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="シナリオごとの Markdown 表を出す")
    args = parser.parse_args(argv)
    if not args.report:
        parser.print_help()
        return 0
    print(report(measure()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
