"""inkfloor's ruler: comparison between two ink prediction maps.

Every number reported by the tool passes through here, so this module is written for
correctness, not elegance or speed.

The central metric is Delta = 1 - IoU between the top-q% of the two maps **with matched
positive budgets**: exactly k = round(n*q) pixels are selected on each side, where n is the
number of valid pixels, and the intersection and union of those two sets are compared.

Why not use a fixed threshold: a fixed threshold conflates a calibration difference with a
localization difference. Two maps can put ink in exactly the same places and have different
histograms, for example because the two volumes have different intensities or because the
model was applied with different normalizations; with a fixed threshold, the "warmer" map
has many more positives than the other, the IoU collapses, and the reader concludes that the
two predictions disagree on where the ink is. The matched budget removes that confounder:
both sides spend the same number of positives, and what remains is a positional difference.
See the villa#191 thread.

Conventions for real data:
- maps arrive from tifffile as 2-D uint8 arrays, with shapes such as (13420, 14940);
- a pixel is valid when its value is > 0: zero means outside the mask, not a prediction
  value. About 91% of pixels are valid in a typical segment;
- two maps of the same segment can have slightly different shapes. Here they are cropped
  to the common region anchored at (0, 0); see `align`.

What this module does NOT do: it does not read files, download anything, print, choose q,
or decide whether a Delta is "large". It accepts arrays and returns data.

A warning about ties, to be read before trusting an IoU: with highly quantized maps (uint8,
256 levels), pixels at the top-k boundary very often have the same value, and which of those
pixels enter the top-k is arbitrary. `delta_at_q` breaks ties deterministically but by
position (lower flat index first, i.e. upper rows), and this can inflate or deflate the IoU
without the number revealing it. The extreme case is a constant map, where the IoU is 1
despite there being no shared information; the realistic case is two maps with the same
tail saturated at 255, where the top-k falls entirely inside the tie and the IoU returns
close to 1 by construction.
`tie_bounds` measures how much of an IoU is arbitrary: it returns the exact interval
[iou_min, iou_max] over all admissible tie-breakings. A wide interval means that the point
estimate should not be reported on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Tolerance for the shape difference between two maps of the same segment. Above this
# threshold it is no longer a matter of rounding the mesh bounding box, and silently
# cropping would conceal an error.
MAX_SHAPE_DIFF_PX = 64
MAX_SHAPE_DIFF_FRAC = 0.01

# Limits for memory-bounded paths: real maps have ~2e8 pixels, and float64 arrays of that
# length cannot be materialized on a laptop.
_HIST_MAX_BINS = 1 << 22
_CHUNK = 1 << 23


class ShapeMismatch(ValueError):
    """The shapes differ too much for cropping to the common region to be defensible."""


@dataclass(frozen=True)
class Delta:
    q: float
    iou: float
    dice: float
    n_valid: int
    k: int  # number of positives per side (matched budget)


@dataclass(frozen=True)
class TieBand:
    """How much of an IoU is decided by ties rather than by the data.

    `iou` is the value returned by `delta_at_q`, i.e. one particular tie-breaking.
    `iou_min` and `iou_max` are the exact minimum and maximum over ALL admissible top-k
    selections for the two maps. If they coincide, the top-k is unique and the IoU is a
    fact; if they are far apart, the point IoU is largely an artifact of index order.
    """

    q: float
    k: int
    n_valid: int
    iou: float
    iou_min: float
    iou_max: float
    thr_a: float  # k-th highest value of a among valid pixels
    thr_b: float
    forced_a: int  # pixels strictly above the threshold: they must be included
    forced_b: int
    ties_needed_a: int  # number of pixels that must be drawn from the tie band
    ties_needed_b: int
    tie_band_a: int  # number of pixels tied at the threshold
    tie_band_b: int

    @property
    def unique(self) -> bool:
        """True if both top-k selections are unique, i.e. involve no arbitrary choices."""
        return self.tie_band_a == self.ties_needed_a and self.tie_band_b == self.ties_needed_b

    @property
    def width(self) -> float:
        """Width of the tie uncertainty interval. 0 means no arbitrariness."""
        return self.iou_max - self.iou_min


# --------------------------------------------------------------------------------------
# array preparation
# --------------------------------------------------------------------------------------


def align(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Crop all arrays to the smallest common shape, anchoring the origin at (0, 0).

    Why this choice instead of padding: maps of a segment are rendered on the same flattened
    grid and anchored at the same corner, and the shape difference between two derivations
    comes from rounding the mesh extent, meaning one or a few rows or columns at the end.
    Cropping keeps corresponding pixels superimposed; padding would invent pixels, and
    center alignment would shift everything by half the difference.

    It does NOT verify that the two maps are actually anchored at the same corner: if they
    are not, cropping compares different regions without complaint. `best_shift_iou` resolves
    that doubt by showing whether a translation is better than no translation.

    Raises ShapeMismatch if the difference on any axis exceeds
    max(MAX_SHAPE_DIFF_PX, MAX_SHAPE_DIFF_FRAC * extent), because at that point it is not a
    rounding difference and stopping is better than cropping.
    """
    arrs = [np.asarray(x) for x in arrays]
    if not arrs:
        raise ValueError("align needs at least one array")
    nd = arrs[0].ndim
    if any(x.ndim != nd for x in arrs):
        raise ShapeMismatch(f"numero di assi diverso: {[x.shape for x in arrs]}")

    target: list[int] = []
    for ax in range(nd):
        sizes = [int(x.shape[ax]) for x in arrs]
        lo, hi = min(sizes), max(sizes)
        tol = max(MAX_SHAPE_DIFF_PX, int(math.ceil(MAX_SHAPE_DIFF_FRAC * hi)))
        if hi - lo > tol:
            raise ShapeMismatch(
                f"asse {ax}: shape {sizes} differiscono di {hi - lo} px, oltre la tolleranza {tol}"
            )
        target.append(lo)

    tt = tuple(target)
    if all(tuple(x.shape) == tt for x in arrs):
        return tuple(arrs)
    sl = tuple(slice(0, t) for t in tt)
    return tuple(x[sl] for x in arrs)


def ink_valid(a: np.ndarray) -> np.ndarray:
    """Mask of prediction pixels in a single map: valid if > 0.

    It does NOT distinguish an out-of-mask pixel from an ink prediction of exactly zero: in
    the published format, zero means outside the mask and that distinction does not exist.
    """
    return np.asarray(a) > 0


def common_valid(maps: list[np.ndarray]) -> np.ndarray:
    """Mask of pixels valid in ALL maps. A pixel is valid if it is > 0 in every map.

    The maps are first cropped to the common region (see `align`), so the mask has the
    smallest common shape, not the shape of the first map.

    It does NOT check that the maps are from the same segment or report how much is lost in
    the intersection: that count is available in `Delta.n_valid`.
    """
    if not maps:
        raise ValueError("common_valid needs at least one map")
    arrs = align(*maps)
    out = ink_valid(arrs[0])
    for x in arrs[1:]:
        out = out & (x > 0)
    return out


def shift_map(
    a: np.ndarray, valid: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a 2-D map by (dy, dx) and return (shifted map, shifted valid pixels).

    The shift uses `np.roll`, but the strip that wrapped around is marked NOT valid, so the
    comparison never uses pixels that reentered from the other side. Convention:
    `out[y, x] = a[y - dy, x - dx]`, meaning positive dy and dx move the content down and to
    the right.

    It does NOT interpolate or handle fractional shifts.
    """
    arr, m = align(a, valid)
    if arr.ndim != 2:
        raise ValueError(f"shift_map works on 2-D maps, got ndim={arr.ndim}")
    m = np.asarray(m).astype(bool)
    out = np.roll(arr, (dy, dx), axis=(0, 1))
    vout = np.roll(m, (dy, dx), axis=(0, 1))
    h, w = arr.shape
    if dy > 0:
        vout[: min(dy, h)] = False
    elif dy < 0:
        vout[max(h + dy, 0) :] = False
    if dx > 0:
        vout[:, : min(dx, w)] = False
    elif dx < 0:
        vout[:, max(w + dx, 0) :] = False
    return out, vout


# --------------------------------------------------------------------------------------
# central metric
# --------------------------------------------------------------------------------------


def delta_at_q(a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float) -> Delta:
    """1 - IoU between the top-q% of a and b, with matched positive budgets.

    It selects k = round(n*q) pixels per side, where n is the number of valid pixels, and
    compares the intersection and union of the two sets. Since both sets have the same
    cardinality k, the identities union = 2k - inter, dice = inter/k, and
    iou = dice / (2 - dice) hold: here IoU and Dice carry the same information; Dice is
    reported only because it is the scale many readers are accustomed to.

    It does NOT use a fixed threshold: a fixed threshold conflates a calibration difference
    with a localization difference. See the villa#191 thread and the module docstring.

    Other things it does NOT do:
    - it does not calculate 1 - IoU for you: it returns the IoU; Delta is 1 - Delta.iou;
    - it does not verify that `valid` is consistent with the two maps; the mask is the one
      you pass in (usually `common_valid([a, b])`);
    - it does not account for ties: if many pixels have the same value at the threshold,
      which ones enter the top-k is arbitrary and this function does not signal it. Use
      `tie_bounds`;
    - it does not handle NaNs: in a float array, a NaN is ordered as the highest value and
      therefore enters the top-k. The corpus maps are uint8, so this case does not arise,
      but if you arrive here with floats, clean them first.

    Raises ValueError if q is not in (0, 1], if there are no valid pixels, or if
    round(n*q) = 0: on a crop that is too small, the metric is undefined, and returning a
    number anyway would be worse than stopping.
    """
    va, vb, n, k = _values_and_budget(a, b, valid, q)
    ma = _topk_mask(va, k)
    mb = _topk_mask(vb, k)
    return _delta_from_masks(ma, mb, q, n, k)


def spearman(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    """Spearman rank correlation between a and b over valid pixels.

    Invariant under a strictly monotonic, order-preserving transformation of the values, it
    therefore tells a milder story than IoU: two maps that order the pixels the same way
    produce 1 even if their calibrations are extremely far apart. Clipping or quantisation is
    only non-decreasing, not strictly monotonic: it creates ties and can change Spearman.
    Both metrics should be reported because they answer different questions: Spearman asks
    "is the order the same?", top-q% IoU asks "are the hottest pixels the same pixels?".

    Ties are handled with midranks, which is the correct adjustment and matters greatly
    here: with uint8 there are 256 levels for millions of pixels, so tie groups are enormous.
    For bounded-range integer inputs, the calculation uses a joint histogram, which gives
    the exact result without materializing rank arrays as long as the map.

    It does NOT perform significance tests (with 1e8 pixels any p-value is zero and means
    nothing), does NOT handle NaNs, and is NOT invariant under non-monotonic remappings.
    Returns nan if either map is constant over the valid pixels, because in that case the
    rank has no variance and the correlation is undefined.
    """
    arr_a, arr_b, m = _prepare(a, b, valid)
    return _spearman_1d(arr_a[m], arr_b[m])


def common_valid_fraction(valid: np.ndarray) -> float:
    """Fraction of valid pixels in the given shape. Handy for reports, nothing more."""
    m = np.asarray(valid)
    return float(np.count_nonzero(m)) / float(m.size) if m.size else 0.0


def chance_iou(q: float) -> float:
    """Expected IoU of two independently and randomly selected top-k sets: q / (2 - q).

    Derivation: two independent subsets of k = n*q pixels out of n have expected intersection
    k^2/n = k*q and expected union 2k - k*q, hence IoU approximately q/(2-q). This holds as a
    ratio of expected values, i.e. asymptotically in k, which is more than sufficient when k
    is on the order of 1e6.

    This serves as a reading reminder: the floor of the metric INCREASES with q. At q = 0.01,
    chance gives 0.005; at q = 0.20, it gives 0.111. An IoU of 0.11 at 20% is not weak
    agreement; it is chance. A Delta should always be read against `null_shift` at the same
    q, not against zero.

    It is NOT a significance threshold and does NOT account for real top-k sets being
    spatially clustered: on structured maps, the value measured by the null control can be
    considerably lower than this number.
    """
    if not (0.0 < float(q) <= 1.0):
        raise ValueError(f"q fuori da (0, 1]: {q}")
    return float(q) / (2.0 - float(q))


# --------------------------------------------------------------------------------------
# null controls
# --------------------------------------------------------------------------------------


def null_self(a: np.ndarray, valid: np.ndarray, q: float) -> Delta:
    """Null control that MUST give IoU = 1: the map against itself.

    This is not a tautology: it follows the same path as `delta_at_q`, so if the top-k
    selection were not deterministic (for example, if it broke ties randomly), this control
    would reveal it. A value other than 1 is a bug in the ruler, not data.

    It says NOTHING about the scale of "good" values: it is only the metric's ceiling.
    """
    return delta_at_q(a, a, valid, q)


def null_shift(a: np.ndarray, valid: np.ndarray, q: float, px: int = 64) -> Delta:
    """Null control that MUST give low IoU: the map against itself shifted by px along x.

    It provides a scale: this is the value obtained when the two maps have the same texture
    and calibration but their positions have nothing to do with each other. A measured Delta
    near this value means "the two predictions share no localization." The returned
    `Delta.n_valid` is for the intersection of valid and shifted-valid pixels, so it is lower
    than that of `null_self`, and k is consequently lower as well.

    It is NOT a guarantee. Two cases in which this control returns high IoU even though the
    shift is meaningless: a constant map, or any map entirely tied at the threshold (ties are
    broken by position and the two selections coincide), and a map with periodic structure
    along x whose period divides px. For the second case, repeat with an incommensurate px;
    for the first, inspect `tie_bounds`.

    It does NOT shift along y: if the segment structure is anisotropic, use `shift_map`
    manually.
    """
    m = np.asarray(valid).astype(bool)
    shifted, vshift = shift_map(a, m, 0, px)
    arr, m2 = align(a, m)
    return delta_at_q(arr, shifted, m2 & vshift, q)


def best_shift_iou(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float, radius: int = 20
) -> tuple[int, int, float]:
    """Best IoU from an exhaustive search of shifts within radius.

    This rules out a difference being merely a misalignment: if the best IoU occurs at
    (0, 0), the two maps are aligned and the measured Delta is a real difference; if it
    occurs elsewhere, the Delta includes a translation component and should be discarded or
    corrected. Returns (dy, dx, iou), using the `shift_map` convention: b must be shifted by
    (dy, dx), meaning `b[y - dy, x - dx]` is compared with `a[y, x]`. If b is a shifted
    version of a, down by 3 and right by 5, the answer is (-3, -5).

    The two top-k sets are computed ONCE on the unshifted common mask and then shifted: they
    are not reselected for every translation. This is a deliberate choice, not a speed
    shortcut: reselecting would change the k budget at every step, and the IoUs of different
    translations would no longer be comparable with each other, which is the only thing
    needed here. Consequently, for translations other than (0, 0), the positive count in the
    overlapping window is slightly lower than k, and IoU is calculated from the window's
    actual counts, not from 2k - inter.

    It does NOT search rotations, scales, or deformations, does NOT interpolate below the
    pixel level, and costs (2*radius+1)^2 passes over the entire map: on a 2e8-pixel map with
    radius 20, that is 1681 passes, measured at about 0.09 s each, or two and a half minutes.
    Lower radius if only a quick check is needed.

    Most importantly: it does NOT say whether the shift it found is real. It always returns
    a maximum, even between two maps that have nothing to do with each other, in which case
    the winning shift is the luckiest noise among (2*radius+1)^2 attempts. The reader must
    compare the best IoU with the IoU at zero shift: if the gain is small, there is no
    misalignment to correct.
    """
    if radius < 0:
        raise ValueError(f"radius negativo: {radius}")
    arr_a, arr_b, m = _prepare(a, b, valid)
    if arr_a.ndim != 2:
        raise ValueError(f"best_shift_iou works on 2-D maps, got ndim={arr_a.ndim}")
    va, vb = arr_a[m], arr_b[m]
    n = int(va.size)
    k = _budget(n, q)

    mask_a = np.zeros(arr_a.shape, dtype=bool)
    mask_a[m] = _topk_mask(va, k)
    mask_b = np.zeros(arr_b.shape, dtype=bool)
    mask_b[m] = _topk_mask(vb, k)

    h, w = arr_a.shape
    shifts = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    # Order by increasing distance from (0, 0), and require a strictly greater improvement:
    # at equal IoU, the smaller shift wins, i.e. the less adventurous hypothesis.
    shifts.sort(key=lambda t: (abs(t[0]) + abs(t[1]), t[0], t[1]))

    best = (0, 0, -1.0)
    for dy, dx in shifts:
        ya0, ya1 = max(0, dy), min(h, h + dy)
        xa0, xa1 = max(0, dx), min(w, w + dx)
        if ya1 <= ya0 or xa1 <= xa0:
            continue
        wa = mask_a[ya0:ya1, xa0:xa1]
        wb = mask_b[ya0 - dy : ya1 - dy, xa0 - dx : xa1 - dx]
        inter = int(np.count_nonzero(wa & wb))
        sa = int(np.count_nonzero(wa))
        sb = int(np.count_nonzero(wb))
        union = sa + sb - inter
        iou = (inter / union) if union else 0.0
        if iou > best[2]:
            best = (dy, dx, iou)
    return best


# --------------------------------------------------------------------------------------
# tie uncertainty
# --------------------------------------------------------------------------------------


def tie_bounds(a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float) -> TieBand:
    """Exact IoU interval over all admissible tie-breakings.

    A top-k is admissible if it contains every pixel strictly above the k-th threshold and
    completes the budget with pixels whose values equal the threshold exactly. When the tie
    band is wider than the number of pixels needed, there are many admissible selections,
    and the IoU is an interval rather than a unique number. This function calculates it in
    closed form, in constant time with respect to the number of possible selections, by
    counting pixels in the nine cells (above threshold / tied / below threshold) for a,
    cross-tabulated against the same states for b.

    How to read it: if `width` is near zero, the reported IoU is a fact; if it is large, the
    point estimate is largely a consequence of index order and should be reported as an
    interval. With uint8 saturated tails (many pixels at 255), `width` can reach 1, meaning
    the data do not constrain the IoU at all.

    It does NOT say which tie-breaking is the "right" one, because there is no such choice,
    and does NOT weight selections with probabilities: this is an admissibility interval,
    not a confidence interval.
    """
    va, vb, n, k = _values_and_budget(a, b, valid, q)
    ma, raw_thr_a = _topk_and_threshold(va, k)
    mb, raw_thr_b = _topk_and_threshold(vb, k)
    delta = _delta_from_masks(ma, mb, q, n, k)

    thr_a = float(raw_thr_a)
    thr_b = float(raw_thr_b)
    cells = _cell_counts(va, vb, raw_thr_a, raw_thr_b)

    core_a = int(cells[0].sum())
    core_b = int(cells[:, 0].sum())
    band_a = int(cells[1].sum())
    band_b = int(cells[:, 1].sum())
    need_a = k - core_a
    need_b = k - core_b

    lo_inter = _min_intersection(cells, need_a, need_b)
    hi_inter = _max_intersection(cells, need_a, need_b)
    return TieBand(
        q=float(q),
        k=k,
        n_valid=n,
        iou=delta.iou,
        iou_min=lo_inter / (2 * k - lo_inter) if (2 * k - lo_inter) else 0.0,
        iou_max=hi_inter / (2 * k - hi_inter) if (2 * k - hi_inter) else 0.0,
        thr_a=thr_a,
        thr_b=thr_b,
        forced_a=core_a,
        forced_b=core_b,
        ties_needed_a=need_a,
        ties_needed_b=need_b,
        tie_band_a=band_a,
        tie_band_b=band_b,
    )


def _cell_counts(va: np.ndarray, vb: np.ndarray, thr_a, thr_b) -> np.ndarray:
    """3x3 count matrix: rows = state in a, columns = state in b.

    States: 0 = value above the threshold (must be included), 1 = tied at the threshold
    (included if selected), 2 = below the threshold (cannot be included).
    """
    out = np.zeros(9, dtype=np.int64)
    for sl in _chunks(int(va.size)):
        xa, xb = va[sl], vb[sl]
        la = np.where(xa > thr_a, 0, np.where(xa == thr_a, 1, 2))
        lb = np.where(xb > thr_b, 0, np.where(xb == thr_b, 1, 2))
        code = (la.astype(np.int64) * 3 + lb.astype(np.int64)).ravel()
        out += np.bincount(code, minlength=9)
    return out.reshape(3, 3)


def _max_intersection(cells: np.ndarray, need_a: int, need_b: int) -> int:
    """Maximum intersection obtainable by selecting tied pixels.

    Pixels above the threshold in both maps (cell 0,0) are in the intersection regardless.
    Then every pixel selected in a that is above the threshold in b (cell 1,0) adds 1, cell
    (0,1) does so symmetrically, and cell (1,1) adds 1 only if selected on both sides. The
    objective is concave and piecewise linear in the share spent on cell (1,1), so evaluating
    it at the breakpoints is sufficient.
    """
    n_cc = int(cells[0, 0])
    n_ct = int(cells[0, 1])
    n_tc = int(cells[1, 0])
    n_tt = int(cells[1, 1])
    hi = min(n_tt, need_a, need_b)
    cands = {0, hi, need_b - n_ct, need_a - n_tc}
    best = 0
    for x in cands:
        x = int(min(max(x, 0), hi))
        val = x + min(n_ct, need_b - x) + min(n_tc, need_a - x)
        best = max(best, val)
    return n_cc + best


def _min_intersection(cells: np.ndarray, need_a: int, need_b: int) -> int:
    """Minimum intersection obtainable by selecting tied pixels.

    Selections for a can go to harmless cells (1,2) or to the shared cell (1,1), and only
    when those run out do they fall into (1,0), where they necessarily count. The objective
    is convex and piecewise linear in the two shares spent on (1,1), so the minimum lies at a
    vertex of the arrangement of breakpoint lines, and all of them are evaluated.
    """
    n_cc = int(cells[0, 0])
    n_ct = int(cells[0, 1])
    n_tc = int(cells[1, 0])
    n_tt = int(cells[1, 1])
    n_to = int(cells[1, 2])
    n_ot = int(cells[2, 1])

    hi_p = min(n_tt, need_a)
    hi_q = min(n_tt, need_b)

    def g(p: int, r: int) -> int:
        return (
            max(0, need_a - n_to - p)
            + max(0, need_b - n_ot - r)
            + max(0, p + r - n_tt)
        )

    ps = {0, hi_p, need_a - n_to}
    rs = {0, hi_q, need_b - n_ot}
    cands: set[tuple[int, int]] = set()
    for p in ps:
        for r in rs:
            cands.add((p, r))
        cands.add((p, n_tt - p))
    for r in rs:
        cands.add((n_tt - r, r))

    best = None
    for p, r in cands:
        p = int(min(max(p, 0), hi_p))
        r = int(min(max(r, 0), hi_q))
        val = g(p, r)
        if best is None or val < best:
            best = val
    # n_tc is an upper bound for selections forced into (1,0), so the minimum is finite.
    return n_cc + int(best if best is not None else 0)


# --------------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------------


def _chunks(n: int, size: int | None = None) -> Iterator[slice]:
    # `size` is read from the module on every call, not frozen as a default, so tests can
    # lower it and exercise the chunked paths on small cases.
    step = _CHUNK if size is None else size
    for start in range(0, n, step):
        yield slice(start, min(n, start + step))


def _prepare(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr_a, arr_b, m = align(a, b, valid)
    return arr_a, arr_b, np.asarray(m).astype(bool, copy=False)


def _budget(n: int, q: float) -> int:
    if not isinstance(q, (int, float)) or isinstance(q, bool):
        raise ValueError(f"q must be a number, got {type(q).__name__}")
    if not (0.0 < float(q) <= 1.0):
        raise ValueError(f"q fuori da (0, 1]: {q}")
    if n <= 0:
        raise ValueError("no valid pixel: the comparison is not defined")
    k = int(round(n * float(q)))
    if k == 0:
        raise ValueError(
            f"q={q} on n={n} valid pixels gives k=0: at least {int(math.ceil(0.5 / q))} "
            "valid pixels are needed, or a larger q"
        )
    return k


def _values_and_budget(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    arr_a, arr_b, m = _prepare(a, b, valid)
    va, vb = arr_a[m], arr_b[m]
    n = int(va.size)
    return va, vb, n, _budget(n, q)


def _kth_largest(values: np.ndarray, k: int):
    """The k-th highest value in a 1-D array, without allocating index arrays.

    For bounded-range integers (the corpus case, uint8), this uses a cumulative histogram:
    exact, with memory independent of the number of pixels. Otherwise it uses `np.partition`,
    which copies the values once.
    """
    n = int(values.size)
    lv = _int_levels(values)
    if lv is not None and lv[1] <= (1 << 16):
        lo, length = lv
        hist = np.zeros(length, dtype=np.int64)
        for sl in _chunks(n):
            hist += np.bincount(
                (values[sl].astype(np.int64, copy=False) - lo).ravel(), minlength=length
            )
        tail = np.cumsum(hist[::-1])[::-1]  # tail[i] = number of values >= lo + i
        i = int(np.flatnonzero(tail >= k)[-1])  # highest level covering k pixels
        return values.dtype.type(lo + i)
    return np.partition(values, n - k)[n - k]


def _topk_and_threshold(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean mask of the k highest values in a 1-D array, with the threshold used.

    Tie rule, explicitly declared rather than inherited from a selection algorithm: every
    pixel strictly above the k-th threshold is included, and the remaining budget is filled
    with tied pixels taken in increasing flat-index order, i.e. from the first row downward.
    The rule is deterministic across all numpy versions and memory layouts, which
    `np.argpartition` does not guarantee, and does not allocate an index array as long as the
    map.

    This is NOT a neutral choice: it systematically favors upper rows and, above all, makes
    the selections of two maps with the same tie band coincide even when the maps share no
    information. See `tie_bounds`, which measures how much of the IoU depends on this rule.
    """
    n = int(values.size)
    out = np.zeros(n, dtype=bool)
    if k >= n:
        out[:] = True
        return out, values.min() if n else values.dtype.type(0)

    thr = _kth_largest(values, k)
    rem = k
    for sl in _chunks(n):
        gt = values[sl] > thr
        out[sl] = gt
        rem -= int(np.count_nonzero(gt))
    for sl in _chunks(n):
        if rem <= 0:
            break
        eq = values[sl] == thr
        count = int(np.count_nonzero(eq))
        if count == 0:
            continue
        if count <= rem:
            out[sl] |= eq
            rem -= count
        else:
            take = np.flatnonzero(eq)[:rem]
            window = out[sl]
            window[take] = True
            rem = 0
    return out, thr


def _topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    return _topk_and_threshold(values, k)[0]


def _delta_from_masks(ma: np.ndarray, mb: np.ndarray, q: float, n: int, k: int) -> Delta:
    inter = int(np.count_nonzero(ma & mb))
    sa = int(np.count_nonzero(ma))
    sb = int(np.count_nonzero(mb))
    union = sa + sb - inter
    iou = (inter / union) if union else 0.0
    dice = (2 * inter / (sa + sb)) if (sa + sb) else 0.0
    return Delta(q=float(q), iou=float(iou), dice=float(dice), n_valid=int(n), k=int(k))


def _midranks_from_counts(counts: np.ndarray) -> np.ndarray:
    """One-based midrank of each tie group, given counts in value order."""
    c = np.asarray(counts, dtype=np.float64)
    starts = np.cumsum(c) - c
    return starts + (c - 1.0) / 2.0 + 1.0


def _midrank(v: np.ndarray) -> np.ndarray:
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    del uniq
    return _midranks_from_counts(cnt)[inv.ravel()]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    sx = math.sqrt(float(np.dot(xm, xm)))
    sy = math.sqrt(float(np.dot(ym, ym)))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.dot(xm, ym)) / (sx * sy)


def _int_levels(v: np.ndarray) -> tuple[int, int] | None:
    if v.dtype.kind not in "bui":
        return None
    if v.size == 0:
        return None
    lo = int(v.min())
    hi = int(v.max())
    return lo, hi - lo + 1


def _spearman_1d(va: np.ndarray, vb: np.ndarray, method: str | None = None) -> float:
    n = int(va.size)
    if n < 2:
        return float("nan")
    if method is None:
        la = _int_levels(va)
        lb = _int_levels(vb)
        use_hist = la is not None and lb is not None and la[1] * lb[1] <= _HIST_MAX_BINS
        method = "hist" if use_hist else "rank"
    if method == "hist":
        return _spearman_hist(va, vb)
    if method == "rank":
        return _pearson(_midrank(va), _midrank(vb))
    raise ValueError(f"method sconosciuto: {method}")


def _spearman_hist(va: np.ndarray, vb: np.ndarray) -> float:
    la = _int_levels(va)
    lb = _int_levels(vb)
    if la is None or lb is None:
        raise ValueError("the histogram path wants non-empty integer arrays")
    a_min, a_len = la
    b_min, b_len = lb
    hist = np.zeros(a_len * b_len, dtype=np.int64)
    for sl in _chunks(int(va.size)):
        ia = va[sl].astype(np.int64, copy=False) - a_min
        ib = vb[sl].astype(np.int64, copy=False) - b_min
        hist += np.bincount((ia * b_len + ib).ravel(), minlength=a_len * b_len)
    return _spearman_from_hist(hist.reshape(a_len, b_len))


def _spearman_from_hist(h: np.ndarray) -> float:
    """Exact Spearman correlation from the joint value histogram.

    With midranks, Spearman is exactly Pearson correlation over the ranks, and the ranks
    depend only on the marginal counts: the joint histogram therefore contains everything
    needed, and this path gives the same number as the full-rank path without allocating
    arrays as long as the map.
    """
    hh = np.asarray(h, dtype=np.float64)
    total = float(hh.sum())
    if total < 2:
        return float("nan")
    wa = hh.sum(axis=1)
    wb = hh.sum(axis=0)
    ra = _midranks_from_counts(wa)
    rb = _midranks_from_counts(wb)
    mean_a = float((wa * ra).sum()) / total
    mean_b = float((wb * rb).sum()) / total
    da = ra - mean_a
    db = rb - mean_b
    var_a = float((wa * da * da).sum()) / total
    var_b = float((wb * db * db).sum()) / total
    if var_a <= 0.0 or var_b <= 0.0:
        return float("nan")
    cov = float((hh * np.outer(da, db)).sum()) / total
    return cov / math.sqrt(var_a * var_b)
