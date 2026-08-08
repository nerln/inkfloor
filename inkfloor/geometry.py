"""Exclusion of geometric and radiometric confounders.

The reproducibility floor is meaningful only if the two derivations being compared look at
the same surface in the same voxels. This module measures the two things that can turn the
floor into an artifact:

* `compare_meshes`: is the flattened mesh the same in both derivations?
* `fit_intensity`: are the two volumes aligned, and how does one transform into the other?

No function in this module prints, trains, or corrects anything. It only measures and
returns results.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numcodecs
import numpy as np
import tifffile

from . import cache

# The ceiling at which the ink detection pipeline clips intensities before feeding the
# volume to the model. A high fraction of voxels at the ceiling means the model sees a flat
# surface, and two volumes with very different fractions are not comparable even if the
# affine relationship between them is perfect.
CLIP_CEIL = 200

# Multiscale level containing the full-resolution voxels.
LEVEL = "0"

CHANNELS = ("x", "y", "z")

# The z-offset scan is dense: every integer offset between -chunk and +chunk. A logarithmic
# scan seemed cheaper but is wrong, because the correlation peak is only one voxel wide
# (r drops from 0.9999 to 0.90 with a shift of 1), and a true maximum at offset 3 would
# remain invisible between the samples at 2 and 4.
#
# Spatial subsampling used ONLY for the scan: with ~10^5 voxels the difference between
# r=0.9999 and r=0.90 is beyond dispute, and the 257 offsets take less than a tenth of a
# second, which is nothing compared with the download.
_SCAN_STRIDE = 4

# Which offsets go into `IntensityFit.r_by_offset`: the maximum and its neighbors, plus a
# logarithmic-scale shoulder. Reporting all 257 would clutter the report without adding
# anything for someone who only needs to decide whether to trust the fit.
_REPORT_NEIGHBOURS = 2
_REPORT_SHOULDER = (0, 1, 2, 4, 8, 16, 32, 64, 128)


class GeometryError(RuntimeError):
    """Data that we cannot read without resorting to guesswork.

    Unlike `None`: `None` means "measurement not applicable to this pair" (mesh not
    published, volumes not corresponding). `GeometryError` means "the format is not one
    this module knows how to read," and must be fixed in the code, not ignored.
    """


# --------------------------------------------------------------------------- mesh


@dataclass(frozen=True)
class MeshCheck:
    identical: bool
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    max_abs_diff: dict[str, float]   # per channel: "x", "y", "z"
    note: str                        # why they are not identical, if they are not


def compare_meshes(segment_prefix: str, vol_a: str, vol_b: str) -> MeshCheck | None:
    """Compare the tifxyz data from both derivations. None if either is not published.

    Does NOT compare file bytes: compression gives files different sizes despite identical
    content (in the corpus, one derivation writes LZW and the other writes uncompressed TIFF,
    with a ratio of nearly 2:1 for the same matrix). It compares the decoded arrays.

    Does NOT look at `meta.json` to decide `identical`: that sidecar contains provenance
    fields (uuid, area, scale precision) that differ even when the coordinates are
    bit-identical. Sidecar differences go into `note`, where they provide information
    without skewing the verdict.

    Does NOT realign or resample: if the two matrices have different shapes, no voxel-wise
    comparison is attempted and `max_abs_diff` is `inf` on every channel.

    `segment_prefix` is the segment's S3 prefix, for example
    "PHerc0172/segments/20251107110950-w064_20251107110950052_flatboi".
    `vol_a` / `vol_b` are the volume IDs, for example "20241024131838".
    """
    prefix = segment_prefix.rstrip("/")
    dir_a = _find_tifxyz(prefix, vol_a)
    dir_b = _find_tifxyz(prefix, vol_b)
    if dir_a is None or dir_b is None:
        return None

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch in CHANNELS:
        a = _read_tif(f"{dir_a}/{ch}.tif")
        b = _read_tif(f"{dir_b}/{ch}.tif")
        if a is None or b is None:
            return None
        arrays[ch] = (a, b)

    shape_a = _shape2(arrays["x"][0])
    shape_b = _shape2(arrays["x"][1])
    notes: list[str] = []

    ragged_a = {ch for ch, (a, _) in arrays.items() if _shape2(a) != shape_a}
    ragged_b = {ch for ch, (_, b) in arrays.items() if _shape2(b) != shape_b}
    if ragged_a or ragged_b:
        notes.append(
            "canali di forma diversa dentro la stessa tifxyz: "
            f"a={sorted(ragged_a)} b={sorted(ragged_b)}"
        )

    if shape_a != shape_b:
        diffs = {ch: float("inf") for ch in CHANNELS}
        identical = False
        notes.append(f"forme diverse {shape_a} contro {shape_b}: confronto per voxel non tentato")
    else:
        diffs = {}
        for ch, (a, b) in arrays.items():
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            diffs[ch] = float(d.max()) if d.size else float("nan")
        identical = all(np.array_equal(a, b) for a, b in arrays.values())
        if identical:
            notes.append(f"coordinates bit-identical on x, y, z ({shape_a[0]}x{shape_a[1]})")
        else:
            worst = max(diffs, key=lambda c: (math.isnan(diffs[c]), diffs[c]))
            notes.append(f"same shape but differing coordinates, worst channel {worst}={diffs[worst]:.6g}")

    meta_note = _meta_note(dir_a, dir_b)
    if meta_note:
        notes.append(meta_note)
    notes.append(f"a={dir_a.rsplit('/', 1)[-1]} b={dir_b.rsplit('/', 1)[-1]}")

    return MeshCheck(
        identical=identical,
        shape_a=shape_a,
        shape_b=shape_b,
        max_abs_diff=diffs,
        note="; ".join(notes),
    )


def _find_tifxyz(segment_prefix: str, vol: str) -> str | None:
    """Prefix of the tifxyz derived from `vol`, without a trailing slash. None if absent.

    Does NOT guess: it requires the "-on-<vol>-" marker in the directory name, which is how
    the corpus records which volume the mesh was flattened on. If multiple directories
    match, it takes the first in lexicographic order so that two runs give the same result.
    """
    try:
        dirs = cache.list_prefixes(f"{segment_prefix}/mesh/")
    except cache.FetchError:
        return None
    hits = sorted(
        d.rstrip("/") for d in dirs
        if d.rstrip("/").endswith(".tifxyz") and f"-on-{vol}-" in d.rsplit("/", 2)[-2]
    )
    return hits[0] if hits else None


def _read_tif(key: str) -> np.ndarray | None:
    """The decoded array, or None if the file is not published."""
    try:
        path = cache.fetch(key)
    except cache.FetchError:
        return None
    return np.asarray(tifffile.imread(path))


def _shape2(a: np.ndarray) -> tuple[int, int]:
    if a.ndim != 2:
        raise GeometryError(f"tifxyz con {a.ndim} dimensioni, attese 2")
    return (int(a.shape[0]), int(a.shape[1]))


def _meta_note(dir_a: str, dir_b: str) -> str:
    """Summary of differences between the two meta.json files. Empty if unreadable."""
    try:
        ma = cache.get_json(f"{dir_a}/meta.json")
        mb = cache.get_json(f"{dir_b}/meta.json")
    except (cache.FetchError, json.JSONDecodeError):
        return ""
    keys = sorted(set(ma) | set(mb))
    differing = [k for k in keys if ma.get(k) != mb.get(k)]
    if not differing:
        return "meta.json identici"
    return "meta.json differs on " + ", ".join(differing)


# --------------------------------------------------------------------------- zarr v2


@dataclass(frozen=True)
class _ZArray:
    """The bare minimum of a zarr v2 needed to read its chunks manually."""

    prefix: str                  # ".../<vol>.zarr/0", without a trailing slash
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    order: str
    fill_value: int
    compressor: dict | None

    @property
    def full_chunks(self) -> tuple[int, ...]:
        """Number of chunks entirely within the array, per axis."""
        return tuple(s // c for s, c in zip(self.shape, self.chunks))


def _read_zarray(vol_prefix: str, level: str = LEVEL) -> _ZArray:
    """Read `<vol>/<level>/.zarray`.

    Does NOT support zarr v3 (`zarr.json`), filters, or arrays with more or fewer than 3
    axes: in those cases it raises GeometryError rather than reading arbitrary bytes.
    """
    prefix = f"{vol_prefix.rstrip('/')}/{level}"
    meta = cache.get_json(f"{prefix}/.zarray")
    if int(meta.get("zarr_format", 0)) != 2:
        raise GeometryError(f"unsupported zarr_format {meta.get('zarr_format')} in {prefix}")
    if meta.get("filters"):
        raise GeometryError(f"unsupported filters in {prefix}: {meta['filters']}")
    if meta.get("dimension_separator", ".") != "/":
        raise GeometryError(f"dimension_separator {meta.get('dimension_separator')!r} in {prefix}")
    shape = tuple(int(v) for v in meta["shape"])
    chunks = tuple(int(v) for v in meta["chunks"])
    if len(shape) != 3 or len(chunks) != 3:
        raise GeometryError(f"array a {len(shape)} assi in {prefix}, attesi 3")
    order = str(meta.get("order", "C"))
    if order not in ("C", "F"):
        raise GeometryError(f"order {order!r} in {prefix}")
    dtype = np.dtype(meta["dtype"])
    if dtype.kind not in ("u", "i") or dtype.itemsize > 2:
        raise GeometryError(
            f"dtype {dtype} in {prefix}: this module accumulates exact histograms and "
            "supporta solo interi da 1 o 2 byte"
        )
    return _ZArray(
        prefix=prefix,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        order=order,
        fill_value=int(meta.get("fill_value") or 0),
        compressor=meta.get("compressor"),
    )


def _decode_chunk(z: _ZArray, raw: bytes) -> np.ndarray:
    """A decoded chunk with shape `z.chunks`.

    Does NOT apply filters and does NOT crop the edge: in zarr v2 even a chunk straddling
    the edge is stored in full, and the tail beyond `shape` remains as the array's producer
    wrote it.
    """
    comp = z.compressor
    if comp is None:
        buf: object = raw
    elif comp.get("id") == "blosc":
        # The blosc header carries clevel, shuffle, and cname, so they need not be reread.
        buf = numcodecs.Blosc().decode(raw)
    else:
        buf = numcodecs.get_codec(comp).decode(raw)
    if isinstance(buf, np.ndarray):
        flat = buf.view(z.dtype).reshape(-1)
    else:
        flat = np.frombuffer(buf, dtype=z.dtype)
    expected = int(np.prod(z.chunks))
    if flat.size != expected:
        raise GeometryError(
            f"chunk di {flat.size} elementi in {z.prefix}, attesi {expected}: "
            "compressore o dtype letti male"
        )
    return flat.reshape(z.chunks, order=z.order)


def _read_chunk(
    z: _ZArray,
    idx: tuple[int, int, int],
    memo: dict[tuple[str, int, int, int], np.ndarray | None],
) -> np.ndarray | None:
    """The `idx` chunk, or None if not stored (in zarr = all `fill_value`).

    Does NOT distinguish a missing chunk from one full of `fill_value`: for this module's
    purpose, where `fill_value` is 0 and valid voxels are > 0, they are the same thing.
    """
    key = (z.prefix, *idx)
    if key in memo:
        return memo[key]
    out: np.ndarray | None
    try:
        raw = cache.get_bytes("/".join((z.prefix, *(str(i) for i in idx))))
    except cache.FetchError as e:
        if not _is_missing(e):
            raise
        out = None
    else:
        out = _decode_chunk(z, raw)
    memo[key] = out
    return out


def _chunk_present(z: _ZArray, idx: tuple[int, int, int]) -> bool:
    """Probe whether the chunk exists with a one-byte Range, without downloading it.

    This discards empty candidates at almost no cost: in the corpus, about 40% of grid
    positions are not stored, and downloading 2 MiB to discover that would be the
    function's dominant source of waste.
    """
    key = "/".join((z.prefix, *(str(i) for i in idx)))
    try:
        cache.get_bytes(key, 0, 0)
    except cache.FetchError as e:
        if _is_missing(e):
            return False
        raise
    return True


def _is_missing(e: cache.FetchError) -> bool:
    return "HTTP 404" in str(e)


def _window_z(
    z: _ZArray,
    idx: tuple[int, int, int],
    dz: int,
    memo: dict[tuple[str, int, int, int], np.ndarray | None],
) -> np.ndarray:
    """The block of `z` corresponding to chunk `idx`, shifted by `dz` voxels along z.

    When `dz` is nonzero, the neighboring chunk must be read: the window straddles two
    chunks. Parts outside the array and missing chunks have the value `fill_value`.

    Does NOT shift in y or x: lateral misalignment, if present, is visible in the ink maps
    and is sought by `metrics.best_shift_iou`, not here.
    """
    cz = z.chunks[0]
    zi, yi, xi = idx
    g0 = zi * cz + dz
    out = np.full(z.chunks, z.fill_value, dtype=z.dtype)
    first = g0 // cz
    last = (g0 + cz - 1) // cz
    for ci in range(first, last + 1):
        lo = max(g0, ci * cz)
        hi = min(g0 + cz, (ci + 1) * cz, z.shape[0])
        if hi <= lo or ci < 0 or ci * cz >= z.shape[0]:
            continue
        src = _read_chunk(z, (ci, yi, xi), memo)
        if src is None:
            continue
        out[lo - g0:hi - g0] = src[lo - ci * cz:hi - ci * cz]
    return out


# --------------------------------------------------------------------------- intensities


@dataclass(frozen=True)
class IntensityFit:
    slope: float
    intercept: float
    r: float
    n_voxel: int
    median_a: float
    median_b: float
    clip_frac_a: float    # fraction of voxels >= 200, the pipeline's clipping ceiling
    clip_frac_b: float
    chunks_used: list[tuple[int, int, int]]
    # Fields appended at the end, all with defaults: calls written against the contract's
    # signature remain valid, and report readers also have proof that z alignment was
    # measured rather than assumed.
    z_offset: int = 0
    r_by_offset: tuple[tuple[int, float], ...] = ()
    note: str = ""


def fit_intensity(
    sample: str,
    vol_a: str,
    vol_b: str,
    n_chunks: int = 5,
    seed: int = 0,
    min_nonzero: float = 0.5,
    max_tries: int = 0,
) -> IntensityFit | None:
    """Sample corresponding chunks from both volumes and estimate A = slope*B + intercept.

    Reads zarr v2 manually: `.zarray` for shape/chunks/compressor, then chunks over HTTP.
    Handles a null compressor (raw bytes) and blosc (numcodecs.Blosc). None if the two
    volumes do not have the same shape in y and x.

    Does NOT assume that the two volumes are aligned in z: it finds the z offset that
    maximizes correlation and reports both `z_offset` and `r_by_offset`, so readers can
    reject the fit if r is low or if the maximum is not at zero.

    Does NOT use a hand-picked chunk list: it draws positions using `seed`, discards those
    where fewer than a `min_nonzero` fraction of either volume's voxels are > 0, and reports
    in `chunks_used` exactly those on which it measured. Changing `seed` changes the chunks
    and must not change the estimate: that is the check.

    This is NOT robust regression and does NOT exclude voxels at the clipping ceiling: the
    estimate is ordinary least squares of A on B over voxels where both are > 0, and the
    fraction at the ceiling is reported separately because that is precisely what makes
    the fitted line optimistic.

    Does NOT sample edge chunks: candidates are only chunks entirely within both arrays, so
    the padding tail beyond `shape` never enters the estimate.

    Does NOT keep more than about `2 * n_chunks` chunks in memory at once (with the corpus's
    128^3 uint8 chunks, about forty MB with the default values).

    Also returns None when either volume is not published under `<sample>/volumes/`, the
    chunk dimensions differ, or no sampled position has data in both volumes.
    """
    prefix_a = _resolve_volume(sample, vol_a)
    prefix_b = _resolve_volume(sample, vol_b)
    if prefix_a is None or prefix_b is None:
        return None

    za = _read_zarray(prefix_a)
    zb = _read_zarray(prefix_b)
    if za.shape[1:] != zb.shape[1:]:
        return None
    if za.chunks != zb.chunks:
        return None
    if za.dtype != zb.dtype:
        return None

    picks, sel_note = _select_chunks(za, zb, n_chunks, seed, min_nonzero, max_tries)
    if not picks:
        return None

    # Chunks already downloaded during selection go into the memo, so the alignment probe
    # pays only for the two z neighbors.
    memo: dict[tuple[str, int, int, int], np.ndarray | None] = {}
    for idx, a_chunk, b_chunk in picks:
        memo[(za.prefix, *idx)] = a_chunk
        memo[(zb.prefix, *idx)] = b_chunk
    scan = _scan_z_offsets(za, zb, picks[0][0], memo)
    z_offset = max(scan, key=lambda d: (-math.inf if math.isnan(scan[d]) else scan[d], -abs(d)))

    memo.clear()   # the probe's two neighbors are no longer needed

    acc = _Accumulator(za.dtype)
    used: list[tuple[int, int, int]] = []
    for idx, a_chunk, b_chunk in picks:
        if z_offset == 0:
            b = b_chunk
        else:
            # The local memo dies at the end of the iteration: the shifted window straddles
            # two chunks, and the first is the one already in hand.
            b = _window_z(zb, idx, z_offset, {(zb.prefix, *idx): b_chunk})
        mask = (a_chunk > 0) & (b > 0)
        if not mask.any():
            continue
        acc.add(a_chunk[mask], b[mask])
        used.append(idx)

    if acc.n == 0:
        return None

    stats = acc.result()
    peak = scan.get(z_offset, float("nan"))
    reported = _report_offsets(scan, z_offset)
    return IntensityFit(
        slope=stats["slope"],
        intercept=stats["intercept"],
        r=stats["r"],
        n_voxel=acc.n,
        median_a=stats["median_a"],
        median_b=stats["median_b"],
        clip_frac_a=stats["clip_frac_a"],
        clip_frac_b=stats["clip_frac_b"],
        chunks_used=used,
        z_offset=z_offset,
        r_by_offset=reported,
        note=(
            f"{sel_note}; scansione z densa su {len(scan)} offset, massimo a {z_offset:+d} "
            f"con r={peak:.5f} (un chunk sonda, sottocampionato 1/{_SCAN_STRIDE} in y,x)"
        ),
    )


def _resolve_volume(sample: str, vol: str) -> str | None:
    """Map a volume ID to its zarr S3 prefix. None if it is not published.

    Does NOT download any volume data: just one LIST of the children of `<sample>/volumes/`.
    """
    if vol.endswith(".zarr") or "/" in vol:
        return vol.rstrip("/")
    try:
        dirs = cache.list_prefixes(f"{sample}/volumes/")
    except cache.FetchError:
        return None
    hits = sorted(
        d.rstrip("/") for d in dirs
        if d.rstrip("/").rsplit("/", 1)[-1].startswith(vol)
    )
    return hits[0] if hits else None


def _select_chunks(
    za: _ZArray,
    zb: _ZArray,
    n_chunks: int,
    seed: int,
    min_nonzero: float,
    max_tries: int,
) -> tuple[list[tuple[tuple[int, int, int], np.ndarray, np.ndarray]], str]:
    """Choose reproducible positions with data in both volumes.

    Does NOT seek the "best" chunks or look at where the mesh runs: it uniformly samples
    the common grid with a seed, probes for presence with a one-byte Range, and accepts only
    positions where at least a `min_nonzero` fraction of each volume's voxels are > 0. It
    also returns the arrays, because downloading them again during estimation would double
    the traffic.
    """
    # Candidates exclude the first and last full chunk on each axis: the first and last in
    # z are needed as neighbors for the offset scan, and the edge brings in the padding.
    hi = [min(a, b) - 1 for a, b in zip(za.full_chunks, zb.full_chunks)]
    if any(h <= 1 for h in hi):
        return [], "common grid too small to sample interior chunks"

    rnd = random.Random(seed)
    budget = max_tries if max_tries > 0 else 8 * n_chunks
    seen: set[tuple[int, int, int]] = set()
    picks: list[tuple[tuple[int, int, int], np.ndarray, np.ndarray]] = []
    tried = absent_a = absent_b = sparse = 0

    while len(picks) < n_chunks and tried < budget:
        idx = (rnd.randrange(1, hi[0]), rnd.randrange(1, hi[1]), rnd.randrange(1, hi[2]))
        if idx in seen:
            continue
        seen.add(idx)
        tried += 1
        if not _chunk_present(za, idx):
            absent_a += 1
            continue
        if not _chunk_present(zb, idx):
            absent_b += 1
            continue
        memo: dict[tuple[str, int, int, int], np.ndarray | None] = {}
        a = _read_chunk(za, idx, memo)
        b = _read_chunk(zb, idx, memo)
        if a is None or b is None:
            absent_a += int(a is None)
            absent_b += int(b is None)
            continue
        if float((a > 0).mean()) < min_nonzero or float((b > 0).mean()) < min_nonzero:
            sparse += 1
            continue
        picks.append((idx, a, b))

    note = (
        f"{len(picks)}/{n_chunks} chunks accepted in {tried} tries "
        f"(seed={seed}, min_nonzero={min_nonzero}); rejected: "
        f"{absent_a} absent in A, {absent_b} present in A but absent in B, {sparse} too empty"
    )
    return picks, note


def _scan_z_offsets(
    za: _ZArray,
    zb: _ZArray,
    idx: tuple[int, int, int],
    memo: dict[tuple[str, int, int, int], np.ndarray | None],
) -> dict[int, float]:
    """Correlation between the volumes at each integer z offset, on one probe chunk.

    Just one probe chunk and its two z neighbors: the probe establishes the offset, and
    repeating it on every chunk would multiply traffic to measure the same thing.

    This is NOT the fit correlation: it is calculated on one chunk and subsampled by
    `_SCAN_STRIDE` in y and x. It selects the offset and shows how narrow the peak is; it
    does not quantify agreement.

    Does NOT search offsets larger than one chunk: beyond that, the two volumes are not the
    same realigned acquisition but two different things, and the fit must be rejected, not
    shifted.
    """
    a_full = _read_chunk(za, idx, memo)
    if a_full is None:
        return {0: float("nan")}
    a = a_full[:, ::_SCAN_STRIDE, ::_SCAN_STRIDE].astype(np.float64)

    cz = zb.chunks[0]
    slab = np.concatenate(
        [_window_z(zb, (idx[0] + d, idx[1], idx[2]), 0, memo) for d in (-1, 0, 1)], axis=0
    )[:, ::_SCAN_STRIDE, ::_SCAN_STRIDE].astype(np.float64)

    scan: dict[int, float] = {}
    for dz in range(-cz, cz + 1):
        b = slab[cz + dz:cz + dz + cz]
        mask = (a > 0) & (b > 0)
        if int(mask.sum()) < 100:
            scan[dz] = float("nan")
            continue
        av, bv = a[mask], b[mask]
        if av.std() == 0 or bv.std() == 0:
            scan[dz] = float("nan")
            continue
        scan[dz] = float(np.corrcoef(av, bv)[0, 1])
    return scan


def _report_offsets(scan: dict[int, float], best: int) -> tuple[tuple[int, float], ...]:
    """Subset of the scan included in the report: the peak and its shoulder."""
    wanted = {best + d for d in range(-_REPORT_NEIGHBOURS, _REPORT_NEIGHBOURS + 1)}
    for s in _REPORT_SHOULDER:
        wanted |= {s, -s}
    return tuple(sorted((d, scan[d]) for d in wanted if d in scan))


class _Accumulator:
    """Exact sums and marginal histograms over valid voxels, chunk by chunk.

    The sums remain Python integers, so slope, intercept, and r are computed from exact
    quantities and do not depend on the order in which chunks arrived. The marginal
    histograms give exact medians and ceiling fractions with constant memory: no value
    array remains alive after the chunk that produced it.

    Does NOT keep the joint histogram: that would be needed for robust regression, which
    this module does not perform.
    """

    def __init__(self, dtype: np.dtype) -> None:
        info = np.iinfo(dtype)
        self._vmin = int(info.min)
        self._nbins = int(info.max) - int(info.min) + 1
        self.n = 0
        self._sa = self._sb = self._saa = self._sbb = self._sab = 0
        self._ha = np.zeros(self._nbins, dtype=np.int64)
        self._hb = np.zeros(self._nbins, dtype=np.int64)

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        ai = a.astype(np.int64, copy=False).ravel()
        bi = b.astype(np.int64, copy=False).ravel()
        self.n += ai.size
        self._sa += int(ai.sum())
        self._sb += int(bi.sum())
        self._saa += int(np.dot(ai, ai))
        self._sbb += int(np.dot(bi, bi))
        self._sab += int(np.dot(ai, bi))
        self._ha += np.bincount(ai - self._vmin, minlength=self._nbins)
        self._hb += np.bincount(bi - self._vmin, minlength=self._nbins)

    def result(self) -> dict[str, float]:
        n = self.n
        cov = n * self._sab - self._sa * self._sb
        var_b = n * self._sbb - self._sb * self._sb
        var_a = n * self._saa - self._sa * self._sa
        slope = cov / var_b if var_b else float("nan")
        intercept = (self._sa - slope * self._sb) / n if var_b else float("nan")
        denom = math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0.0
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r": float(cov / denom) if denom else float("nan"),
            "median_a": _median_from_hist(self._ha, self._vmin),
            "median_b": _median_from_hist(self._hb, self._vmin),
            "clip_frac_a": _tail_frac(self._ha, self._vmin, CLIP_CEIL),
            "clip_frac_b": _tail_frac(self._hb, self._vmin, CLIP_CEIL),
        }


def _median_from_hist(counts: np.ndarray, vmin: int) -> float:
    """Exact median from a per-value histogram, using the `numpy.median` convention.

    Does NOT interpolate: values are integers, so for even n the median is the mean of the
    two central elements and nothing else.
    """
    n = int(counts.sum())
    if n == 0:
        return float("nan")
    cum = np.cumsum(counts)
    lo = int(np.searchsorted(cum, (n - 1) // 2, side="right")) + vmin
    hi = int(np.searchsorted(cum, n // 2, side="right")) + vmin
    return (lo + hi) / 2.0


def _tail_frac(counts: np.ndarray, vmin: int, threshold: int) -> float:
    """Fraction of voxels with value >= `threshold`."""
    n = int(counts.sum())
    if n == 0:
        return float("nan")
    start = max(0, threshold - vmin)
    return float(int(counts[start:].sum()) / n)
