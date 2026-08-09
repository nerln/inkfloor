"""Test di `inkfloor.geometry`.

La parte logica non tocca la rete: i volumi e le mesh sono finti, costruiti in memoria, e
la relazione affine fra i due volumi e' nota per costruzione, cosi' un errore nel fit si
vede come uno scostamento da un numero che sappiamo.

I due test che riproducono le misure sul corpus vero sono marcati e saltati per default.
Per eseguirli:

    INKFLOOR_NETWORK=1 .venv/bin/python -m pytest tests/test_geometry.py -v
"""

from __future__ import annotations

import io
import json
import os

import numcodecs
import numpy as np
import pytest
import tifffile

from inkfloor import cache, geometry

requires_network = pytest.mark.skipif(
    os.environ.get("INKFLOOR_NETWORK") != "1",
    reason="tocca la rete: esegui con INKFLOOR_NETWORK=1",
)


# --------------------------------------------------------------------- finto bucket S3


class FakeBucket:
    """Un bucket in memoria, con la stessa superficie di `cache` che geometry usa."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.range_gets: list[str] = []
        self.bytes_out = 0

    def put(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(geometry.cache, "get_bytes", self.get_bytes)
        monkeypatch.setattr(geometry.cache, "get_json", self.get_json)
        monkeypatch.setattr(geometry.cache, "list_prefixes", self.list_prefixes)
        monkeypatch.setattr(geometry.cache, "fetch", self._no_fetch)

    # --- superficie di cache

    def get_bytes(self, key: str, start: int | None = None, end: int | None = None) -> bytes:
        if key not in self.blobs:
            raise cache.FetchError(f"HTTP 404 su {key}")
        data = self.blobs[key]
        if start is None:
            self.gets.append(key)
            self.bytes_out += len(data)
            return data
        self.range_gets.append(key)
        stop = len(data) if end is None else end + 1
        self.bytes_out += max(0, stop - start)
        return data[start:stop]

    def get_json(self, key: str) -> dict:
        return json.loads(self.get_bytes(key))

    def list_prefixes(self, prefix: str) -> list[str]:
        out = set()
        for key in self.blobs:
            if key.startswith(prefix):
                rest = key[len(prefix):]
                if "/" in rest:
                    out.add(prefix + rest.split("/", 1)[0] + "/")
        return sorted(out)

    def _no_fetch(self, key: str):  # pragma: no cover - solo per accorgersi se serve
        raise AssertionError(f"fetch() inattesa su {key}")


def make_volume(
    bucket: FakeBucket,
    prefix: str,
    data: np.ndarray,
    chunks: tuple[int, int, int],
    compressor: dict | None,
    *,
    skip: set[tuple[int, int, int]] = frozenset(),
) -> None:
    """Scrive `data` come zarr v2 nel finto bucket, con dimension_separator "/".

    `skip` elenca i chunk da NON scrivere: in zarr valgono `fill_value`, ed e' cosi' che il
    corpus vero rappresenta le zone mascherate.
    """
    bucket.put(
        f"{prefix}/0/.zarray",
        json.dumps(
            {
                "shape": list(data.shape),
                "chunks": list(chunks),
                "dtype": data.dtype.str,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": "/",
                "compressor": compressor,
                "zarr_format": 2,
            }
        ).encode(),
    )
    grid = [-(-s // c) for s, c in zip(data.shape, chunks)]
    codec = numcodecs.get_codec(compressor) if compressor else None
    for zi in range(grid[0]):
        for yi in range(grid[1]):
            for xi in range(grid[2]):
                if (zi, yi, xi) in skip:
                    continue
                block = np.zeros(chunks, dtype=data.dtype)
                src = data[
                    zi * chunks[0]:(zi + 1) * chunks[0],
                    yi * chunks[1]:(yi + 1) * chunks[1],
                    xi * chunks[2]:(xi + 1) * chunks[2],
                ]
                block[: src.shape[0], : src.shape[1], : src.shape[2]] = src
                raw = block.tobytes(order="C")
                bucket.put(f"{prefix}/0/{zi}/{yi}/{xi}", codec.encode(raw) if codec else raw)


BLOSC = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0}


# --------------------------------------------------------------- decodifica dei chunk


def _zarray(**over) -> geometry._ZArray:
    base = dict(
        prefix="fake/0",
        shape=(8, 8, 8),
        chunks=(4, 4, 4),
        dtype=np.dtype("|u1"),
        order="C",
        fill_value=0,
        compressor=None,
    )
    base.update(over)
    return geometry._ZArray(**base)


def test_decode_chunk_raw_bytes():
    """Compressor null: i byte grezzi vanno riletti nell'ordine in cui stanno."""
    block = (np.arange(64, dtype=np.uint8) * 3).reshape(4, 4, 4)
    got = geometry._decode_chunk(_zarray(), block.tobytes(order="C"))
    assert got.shape == (4, 4, 4)
    assert np.array_equal(got, block)


def test_decode_chunk_blosc():
    """Blosc: l'header porta i parametri, non serve rileggerli dal .zarray."""
    rng = np.random.default_rng(1)
    block = rng.integers(0, 256, size=(4, 4, 4), dtype=np.uint8)
    raw = numcodecs.get_codec(BLOSC).encode(block.tobytes(order="C"))
    assert raw != block.tobytes(order="C")
    got = geometry._decode_chunk(_zarray(compressor=BLOSC), raw)
    assert np.array_equal(got, block)


def test_decode_chunk_order_f():
    block = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    got = geometry._decode_chunk(_zarray(order="F"), block.tobytes(order="F"))
    assert np.array_equal(got, block)


def test_decode_chunk_wrong_size_raises():
    """A buffer of the wrong length must not turn into a plausible array."""
    with pytest.raises(geometry.GeometryError):
        geometry._decode_chunk(_zarray(), b"\x00" * 63)


def test_read_zarray_rejects_unsupported(monkeypatch):
    bucket = FakeBucket()
    bucket.install(monkeypatch)
    good = {
        "shape": [8, 8, 8], "chunks": [4, 4, 4], "dtype": "|u1", "fill_value": 0,
        "order": "C", "filters": None, "dimension_separator": "/",
        "compressor": None, "zarr_format": 2,
    }
    bucket.put("v/0/.zarray", json.dumps(good).encode())
    z = geometry._read_zarray("v")
    assert (z.shape, z.chunks, z.dtype, z.compressor) == ((8, 8, 8), (4, 4, 4), np.dtype("|u1"), None)
    assert z.full_chunks == (2, 2, 2)

    for bad in (
        {"zarr_format": 3},
        {"filters": [{"id": "delta"}]},
        {"dimension_separator": "."},
        {"shape": [8, 8], "chunks": [4, 4]},
        {"dtype": "<f4"},
        {"order": "X"},
    ):
        bucket.put("w/0/.zarray", json.dumps({**good, **bad}).encode())
        with pytest.raises(geometry.GeometryError):
            geometry._read_zarray("w")


# ------------------------------------------------------------------------- statistiche


@pytest.mark.parametrize("n", [1, 2, 3, 10, 11, 257])
def test_median_from_hist_matches_numpy(n):
    rng = np.random.default_rng(n)
    values = rng.integers(0, 256, size=n, dtype=np.uint8)
    counts = np.bincount(values.astype(np.int64), minlength=256)
    assert geometry._median_from_hist(counts, 0) == pytest.approx(float(np.median(values)))


def test_median_from_hist_empty_is_nan():
    assert np.isnan(geometry._median_from_hist(np.zeros(256, dtype=np.int64), 0))


def test_tail_frac():
    counts = np.zeros(256, dtype=np.int64)
    counts[199] = 3
    counts[200] = 1
    assert geometry._tail_frac(counts, 0, geometry.CLIP_CEIL) == pytest.approx(0.25)


def test_accumulator_matches_numpy_and_is_order_independent():
    """Le somme sono interi esatti: due ordini di arrivo devono dare lo stesso numero."""
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, size=5000, dtype=np.uint8)
    b = rng.integers(1, 256, size=5000, dtype=np.uint8)

    one = geometry._Accumulator(np.dtype("|u1"))
    one.add(a, b)
    many = geometry._Accumulator(np.dtype("|u1"))
    for lo in range(0, 5000, 700):
        many.add(a[lo:lo + 700], b[lo:lo + 700])
    assert one.result() == many.result()

    slope, intercept = np.polyfit(b.astype(np.float64), a.astype(np.float64), 1)
    got = one.result()
    assert got["slope"] == pytest.approx(slope, abs=1e-9)
    assert got["intercept"] == pytest.approx(intercept, abs=1e-7)
    assert got["r"] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)
    assert got["median_a"] == pytest.approx(float(np.median(a)))


# --------------------------------------------------------------------- fit su sintetico

CH = (32, 32, 32)
SHAPE = (32 * 6, 32 * 4, 32 * 4)
TRUE_SLOPE = 0.6154
TRUE_INTERCEPT = 104.32


def _affine_pair(seed: int = 3, z_shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """B casuale in [1, 240], A = TRUE_SLOPE*B + TRUE_INTERCEPT arrotondato.

    Con `z_shift = k`, A(z) e' derivato da B(z + k): il codice deve trovare `z_offset = k`.
    Il range di B tiene A dentro [105, 253], cosi' nessun valore satura in uint8 e la
    relazione resta affine su tutto il dominio.
    """
    rng = np.random.default_rng(seed)
    pad = abs(z_shift)
    tall = rng.integers(1, 241, size=(SHAPE[0] + 2 * pad, SHAPE[1], SHAPE[2]), dtype=np.uint8)
    b = tall[pad:pad + SHAPE[0]]
    src = tall[pad + z_shift:pad + z_shift + SHAPE[0]].astype(np.float64)
    a = np.rint(TRUE_SLOPE * src + TRUE_INTERCEPT).astype(np.uint8)
    return a, b


def _install_pair(
    monkeypatch,
    a: np.ndarray,
    b: np.ndarray,
    *,
    skip_b: set[tuple[int, int, int]] = frozenset(),
    b_shape: tuple[int, int, int] | None = None,
) -> FakeBucket:
    bucket = FakeBucket()
    bucket.install(monkeypatch)
    make_volume(bucket, "S/volumes/va-7.910um.zarr", a, CH, BLOSC)
    make_volume(bucket, "S/volumes/vb-7.910um.zarr", b if b_shape is None else b[: b_shape[0]],
                CH, None, skip=skip_b)
    return bucket


def test_fit_recovers_known_affine(monkeypatch):
    """Il numero che conta: se la relazione e' nota, il fit la ritrova."""
    a, b = _affine_pair()
    _install_pair(monkeypatch, a, b)
    fit = geometry.fit_intensity("S", "va", "vb", n_chunks=3, seed=0)
    assert fit is not None
    assert fit.slope == pytest.approx(TRUE_SLOPE, abs=2e-3)
    assert fit.intercept == pytest.approx(TRUE_INTERCEPT, abs=2e-2)
    assert fit.r > 0.999
    assert fit.z_offset == 0
    assert len(fit.chunks_used) == 3
    assert len(set(fit.chunks_used)) == 3
    assert fit.n_voxel == 3 * 32 ** 3


def test_fit_is_reproducible_and_seed_changes_chunks(monkeypatch):
    """Stesso seed, stessi chunk. Seed diverso, chunk diversi e stessa stima."""
    a, b = _affine_pair()
    _install_pair(monkeypatch, a, b)
    one = geometry.fit_intensity("S", "va", "vb", n_chunks=3, seed=0)
    again = geometry.fit_intensity("S", "va", "vb", n_chunks=3, seed=0)
    other = geometry.fit_intensity("S", "va", "vb", n_chunks=3, seed=11)
    assert one is not None and again is not None and other is not None
    assert one.chunks_used == again.chunks_used
    assert one.slope == again.slope
    assert other.chunks_used != one.chunks_used
    assert other.slope == pytest.approx(one.slope, abs=5e-3)


def test_fit_finds_z_offset(monkeypatch):
    """Requisito: l'allineamento in z va misurato, non assunto."""
    a, b = _affine_pair(z_shift=3)
    _install_pair(monkeypatch, a, b)
    fit = geometry.fit_intensity("S", "va", "vb", n_chunks=3, seed=0)
    assert fit is not None
    assert fit.z_offset == 3
    # Il picco e' stretto: a offset zero la correlazione deve essere quasi nulla.
    scan = dict(fit.r_by_offset)
    assert scan[3] > 0.99
    assert abs(scan[0]) < 0.1
    # E il fit deve essere quello dell'offset giusto, non quello dei voxel disallineati.
    assert fit.slope == pytest.approx(TRUE_SLOPE, abs=2e-3)
    assert fit.intercept == pytest.approx(TRUE_INTERCEPT, abs=2e-2)
    assert fit.r > 0.999


def test_fit_returns_none_on_yx_mismatch(monkeypatch):
    a, b = _affine_pair()
    bucket = FakeBucket()
    bucket.install(monkeypatch)
    make_volume(bucket, "S/volumes/va.zarr", a, CH, BLOSC)
    make_volume(bucket, "S/volumes/vb.zarr", b[:, :, : SHAPE[2] - 32], CH, None)
    assert geometry.fit_intensity("S", "va", "vb") is None


def test_fit_tolerates_different_z_extent(monkeypatch):
    """Come nel corpus: i due volumi hanno z diverso ma partono dallo stesso voxel."""
    a, b = _affine_pair()
    bucket = FakeBucket()
    bucket.install(monkeypatch)
    make_volume(bucket, "S/volumes/va.zarr", a, CH, BLOSC)
    make_volume(bucket, "S/volumes/vb.zarr", b[: SHAPE[0] - 32], CH, None)
    fit = geometry.fit_intensity("S", "va", "vb", n_chunks=2, seed=0)
    assert fit is not None
    assert fit.z_offset == 0
    assert fit.slope == pytest.approx(TRUE_SLOPE, abs=3e-3)


def test_fit_returns_none_when_volume_absent(monkeypatch):
    a, b = _affine_pair()
    _install_pair(monkeypatch, a, b)
    assert geometry.fit_intensity("S", "va", "nonesiste") is None


def test_fit_skips_chunks_missing_in_one_volume(monkeypatch):
    """Nel corpus vero ~16% dei chunk presenti in A manca in B: non vanno usati."""
    a, b = _affine_pair()
    absent = {(zi, yi, xi) for zi in range(1, 5) for yi in range(1, 3) for xi in (1,)}
    bucket = _install_pair(monkeypatch, a, b, skip_b=absent)
    fit = geometry.fit_intensity("S", "va", "vb", n_chunks=3, seed=0)
    assert fit is not None
    assert not (set(fit.chunks_used) & absent)
    assert "absent in B" in fit.note


def test_fit_probes_presence_before_downloading(monkeypatch):
    """Un chunk assente in B non deve costare il download del chunk di A."""
    a, b = _affine_pair()
    absent = {(zi, yi, xi) for zi in range(6) for yi in range(4) for xi in range(4)} - {
        (2, 2, 2), (3, 1, 2), (4, 2, 1), (1, 1, 1), (2, 1, 1), (3, 2, 2)
    }
    bucket = _install_pair(monkeypatch, a, b, skip_b=absent)
    fit = geometry.fit_intensity("S", "va", "vb", n_chunks=2, seed=0)
    assert fit is not None
    a_full_gets = [k for k in bucket.gets if "va-" in k and k.count("/") == 6]
    # Ogni chunk di A scaricato per intero deve corrispondere a un chunk poi usato o
    # scartato per sparsita', non a uno che si sapeva gia' assente in B.
    assert len(a_full_gets) <= len(fit.chunks_used) + 1
    assert len(bucket.range_gets) > len(a_full_gets)


def test_fit_returns_none_when_nothing_overlaps(monkeypatch):
    """Nessun chunk in comune: None, non una retta stimata su niente."""
    a, b = _affine_pair()
    everything = {(zi, yi, xi) for zi in range(6) for yi in range(4) for xi in range(4)}
    _install_pair(monkeypatch, a, b, skip_b=everything)
    assert geometry.fit_intensity("S", "va", "vb", n_chunks=2, seed=0) is None


def test_fit_clip_and_median_are_on_the_common_mask(monkeypatch):
    """Mediane e frazioni al tetto sono calcolate sui voxel validi in entrambi i volumi."""
    a, b = _affine_pair()
    _install_pair(monkeypatch, a, b)
    fit = geometry.fit_intensity("S", "va", "vb", n_chunks=2, seed=0)
    assert fit is not None
    zi, yi, xi = fit.chunks_used[0]
    sl = (slice(zi * 32, zi * 32 + 32), slice(yi * 32, yi * 32 + 32), slice(xi * 32, xi * 32 + 32))
    mask = (a[sl] > 0) & (b[sl] > 0)
    # Un solo chunk non basta per l'aggregato, ma l'ordine di grandezza deve tornare.
    assert fit.median_a == pytest.approx(float(np.median(a[sl][mask])), abs=2.0)
    assert fit.clip_frac_a == pytest.approx(float((a[sl][mask] >= 200).mean()), abs=0.02)
    assert fit.clip_frac_b == pytest.approx(float((b[sl][mask] >= 200).mean()), abs=0.02)


# ----------------------------------------------------------------------------- mesh


def _tif(array: np.ndarray, compression) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, array, compression=compression)
    return buf.getvalue()


def _install_mesh(
    monkeypatch,
    arr_a: dict[str, np.ndarray],
    arr_b: dict[str, np.ndarray],
    *,
    meta_a: dict | None = None,
    meta_b: dict | None = None,
    drop: set[str] = frozenset(),
) -> tuple[FakeBucket, str]:
    seg = "P/segments/SEG"
    da = f"{seg}/mesh/T-on-VA-7.91um.tifxyz"
    db = f"{seg}/mesh/T-on-VB-7.91um.tifxyz"
    bucket = FakeBucket()
    bucket.blobs[f"{seg}/mesh/intermediate/tifxyz_flattened/x.tif"] = b"rumore"
    files: dict[str, bytes] = {}
    for d, arrays, comp in ((da, arr_a, "lzw"), (db, arr_b, None)):
        for ch, a in arrays.items():
            files[f"{d}/{ch}.tif"] = _tif(a, comp)
    for k, v in files.items():
        if k not in drop:
            bucket.put(k, v)
    bucket.put(f"{da}/meta.json", json.dumps(meta_a or {"format": "tifxyz"}).encode())
    bucket.put(f"{db}/meta.json", json.dumps(meta_b or {"format": "tifxyz"}).encode())

    paths = {k: v for k, v in bucket.blobs.items()}

    def fake_fetch(key: str):
        if key not in paths:
            raise cache.FetchError(f"HTTP 404 su {key}")
        return io.BytesIO(paths[key])

    monkeypatch.setattr(geometry.cache, "get_bytes", bucket.get_bytes)
    monkeypatch.setattr(geometry.cache, "get_json", bucket.get_json)
    monkeypatch.setattr(geometry.cache, "list_prefixes", bucket.list_prefixes)
    monkeypatch.setattr(geometry.cache, "fetch", fake_fetch)
    return bucket, seg


def test_compare_meshes_compares_arrays_not_bytes(monkeypatch):
    """La trappola: stessi valori, file di dimensione molto diversa."""
    rng = np.random.default_rng(4)
    arrays = {ch: rng.random((40, 60)).astype(np.float32) * 1000 for ch in "xyz"}
    bucket, seg = _install_mesh(monkeypatch, arrays, arrays)
    for ch in "xyz":
        lzw = bucket.blobs[f"{seg}/mesh/T-on-VA-7.91um.tifxyz/{ch}.tif"]
        raw = bucket.blobs[f"{seg}/mesh/T-on-VB-7.91um.tifxyz/{ch}.tif"]
        assert lzw != raw

    check = geometry.compare_meshes(seg, "VA", "VB")
    assert check is not None
    assert check.identical is True
    assert check.shape_a == check.shape_b == (40, 60)
    assert check.max_abs_diff == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_compare_meshes_detects_one_changed_voxel(monkeypatch):
    rng = np.random.default_rng(5)
    a = {ch: rng.random((20, 30)).astype(np.float32) for ch in "xyz"}
    b = {ch: v.copy() for ch, v in a.items()}
    b["y"][7, 9] += 0.25
    _install_mesh(monkeypatch, a, b)
    check = geometry.compare_meshes("P/segments/SEG", "VA", "VB")
    assert check is not None
    assert check.identical is False
    assert check.max_abs_diff["x"] == 0.0
    assert check.max_abs_diff["y"] == pytest.approx(0.25, abs=1e-6)


def test_compare_meshes_shape_mismatch(monkeypatch):
    a = {ch: np.zeros((20, 30), np.float32) for ch in "xyz"}
    b = {ch: np.zeros((20, 31), np.float32) for ch in "xyz"}
    _install_mesh(monkeypatch, a, b)
    check = geometry.compare_meshes("P/segments/SEG", "VA", "VB")
    assert check is not None
    assert check.identical is False
    assert check.shape_a == (20, 30) and check.shape_b == (20, 31)
    assert all(np.isinf(v) for v in check.max_abs_diff.values())
    assert "forme diverse" in check.note


def test_compare_meshes_none_when_channel_missing(monkeypatch):
    a = {ch: np.zeros((8, 8), np.float32) for ch in "xyz"}
    _install_mesh(monkeypatch, a, a, drop={"P/segments/SEG/mesh/T-on-VB-7.91um.tifxyz/z.tif"})
    assert geometry.compare_meshes("P/segments/SEG", "VA", "VB") is None


def test_compare_meshes_none_when_derivation_missing(monkeypatch):
    a = {ch: np.zeros((8, 8), np.float32) for ch in "xyz"}
    _install_mesh(monkeypatch, a, a)
    assert geometry.compare_meshes("P/segments/SEG", "VA", "VC") is None


def test_compare_meshes_meta_difference_does_not_flip_identical(monkeypatch):
    """The sidecar may differ on uuid or scale precision: that is not the geometry."""
    rng = np.random.default_rng(6)
    arrays = {ch: rng.random((10, 10)).astype(np.float32) for ch in "xyz"}
    _install_mesh(
        monkeypatch, arrays, arrays,
        meta_a={"uuid": "output_tifxyz", "scale": [0.05, 0.05]},
        meta_b={"uuid": "out", "scale": [0.05000000074505806, 0.05000000074505806], "area_vx2": 1.0},
    )
    check = geometry.compare_meshes("P/segments/SEG", "VA", "VB")
    assert check is not None
    assert check.identical is True
    assert "meta.json differs on" in check.note
    assert "uuid" in check.note and "area_vx2" in check.note


def test_compare_meshes_ignores_intermediate_dirs(monkeypatch):
    """Sotto mesh/ ci sono anche le tifxyz intermedie: non sono derivazioni di un volume."""
    arrays = {ch: np.zeros((6, 6), np.float32) for ch in "xyz"}
    _install_mesh(monkeypatch, arrays, arrays)
    assert geometry._find_tifxyz("P/segments/SEG", "VA").endswith("T-on-VA-7.91um.tifxyz")
    assert geometry._find_tifxyz("P/segments/SEG", "tifxyz_flattened") is None


# ------------------------------------------------------------------- corpus, con rete

SEGMENT = "PHerc0172/segments/20251107110950-w064_20251107110950052_flatboi"
VOL_A = "20241024131838"
VOL_B = "20241024131839"


@requires_network
def test_corpus_mesh_is_identical():
    """Misura 1: shape (671, 747) e maxabsdiff 0 su x, y, z."""
    check = geometry.compare_meshes(SEGMENT, VOL_A, VOL_B)
    assert check is not None
    assert check.shape_a == check.shape_b == (671, 747)
    assert check.max_abs_diff == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert check.identical is True


@requires_network
def test_corpus_intensity_fit():
    """Misura 2: A = 0.6154*B + 104.32 con r = 0.9999, e allineamento in z a offset 0."""
    fit = geometry.fit_intensity("PHerc0172", VOL_A, VOL_B, n_chunks=4, seed=0)
    assert fit is not None
    assert fit.z_offset == 0
    assert fit.r > 0.999
    assert fit.slope == pytest.approx(0.6154, abs=2e-3)
    assert fit.intercept == pytest.approx(104.32, abs=0.2)
    assert 100 < fit.median_a < 200
    assert 30 < fit.median_b < 120
    assert fit.clip_frac_a > fit.clip_frac_b
    assert len(fit.chunks_used) == 4
    scan = dict(fit.r_by_offset)
    assert scan[0] > 0.99
    assert scan[1] < 0.99   # il picco e' largo un voxel
