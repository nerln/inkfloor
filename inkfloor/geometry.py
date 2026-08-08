"""Esclusione dei confondenti geometrici e radiometrici.

Il pavimento di riproducibilita' ha senso solo se le due derivazioni che confrontiamo
guardano la stessa superficie negli stessi voxel. Qui si misurano le due cose che possono
rendere il pavimento un artefatto:

* `compare_meshes`: la mesh appiattita e' la stessa nelle due derivazioni?
* `fit_intensity`: i due volumi sono allineati, e come si trasformano l'uno nell'altro?

Nessuna funzione di questo modulo stampa, addestra o corregge qualcosa. Misura e restituisce.
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

# Il tetto a cui la pipeline di ink detection satura le intensita' prima di dare in pasto
# il volume al modello. Una frazione alta di voxel al tetto significa che il modello vede
# una superficie piatta, e due volumi con frazioni molto diverse non sono confrontabili
# anche se la relazione affine fra loro e' perfetta.
CLIP_CEIL = 200

# Livello del multiscale che contiene i voxel a piena risoluzione.
LEVEL = "0"

CHANNELS = ("x", "y", "z")

# La scansione degli offset in z e' densa: ogni offset intero fra -chunk e +chunk. Una
# scansione a scala logaritmica sembrava piu' economica ma e' sbagliata, perche' il picco di
# correlazione e' larghissimo un voxel (r crolla da 0.9999 a 0.90 spostandosi di 1) e un
# massimo vero a offset 3 resterebbe invisibile fra i campioni a 2 e a 4.
#
# Sottocampionamento spaziale usato SOLO per la scansione: con ~10^5 voxel la differenza fra
# r=0.9999 e r=0.90 e' fuori discussione, e i 257 offset costano meno di un decimo di
# secondo, cioe' niente rispetto al download.
_SCAN_STRIDE = 4

# Quali offset finiscono in `IntensityFit.r_by_offset`: il massimo con i suoi vicini, piu'
# una spalla a scala logaritmica. Riportarli tutti e 257 sporcherebbe il report senza
# aggiungere nulla a chi deve solo decidere se fidarsi del fit.
_REPORT_NEIGHBOURS = 2
_REPORT_SHOULDER = (0, 1, 2, 4, 8, 16, 32, 64, 128)


class GeometryError(RuntimeError):
    """Dati che non sappiamo leggere senza tirare a indovinare.

    Diverso da `None`: `None` vuol dire "misura non applicabile a questa coppia" (mesh non
    pubblicata, volumi non omologhi). `GeometryError` vuol dire "il formato non e' quello
    che questo modulo sa leggere", e va corretto nel codice, non ignorato.
    """


# --------------------------------------------------------------------------- mesh


@dataclass(frozen=True)
class MeshCheck:
    identical: bool
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    max_abs_diff: dict[str, float]   # per canale "x", "y", "z"
    note: str                        # perche' non identiche, se non lo sono


def compare_meshes(segment_prefix: str, vol_a: str, vol_b: str) -> MeshCheck | None:
    """Confronta le tifxyz delle due derivazioni. None se una delle due non e' pubblicata.

    NON confronta i byte dei file: la compressione da' dimensioni diverse a contenuto
    identico (nel corpus una derivazione scrive LZW e l'altra TIFF non compresso, con un
    rapporto di quasi 2:1 sulla stessa matrice). Confronta gli array decodificati.

    NON guarda il `meta.json` per decidere `identical`: quel sidecar contiene campi di
    provenienza (uuid, area, precisione della scala) che differiscono anche quando le
    coordinate sono identiche al bit. Le differenze del sidecar finiscono in `note`, dove
    informano senza falsare il verdetto.

    NON riallinea e non ricampiona: se le due matrici hanno forma diversa, il confronto
    voxel per voxel non viene tentato e `max_abs_diff` vale `inf` su tutti i canali.

    `segment_prefix` e' il prefisso S3 del segmento, per esempio
    "PHerc0172/segments/20251107110950-w064_20251107110950052_flatboi".
    `vol_a` / `vol_b` sono gli id di volume, per esempio "20241024131838".
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
    """Il prefisso della tifxyz derivata da `vol`, senza slash finale. None se assente.

    NON indovina: pretende il marcatore "-on-<vol>-" nel nome della cartella, che e' come
    il corpus registra su quale volume e' stata appiattita la mesh. Se piu' cartelle
    corrispondono prende la prima in ordine lessicografico, in modo che due esecuzioni
    diano lo stesso risultato.
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
    """L'array decodificato, o None se il file non e' pubblicato."""
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
    """Riassunto delle differenze fra i due meta.json. Stringa vuota se non leggibili."""
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
    """Il minimo indispensabile di uno zarr v2 per leggerne i chunk a mano."""

    prefix: str                  # ".../<vol>.zarr/0", senza slash finale
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    order: str
    fill_value: int
    compressor: dict | None

    @property
    def full_chunks(self) -> tuple[int, ...]:
        """Quanti chunk sono interamente dentro l'array, per asse."""
        return tuple(s // c for s, c in zip(self.shape, self.chunks))


def _read_zarray(vol_prefix: str, level: str = LEVEL) -> _ZArray:
    """Legge `<vol>/<level>/.zarray`.

    NON supporta zarr v3 (`zarr.json`), i filtri, e gli array di piu' o meno di 3 assi:
    su quei casi solleva GeometryError invece di leggere byte a caso.
    """
    prefix = f"{vol_prefix.rstrip('/')}/{level}"
    meta = cache.get_json(f"{prefix}/.zarray")
    if int(meta.get("zarr_format", 0)) != 2:
        raise GeometryError(f"zarr_format {meta.get('zarr_format')} non supportato in {prefix}")
    if meta.get("filters"):
        raise GeometryError(f"filtri non supportati in {prefix}: {meta['filters']}")
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
            f"dtype {dtype} in {prefix}: questo modulo accumula istogrammi esatti e "
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
    """Un chunk decodificato, di forma `z.chunks`.

    NON applica filtri e NON ritaglia il bordo: in zarr v2 anche il chunk a cavallo del
    bordo e' memorizzato per intero, e la coda oltre `shape` resta come l'ha scritta chi
    ha prodotto l'array.
    """
    comp = z.compressor
    if comp is None:
        buf: object = raw
    elif comp.get("id") == "blosc":
        # L'header blosc porta con se' clevel, shuffle e cname: non serve rileggerli.
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
    """Il chunk `idx`, o None se non e' memorizzato (in zarr = tutto `fill_value`).

    NON distingue un chunk assente da uno pieno di `fill_value`: per lo scopo di questo
    modulo, dove `fill_value` e' 0 e i voxel validi sono > 0, sono la stessa cosa.
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
    """Sonda con una Range di un byte se il chunk esiste, senza scaricarlo.

    Serve a scartare i candidati vuoti a costo quasi zero: nel corpus circa il 40% delle
    posizioni della griglia non e' memorizzato, e scaricare 2 MiB per accorgersene sarebbe
    lo spreco dominante della funzione.
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
    """Il blocco di `z` omologo al chunk `idx`, traslato di `dz` voxel lungo z.

    Per `dz` diverso da zero serve leggere il chunk vicino: la finestra cade a cavallo di
    due chunk. Le parti fuori dall'array e i chunk assenti valgono `fill_value`.

    NON trasla in y e in x: il disallineamento laterale, se c'e', si vede nelle mappe di
    ink e lo cerca `metrics.best_shift_iou`, non qui.
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


# --------------------------------------------------------------------------- intensita'


@dataclass(frozen=True)
class IntensityFit:
    slope: float
    intercept: float
    r: float
    n_voxel: int
    median_a: float
    median_b: float
    clip_frac_a: float    # frazione di voxel >= 200, il tetto del clip della pipeline
    clip_frac_b: float
    chunks_used: list[tuple[int, int, int]]
    # Campi aggiunti in coda, tutti con default: le chiamate scritte contro la firma del
    # contratto continuano a valere, e chi legge il report ha anche la prova che
    # l'allineamento in z e' stato misurato invece di assunto.
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
    """Campiona chunk omologhi dai due volumi e stima A = slope*B + intercept.

    Legge zarr v2 a mano: `.zarray` per shape/chunks/compressor, poi i chunk via HTTP.
    Gestisce compressor null (byte grezzi) e blosc (numcodecs.Blosc). None se i due
    volumi non hanno la stessa shape in y e x.

    NON assume che i due volumi siano allineati in z: cerca l'offset in z che massimizza
    la correlazione e riporta sia `z_offset` sia `r_by_offset`, cosi' chi legge puo'
    scartare il fit se r e' basso o se il massimo non cade a zero.

    NON usa una lista di chunk scelta a mano: estrae posizioni con `seed`, scarta quelle
    dove uno dei due volumi non ha almeno `min_nonzero` di voxel > 0, e riporta in
    `chunks_used` esattamente quelle su cui ha misurato. Cambiando `seed` cambiano i chunk
    e non deve cambiare la stima: e' quello il controllo.

    NON e' una regressione robusta e NON ritaglia i voxel al tetto del clip: la stima e' un
    minimi-quadrati ordinario di A su B sui voxel dove entrambi sono > 0, e la frazione al
    tetto viene riportata a parte perche' e' proprio quella a rendere la retta ottimistica.

    NON campiona i chunk di bordo: i candidati sono solo i chunk interamente dentro
    entrambi gli array, cosi' la coda di padding oltre `shape` non entra mai nella stima.

    NON tiene in memoria piu' di circa `2 * n_chunks` chunk alla volta (con i 128^3 uint8
    del corpus, una quarantina di MB con i valori di default).

    Restituisce None anche quando: uno dei due volumi non e' pubblicato sotto
    `<sample>/volumes/`, le dimensioni dei chunk differiscono, o nessuna posizione
    campionata ha dati in entrambi i volumi.
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

    # I chunk gia' scaricati durante la selezione entrano nel memo, cosi' la sonda di
    # allineamento paga soltanto i due vicini in z.
    memo: dict[tuple[str, int, int, int], np.ndarray | None] = {}
    for idx, a_chunk, b_chunk in picks:
        memo[(za.prefix, *idx)] = a_chunk
        memo[(zb.prefix, *idx)] = b_chunk
    scan = _scan_z_offsets(za, zb, picks[0][0], memo)
    z_offset = max(scan, key=lambda d: (-math.inf if math.isnan(scan[d]) else scan[d], -abs(d)))

    memo.clear()   # i due vicini della sonda non servono piu'

    acc = _Accumulator(za.dtype)
    used: list[tuple[int, int, int]] = []
    for idx, a_chunk, b_chunk in picks:
        if z_offset == 0:
            b = b_chunk
        else:
            # Il memo locale muore a fine iterazione: la finestra traslata cade a cavallo di
            # due chunk, e il primo dei due e' quello che abbiamo gia' in mano.
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
    """Dall'id di volume al prefisso S3 dello zarr. None se non e' pubblicato.

    NON scarica nulla del volume: una sola LIST sui figli di `<sample>/volumes/`.
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
    """Sceglie posizioni riproducibili con dati in entrambi i volumi.

    NON cerca i chunk "migliori" e NON guarda dove passa la mesh: campiona uniformemente
    la griglia comune con un seed, sonda la presenza con una Range di un byte, e accetta
    solo dove entrambi i volumi hanno almeno `min_nonzero` di voxel > 0. Restituisce anche
    gli array, perche' riscaricarli nella fase di stima sarebbe il doppio del traffico.
    """
    # I candidati escludono il primo e l'ultimo chunk pieno di ogni asse: il primo e
    # l'ultimo in z servono come vicini per la scansione degli offset, e il bordo porta
    # dentro il padding.
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
    """Correlazione fra i due volumi per ogni offset intero in z, su un chunk sonda.

    Un solo chunk sonda, e i suoi due vicini in z: la sonda serve a stabilire l'offset, e
    ripeterla su tutti i chunk moltiplicherebbe il traffico per misurare la stessa cosa.

    NON e' la correlazione del fit: e' calcolata su un solo chunk e sottocampionata di
    `_SCAN_STRIDE` in y e x. Serve a scegliere l'offset e a mostrare quanto e' stretto il
    picco, non a quantificare l'accordo.

    NON cerca offset piu' grandi di un chunk: oltre quello i due volumi non sono la stessa
    acquisizione riallineata ma due cose diverse, e il fit va scartato, non spostato.
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
    """Il sottoinsieme della scansione che finisce nel report: il picco e la sua spalla."""
    wanted = {best + d for d in range(-_REPORT_NEIGHBOURS, _REPORT_NEIGHBOURS + 1)}
    for s in _REPORT_SHOULDER:
        wanted |= {s, -s}
    return tuple(sorted((d, scan[d]) for d in wanted if d in scan))


class _Accumulator:
    """Somme esatte e istogrammi marginali sui voxel validi, chunk dopo chunk.

    Le somme restano interi Python, quindi slope, intercept e r si calcolano da quantita'
    esatte e non dipendono dall'ordine in cui sono arrivati i chunk. Gli istogrammi
    marginali danno mediane e frazioni al tetto esatte con memoria costante: nessun array
    di valori resta in vita dopo il chunk che l'ha prodotto.

    NON tiene l'istogramma congiunto: servirebbe per una regressione robusta, che questo
    modulo non fa.
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
    """Mediana esatta da un istogramma per valore, con la convenzione di `numpy.median`.

    NON interpola: i valori sono interi, quindi per n pari la mediana e' la media dei due
    elementi centrali e nient'altro.
    """
    n = int(counts.sum())
    if n == 0:
        return float("nan")
    cum = np.cumsum(counts)
    lo = int(np.searchsorted(cum, (n - 1) // 2, side="right")) + vmin
    hi = int(np.searchsorted(cum, n // 2, side="right")) + vmin
    return (lo + hi) / 2.0


def _tail_frac(counts: np.ndarray, vmin: int, threshold: int) -> float:
    """Frazione di voxel con valore >= `threshold`."""
    n = int(counts.sum())
    if n == 0:
        return float("nan")
    start = max(0, threshold - vmin)
    return float(int(counts[start:].sum()) / n)
