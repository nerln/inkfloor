"""Orchestration and formatting: one record per segment, rendered as Markdown or JSON.

This module glues `census` (what is comparable), `metrics` (the ruler) and `geometry` (the
confounder checks) into a `SegmentFloor`, and builds the download plan that `cli.py` prints
before anything crosses the network.

What this module does NOT do:

- it does not print: every function returns data or a string, printing lives in `cli.py`;
- it does not open sockets of its own: every byte goes through `cache.fetch` /
  `cache.get_bytes`, and the plan builders never fetch, they only list or read the cache;
- it does not attribute a cause. It puts the floor, the high anchor and the confounder
  checks next to each other and stops there. A floor close to its anchor is a measurement,
  not an explanation.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inkfloor import cache

if TYPE_CHECKING:  # imported for type hints only, so this module loads on its own
    import numpy as np

    from inkfloor.census import Pair, Prediction
    from inkfloor.geometry import IntensityFit, MeshCheck
    from inkfloor.metrics import Delta, TieBand

DEFAULT_QS: tuple[float, ...] = (0.01, 0.05, 0.20)

# 1 MB = 1024 * 1024 bytes, everywhere in this tool.
KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

# Nominal sizes, used only when a plan is built offline (`--dry-run`) and the real size is
# not known yet. Grounded on measured keys of PHerc0172/.../flatboi: the four published
# predictions of that segment are 32.3, 42.8, 44.2 and 46.0 MB; its two tifxyz derivations
# are 3.5 and 5.7 MB; its surface-volume zarr chunks are 540672 B each.
NOMINAL_PREDICTION_BYTES = 40 * MB
NOMINAL_PREDICTION_RANGE = (30 * MB, 50 * MB)
NOMINAL_MESH_BYTES = 5 * MB
NOMINAL_ZARR_CHUNK_BYTES = 528 * KB

PREDICTION_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Above this width the point Δ is mostly a consequence of index order, so the report stops
# presenting it as a number and starts presenting it as an interval.
TIE_WARN_WIDTH = 0.05


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class SegmentFloor:
    """Everything measured on one segment. Optional fields are None when not measurable.

    The three fields after `nulls` are declared in the tail with defaults, so a caller
    written against the signature in CONTRACTS.md keeps working. They are not decoration: a
    Delta without its tie band is a point value whose precision may be invented, and Deltas
    at different q are not commensurable without the chance level of each q. `to_markdown`
    and `to_json` render them whenever they are present.

    Keys of `ties` and `spearman` are `(a.key, b.key)` of the pair, which is the only
    identifier a pair has that survives a round trip through JSON.
    """

    sample: str
    segment: str
    volume_pairs: list[tuple[Pair, dict[float, Delta]]]
    model_pairs: list[tuple[Pair, dict[float, Delta]]]
    mesh: MeshCheck | None
    intensity: IntensityFit | None
    nulls: dict[str, Delta]
    ties: dict[tuple[str, str], dict[float, TieBand]] = field(default_factory=dict)
    chance: dict[float, float] = field(default_factory=dict)
    spearman: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanItem:
    """One line of a download plan.

    `new_bytes` is what will cross the network, `cached_bytes` what is already local.
    Either may be None when the count is only known after the listing phase; `exact` says
    whether the numbers come from real object sizes or from a nominal estimate.
    """

    step: str
    what: str
    count: int | None = None
    new_bytes: int | None = None
    cached_bytes: int = 0
    exact: bool = True


@dataclass(frozen=True)
class DownloadPlan:
    """What a command is about to do, in bytes, before it does it."""

    title: str
    items: list[PlanItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def new_bytes(self) -> int:
        """Known bytes to download. Items with an unknown size are not counted here."""
        return sum(i.new_bytes or 0 for i in self.items)

    @property
    def cached_bytes(self) -> int:
        return sum(i.cached_bytes for i in self.items)

    @property
    def exact(self) -> bool:
        """True when every item with a size knows its real size."""
        return all(i.exact for i in self.items if i.new_bytes)

    @property
    def unknown(self) -> list[PlanItem]:
        return [i for i in self.items if i.new_bytes is None and i.step != "compute"]


# --------------------------------------------------------------------------- byte plans


def human_bytes(n: int | None) -> str:
    """Bytes as MB/GB with one decimal. 'unknown' for None, so a plan never prints a lie."""
    if n is None:
        return "unknown"
    if n < 0:
        return "unknown"
    for unit, div in (("GB", GB), ("MB", MB), ("KB", KB)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def cached_path(key: str) -> Path:
    """Where `cache.fetch` would put this key. Does not check that it exists."""
    return cache.CACHE_ROOT / key


def is_cached(key: str) -> bool:
    p = cached_path(key)
    return p.exists() and p.stat().st_size > 0


def _split_cached(keys_sizes: Sequence[tuple[str, int]]) -> tuple[int, int, int]:
    """(n_to_fetch, bytes_to_fetch, bytes_already_local) for a list of (key, size)."""
    new_n = new_b = have_b = 0
    for key, size in keys_sizes:
        if is_cached(key):
            have_b += size
        else:
            new_n += 1
            new_b += size
    return new_n, new_b, have_b


def _cached_segment_bytes(sample: str, segment: str) -> dict[str, int]:
    """Bytes already in cache for a segment, split by role, without touching the network.

    The split is by path substring, so it follows the published layout
    (`<sample>/segments/<segment>/{ink-detection,mesh,surface-volumes}/...`). A layout
    change would push bytes into "other"; it never invents bytes that are not on disk.
    """
    out = {"prediction": 0, "mesh": 0, "chunk": 0, "other": 0}
    root = cache.CACHE_ROOT / sample
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file() or segment not in str(p):
            continue
        s = str(p)
        size = p.stat().st_size
        if "/ink-detection/" in s and s.endswith(PREDICTION_SUFFIXES):
            out["prediction"] += size
        elif "/mesh/" in s:
            out["mesh"] += size
        elif ".zarr/" in s:
            out["chunk"] += size
        else:
            out["other"] += size
    return out


def plan_census(samples: Sequence[str] | None = None) -> DownloadPlan:
    """The plan for `inkfloor census`: listings only, no file payload at all."""
    target = "all samples in the bucket" if not samples else ", ".join(samples)
    n = len(samples) if samples else None
    plan = DownloadPlan(title=f"census of {target}")
    plan.items.append(
        PlanItem(
            step="list",
            what=f"S3 ListObjectsV2 over {target} (object metadata, no payload)",
            count=n,
            new_bytes=0,
            exact=True,
        )
    )
    plan.notes.append(
        "census downloads no prediction file: it reads key names and sizes only. "
        "The listing is paginated at 1000 keys per request and a busy segment has "
        "tens of thousands of keys, so a full-bucket census takes minutes of requests."
    )
    if not samples:
        plan.notes.append(
            "Without --samples the whole bucket is enumerated. Use --samples PHerc0172 "
            "to keep it to one scroll."
        )
    return plan


def plan_segment(
    sample: str,
    segment: str,
    *,
    preds: Sequence[Prediction] | None = None,
    geometry_checks: bool = True,
    n_chunks: int = 5,
    allow_listing: bool = False,
) -> DownloadPlan:
    """The plan for one segment.

    With `preds` (from `census.census`) the prediction bytes are exact. Without it, the
    plan is built offline: it counts what is already in the cache and marks the rest as an
    estimate. This function never calls `cache.fetch`; with `allow_listing=True` it issues
    one small ListObjectsV2 on the mesh prefix to make the mesh line exact.
    """
    plan = DownloadPlan(title=f"floor {sample} / {segment}")

    if preds is None:
        have = _cached_segment_bytes(sample, segment)
        lo, hi = NOMINAL_PREDICTION_RANGE
        plan.items.append(
            PlanItem(
                step="list",
                what=f"S3 prefix {sample}/segments/{segment}/ (metadata, no payload)",
                count=1,
                new_bytes=0,
            )
        )
        plan.items.append(
            PlanItem(
                step="fetch",
                what=(
                    "ink predictions: count resolved by the listing, "
                    f"{human_bytes(lo)} to {human_bytes(hi)} each"
                ),
                count=None,
                new_bytes=None,
                cached_bytes=have["prediction"],
                exact=False,
            )
        )
        if geometry_checks:
            plan.items.append(
                PlanItem(
                    step="fetch",
                    what="tifxyz mesh of each derivation (x/y/z + meta.json)",
                    count=None,
                    new_bytes=None,
                    cached_bytes=have["mesh"],
                    exact=False,
                )
            )
            plan.items.append(
                PlanItem(
                    step="range",
                    what=(
                        f"{2 * n_chunks} surface-volume zarr chunks via HTTP Range "
                        f"(~{human_bytes(NOMINAL_ZARR_CHUNK_BYTES)} each, not cached)"
                    ),
                    count=2 * n_chunks,
                    new_bytes=2 * n_chunks * NOMINAL_ZARR_CHUNK_BYTES,
                    exact=False,
                )
            )
        plan.notes.append(
            "Offline plan: sizes are nominal. Run without --dry-run for exact bytes, the "
            "listing phase prints them before the first fetch and downloads no payload."
        )
        _add_compute_steps(plan, geometry_checks)
        return plan

    seg_preds = [p for p in preds if p.sample == sample and p.segment == segment]
    keys_sizes = [(p.key, p.size_bytes) for p in seg_preds]
    new_n, new_b, have_b = _split_cached(keys_sizes)
    plan.items.append(
        PlanItem(
            step="fetch",
            what=f"ink predictions ({len(seg_preds)} published, {new_n} not in cache)",
            count=new_n,
            new_bytes=new_b,
            cached_bytes=have_b,
        )
    )

    if geometry_checks:
        volumes = sorted({p.volume for p in seg_preds})
        mesh_keys: list[tuple[str, int]] = []
        listed = False
        if allow_listing and seg_preds:
            mesh_keys = _list_mesh_keys(segment_prefix(seg_preds), volumes)
            listed = bool(mesh_keys)
        if listed:
            m_new_n, m_new_b, m_have_b = _split_cached(mesh_keys)
            plan.items.append(
                PlanItem(
                    step="fetch",
                    what=f"tifxyz mesh, {len(volumes)} derivations ({len(mesh_keys)} files)",
                    count=m_new_n,
                    new_bytes=m_new_b,
                    cached_bytes=m_have_b,
                )
            )
        else:
            plan.items.append(
                PlanItem(
                    step="fetch",
                    what=f"tifxyz mesh, {len(volumes)} derivations (size not listed)",
                    count=len(volumes),
                    new_bytes=len(volumes) * NOMINAL_MESH_BYTES,
                    exact=False,
                )
            )
        plan.items.append(
            PlanItem(
                step="range",
                what=(
                    f"{2 * n_chunks} surface-volume zarr chunks via HTTP Range "
                    f"(~{human_bytes(NOMINAL_ZARR_CHUNK_BYTES)} each, not cached)"
                ),
                count=2 * n_chunks,
                new_bytes=2 * n_chunks * NOMINAL_ZARR_CHUNK_BYTES,
                exact=False,
            )
        )

    _add_compute_steps(plan, geometry_checks)
    if len(seg_preds) > 1:
        plan.notes.append(
            "Every map of the segment is held in memory at once to build the common valid "
            "mask: expect roughly one byte of RSS per pixel per map."
        )
    return plan


def _kind_phrase(kinds: Sequence[str]) -> str:
    """How to name the pair a segment must carry, in a sentence about downloads."""
    wanted = frozenset(kinds)
    if wanted == {"volume"}:
        return "volume pair (two derivations of one scan)"
    if wanted == {"model"}:
        return "model pair (two checkpoints on one derivation)"
    return "comparable pair"


def plan_corpus(
    samples: Sequence[str] | None = None,
    *,
    preds: Sequence[Prediction] | None = None,
    segments: Sequence[tuple[str, str]] | None = None,
    geometry_checks: bool = True,
    kinds: Sequence[str] = ("volume",),
    n_chunks: int = 5,
) -> DownloadPlan:
    """The plan for a corpus run. `segments` restricts it to the segments actually kept."""
    target = "all samples" if not samples else ", ".join(samples)
    # Name what is actually being measured. The floor and the anchor are different quantities
    # and the whole tool exists to keep them apart, so a plan header that says "floor" while
    # fetching model pairs contradicts the point on its first line.
    what = {frozenset({"volume"}): "floor", frozenset({"model"}): "anchor"}.get(
        frozenset(kinds), "floor and anchor"
    )
    plan = DownloadPlan(title=f"corpus {what} over {target}")

    if preds is None:
        lo, hi = NOMINAL_PREDICTION_RANGE
        plan.items.append(
            PlanItem(step="list", what=f"S3 listing of {target} (no payload)", new_bytes=0)
        )
        plan.items.append(
            PlanItem(
                step="fetch",
                what=(
                    "every prediction of every segment that has a comparable pair, "
                    f"{human_bytes(lo)} to {human_bytes(hi)} each"
                ),
                count=None,
                new_bytes=None,
                exact=False,
            )
        )
        plan.notes.append(
            "Offline plan: the number of predictions is only known after the listing. Only "
            f"segments that carry a {_kind_phrase(kinds)} are fetched, so the total follows "
            "how many such pairs are published. At "
            f"{human_bytes(NOMINAL_PREDICTION_BYTES)} each, 100 predictions are "
            f"{human_bytes(100 * NOMINAL_PREDICTION_BYTES)} and 500 are "
            f"{human_bytes(500 * NOMINAL_PREDICTION_BYTES)}."
        )
        plan.notes.append(
            "Run without --dry-run to get the exact total: the listing phase prints it "
            "and the run stops before the first fetch unless --yes is given."
        )
        _add_compute_steps(plan, geometry_checks)
        return plan

    keep = set(segments) if segments is not None else None
    chosen = [p for p in preds if keep is None or (p.sample, p.segment) in keep]
    by_segment: dict[tuple[str, str], set[str]] = {}
    for p in chosen:
        by_segment.setdefault((p.sample, p.segment), set()).add(p.volume)
    n_segments = len(by_segment)
    new_n, new_b, have_b = _split_cached([(p.key, p.size_bytes) for p in chosen])
    plan.items.append(
        PlanItem(
            step="fetch",
            what=(
                f"ink predictions of {n_segments} segments "
                f"({len(chosen)} published, {new_n} not in cache)"
            ),
            count=new_n,
            new_bytes=new_b,
            cached_bytes=have_b,
        )
    )
    if geometry_checks and n_segments:
        n_vol = sum(len(v) for v in by_segment.values())
        plan.items.append(
            PlanItem(
                step="fetch",
                what=f"tifxyz mesh of {n_vol} derivations across {n_segments} segments",
                count=n_vol,
                new_bytes=n_vol * NOMINAL_MESH_BYTES,
                exact=False,
            )
        )
        plan.items.append(
            PlanItem(
                step="range",
                what=(
                    f"{2 * n_chunks * n_segments} zarr chunks via HTTP Range "
                    f"({n_segments} segments, not cached)"
                ),
                count=2 * n_chunks * n_segments,
                new_bytes=2 * n_chunks * n_segments * NOMINAL_ZARR_CHUNK_BYTES,
                exact=False,
            )
        )
    _add_compute_steps(plan, geometry_checks)
    return plan


def _add_compute_steps(plan: DownloadPlan, geometry_checks: bool) -> None:
    plan.items.append(
        PlanItem(
            step="compute",
            what=(
                "common valid mask, Delta@q per pair, tie bands, rank correlation, "
                "null controls"
            ),
            new_bytes=0,
        )
    )
    if geometry_checks:
        plan.items.append(
            PlanItem(step="compute", what="mesh identity and intensity fit", new_bytes=0)
        )


def _list_mesh_keys(prefix: str, volumes: Sequence[str]) -> list[tuple[str, int]]:
    """The tifxyz keys of the given derivations. Empty list on any surprise, never raises.

    Planning must not fail the run: if the mesh prefix does not answer or the layout is not
    the one we know, the caller falls back to the nominal estimate and says so.
    """
    try:
        keys = cache.list_keys(f"{prefix}mesh/")
    except Exception:
        return []
    out = []
    for key, size in keys:
        if "/intermediate/" in key or ".tifxyz/" not in key:
            continue
        if any(f"-on-{v}-" in key for v in volumes):
            out.append((key, size))
    return out


def format_plan(plan: DownloadPlan, *, cache_root: Path | None = None) -> str:
    """The plan as text: what will be fetched, in bytes, before anything is fetched.

    Sizes come first and the description last, so a long S3 prefix cannot push the numbers
    off the eye's path. An unknown size prints as 'unknown', never as 0.
    """
    root = cache.CACHE_ROOT if cache_root is None else cache_root
    rows = [("step", "fetch", "cached", "what")]
    for i in plan.items:
        size = "-" if i.step == "compute" else human_bytes(i.new_bytes)
        if not i.exact and i.new_bytes:
            size = "~" + size
        have = human_bytes(i.cached_bytes) if i.cached_bytes else "-"
        rows.append((i.step, size, have, i.what))
    w_step = max(len(r[0]) for r in rows)
    w_size = max(len(r[1]) for r in rows)
    w_have = max(len(r[2]) for r in rows)
    lines = [f"plan: {plan.title}", ""]
    for step, size, have, what in rows:
        lines.append(f"  {step:<{w_step}}  {size:>{w_size}}  {have:>{w_have}}  {what}")
    lines.append("")
    if plan.unknown:
        known = human_bytes(plan.new_bytes)
        if not plan.exact:
            known = "~" + known
        lines.append(
            f"  to download now: UNKNOWN until the listing. "
            f"Sized steps: {known}. Steps with no size yet: {len(plan.unknown)}."
        )
    else:
        total = human_bytes(plan.new_bytes)
        if not plan.exact:
            total = "~" + total
        lines.append(f"  to download now: {total}")
    lines.append(f"  of which already in cache: {human_bytes(plan.cached_bytes)}")
    try:
        here = cache.cache_size_bytes()
    except OSError:
        here = None
    lines.append(f"  cache: {root} ({human_bytes(here)} on disk)")
    for n in plan.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- measuring


def segment_prefix(preds: Sequence[Prediction]) -> str:
    """The S3 prefix of the segment directory, for `geometry.compare_meshes`.

    Prefers the `Prediction.segment_prefix` property when the census module offers one: the
    layout belongs to census, not here. Otherwise it cuts a key right after its
    `/<segment>/` component, so the layout is still read from the data instead of being
    hardcoded, and falls back to `<sample>/segments/<segment>/`.
    """
    if not preds:
        raise ValueError("segment_prefix needs at least one prediction")
    p = sorted(preds, key=lambda x: x.key)[0]
    own = _g(p, "segment_prefix")
    if isinstance(own, str) and own:
        return own
    marker = f"/{p.segment}/"
    i = p.key.find(marker)
    if i >= 0:
        return p.key[: i + len(marker)]
    return f"{p.sample}/segments/{p.segment}/"


def load_map(key: str) -> np.ndarray:
    """Fetch a published prediction and decode it to a 2D array.

    Does not rescale, does not normalise and does not touch the dtype: the whole point of
    the measurement is that the published 8 bit values are what people look at. Multi-channel
    files are reduced to their first channel.
    """
    import numpy as np

    path = cache.fetch(key)
    low = key.lower()
    if low.endswith((".tif", ".tiff")):
        import tifffile

        arr = tifffile.imread(path)
    elif low.endswith(".png"):
        import imagecodecs

        arr = imagecodecs.png_decode(path.read_bytes())
    elif low.endswith((".jpg", ".jpeg")):
        import imagecodecs

        arr = imagecodecs.jpeg_decode(path.read_bytes())
    else:
        raise ValueError(f"no decoder for {key}")
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return arr


def _primary_q(qs: Sequence[float]) -> float:
    if not qs:
        return 0.05
    return 0.05 if 0.05 in tuple(qs) else tuple(qs)[len(qs) // 2]


def floor_for_segment(
    sample: str,
    segment: str,
    qs: Sequence[float] = DEFAULT_QS,
    *,
    preds: Sequence[Prediction] | None = None,
    geometry_checks: bool = True,
    null_px: int = 64,
) -> SegmentFloor:
    """Measure one segment: the volume pairs (the floor) and the model pairs (the anchor).

    Does not choose a winner and does not exclude a pair on its own. Pairs of kind "both"
    never arrive here, `census.pairs` drops them. Does not reconcile shapes either: the maps
    go to `metrics` as they were published, and `metrics.align` crops them to the common
    region or raises `ShapeMismatch` when the difference is too large to crop honestly.

    The valid mask is built once over every map of the segment, not per pair, so the floor
    and the anchor are measured on the same pixels and their numbers can be compared.

    Does not offer a way to skip the tie bands, the chance levels or the rank correlation.
    They cost roughly as much again as the Deltas themselves, and they are what keeps a Delta
    from being read as more precise than it is: a tie band wider than the difference under
    discussion, or a Delta compared against zero instead of against the chance level of its
    own q, is how this tool would produce a wrong number that looks credible.
    """
    from inkfloor import census as census_mod
    from inkfloor import metrics

    qs = tuple(qs)
    if preds is None:
        preds = census_mod.census([sample])
    seg_preds = [p for p in preds if p.sample == sample and p.segment == segment]
    if not seg_preds:
        raise LookupError(f"no published prediction for {sample} / {segment}")

    all_pairs = list(census_mod.pairs(seg_preds))
    vol_pairs = [p for p in all_pairs if p.kind == "volume"]
    mod_pairs = [p for p in all_pairs if p.kind == "model"]

    used_keys: list[str] = []
    for pair in vol_pairs + mod_pairs:
        for pred in (pair.a, pair.b):
            if pred.key not in used_keys:
                used_keys.append(pred.key)
    maps = {k: load_map(k) for k in used_keys}
    valid = metrics.common_valid([maps[k] for k in used_keys]) if maps else None

    def deltas(pair: Pair) -> dict[float, Delta]:
        return {q: metrics.delta_at_q(maps[pair.a.key], maps[pair.b.key], valid, q) for q in qs}

    volume_pairs = [(p, deltas(p)) for p in vol_pairs]
    model_pairs = [(p, deltas(p)) for p in mod_pairs]

    ties: dict[tuple[str, str], dict[float, TieBand]] = {}
    rank: dict[tuple[str, str], float] = {}
    for pair in vol_pairs + mod_pairs:
        a, b = maps[pair.a.key], maps[pair.b.key]
        ties[(pair.a.key, pair.b.key)] = {q: metrics.tie_bounds(a, b, valid, q) for q in qs}
        rank[(pair.a.key, pair.b.key)] = metrics.spearman(a, b, valid)
    chance = {q: metrics.chance_iou(q) for q in qs}

    mesh = None
    intensity = None
    if geometry_checks and vol_pairs:
        from inkfloor import geometry

        ref = vol_pairs[0]
        mesh = geometry.compare_meshes(segment_prefix(seg_preds), ref.a.volume, ref.b.volume)
        intensity = geometry.fit_intensity(sample, ref.a.volume, ref.b.volume)

    nulls: dict[str, Delta] = {}
    if used_keys and valid is not None:
        q0 = _primary_q(qs)
        ref_map = maps[used_keys[0]]
        nulls["self"] = metrics.null_self(ref_map, valid, q0)
        nulls[f"shift_{null_px}px"] = metrics.null_shift(ref_map, valid, q0, null_px)

    return SegmentFloor(
        sample=sample,
        segment=segment,
        volume_pairs=volume_pairs,
        model_pairs=model_pairs,
        mesh=mesh,
        intensity=intensity,
        nulls=nulls,
        ties=ties,
        chance=chance,
        spearman=rank,
    )


def corpus_floor(
    samples: Sequence[str] | None = None,
    qs: Sequence[float] = DEFAULT_QS,
    *,
    preds: Sequence[Prediction] | None = None,
    segments: Sequence[tuple[str, str]] | None = None,
    geometry_checks: bool = True,
    kinds: Sequence[str] = ("volume",),
    on_segment: Callable[[int, int, str, str], None] | None = None,
    on_error: Callable[[str, str, Exception], None] | None = None,
) -> list[SegmentFloor]:
    """Measure every segment that carries at least one pair of a kind in `kinds`.

    Does not print progress: pass `on_segment` / `on_error` if you want to see it. Without
    `on_error` a failing segment aborts the run; with it, the segment is skipped and the
    caller decides what to say.

    `kinds` defaults to the floor alone, which is one segment in the corpus published today.
    Pass `("model",)` for the anchor instead: how far apart two checkpoints land on the same
    derivation. That is the scale the floor has to be read against, and it exists on far more
    segments, so it is the only way to say whether one floor is large without leaning on n=1.
    Every measured segment still reports both its volume and its model pairs; `kinds` only
    decides which segments are worth downloading.
    """
    from inkfloor import census as census_mod

    wanted_kinds = frozenset(kinds)

    if preds is None:
        preds = census_mod.census(list(samples) if samples else None)

    if segments is None:
        wanted: list[tuple[str, str]] = []
        by_segment: dict[tuple[str, str], list[Prediction]] = {}
        for p in preds:
            by_segment.setdefault((p.sample, p.segment), []).append(p)
        for seg, group in sorted(by_segment.items()):
            if any(pair.kind in wanted_kinds for pair in census_mod.pairs(group)):
                wanted.append(seg)
    else:
        wanted = list(segments)

    out: list[SegmentFloor] = []
    for i, (sample, segment) in enumerate(wanted, start=1):
        if on_segment is not None:
            on_segment(i, len(wanted), sample, segment)
        try:
            out.append(
                floor_for_segment(
                    sample,
                    segment,
                    qs,
                    preds=preds,
                    geometry_checks=geometry_checks,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad segment must not kill the corpus
            if on_error is None:
                raise
            on_error(sample, segment, exc)
    return out


# --------------------------------------------------------------------------- formatting


def _g(obj: Any, name: str) -> Any:
    """Field access that tolerates a missing field, so a partial record still renders."""
    return getattr(obj, name, None)


def _f(x: Any) -> float | None:
    """A JSON-safe float. NaN and inf become null: a report must stay valid JSON."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _i(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _num(x: Any, digits: int = 3) -> str:
    """A number for a table cell. None, NaN and non-numbers all print as 'n/a'."""
    v = _f(x)
    return "n/a" if v is None else f"{v:.{digits}f}"


def _int(x: Any) -> str:
    v = _i(x)
    return "n/a" if v is None else f"{v:,}"


def _pct(x: Any, digits: int = 2) -> str:
    v = _f(x)
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _qlabel(q: float) -> str:
    return f"{q * 100:g}%"


def _qkey(q: float) -> str:
    return f"{float(q):.6g}"


def _delta(deltas: Mapping[float, Delta] | None, q: float) -> Delta | None:
    if not deltas:
        return None
    if q in deltas:
        return deltas[q]
    for k, v in deltas.items():
        try:
            if abs(float(k) - float(q)) < 1e-12:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _delta_value(deltas: Mapping[float, Delta] | None, q: float) -> float | None:
    """1 - IoU at q, or None when that q was not measured or IoU is missing."""
    d = _delta(deltas, q)
    if d is None:
        return None
    iou = _g(d, "iou")
    if iou is None:
        return None
    try:
        return 1.0 - float(iou)
    except (TypeError, ValueError):
        return None


def _qs_of(pairs: Sequence[tuple[Pair, Mapping[float, Delta]]]) -> list[float]:
    qs: set[float] = set()
    for _, deltas in pairs or ():
        for k in (deltas or {}):
            try:
                qs.add(float(k))
            except (TypeError, ValueError):
                continue
    return sorted(qs)


def _median(values: Sequence[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    if not real:
        return None
    return statistics.median(real)


def _cell(value: Any) -> str:
    """One table cell. A pipe in a model name would split the row, so it is escaped."""
    text = "n/a" if value is None else str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = [_row(header), _row(["---"] * len(header))]
    out.extend(_row(r) for r in rows)
    return out


def _pair_key(pair: Pair) -> tuple[str, str]:
    return (str(_g(_g(pair, "a"), "key")), str(_g(_g(pair, "b"), "key")))


def _tie_of(
    ties: Mapping[tuple[str, str], Mapping[float, TieBand]] | None,
    pair: Pair,
    q: float,
) -> TieBand | None:
    per_pair = (ties or {}).get(_pair_key(pair))
    if not per_pair:
        return None
    if q in per_pair:
        return per_pair[q]
    for k, v in per_pair.items():
        try:
            if abs(float(k) - float(q)) < 1e-12:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _band_width(tb: TieBand | None) -> float | None:
    """Width of the admissible IoU interval, computed from the bounds when needed.

    Prefers the module's own `width`, and falls back to `iou_max - iou_min` so a stub that
    carries only the bounds still gets flagged.
    """
    if tb is None:
        return None
    w = _f(_g(tb, "width"))
    if w is not None:
        return w
    lo, hi = _f(_g(tb, "iou_min")), _f(_g(tb, "iou_max"))
    if lo is None or hi is None:
        return None
    return hi - lo


def _is_degenerate(tb: TieBand | None) -> bool:
    """True when no pixel of either selection was forced by the data.

    This is the case that looks like the best possible result and means the least: if the
    whole budget k is drawn from one plateau of equal values on both sides, the two maps can
    be made to agree completely by index order alone, so a Δ of 0 says nothing about where
    either map put ink.
    """
    if tb is None:
        return False
    k = _i(_g(tb, "k"))
    need_a = _i(_g(tb, "ties_needed_a"))
    need_b = _i(_g(tb, "ties_needed_b"))
    if not k or need_a is None or need_b is None:
        return False
    return need_a >= k and need_b >= k


def _tie_flag(tb: TieBand | None) -> str:
    """The marker that goes in a Δ cell: '!!' degenerate, '!' wide, '' otherwise."""
    if tb is None:
        return ""
    if _is_degenerate(tb):
        return "!!"
    width = _band_width(tb)
    if width is not None and width > TIE_WARN_WIDTH:
        return "!"
    return ""


def _delta_cell(
    deltas: Mapping[float, Delta] | None,
    tb: TieBand | None,
    q: float,
) -> str:
    """`Δ [low, high] flag`. The bracket is the tie interval mirrored onto Δ.

    Δ = 1 - IoU, so the interval on Δ runs from 1 - iou_max to 1 - iou_min. Reporting it in Δ
    units keeps one unit per column; the JSON carries the IoU bounds as `metrics` produced
    them.
    """
    point = _num(_delta_value(deltas, q))
    lo, hi = _f(_g(tb, "iou_max")), _f(_g(tb, "iou_min"))
    if lo is None or hi is None:
        return point
    flag = _tie_flag(tb)
    cell = f"{point} [{1.0 - lo:.3f}, {1.0 - hi:.3f}]"
    return f"{cell} {flag}" if flag else cell


def _pair_table(
    pairs: Sequence[tuple[Pair, Mapping[float, Delta]]],
    qs: Sequence[float],
    kind: str,
    ties: Mapping[tuple[str, str], Mapping[float, TieBand]] | None = None,
    rank: Mapping[tuple[str, str], float] | None = None,
) -> list[str]:
    """One table per pair family. `kind` picks which column is the constant one."""
    if not pairs:
        missing = (
            "_No volume pair on this segment: there is no floor to measure here, only one "
            "derivation of the scan carries a prediction._"
            if kind == "volume"
            else "_No model pair on this segment: the floor has no high anchor to be read "
            "against here._"
        )
        return [missing]
    fixed = "model" if kind == "volume" else "volume"
    varied = "volume" if kind == "volume" else "model"
    q_ref = _primary_q(qs) if qs else 0.05
    header = (
        [f"same {fixed}", f"{varied} A", f"{varied} B"]
        + [f"Δ@{_qlabel(q)} [tie band]" for q in qs]
        + [
            f"IoU@{_qlabel(q_ref)}",
            f"Dice@{_qlabel(q_ref)}",
            "ρ (rank)",
            f"k@{_qlabel(q_ref)}",
            "valid px",
        ]
    )
    rows = []
    for pair, deltas in pairs:
        a = _g(pair, "a")
        b = _g(pair, "b")
        ref = _delta(deltas, q_ref)
        cells = [
            str(_g(a, fixed) or "n/a"),
            str(_g(a, varied) or "n/a"),
            str(_g(b, varied) or "n/a"),
        ]
        cells += [_delta_cell(deltas, _tie_of(ties, pair, q), q) for q in qs]
        cells += [
            _num(_g(ref, "iou")),
            _num(_g(ref, "dice")),
            _num((rank or {}).get(_pair_key(pair))),
            _int(_g(ref, "k")),
            _int(_g(ref, "n_valid")),
        ]
        rows.append(cells)
    return _table(header, rows)


def _tie_section(floor: SegmentFloor, qs: Sequence[float]) -> list[str]:
    """The tie report for one segment. Always says something, including "all narrow"."""
    ties = _g(floor, "ties") or {}
    if not ties:
        return [
            "_No tie band recorded for this segment. A Δ without its band is a point value "
            "whose precision is unverified: rerun with a version that computes "
            "`metrics.tie_bounds`._"
        ]
    families = (("floor", _g(floor, "volume_pairs") or []), ("anchor", _g(floor, "model_pairs") or []))
    bullets: list[str] = []
    total = 0
    widest = 0.0
    for label, pairs in families:
        fixed = "model" if label == "floor" else "volume"
        for i, (pair, deltas) in enumerate(pairs, start=1):
            for q in qs:
                tb = _tie_of(ties, pair, q)
                if tb is None:
                    continue
                total += 1
                width = _band_width(tb)
                if width is not None:
                    widest = max(widest, width)
                flag = _tie_flag(tb)
                if not flag:
                    continue
                who = f"{label} pair {i} ({_g(_g(pair, 'a'), fixed) or 'n/a'})"
                lo = _f(_g(tb, "iou_max"))
                hi = _f(_g(tb, "iou_min"))
                interval = (
                    f"[{1.0 - lo:.3f}, {1.0 - hi:.3f}]"
                    if lo is not None and hi is not None
                    else "unknown"
                )
                line = (
                    f"- {who}, Δ@{_qlabel(q)} = {_num(_delta_value(deltas, q))} in {interval}, "
                    f"band {_num(width)}"
                )
                if flag == "!!":
                    line += (
                        ". The whole budget k came from one plateau of equal values on both "
                        "sides, so this Δ is an artifact of a saturated tail shared by the "
                        "two maps and not a measurement of placement"
                    )
                bullets.append(line + ".")
    if not bullets:
        return [
            f"All {total} tie bands are at or below {TIE_WARN_WIDTH:.3f} "
            f"(widest {widest:.3f}), so the Δ values above are facts about the data rather "
            "than consequences of index order."
        ]
    out = [
        f"**{len(bullets)} of {total} Δ values must be read as intervals.** Their tie band is "
        f"wider than {TIE_WARN_WIDTH:.3f}, which means the point value is largely a "
        "consequence of which pixels of a plateau of equal values happened to be taken:",
        "",
    ]
    out.extend(bullets)
    return out


NOTE_CHARS = 140


def _short_note(value: Any) -> str:
    """A diagnostic note, trimmed for a one-line bullet. The full text stays in the JSON.

    Does not reword and does not translate: the note belongs to the module that produced it.
    """
    text = " ".join((value or "").split())
    if len(text) <= NOTE_CHARS:
        return text
    cut = text[:NOTE_CHARS].rsplit(" ", 1)[0]
    return f"{cut} [...], full note in the JSON report"


def _mesh_line(mesh: MeshCheck | None) -> str:
    """Always one line, whatever `mesh` is. A missing check is a result, not a blank."""
    if mesh is None:
        return (
            "- mesh: **not checked** (no volume pair on this segment, one tifxyz is not "
            "published, or the run used --no-geometry)"
        )
    identical = _g(mesh, "identical")
    verdict = "identical" if identical else "different"
    if identical is None:
        verdict = "unknown"
    shape_a = _g(mesh, "shape_a")
    shape_b = _g(mesh, "shape_b")
    raw = _g(mesh, "max_abs_diff")
    diffs = dict(raw) if isinstance(raw, Mapping) else {}
    parts = [f"{ch}={_num(diffs.get(ch), 6)}" for ch in ("x", "y", "z")]
    note = _short_note(_g(mesh, "note"))
    line = (
        f"- mesh: **{verdict}**, shape {shape_a or 'n/a'} vs {shape_b or 'n/a'}, "
        f"max abs diff {' '.join(parts)}"
    )
    return line + (f" ({note})" if note else "")


def _intensity_line(fit: IntensityFit | None) -> str:
    """Always one line, whatever `fit` is."""
    if fit is None:
        return (
            "- intensity: **not fitted** (no volume pair, the two volumes disagree on "
            "shape, or the run used --no-geometry)"
        )
    slope = _num(_g(fit, "slope"), 4)
    intercept = _num(_g(fit, "intercept"), 2)
    r = _num(_g(fit, "r"), 5)
    n = _int(_g(fit, "n_voxel"))
    med_a = _num(_g(fit, "median_a"), 1)
    med_b = _num(_g(fit, "median_b"), 1)
    clip_a = _pct(_g(fit, "clip_frac_a"))
    clip_b = _pct(_g(fit, "clip_frac_b"))
    chunks = _g(fit, "chunks_used") or []
    chunk_txt = (
        ", ".join("(" + ",".join(str(v) for v in _chunk_json(c)) + ")" for c in chunks)
        if chunks
        else "n/a"
    )
    line = (
        f"- intensity: A = {slope}*B + {intercept} (r = {r}, n = {n} voxels), "
        f"median {med_a} vs {med_b}, at or above the clip ceiling 200: {clip_a} vs {clip_b}"
    )
    z = _i(_g(fit, "z_offset"))
    if z is not None:
        line += f", best z offset {z}"
    line += f", chunks {chunk_txt}"
    note = _short_note(_g(fit, "note"))
    return line + (f" ({note})" if note else "")


def _nulls_table(nulls: Mapping[str, Delta] | None, floor_iou: float | None) -> list[str]:
    if not nulls:
        return ["_No null control recorded for this segment._"]
    header = ["control", "expected", "Δ", "IoU", "Dice", "k", "valid px", "verdict"]
    rows = []
    for name in sorted(nulls):
        d = nulls[name]
        iou = _g(d, "iou")
        iou_f: float | None
        try:
            iou_f = None if iou is None else float(iou)
        except (TypeError, ValueError):
            iou_f = None
        if name == "self":
            expected = "IoU = 1"
            if iou_f is None:
                verdict = "unknown"
            elif abs(iou_f - 1.0) <= 1e-6:
                verdict = "ok"
            else:
                verdict = "FAIL: the ruler disagrees with itself"
        elif name.startswith("shift"):
            expected = "IoU well below the pair IoU"
            if iou_f is None:
                verdict = "unknown"
            elif floor_iou is None:
                verdict = "no floor pair to compare against"
            elif iou_f < floor_iou:
                verdict = "ok"
            else:
                verdict = "SUSPECT: a rigid shift agrees as much as the two derivations"
        else:
            # A control this renderer was not told how to read. Printing "ok" would be a
            # claim nobody made.
            expected = "not declared"
            verdict = "not interpreted here"
        delta = None if iou_f is None else 1.0 - iou_f
        rows.append(
            [
                name,
                expected,
                _num(delta),
                _num(iou_f),
                _num(_g(d, "dice")),
                _int(_g(d, "k")),
                _int(_g(d, "n_valid")),
                verdict,
            ]
        )
    return _table(header, rows)


def _summary_row(floor: SegmentFloor, q: float) -> list[str]:
    vol = _g(floor, "volume_pairs") or []
    mod = _g(floor, "model_pairs") or []
    f_med = _median([_delta_value(d, q) for _, d in vol])
    a_med = _median([_delta_value(d, q) for _, d in mod])
    if f_med is None or a_med is None or a_med == 0:
        ratio = "n/a"
    else:
        ratio = _num(f_med / a_med, 2)
    mesh = _g(floor, "mesh")
    if mesh is None:
        mesh_txt = "not checked"
    else:
        ident = _g(mesh, "identical")
        mesh_txt = "yes" if ident else ("no" if ident is not None else "unknown")
    return [
        f"{_g(floor, 'sample')} / {_g(floor, 'segment')}",
        _num(f_med),
        _num(a_med),
        ratio,
        mesh_txt,
        f"{len(vol)} / {len(mod)}",
    ]


HEADER_NOTE = (
    "Δ@q = 1 - IoU between the top-q% of two ink predictions, at a matched positive budget: "
    "the same number of positives k on each side, so a difference in calibration is not read "
    "as a difference in placement. 0 means the two maps put ink in the same pixels, 1 means "
    "they are disjoint.\n\n"
    "**Floor** = same model, two derivations of the same scan. **Anchor** = same derivation, "
    "two models. The anchor is the difference the community already treats as real, so it is "
    "the scale the floor is read against. A floor near its anchor is a measurement, not an "
    "explanation: inkfloor does not establish a cause.\n\n"
    "**How to read a Δ cell.** Each one is `Δ [low, high]`, where the bracket is the exact "
    "interval the Δ can take over every admissible tie-break, mirrored onto Δ from the IoU "
    "bounds that `metrics.tie_bounds` returns. Published maps are 8 bit, so the top-q% cut "
    "usually lands inside a plateau of pixels that share one value, and which members of the "
    "plateau get taken is arbitrary. A narrow bracket means the Δ is a fact about the data. "
    f"`!` marks a band wider than {TIE_WARN_WIDTH:.3f}, which has to be read as an interval. "
    "`!!` marks the degenerate case: the whole budget came from one plateau on both sides, so "
    "the Δ is an artifact of a shared saturated tail and not a measurement of placement. That "
    "case can print a Δ of 0.000, which looks like the best possible result and is worth "
    "nothing.\n\n"
    "**ρ (rank)** is Spearman's rank correlation over the valid pixels. It is invariant to "
    "any monotone rescaling of either map, so it answers a different question from Δ: ρ asks "
    "whether the two maps agree on the ordering of every pixel, Δ asks whether they agree on "
    "which pixels make the top of the list. High ρ with high Δ is informative and not a "
    "contradiction: it says the two maps rank the surface almost identically and still "
    "disagree about where the ink is."
)


def _chance_block(floors: Sequence[SegmentFloor], qs: Sequence[float]) -> list[str]:
    """The chance level for each q, once per report.

    Deltas at different q are not commensurable. The expected IoU of two independent
    selections of the same size is q/(2-q), so it grows with q, and a Δ has to be read
    against the chance level of its own q instead of against zero. The values come from
    `metrics.chance_iou`, carried in the record: this renderer does not recompute the formula.
    """
    merged: dict[float, float] = {}
    for f in floors:
        for q, v in (_g(f, "chance") or {}).items():
            fq, fv = _f(q), _f(v)
            if fq is not None and fv is not None:
                merged[fq] = fv
    if not merged:
        return [
            "_Chance level not recorded. Δ values at different q cannot be compared against "
            "each other without it._"
        ]
    order = [q for q in qs if q in merged] or sorted(merged)
    parts = [
        f"{_qlabel(q)}: IoU {merged[q]:.3f}, Δ {1.0 - merged[q]:.3f}" for q in order
    ]
    return [
        "**Chance level per q** (two independent selections of k pixels, expected IoU "
        "q/(2-q)): " + "; ".join(parts) + ". The floor of the metric grows with q, so a Δ is "
        "read against the chance level of its own q and never against zero. Δ values at "
        "different q are not commensurable with each other."
    ]


def to_markdown(floors: Sequence[SegmentFloor]) -> str:
    """Render the report. Never omits a row: a None field prints as an explicit non-result.

    Does not sort by severity and does not hide a segment whose numbers look uninteresting.
    """
    floors = list(floors or [])
    out = ["# inkfloor report", "", HEADER_NOTE, ""]
    if not floors:
        out += ["No segment measured. Nothing to report.", ""]
        return "\n".join(out)

    seen_qs: set[float] = set()
    for f in floors:
        seen_qs.update(_qs_of(_g(f, "volume_pairs") or []))
        seen_qs.update(_qs_of(_g(f, "model_pairs") or []))
    all_qs = sorted(seen_qs)
    q_ref = _primary_q(all_qs) if all_qs else 0.05
    with_both = sum(
        1 for f in floors if (_g(f, "volume_pairs") or []) and (_g(f, "model_pairs") or [])
    )
    out += [
        f"Segments measured: {len(floors)} ({with_both} with both a floor pair and an anchor pair)",
        "",
        f"q grid: {', '.join(_qlabel(q) for q in all_qs) if all_qs else 'none'}",
        "",
    ]
    out += _chance_block(floors, all_qs)
    out.append("")
    out += _table(
        [
            "segment",
            f"floor Δ@{_qlabel(q_ref)} (median)",
            f"anchor Δ@{_qlabel(q_ref)} (median)",
            "floor / anchor",
            "mesh identical",
            "pairs floor / anchor",
        ],
        [_summary_row(f, q_ref) for f in floors],
    )
    out.append("")

    for f in floors:
        vol = _g(f, "volume_pairs") or []
        mod = _g(f, "model_pairs") or []
        qs = sorted(set(_qs_of(vol)) | set(_qs_of(mod))) or list(all_qs)
        ties = _g(f, "ties")
        rank = _g(f, "spearman")
        out += [f"## {_g(f, 'sample')} / {_g(f, 'segment')}", ""]
        out += ["### Floor: same model, different derivation of the same scan", ""]
        out += _pair_table(vol, qs, "volume", ties, rank)
        out += ["", "### Anchor: same derivation, different model", ""]
        out += _pair_table(mod, qs, "model", ties, rank)
        out += ["", "### Tie bands", ""]
        out += _tie_section(f, qs)
        out += ["", "### Confounder checks", ""]
        out += [_mesh_line(_g(f, "mesh")), _intensity_line(_g(f, "intensity")), ""]
        floor_iou = _median(
            [
                None if (d := _delta(deltas, _primary_q(qs))) is None else _g(d, "iou")
                for _, deltas in vol
            ]
        )
        out += ["### Null controls", ""]
        out += _nulls_table(_g(f, "nulls"), floor_iou)
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- json


def _pred_json(p: Prediction | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {
        "key": _g(p, "key"),
        "sample": _g(p, "sample"),
        "segment": _g(p, "segment"),
        "volume": _g(p, "volume"),
        "model": _g(p, "model"),
        "voxel_um": _f(_g(p, "voxel_um")),
        "tile": _i(_g(p, "tile")),
        "stride": _i(_g(p, "stride")),
        "size_bytes": _i(_g(p, "size_bytes")),
        # Null when the file name does not declare a pyramid level. Two predictions are
        # comparable pixel by pixel only at the same (voxel_um, level).
        "level": _i(_g(p, "level")),
    }


def _delta_json(d: Delta | None) -> dict[str, Any] | None:
    if d is None:
        return None
    iou = _f(_g(d, "iou"))
    return {
        "q": _f(_g(d, "q")),
        "iou": iou,
        "delta": None if iou is None else 1.0 - iou,
        "dice": _f(_g(d, "dice")),
        "n_valid": _i(_g(d, "n_valid")),
        "k": _i(_g(d, "k")),
    }


def _tie_json(tb: TieBand | None) -> dict[str, Any] | None:
    """The tie band as `metrics` produced it, plus the same interval mirrored onto Δ.

    `unique` and `width` are properties on the real dataclass; `width` is recomputed from the
    bounds when a record does not carry it, so the field is never silently absent.
    """
    if tb is None:
        return None
    lo, hi = _f(_g(tb, "iou_min")), _f(_g(tb, "iou_max"))
    return {
        "iou": _f(_g(tb, "iou")),
        "iou_min": lo,
        "iou_max": hi,
        "delta_min": None if hi is None else 1.0 - hi,
        "delta_max": None if lo is None else 1.0 - lo,
        "width": _band_width(tb),
        "unique": _g(tb, "unique"),
        "degenerate": _is_degenerate(tb),
        "wide": _tie_flag(tb) != "",
        "q": _f(_g(tb, "q")),
        "k": _i(_g(tb, "k")),
        "n_valid": _i(_g(tb, "n_valid")),
        "thr_a": _f(_g(tb, "thr_a")),
        "thr_b": _f(_g(tb, "thr_b")),
        "forced_a": _i(_g(tb, "forced_a")),
        "forced_b": _i(_g(tb, "forced_b")),
        "ties_needed_a": _i(_g(tb, "ties_needed_a")),
        "ties_needed_b": _i(_g(tb, "ties_needed_b")),
        "tie_band_a": _i(_g(tb, "tie_band_a")),
        "tie_band_b": _i(_g(tb, "tie_band_b")),
    }


def _pair_json(
    pair: Pair,
    deltas: Mapping[float, Delta] | None,
    ties: Mapping[tuple[str, str], Mapping[float, TieBand]] | None = None,
    rank: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for q, d in sorted((deltas or {}).items()):
        entry = _delta_json(d)
        if entry is not None:
            # The band lives inside its own q, so a consumer cannot read a Δ without meeting
            # the interval it is allowed to move in.
            entry["tie"] = _tie_json(_tie_of(ties, pair, q))
        entries[_qkey(q)] = entry
    return {
        "kind": _g(pair, "kind"),
        "a": _pred_json(_g(pair, "a")),
        "b": _pred_json(_g(pair, "b")),
        "spearman": _f((rank or {}).get(_pair_key(pair))),
        "deltas": entries,
    }


def _mesh_json(mesh: MeshCheck | None) -> dict[str, Any] | None:
    if mesh is None:
        return None
    raw = _g(mesh, "max_abs_diff")
    diffs = dict(raw) if isinstance(raw, Mapping) else {}
    return {
        "identical": _g(mesh, "identical"),
        "shape_a": [_i(v) for v in (_g(mesh, "shape_a") or ())],
        "shape_b": [_i(v) for v in (_g(mesh, "shape_b") or ())],
        "max_abs_diff": {str(k): _f(v) for k, v in diffs.items()},
        "note": _g(mesh, "note"),
    }


def _pairs_of(value: Any) -> list[tuple[Any, Any]]:
    """Read a sequence of two-item pairs, or nothing at all. Never raises on a stub."""
    out: list[tuple[Any, Any]] = []
    for item in value or ():
        try:
            a, b = item
        except (TypeError, ValueError):
            continue
        out.append((a, b))
    return out


def _chunk_json(c: Any) -> list[int | None]:
    try:
        return [_i(v) for v in c]
    except TypeError:
        return [_i(c)]


def _intensity_json(fit: IntensityFit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {
        "slope": _f(_g(fit, "slope")),
        "intercept": _f(_g(fit, "intercept")),
        "r": _f(_g(fit, "r")),
        "n_voxel": _i(_g(fit, "n_voxel")),
        "median_a": _f(_g(fit, "median_a")),
        "median_b": _f(_g(fit, "median_b")),
        "clip_frac_a": _f(_g(fit, "clip_frac_a")),
        "clip_frac_b": _f(_g(fit, "clip_frac_b")),
        "chunks_used": [_chunk_json(c) for c in (_g(fit, "chunks_used") or ())],
        # Declared after the contract fields, with defaults on the geometry side: the z
        # alignment was measured rather than assumed, and the report carries the proof.
        "z_offset": _i(_g(fit, "z_offset")),
        "r_by_offset": [
            [_i(o), _f(v)] for o, v in _pairs_of(_g(fit, "r_by_offset"))
        ],
        "note": _g(fit, "note"),
    }


def _segment_json(f: SegmentFloor) -> dict[str, Any]:
    ties = _g(f, "ties")
    rank = _g(f, "spearman")
    vol = _g(f, "volume_pairs") or []
    mod = _g(f, "model_pairs") or []
    qs = sorted(set(_qs_of(vol)) | set(_qs_of(mod)))
    chance = {
        _qkey(q): _f(v)
        for q, v in sorted((_g(f, "chance") or {}).items(), key=lambda kv: _f(kv[0]) or 0.0)
    }
    return {
        "sample": _g(f, "sample"),
        "segment": _g(f, "segment"),
        # Present so a consumer never has to recompute q/(2-q) to know what a Δ is worth.
        "chance_iou": chance,
        "chance_delta": {k: (None if v is None else 1.0 - v) for k, v in chance.items()},
        "tie_warn_width": TIE_WARN_WIDTH,
        "tie_warnings": _tie_warnings_json(f, qs),
        "volume_pairs": [_pair_json(p, d, ties, rank) for p, d in vol],
        "model_pairs": [_pair_json(p, d, ties, rank) for p, d in mod],
        "mesh": _mesh_json(_g(f, "mesh")),
        "intensity": _intensity_json(_g(f, "intensity")),
        "nulls": {str(k): _delta_json(v) for k, v in sorted((_g(f, "nulls") or {}).items())},
    }


def _tie_warnings_json(f: SegmentFloor, qs: Sequence[float]) -> list[dict[str, Any]]:
    """The flagged cells, so a machine reader meets them without scanning every band."""
    ties = _g(f, "ties") or {}
    out: list[dict[str, Any]] = []
    for family, pairs in (
        ("volume", _g(f, "volume_pairs") or []),
        ("model", _g(f, "model_pairs") or []),
    ):
        for i, (pair, _deltas) in enumerate(pairs):
            for q in qs:
                tb = _tie_of(ties, pair, q)
                if tb is None or not _tie_flag(tb):
                    continue
                out.append(
                    {
                        "family": family,
                        "index": i,
                        "a": _g(_g(pair, "a"), "key"),
                        "b": _g(_g(pair, "b"), "key"),
                        "q": _f(q),
                        "width": _band_width(tb),
                        "degenerate": _is_degenerate(tb),
                    }
                )
    return out


def to_dict(floors: Sequence[SegmentFloor]) -> dict[str, Any]:
    """The report as plain data. Optional fields are present and null, never dropped."""
    floors = list(floors or [])
    return {
        "schema": "inkfloor/report/2",
        "metric": "delta_at_q = 1 - IoU of the top-q% at a matched positive budget",
        "reading": (
            "Every delta carries a 'tie' object with the exact interval it can take over all "
            "admissible tie-breaks. A delta whose tie.wide is true is an interval, not a "
            "point. A delta whose tie.degenerate is true says nothing about placement. "
            "Compare a delta against chance_delta at the same q, never against zero, and "
            "never against a delta at a different q."
        ),
        "segments": [_segment_json(f) for f in floors],
    }


def to_json(floors: Sequence[SegmentFloor], *, indent: int = 2) -> str:
    """The report as JSON. Deterministic: no timestamps, no paths, no run metadata."""
    return json.dumps(to_dict(floors), indent=indent, sort_keys=False, allow_nan=False)
