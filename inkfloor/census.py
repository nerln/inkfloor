"""Census of published ink predictions and classification of pairs.

The corpus publishes predictions as TIFFs under
`<sample>/segments/<segment>/ink-detection/`, with all metadata in the filename. This module
reads those names and DOES NOT open the files: it does not decode a single pixel or download
a single TIFF. Only S3 listings and string parsing.

The rule governing everything: `parse_prediction` returns None rather than guessing. A name
that does not match the known schema is discarded and counted, never partially interpreted.
A parser that guesses corrupts the downstream census and leaves no trace.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import cache

# ---------------------------------------------------------------------------
# Bucket layout
# ---------------------------------------------------------------------------

SEGMENTS_DIR = "segments"
INK_DIR = "ink-detection"

#: Directory names that contain artifacts, rather than other segments, within a segment.
#: This only avoids wasting requests: if a name is missing here, the census stays correct
#: but makes a few more GETs. If there is one name too many, a nested segment with that name
#: would be skipped, so the list contains only names actually seen in the corpus.
ARTIFACT_DIRS = frozenset(
    {
        INK_DIR,
        "mesh",
        "surface-volumes",
        "surface-renders",
        "layers",
        "layers_ink",
        "overlapping",
        "versions",
        "downsampled",
    }
)

#: How many S3 listings to run in parallel. They are anonymous GETs to a public bucket;
#: nothing is written.
MAX_WORKERS = 8

#: How many discarded key names to retain as examples in the report.
MAX_DISCARDED_EXAMPLES = 12


# ---------------------------------------------------------------------------
# Filename schema
# ---------------------------------------------------------------------------

# PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838
#   -20250713185324-timesformer_scroll5_july_retreat-tile64-stride16.tif
#
# PHerc0139-20250108000000-1.129um-0.22m-59keV-volume-20260413113053-L1
#   -20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif
#
# Optional parts verified in the corpus: source-to-object distance (`-0.22m`), pyramid level
# (`-L1`), and the tile/stride pair. The model name can contain hyphens and even fragments
# such as `1um`, so the voxel is anchored immediately after the segment timestamp and the
# model immediately after the volume ID (or after the level, when present).
_NAME_RX = re.compile(
    r"^(?P<sample>[A-Za-z0-9]+)"
    r"-(?P<segment_ts>\d{14})"
    r"-(?P<voxel_um>\d+(?:\.\d+)?)um"
    r"(?:-(?P<dist_m>\d+(?:\.\d+)?)m)?"
    r"-(?P<kev>\d+(?:\.\d+)?)keV"
    r"-volume-(?P<volume>\d{14})"
    r"(?:-L(?P<level>\d+))?"
    r"-(?P<model>\d{14}-.+?)"
    r"(?:-tile(?P<tile>\d+)-stride(?P<stride>\d+))?"
    r"\.tif$"
)

# `.+?` for the model name is permissive: if the filename tail is malformed (for example,
# `-tile64` without `-stride16`, or a `-ds8` suffix after the stride), the malformed piece
# would end up in the model name and tile/stride would remain None, with nothing flagging the
# problem. Two predictions with different strides would appear to use the same model.
# Therefore, if a `-tile<digits>` or `-stride<digits>` token remains in the model, the name
# is discarded.
_LEFTOVER_RX = re.compile(r"-(?:tile|stride)\d")

# Subdirectory for reduced previews: the same prediction, with a smaller raster.
DOWNSAMPLED_DIR = "downsampled"

# Image extensions that are NOT .tif. They are used only to distinguish an unexpected format
# from an arbitrary file: they are never read, because the census reads only names and sizes.
_IMAGE_SUFFIXES = (".tiff", ".png", ".jpg", ".jpeg")

# Discard reasons, in check order.
R_NOT_TIF = "not-a-tif"
R_OTHER_IMAGE = "image-under-ink-detection-that-is-not-a-tif"
R_NOT_INK = "not-under-ink-detection"
R_NESTED = "nested-under-ink-detection"
R_NAME = "name-not-recognised"
R_SAMPLE = "sample-mismatch-with-path"
R_SEGMENT = "segment-mismatch-with-path"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """A published ink prediction, as described by its filename.

    Says NOTHING about the content: shape, dtype, and fraction of valid pixels are known only
    by opening the TIFF, which is not touched here. `size_bytes` is the size of the S3 object,
    not the pixel count: compression makes object sizes incomparable across predictions.
    """

    key: str
    sample: str
    segment: str
    volume: str
    model: str
    voxel_um: float
    tile: int | None
    stride: int | None
    size_bytes: int
    # Additional fields, all with defaults: code constructing a Prediction with only the
    # contract fields keeps working. `level` is None when the name does not declare it, and
    # remains unknown in that case: 0 is not assumed.
    level: int | None = None
    kev: float | None = None
    dist_m: float | None = None

    @property
    def segment_prefix(self) -> str:
        """The segment's S3 prefix, including the trailing slash. Does not check if it exists."""
        return self.key.split(f"/{INK_DIR}/", 1)[0] + "/"

    @property
    def step_um(self) -> float | None:
        """Effective raster step: `voxel_um * 2**level`.

        None when the pyramid level is not declared in the name. It DOES NOT guess L=0: a
        name without a level does not say the level is zero; it says that we do not know it.
        """
        if self.level is None:
            return None
        return self.voxel_um * float(2**self.level)

    @property
    def raster(self) -> tuple[float, int | None]:
        """Identity of the raster on which the map lives: (voxel_um, level).

        Two predictions can be overlaid pixel for pixel only if this pair matches. It is NOT
        a shape: it does not guarantee that the two TIFFs have the same dimensions.
        """
        return (self.voxel_um, self.level)


@dataclass(frozen=True)
class Pair:
    """Two predictions for the same segment, and what differs between them."""

    a: Prediction
    b: Prediction
    kind: str  # "volume" | "model" | "both"


@dataclass(frozen=True)
class CensusReport:
    """Census result, including an accounting of what was discarded.

    Prints NOTHING: the fields are intended to be formatted by `cli.py`.
    """

    predictions: list[Prediction]
    samples_scanned: list[str]
    n_keys_seen: int
    n_kept: int
    n_discarded: int
    discarded_by_reason: dict[str, int]
    discarded_examples: list[tuple[str, str]]  # (key, reason), example
    n_ink_dirs: int
    n_segments_with_predictions: int
    samples_with_predictions: dict[str, int]


@dataclass(frozen=True)
class PairStats:
    """The retained pairs and, above all, the excluded ones and why."""

    pairs: list[Pair]
    by_kind: dict[str, int]
    n_candidate_pairs: int
    n_excluded_both: int
    n_excluded_raster: int
    n_excluded_duplicate: int
    excluded_raster_examples: list[tuple[str, str]] = field(default_factory=list)
    segments_with_volume_pair: list[tuple[str, str]] = field(default_factory=list)
    segments_with_model_pair: list[tuple[str, str]] = field(default_factory=list)
    n_segments_grouped: int = 0
    n_segments_single: int = 0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_prediction(key: str, size_bytes: int) -> Prediction | None:
    """Extract fields from the filename. None if the name does not follow the known schema.

    Does NOT guess: if a required part is missing, if the sample or segment timestamp in the
    name does not match the path, or if the file is not a TIFF directly inside
    `ink-detection/`, it returns None. In particular, it discards `downsampled/*.jpg`
    previews, which are the same prediction reduced 8x, not the prediction itself.

    Does NOT read the file and does NOT touch the network.
    """
    pred, _ = _parse_with_reason(key, size_bytes)
    return pred


def _parse_with_reason(key: str, size_bytes: int) -> tuple[Prediction | None, str | None]:
    """Like `parse_prediction`, but also says why it discarded a key. Internal report use.

    A non-.tif file under ink-detection is distinguished from previews: today every discard
    is a .jpg inside `downsampled/`, meaning the same reduced prediction, and ignoring it is
    correct. But the census supports an exhaustive claim ("only one segment has a pair
    of derivations"), and that claim would silently become false the day a prediction with
    another extension was published. That case has its own discard reason, so the report can
    shout about it instead of adding it to the previews.
    """
    if not key.endswith(".tif"):
        under_ink = f"/{INK_DIR}/" in key
        is_preview = f"/{DOWNSAMPLED_DIR}/" in key
        if under_ink and not is_preview and key.lower().endswith(_IMAGE_SUFFIXES):
            return None, R_OTHER_IMAGE
        return None, R_NOT_TIF

    marker = f"/{INK_DIR}/"
    if marker not in key or f"/{SEGMENTS_DIR}/" not in key:
        return None, R_NOT_INK

    head, tail = key.split(marker, 1)
    if "/" in tail:
        return None, R_NESTED

    m = _NAME_RX.match(tail)
    if m is None:
        return None, R_NAME
    if _LEFTOVER_RX.search(m["model"]):
        return None, R_NAME

    parts = head.split("/")
    if len(parts) < 3:
        return None, R_NOT_INK
    sample_dir, segment_dir = parts[0], parts[-1]

    if m["sample"] != sample_dir:
        return None, R_SAMPLE
    if not segment_dir.startswith(m["segment_ts"]):
        return None, R_SEGMENT

    tile = m["tile"]
    stride = m["stride"]
    level = m["level"]
    return (
        Prediction(
            key=key,
            sample=sample_dir,
            segment=segment_dir,
            volume=m["volume"],
            model=m["model"],
            voxel_um=float(m["voxel_um"]),
            tile=int(tile) if tile is not None else None,
            stride=int(stride) if stride is not None else None,
            size_bytes=size_bytes,
            level=int(level) if level is not None else None,
            kev=float(m["kev"]),
            dist_m=float(m["dist_m"]) if m["dist_m"] is not None else None,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def samples_in_corpus() -> list[str]:
    """The published samples, in alphabetical order.

    Does NOT include service prefixes beginning with `_` (for example, `_thumbnails/`).
    """
    return sorted(
        p.rstrip("/") for p in cache.list_prefixes("") if p and not p.startswith("_")
    )


def _ink_keys_for_sample(sample: str) -> tuple[list[tuple[str, int]], int]:
    """All keys under a sample's `ink-detection/` directories, and how many it found.

    First tries `<sample>/segments/<candidate>/ink-detection/`. If a candidate has no
    predictions, it looks one level lower: the corpus contains intermediate containers
    (`segments/raw/<segment>/`). It does NOT descend beyond that level.
    """
    candidates = cache.list_prefixes(f"{sample}/{SEGMENTS_DIR}/")
    keys: list[tuple[str, int]] = []
    n_dirs = 0
    empty: list[str] = []

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for cand, found in zip(
            candidates, pool.map(lambda c: cache.list_keys(f"{c}{INK_DIR}/"), candidates)
        ):
            if found:
                keys.extend(found)
                n_dirs += 1
            else:
                empty.append(cand)

        nested: list[str] = []
        for cand, children in zip(empty, pool.map(cache.list_prefixes, empty)):
            for child in children:
                if child.rstrip("/").rsplit("/", 1)[-1] in ARTIFACT_DIRS:
                    continue
                nested.append(child)

        for found in pool.map(lambda c: cache.list_keys(f"{c}{INK_DIR}/"), nested):
            if found:
                keys.extend(found)
                n_dirs += 1

    return keys, n_dirs


def census_report(
    samples: list[str] | None = None,
    max_examples: int = MAX_DISCARDED_EXAMPLES,
) -> CensusReport:
    """Like `census`, but also returns an accounting of discards.

    Does NOT print or download TIFFs: it only calls ListObjectsV2 on the bucket. The count of
    keys seen includes everything under `ink-detection/`, previews included, so that
    `n_keys_seen == n_kept + n_discarded` can be checked at a glance.
    """
    names = samples_in_corpus() if samples is None else [s.rstrip("/") for s in samples]

    kept: list[Prediction] = []
    reasons: Counter[str] = Counter()
    # The example set guarantees one name for each discard reason, then fills to the limit,
    # so a rare reason is never hidden by hundreds of discards of the same type.
    first_of_reason: dict[str, str] = {}
    extra_examples: list[tuple[str, str]] = []
    n_keys = 0
    n_ink_dirs = 0

    for sample in names:
        keys, n_dirs = _ink_keys_for_sample(sample)
        n_ink_dirs += n_dirs
        for key, size in sorted(keys):
            n_keys += 1
            pred, reason = _parse_with_reason(key, size)
            if pred is not None:
                kept.append(pred)
                continue
            reason = reason or "unknown"
            reasons[reason] += 1
            if reason not in first_of_reason:
                first_of_reason[reason] = key
            elif len(extra_examples) < max_examples:
                extra_examples.append((key, reason))

    examples = [(k, r) for r, k in sorted(first_of_reason.items())]
    room = max(0, max_examples - len(examples))
    examples.extend(extra_examples[:room])

    kept.sort(key=lambda p: p.key)
    per_sample: Counter[str] = Counter(p.sample for p in kept)
    segments = {(p.sample, p.segment) for p in kept}

    return CensusReport(
        predictions=kept,
        samples_scanned=names,
        n_keys_seen=n_keys,
        n_kept=len(kept),
        n_discarded=sum(reasons.values()),
        discarded_by_reason=dict(reasons),
        discarded_examples=examples,
        n_ink_dirs=n_ink_dirs,
        n_segments_with_predictions=len(segments),
        samples_with_predictions=dict(sorted(per_sample.items())),
    )


#: Report from the latest census run in this process. It is used only by `census_stats()`,
#: because the `census()` signature returns predictions and has no room for the accounting.
#: No module logic reads it: `census_report` and `pair_stats` remain pure functions.
_last_report: CensusReport | None = None


def census(samples: list[str] | None = None) -> list[Prediction]:
    """Enumerate ink predictions for the entire corpus (or only the specified samples).

    Does NOT return discards: use `census_report` for those; it gives the same list plus an
    accounting of how many keys were discarded and why. Code with access to only this
    signature can read `census_stats()` immediately after the call.
    """
    global _last_report
    _last_report = census_report(samples)
    return _last_report.predictions


def census_stats() -> dict[str, object]:
    """Accounting for this process's latest `census()`, for code that prints it.

    Empty dictionary if `census()` has not been called yet: it does NOT run a census to answer
    and does not touch the network. The `skipped`, `n_skipped`, and `unparsed` keys carry the
    same number: how many keys under `ink-detection/` were rejected.
    """
    r = _last_report
    if r is None:
        return {}
    return {
        "skipped": r.n_discarded,
        "n_skipped": r.n_discarded,
        "unparsed": r.n_discarded,
        "keys_seen": r.n_keys_seen,
        "kept": r.n_kept,
        "by_reason": dict(r.discarded_by_reason),
        "examples": list(r.discarded_examples),
        "segments_with_predictions": r.n_segments_with_predictions,
        "ink_dirs": r.n_ink_dirs,
    }


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def comparable(a: Prediction, b: Prediction) -> bool:
    """True if both maps live on the same raster: same voxel and same level.

    Does NOT check TIFF shapes or mesh alignment: it only says that comparing them pixel for
    pixel makes sense. An unknown level (None) is considered equal only to another unknown
    level, because two names that omit the level omit the same information.
    """
    return abs(a.voxel_um - b.voxel_um) < 1e-9 and a.level == b.level


def _kind(a: Prediction, b: Prediction) -> str | None:
    same_volume = a.volume == b.volume
    same_model = a.model == b.model
    if same_volume and same_model:
        return None  # same volume and same model: it is not a pair, but a duplicate
    if same_model:
        return "volume"
    if same_volume:
        return "model"
    return "both"


def pair_stats(preds: list[Prediction]) -> PairStats:
    """Like `pairs`, but also returns what was excluded and why.

    Does NOT print. Does NOT pair predictions from different segments: the floor is measured
    on the same surface, and two different segments have different meshes.
    """
    groups: dict[tuple[str, str], list[Prediction]] = defaultdict(list)
    for p in preds:
        groups[(p.sample, p.segment)].append(p)

    out: list[Pair] = []
    by_kind: Counter[str] = Counter()
    n_candidates = 0
    n_both = 0
    n_raster = 0
    n_dup = 0
    raster_examples: list[tuple[str, str]] = []
    vol_segments: list[tuple[str, str]] = []
    mod_segments: list[tuple[str, str]] = []

    for seg_id in sorted(groups):
        members = sorted(groups[seg_id], key=lambda p: p.key)
        has_vol = has_mod = False
        for a, b in itertools.combinations(members, 2):
            n_candidates += 1
            kind = _kind(a, b)
            if kind is None:
                n_dup += 1
                continue
            if kind == "both":
                n_both += 1
                continue
            if not comparable(a, b):
                n_raster += 1
                if len(raster_examples) < MAX_DISCARDED_EXAMPLES:
                    raster_examples.append((a.key, b.key))
                continue
            out.append(Pair(a=a, b=b, kind=kind))
            by_kind[kind] += 1
            has_vol |= kind == "volume"
            has_mod |= kind == "model"
        if has_vol:
            vol_segments.append(seg_id)
        if has_mod:
            mod_segments.append(seg_id)

    return PairStats(
        pairs=out,
        by_kind=dict(by_kind),
        n_candidate_pairs=n_candidates,
        n_excluded_both=n_both,
        n_excluded_raster=n_raster,
        n_excluded_duplicate=n_dup,
        excluded_raster_examples=raster_examples,
        segments_with_volume_pair=vol_segments,
        segments_with_model_pair=mod_segments,
        n_segments_grouped=len(groups),
        n_segments_single=sum(1 for v in groups.values() if len(v) == 1),
    )


def pairs(preds: list[Prediction]) -> list[Pair]:
    """All comparable pairs, grouped by segment. Excludes kind='both'.

    Does NOT include pairs on different rasters (different voxel or pyramid level): those
    maps have different steps, and overlaying them pixel for pixel would measure resampling,
    not the floor. `pair_stats` reports how many were excluded.
    """
    return pair_stats(preds).pairs
