"""Formatting tests: hand-built records, no network, no other inkfloor module.

The stubs below carry the field names declared in CONTRACTS.md and nothing else. If a real
dataclass drops or renames a field, these tests keep passing while the renderer starts
printing 'n/a', which is the intended behaviour: a report must degrade to an explicit
non-result, never to a missing row.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

import pytest

from inkfloor import report

# --------------------------------------------------------------------------- stubs


@dataclass(frozen=True)
class FakePrediction:
    key: str
    sample: str
    segment: str
    volume: str
    model: str
    voxel_um: float = 7.91
    tile: int | None = 64
    stride: int | None = 16
    size_bytes: int = 42 * 1024 * 1024
    level: int | None = None


@dataclass(frozen=True)
class FakePair:
    a: FakePrediction
    b: FakePrediction
    kind: str


@dataclass(frozen=True)
class FakeDelta:
    q: float
    iou: float
    dice: float
    n_valid: int
    k: int


@dataclass(frozen=True)
class FakeTieBand:
    """The fields of metrics.TieBand, with `width` and `unique` as real properties."""

    q: float
    k: int
    n_valid: int
    iou: float
    iou_min: float
    iou_max: float
    thr_a: float = 128.0
    thr_b: float = 96.0
    forced_a: int = 100
    forced_b: int = 100
    ties_needed_a: int = 10
    ties_needed_b: int = 10
    tie_band_a: int = 40
    tie_band_b: int = 40

    @property
    def unique(self) -> bool:
        return self.tie_band_a == self.ties_needed_a and self.tie_band_b == self.ties_needed_b

    @property
    def width(self) -> float:
        return self.iou_max - self.iou_min


@dataclass(frozen=True)
class FakeMesh:
    identical: bool
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    max_abs_diff: dict[str, float]
    note: str


@dataclass(frozen=True)
class FakeIntensity:
    slope: float
    intercept: float
    r: float
    n_voxel: int
    median_a: float
    median_b: float
    clip_frac_a: float
    clip_frac_b: float
    chunks_used: list[tuple[int, int, int]] = field(default_factory=list)


SEG = "20251107110950-w064_20251107110950052_flatboi"
VOL_A = "20241024131838"
VOL_B = "20241024131839"
MODEL_1 = "20250713185324-timesformer_scroll5_july_retreat"
MODEL_2 = "20251222202946-timesformer_scroll5_november19"
QS = (0.01, 0.05, 0.20)


def pred(volume: str, model: str) -> FakePrediction:
    return FakePrediction(
        key=f"PHerc0172/segments/{SEG}/ink-detection/PHerc0172-x-volume-{volume}-{model}.tif",
        sample="PHerc0172",
        segment=SEG,
        volume=volume,
        model=model,
    )


def deltas(iou_by_q: dict[float, float], n_valid: int = 182_274_183) -> dict[float, FakeDelta]:
    return {
        q: FakeDelta(q=q, iou=iou, dice=2 * iou / (1 + iou), n_valid=n_valid, k=int(n_valid * q))
        for q, iou in iou_by_q.items()
    }


def bands(
    iou_by_q: dict[float, float],
    width_by_q: dict[float, float] | None = None,
    bounds_by_q: dict[float, tuple[float, float]] | None = None,
    n_valid: int = 182_274_183,
) -> dict[float, FakeTieBand]:
    """Tie bands per q: explicit bounds when given, otherwise centred on the IoU.

    Real bands are not centred on the point value, which is the whole reason the report
    prints an interval instead of a plus-or-minus.
    """
    widths = width_by_q or {}
    bounds = bounds_by_q or {}
    out = {}
    for q, iou in iou_by_q.items():
        if q in bounds:
            lo, hi = bounds[q]
        else:
            w = widths.get(q, 0.004)
            lo, hi = iou - w / 2, iou + w / 2
        out[q] = FakeTieBand(
            q=q, k=int(n_valid * q), n_valid=n_valid, iou=iou, iou_min=lo, iou_max=hi
        )
    return out


# IoU at each q from the real run of 2026-08-08. Δ is 1 - IoU, so these are the four published
# villa#1372 numbers at q = 5%: 0.620, 0.713 for the floor and 0.580, 0.750 for the anchor.
FLOOR_1 = {0.01: 0.306, 0.05: 0.380, 0.20: 0.481}
FLOOR_2 = {0.01: 0.236, 0.05: 0.287, 0.20: 0.343}
ANCHOR_1 = {0.01: 0.342, 0.05: 0.420, 0.20: 0.427}
ANCHOR_2 = {0.01: 0.174, 0.05: 0.250, 0.20: 0.327}

# The one measured band reported on the issue: anchor on volume ...838 at q = 1% has IoU 0.342
# inside [0.304, 0.408], a width of 0.104 and not centred on the point value. The q = 5% width
# is the reported upper bound for that q. Every other band here is constructed, narrow on
# purpose, so the wide one is the only thing the warning has to find.
ANCHOR_1_BOUNDS = {0.01: (0.304, 0.408)}
ANCHOR_1_WIDTHS = {0.05: 0.015}


def full_floor(**overrides) -> report.SegmentFloor:
    """The published villa#1372 measurement, as a record. Numbers are the real ones."""
    base = dict(
        sample="PHerc0172",
        segment=SEG,
        volume_pairs=[
            (
                FakePair(pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1), "volume"),
                deltas(FLOOR_1),
            ),
            (
                FakePair(pred(VOL_A, MODEL_2), pred(VOL_B, MODEL_2), "volume"),
                deltas(FLOOR_2),
            ),
        ],
        model_pairs=[
            (
                FakePair(pred(VOL_A, MODEL_1), pred(VOL_A, MODEL_2), "model"),
                deltas(ANCHOR_1),
            ),
            (
                FakePair(pred(VOL_B, MODEL_1), pred(VOL_B, MODEL_2), "model"),
                deltas(ANCHOR_2),
            ),
        ],
        ties={
            (pred(VOL_A, MODEL_1).key, pred(VOL_B, MODEL_1).key): bands(FLOOR_1),
            (pred(VOL_A, MODEL_2).key, pred(VOL_B, MODEL_2).key): bands(FLOOR_2),
            (pred(VOL_A, MODEL_1).key, pred(VOL_A, MODEL_2).key): bands(
                ANCHOR_1, ANCHOR_1_WIDTHS, ANCHOR_1_BOUNDS
            ),
            (pred(VOL_B, MODEL_1).key, pred(VOL_B, MODEL_2).key): bands(ANCHOR_2),
        },
        chance={0.01: 0.01 / 1.99, 0.05: 0.05 / 1.95, 0.20: 0.20 / 1.80},
        spearman={
            (pred(VOL_A, MODEL_1).key, pred(VOL_B, MODEL_1).key): 0.921,
            (pred(VOL_A, MODEL_2).key, pred(VOL_B, MODEL_2).key): 0.887,
            (pred(VOL_A, MODEL_1).key, pred(VOL_A, MODEL_2).key): 0.734,
            (pred(VOL_B, MODEL_1).key, pred(VOL_B, MODEL_2).key): 0.702,
        },
        mesh=FakeMesh(
            identical=True,
            shape_a=(671, 747),
            shape_b=(671, 747),
            max_abs_diff={"x": 0.0, "y": 0.0, "z": 0.0},
            note="",
        ),
        intensity=FakeIntensity(
            slope=0.6154,
            intercept=104.32,
            r=0.99987,
            n_voxel=5_242_880,
            median_a=132.0,
            median_b=44.9,
            clip_frac_a=0.031,
            clip_frac_b=0.0042,
            chunks_used=[(0, 0, 94), (0, 0, 108), (0, 1, 110), (0, 1, 112), (0, 2, 116)],
        ),
        nulls={
            "self": FakeDelta(q=0.05, iou=1.0, dice=1.0, n_valid=182_274_183, k=9_113_709),
            "shift_64px": FakeDelta(
                q=0.05, iou=0.071, dice=0.133, n_valid=182_274_183, k=9_113_709
            ),
        },
    )
    base.update(overrides)
    return report.SegmentFloor(**base)


# --------------------------------------------------------------------------- helpers


CELL_SPLIT = re.compile(r"(?<!\\)\|")


def tables(md: str) -> list[list[list[str]]]:
    """Every Markdown table in the text, as a list of rows, as a list of cells.

    Splits on unescaped pipes only, the way a Markdown renderer does, so an escaped pipe
    inside a cell stays inside that cell.
    """
    out: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in CELL_SPLIT.split(stripped)[1:-1]]
            if current is None:
                current = []
                out.append(current)
            current.append(cells)
        else:
            current = None
    return out


# --------------------------------------------------------------------------- isolation


def test_report_imports_without_the_other_modules():
    """report must load, render and plan on its own.

    Runs in a subprocess: inside this suite the other test modules have already imported
    census and metrics, so checking sys.modules here would only measure collection order.
    The point of the check is that formatting and planning pull in nothing else, which is
    also what makes --dry-run work before the analysis modules are finished.
    """
    code = (
        "import sys; from inkfloor import report;"
        "report.to_markdown([]); report.to_json([]);"
        "report.format_plan(report.plan_corpus());"
        "leaked = [m for m in ('inkfloor.census','inkfloor.metrics','inkfloor.geometry')"
        " if m in sys.modules];"
        "print(leaked)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_cli_dry_run_imports_neither_the_network_nor_the_analysis_modules(tmp_path):
    """The promise on the tin: --dry-run shows the plan without opening a socket.

    Checked by running the real command in a subprocess with urlopen replaced by a raiser,
    and by looking at what got imported afterwards.
    """
    code = (
        "import sys, urllib.request\n"
        "def boom(*a, **k): raise AssertionError('dry run opened a socket')\n"
        "urllib.request.urlopen = boom\n"
        "from inkfloor import cli\n"
        "rc = cli.main(['floor', 'PHerc0172', 'seg-x', '--dry-run'])\n"
        "leaked = [m for m in ('inkfloor.census','inkfloor.metrics','inkfloor.geometry')"
        " if m in sys.modules]\n"
        "print('rc', rc, 'leaked', leaked)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
        env={"PATH": "/usr/bin:/bin", "INKFLOOR_CACHE": str(tmp_path)},
    )
    assert out.returncode == 0, out.stderr
    assert "rc 0 leaked []" in out.stdout, out.stdout


def test_formatting_never_touches_the_network(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - only runs if formatting hits the net
        raise AssertionError("formatting opened a socket")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    report.to_markdown([full_floor()])
    report.to_json([full_floor()])


# --------------------------------------------------------------------------- to_json


def test_to_json_is_valid_json_and_keeps_the_shape():
    doc = json.loads(report.to_json([full_floor()]))
    assert doc["schema"] == "inkfloor/report/2"
    assert len(doc["segments"]) == 1
    seg = doc["segments"][0]
    assert seg["sample"] == "PHerc0172"
    assert seg["segment"] == SEG
    assert len(seg["volume_pairs"]) == 2
    assert len(seg["model_pairs"]) == 2
    pair = seg["volume_pairs"][0]
    assert pair["kind"] == "volume"
    assert pair["a"]["volume"] == VOL_A
    assert pair["b"]["volume"] == VOL_B
    assert set(pair["deltas"]) == {"0.01", "0.05", "0.2"}
    d = pair["deltas"]["0.05"]
    assert d["iou"] == pytest.approx(0.380)
    assert d["delta"] == pytest.approx(0.620)
    assert d["k"] == 9_113_709
    assert seg["mesh"]["identical"] is True
    assert seg["intensity"]["slope"] == pytest.approx(0.6154)
    assert set(seg["nulls"]) == {"self", "shift_64px"}


def test_to_json_keeps_optional_keys_present_as_null():
    """mesh and intensity may be None. The key stays, so a consumer can tell 'not measured'
    from 'not in this version of the schema'."""
    doc = json.loads(report.to_json([full_floor(mesh=None, intensity=None)]))
    seg = doc["segments"][0]
    assert "mesh" in seg and seg["mesh"] is None
    assert "intensity" in seg and seg["intensity"] is None


def test_to_json_of_an_empty_report_is_still_valid():
    doc = json.loads(report.to_json([]))
    assert doc["segments"] == []


def test_to_json_survives_numpy_scalars():
    np = pytest.importorskip("numpy")
    floor = full_floor(
        nulls={
            "self": FakeDelta(
                q=np.float32(0.05),
                iou=np.float64(1.0),
                dice=np.float32(1.0),
                n_valid=np.int64(10),
                k=np.int32(2),
            )
        }
    )
    doc = json.loads(report.to_json([floor]))
    assert doc["segments"][0]["nulls"]["self"]["iou"] == 1.0
    assert doc["segments"][0]["nulls"]["self"]["n_valid"] == 10


def test_to_json_turns_nan_into_null_instead_of_breaking_the_file():
    """A degenerate intensity fit gives r = nan. NaN is not JSON, so it must become null."""
    floor = full_floor(
        intensity=FakeIntensity(
            slope=float("nan"),
            intercept=0.0,
            r=float("nan"),
            n_voxel=0,
            median_a=float("inf"),
            median_b=0.0,
            clip_frac_a=0.0,
            clip_frac_b=0.0,
        )
    )
    text = report.to_json([floor])
    assert "NaN" not in text and "Infinity" not in text
    fit = json.loads(text)["segments"][0]["intensity"]
    assert fit["r"] is None
    assert fit["median_a"] is None


def test_to_json_tolerates_a_record_missing_a_field():
    @dataclass(frozen=True)
    class Thin:
        q: float
        iou: float

    doc = json.loads(report.to_json([full_floor(nulls={"self": Thin(0.05, 1.0)})]))
    entry = doc["segments"][0]["nulls"]["self"]
    assert entry["iou"] == 1.0
    assert entry["dice"] is None
    assert entry["k"] is None


# --------------------------------------------------------------------------- to_markdown


def test_to_markdown_has_a_row_per_pair_and_a_summary():
    md = report.to_markdown([full_floor()])
    assert "# inkfloor report" in md
    assert f"## PHerc0172 / {SEG}" in md
    assert MODEL_1 in md and MODEL_2 in md
    # summary + floor + anchor + nulls
    assert len(tables(md)) == 4
    summary, floor_t, anchor_t, nulls_t = tables(md)
    assert len(floor_t) == 2 + 2  # header, separator, two pairs
    assert len(anchor_t) == 2 + 2
    assert len(nulls_t) == 2 + 2
    assert "0.620" in md  # 1 - 0.380 at q=5%


def test_every_table_row_has_the_header_column_count():
    md = report.to_markdown([full_floor()])
    for table in tables(md):
        widths = {len(row) for row in table}
        assert len(widths) == 1, f"ragged table: {widths} in {table[0]}"


def test_none_mesh_and_none_intensity_still_print_their_line():
    """The case that breaks tables: an optional field is None. Nothing may be skipped."""
    full = report.to_markdown([full_floor()])
    empty = report.to_markdown([full_floor(mesh=None, intensity=None)])

    assert "- mesh:" in empty
    assert "- intensity:" in empty
    assert "not checked" in empty
    assert "not fitted" in empty
    # same structure: same number of tables, same rows in each
    assert [len(t) for t in tables(full)] == [len(t) for t in tables(empty)]
    # and the summary row keeps its column count with mesh unknown
    assert "not checked" in tables(empty)[0][2][4]
    for table in tables(empty):
        assert len({len(row) for row in table}) == 1


def test_no_pairs_at_all_says_so_and_keeps_the_sections():
    md = report.to_markdown([full_floor(volume_pairs=[], model_pairs=[], nulls={})])
    assert "### Floor" in md and "### Anchor" in md
    assert "No volume pair on this segment" in md
    assert "No model pair on this segment" in md
    assert "No null control recorded" in md
    assert "n/a" in md  # the summary row cannot state a median it does not have


def test_a_pair_missing_one_q_gets_an_explicit_na_cell():
    floor = full_floor(
        volume_pairs=[
            (
                FakePair(pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1), "volume"),
                deltas({0.01: 0.288, 0.20: 0.596}),  # 5% not measured for this pair
            )
        ]
    )
    md = report.to_markdown([floor])
    floor_table = tables(md)[1]
    header, _, row = floor_table[0], floor_table[1], floor_table[2]
    assert len(row) == len(header)
    assert "n/a" in row
    assert "Δ@5% [tie band]" in header  # the column stays, the cell says n/a


def test_empty_report_is_not_an_empty_string():
    md = report.to_markdown([])
    assert "# inkfloor report" in md
    assert "No segment measured" in md


def test_markdown_never_leaks_a_python_none():
    md = report.to_markdown([full_floor(mesh=None, intensity=None, nulls={})])
    assert "None" not in md


def test_nan_in_markdown_prints_na_not_nan():
    floor = full_floor(
        nulls={"self": FakeDelta(q=0.05, iou=float("nan"), dice=0.0, n_valid=0, k=0)}
    )
    md = report.to_markdown([floor])
    assert "nan" not in md.lower().replace("n/a", "")
    assert "n/a" in md


def test_a_pipe_in_a_model_name_does_not_split_the_row():
    floor = full_floor(
        volume_pairs=[
            (
                FakePair(pred(VOL_A, "model|with|pipes"), pred(VOL_B, "model|with|pipes"), "volume"),
                deltas({0.05: 0.5}),
            )
        ],
        model_pairs=[],
    )
    md = report.to_markdown([floor])
    for table in tables(md):
        assert len({len(row) for row in table}) == 1


# --------------------------------------------------------------------------- tie bands


def anchor_row(md: str) -> list[str]:
    return tables(md)[2][2]


def test_every_delta_cell_carries_its_tie_interval():
    md = report.to_markdown([full_floor()])
    floor_row = tables(md)[1][2]
    # Δ = 1 - IoU, so a band of [iou_min, iou_max] prints as [1-iou_max, 1-iou_min].
    assert floor_row[3] == "0.694 [0.692, 0.696]"
    assert floor_row[4] == "0.620 [0.618, 0.622]"


def test_a_wide_band_is_marked_in_the_cell_and_listed_below():
    """The measured case: anchor on volume ...838 at q=1%, IoU 0.342 in [0.304, 0.408]."""
    md = report.to_markdown([full_floor()])
    assert anchor_row(md)[3] == "0.658 [0.592, 0.696] !"
    assert "1 of 12 Δ values must be read as intervals" in md
    assert "anchor pair 1 (20241024131838), Δ@1% = 0.658 in [0.592, 0.696], band 0.104" in md


def test_narrow_bands_everywhere_says_so_instead_of_staying_silent():
    floor = full_floor(
        ties={
            (pred(VOL_A, MODEL_1).key, pred(VOL_B, MODEL_1).key): bands(FLOOR_1),
            (pred(VOL_A, MODEL_2).key, pred(VOL_B, MODEL_2).key): bands(FLOOR_2),
            (pred(VOL_A, MODEL_1).key, pred(VOL_A, MODEL_2).key): bands(ANCHOR_1),
            (pred(VOL_B, MODEL_1).key, pred(VOL_B, MODEL_2).key): bands(ANCHOR_2),
        }
    )
    md = report.to_markdown([floor])
    assert "All 12 tie bands are at or below 0.050" in md
    assert "widest 0.004" in md
    assert "!" not in md.split("### Floor")[1].split("### Tie bands")[0]


def test_the_degenerate_case_that_looks_perfect_is_called_out():
    """The whole budget drawn from one plateau on both sides: IoU 1.000, Δ 0.000, worth zero.

    This is the failure the metrics module warned about. A Δ of 0.000 is the best-looking
    number the tool can print, so it has to be the loudest one to carry a flag.
    """
    k = 9_113_709
    degenerate = FakeTieBand(
        q=0.05,
        k=k,
        n_valid=182_274_183,
        iou=1.0,
        iou_min=0.02,
        iou_max=1.0,
        forced_a=0,
        forced_b=0,
        ties_needed_a=k,
        ties_needed_b=k,
        tie_band_a=k * 3,
        tie_band_b=k * 3,
    )
    pair = FakePair(pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1), "volume")
    floor = full_floor(
        volume_pairs=[(pair, deltas({0.05: 1.0}))],
        model_pairs=[],
        ties={(pair.a.key, pair.b.key): {0.05: degenerate}},
    )
    md = report.to_markdown([floor])
    assert "0.000 [0.000, 0.980] !!" in md
    assert "artifact of a saturated tail shared by the two maps" in md
    assert report._is_degenerate(degenerate) is True
    assert report._tie_flag(degenerate) == "!!"

    tie = json.loads(report.to_json([floor]))["segments"][0]["volume_pairs"][0]
    assert tie["deltas"]["0.05"]["tie"]["degenerate"] is True
    assert tie["deltas"]["0.05"]["tie"]["wide"] is True


def test_a_record_without_tie_bands_says_the_precision_is_unverified():
    floor = full_floor(ties={})
    md = report.to_markdown([floor])
    assert "No tie band recorded for this segment" in md
    assert tables(md)[1][2][3] == "0.694"  # no bracket invented
    for table in tables(md):
        assert len({len(row) for row in table}) == 1


def test_tie_band_reaches_the_json_inside_its_own_q():
    doc = json.loads(report.to_json([full_floor()]))
    anchor = doc["segments"][0]["model_pairs"][0]
    tie = anchor["deltas"]["0.01"]["tie"]
    assert tie["iou_min"] == pytest.approx(0.304)
    assert tie["iou_max"] == pytest.approx(0.408)
    assert tie["delta_min"] == pytest.approx(0.592)
    assert tie["delta_max"] == pytest.approx(0.696)
    assert tie["width"] == pytest.approx(0.104)
    assert tie["wide"] is True
    assert tie["degenerate"] is False
    assert anchor["deltas"]["0.05"]["tie"]["wide"] is False


def test_json_lists_the_tie_warnings_for_a_machine_reader():
    seg = json.loads(report.to_json([full_floor()]))["segments"][0]
    assert seg["tie_warn_width"] == 0.05
    assert len(seg["tie_warnings"]) == 1
    warning = seg["tie_warnings"][0]
    assert warning["family"] == "model"
    assert warning["q"] == pytest.approx(0.01)
    assert warning["width"] == pytest.approx(0.104)
    assert warning["degenerate"] is False


def test_json_tie_is_null_when_the_band_was_not_measured():
    seg = json.loads(report.to_json([full_floor(ties={})]))["segments"][0]
    entry = seg["volume_pairs"][0]["deltas"]["0.05"]
    assert "tie" in entry and entry["tie"] is None
    assert seg["tie_warnings"] == []


def test_band_width_falls_back_to_the_bounds_when_width_is_absent():
    @dataclass(frozen=True)
    class BoundsOnly:
        q: float
        k: int
        iou: float
        iou_min: float
        iou_max: float

    tb = BoundsOnly(q=0.01, k=10, iou=0.5, iou_min=0.4, iou_max=0.7)
    assert report._band_width(tb) == pytest.approx(0.3)
    assert report._tie_flag(tb) == "!"


# --------------------------------------------------------------------------- chance level


def test_chance_level_is_reported_once_per_report_for_every_q():
    md = report.to_markdown([full_floor()])
    assert "1%: IoU 0.005, Δ 0.995" in md
    assert "5%: IoU 0.026, Δ 0.974" in md
    assert "20%: IoU 0.111, Δ 0.889" in md
    assert "not commensurable" in md
    assert md.count("Chance level per q") == 1  # per report, not per row


def test_chance_level_absent_is_stated_and_not_recomputed():
    md = report.to_markdown([full_floor(chance={})])
    assert "Chance level not recorded" in md
    assert "IoU 0.005" not in md


def test_chance_reaches_the_json_in_both_units():
    seg = json.loads(report.to_json([full_floor()]))["segments"][0]
    assert seg["chance_iou"]["0.05"] == pytest.approx(0.05 / 1.95)
    assert seg["chance_delta"]["0.05"] == pytest.approx(1 - 0.05 / 1.95)
    assert set(seg["chance_iou"]) == {"0.01", "0.05", "0.2"}


# --------------------------------------------------------------------------- rank correlation


def test_spearman_has_its_own_column_and_json_field():
    md = report.to_markdown([full_floor()])
    header = tables(md)[1][0]
    assert "ρ (rank)" in header
    assert tables(md)[1][2][header.index("ρ (rank)")] == "0.921"
    pair = json.loads(report.to_json([full_floor()]))["segments"][0]["volume_pairs"][0]
    assert pair["spearman"] == pytest.approx(0.921)


def test_a_pair_without_a_rank_correlation_prints_na():
    floor = full_floor(spearman={})
    md = report.to_markdown([floor])
    header = tables(md)[1][0]
    assert tables(md)[1][2][header.index("ρ (rank)")] == "n/a"
    pair = json.loads(report.to_json([floor]))["segments"][0]["volume_pairs"][0]
    assert "spearman" in pair and pair["spearman"] is None


# --------------------------------------------------------------------------- null controls


def test_self_control_that_is_not_one_is_marked_as_a_failure():
    floor = full_floor(
        nulls={"self": FakeDelta(q=0.05, iou=0.97, dice=0.98, n_valid=100, k=5)}
    )
    md = report.to_markdown([floor])
    assert "FAIL" in md


def test_self_control_at_one_passes():
    md = report.to_markdown([full_floor()])
    nulls_table = tables(md)[3]
    self_row = next(r for r in nulls_table if r[0] == "self")
    assert self_row[1] == "IoU = 1"
    assert self_row[3] == "1.000"
    assert self_row[-1] == "ok"


def test_shift_control_that_agrees_as_much_as_the_pair_is_flagged():
    """A shifted copy of a map must disagree more than the two derivations do. If it does
    not, the pair difference is not evidence of anything."""
    floor = full_floor(
        nulls={"shift_64px": FakeDelta(q=0.05, iou=0.9, dice=0.94, n_valid=100, k=5)}
    )
    md = report.to_markdown([floor])
    assert "SUSPECT" in md


def test_shift_control_without_a_floor_pair_says_it_has_no_reference():
    floor = full_floor(
        volume_pairs=[],
        nulls={"shift_64px": FakeDelta(q=0.05, iou=0.07, dice=0.13, n_valid=100, k=5)},
    )
    md = report.to_markdown([floor])
    assert "no floor pair to compare against" in md


def test_an_unrecognised_control_is_not_given_a_verdict():
    """A control the renderer was not told how to read gets no pass and no fail."""
    floor = full_floor(
        nulls={"rotate_90": FakeDelta(q=0.05, iou=0.02, dice=0.04, n_valid=100, k=5)}
    )
    md = report.to_markdown([floor])
    row = next(r for r in tables(md)[3] if r[0] == "rotate_90")
    assert row[1] == "not declared"
    assert row[-1] == "not interpreted here"
    assert "ok" not in row[-1]


def test_a_null_with_no_iou_is_unknown_not_a_pass():
    @dataclass(frozen=True)
    class NoIou:
        q: float

    md = report.to_markdown([full_floor(nulls={"self": NoIou(0.05)})])
    assert "unknown" in md
    assert "FAIL" not in md


# --------------------------------------------------------------------------- summary row


def test_summary_ratio_compares_the_floor_to_the_anchor():
    md = report.to_markdown([full_floor()])
    summary = tables(md)[0]
    row = summary[2]
    # floor deltas at 5%: 0.620 and 0.713 -> median 0.6665
    # anchor deltas at 5%: 0.580 and 0.750 -> median 0.665
    assert row[1] == "0.667"
    assert row[2] == "0.665"
    assert row[3] == "1.00"
    assert row[4] == "yes"
    assert row[5] == "2 / 2"


def test_summary_ratio_is_na_when_there_is_no_anchor():
    md = report.to_markdown([full_floor(model_pairs=[])])
    row = tables(md)[0][2]
    assert row[2] == "n/a"
    assert row[3] == "n/a"


def test_two_segments_with_different_q_grids_share_one_summary_table():
    a = full_floor()
    b = full_floor(
        segment="other-segment",
        volume_pairs=[
            (
                FakePair(pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1), "volume"),
                deltas({0.05: 0.4}),
            )
        ],
        model_pairs=[],
        mesh=None,
        intensity=None,
        nulls={},
    )
    md = report.to_markdown([a, b])
    summary = tables(md)[0]
    assert len(summary) == 2 + 2
    assert len({len(r) for r in summary}) == 1
    assert "Segments measured: 2 (1 with both" in md


# --------------------------------------------------------------------------- plans


def test_human_bytes():
    assert report.human_bytes(0) == "0 B"
    assert report.human_bytes(1023) == "1023 B"
    assert report.human_bytes(1024) == "1.0 KB"
    assert report.human_bytes(45 * 1024 * 1024) == "45.0 MB"
    assert report.human_bytes(3 * 1024**3) == "3.0 GB"
    assert report.human_bytes(None) == "unknown"


def test_plan_reports_unknown_sizes_as_unknown_never_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)
    plan = report.DownloadPlan(
        title="t",
        items=[
            report.PlanItem(step="fetch", what="four predictions", new_bytes=180 * report.MB),
            report.PlanItem(step="fetch", what="meshes", new_bytes=None, exact=False),
        ],
    )
    text = report.format_plan(plan)
    assert "180.0 MB" in text
    assert "UNKNOWN until the listing" in text
    assert "Steps with no size yet: 1" in text
    assert "to download now: 0 B" not in text  # an unknown size is never reported as zero


def test_plan_totals_and_cached_split(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)
    plan = report.DownloadPlan(
        title="floor x / y",
        items=[
            report.PlanItem(step="fetch", what="predictions", new_bytes=100, cached_bytes=50),
            report.PlanItem(step="range", what="chunks", new_bytes=25, exact=False),
            report.PlanItem(step="compute", what="metrics", new_bytes=0),
        ],
        notes=["a note"],
    )
    assert plan.new_bytes == 125
    assert plan.cached_bytes == 50
    assert plan.exact is False
    text = report.format_plan(plan)
    assert "to download now: ~125 B" in text
    assert "already in cache: 50 B" in text
    assert "note: a note" in text
    assert str(tmp_path) in text


def test_census_plan_downloads_nothing():
    plan = report.plan_census(["PHerc0172"])
    assert plan.new_bytes == 0
    assert plan.unknown == []
    assert "no payload" in report.format_plan(plan)


def test_segment_plan_is_exact_when_given_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)
    preds = [pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1)]
    plan = report.plan_segment(
        "PHerc0172", SEG, preds=preds, geometry_checks=False, allow_listing=False
    )
    assert plan.new_bytes == 2 * 42 * 1024 * 1024
    assert plan.cached_bytes == 0
    assert plan.unknown == []


def test_segment_plan_counts_what_is_already_in_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)
    preds = [pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1)]
    local = tmp_path / preds[0].key
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    plan = report.plan_segment(
        "PHerc0172", SEG, preds=preds, geometry_checks=False, allow_listing=False
    )
    assert plan.new_bytes == 42 * 1024 * 1024
    assert plan.cached_bytes == 42 * 1024 * 1024
    assert report.is_cached(preds[0].key) is True
    assert report.is_cached(preds[1].key) is False


def test_offline_segment_plan_never_lists_or_fetches(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)

    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("the offline plan touched the network")

    monkeypatch.setattr(report.cache, "list_keys", boom)
    monkeypatch.setattr(report.cache, "fetch", boom)
    plan = report.plan_segment("PHerc0172", SEG)
    text = report.format_plan(plan)
    assert "30.0 MB to 50.0 MB each" in text
    assert plan.unknown  # the count is not knowable offline, and it says so


def test_corpus_plan_states_the_scale_before_anything_is_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)
    text = report.format_plan(report.plan_corpus())
    assert "30.0 MB to 50.0 MB each" in text
    assert "500 are 19.5 GB" in text
    assert "UNKNOWN until the listing" in text


def test_a_long_note_is_trimmed_in_markdown_and_kept_whole_in_json():
    long = "coordinate identiche al bit " * 12
    floor = full_floor(
        mesh=FakeMesh(
            identical=True,
            shape_a=(671, 747),
            shape_b=(671, 747),
            max_abs_diff={"x": 0.0, "y": 0.0, "z": 0.0},
            note=long,
        )
    )
    md = report.to_markdown([floor])
    mesh_line = next(line for line in md.splitlines() if line.startswith("- mesh:"))
    assert len(mesh_line) < 400
    assert "full note in the JSON report" in mesh_line
    assert json.loads(report.to_json([floor]))["segments"][0]["mesh"]["note"] == long


def test_corpus_plan_with_predictions_is_a_real_total(tmp_path, monkeypatch):
    monkeypatch.setattr(report.cache, "CACHE_ROOT", tmp_path)
    preds = [pred(VOL_A, MODEL_1), pred(VOL_B, MODEL_1), pred(VOL_A, MODEL_2)]
    plan = report.plan_corpus(
        ["PHerc0172"], preds=preds, segments=[("PHerc0172", SEG)], geometry_checks=False
    )
    assert plan.new_bytes == 3 * 42 * 1024 * 1024
    assert plan.unknown == []


def test_segment_prefix_is_read_from_the_keys():
    got = report.segment_prefix([pred(VOL_A, MODEL_1)])
    assert got == f"PHerc0172/segments/{SEG}/"


def test_segment_prefix_falls_back_when_the_key_has_no_segment_in_it():
    odd = FakePrediction(
        key="somewhere/else/file.tif",
        sample="PHerc0172",
        segment=SEG,
        volume=VOL_A,
        model=MODEL_1,
    )
    assert report.segment_prefix([odd]) == f"PHerc0172/segments/{SEG}/"


def test_segment_prefix_needs_a_prediction():
    with pytest.raises(ValueError):
        report.segment_prefix([])


def test_primary_q_prefers_five_percent():
    assert report._primary_q((0.01, 0.05, 0.20)) == 0.05
    assert report._primary_q((0.02, 0.10)) == 0.10
    assert report._primary_q(()) == 0.05


def test_number_coercion_rejects_what_json_cannot_carry():
    assert math.isfinite(report._f(1.0))
    assert report._f(float("inf")) is None
    assert report._f(float("nan")) is None
    assert report._f("not a number") is None
    assert report._i(None) is None


# --------------------------------------------------------------------------- corpus kinds


def pred_in(segment: str, volume: str, model: str) -> FakePrediction:
    """Like `pred`, but on a named segment, so a corpus can hold more than one."""
    return FakePrediction(
        key=f"PHerc0172/segments/{segment}/ink-detection/PHerc0172-x-volume-{volume}-{model}.tif",
        sample="PHerc0172",
        segment=segment,
        volume=volume,
        model=model,
    )


def _corpus_of_two() -> list[FakePrediction]:
    """One segment with only a volume pair, one with only a model pair."""
    return [
        pred_in("seg-floor", VOL_A, MODEL_1),
        pred_in("seg-floor", VOL_B, MODEL_1),
        pred_in("seg-anchor", VOL_A, MODEL_1),
        pred_in("seg-anchor", VOL_A, MODEL_2),
    ]


def _segments_selected(monkeypatch, kinds) -> list[str]:
    """Which segments corpus_floor decides to measure, without measuring them."""
    from inkfloor import report as report_mod

    asked: list[str] = []

    def fake_floor_for_segment(sample, segment, *a, **kw):
        asked.append(segment)
        return report_mod.SegmentFloor(
            sample=sample, segment=segment, volume_pairs=[], model_pairs=[],
            mesh=None, intensity=None, nulls={},
        )

    monkeypatch.setattr(report_mod, "floor_for_segment", fake_floor_for_segment)
    report_mod.corpus_floor(preds=_corpus_of_two(), kinds=kinds)
    return asked


def test_corpus_kinds_volume_takes_only_the_floor_segment(monkeypatch):
    assert _segments_selected(monkeypatch, ("volume",)) == ["seg-floor"]


def test_corpus_kinds_model_takes_only_the_anchor_segment(monkeypatch):
    assert _segments_selected(monkeypatch, ("model",)) == ["seg-anchor"]


def test_corpus_kinds_both_takes_both(monkeypatch):
    assert sorted(_segments_selected(monkeypatch, ("volume", "model"))) == [
        "seg-anchor", "seg-floor",
    ]


def test_corpus_kinds_defaults_to_the_floor(monkeypatch):
    """The default must not change under anyone who was calling this before --kind existed."""
    assert _segments_selected(monkeypatch, ("volume",)) == _segments_selected(
        monkeypatch, ("volume",)
    )
    from inkfloor import report as report_mod

    asked: list[str] = []
    monkeypatch.setattr(
        report_mod, "floor_for_segment",
        lambda sample, segment, *a, **kw: (
            asked.append(segment),
            report_mod.SegmentFloor(
                sample=sample, segment=segment, volume_pairs=[], model_pairs=[],
                mesh=None, intensity=None, nulls={},
            ),
        )[1],
    )
    report_mod.corpus_floor(preds=_corpus_of_two())
    assert asked == ["seg-floor"]


def test_plan_corpus_names_the_pair_it_will_fetch():
    """The plan must not say 'volume pair' when it is about to fetch model pairs."""
    from inkfloor import report as report_mod

    vol = " ".join(report_mod.plan_corpus(kinds=("volume",)).notes)
    mod = " ".join(report_mod.plan_corpus(kinds=("model",)).notes)
    assert "volume pair" in vol and "model pair" not in vol
    assert "model pair" in mod and "volume pair" not in mod
