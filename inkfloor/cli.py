"""The command line: three subcommands, and no download that was not announced first.

Two rules hold everywhere in here:

1. every command prints what it is about to download, in MB, before the first byte moves,
   and how much of it is already in the cache;
2. `--dry-run` prints the same plan and returns without opening a socket.

This is the only module that prints. It does not compute anything: it asks `report` for
data and strings and puts them on stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inkfloor import cache, report

DEFAULT_MAX_DOWNLOAD_MB = 1024

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2


def _say(*parts: object) -> None:
    print(*parts, file=sys.stdout, flush=True)


def _warn(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def _print_plan(plan: report.DownloadPlan) -> None:
    _say(report.format_plan(plan))
    _say("")


def _gate(plan: report.DownloadPlan, args: argparse.Namespace) -> bool:
    """True when the run may proceed. Refuses a download bigger than the budget."""
    limit = int(args.max_download_mb) * report.MB
    if args.yes or plan.new_bytes <= limit:
        return True
    _warn(
        f"refusing to download {report.human_bytes(plan.new_bytes)}: over the "
        f"{args.max_download_mb} MB budget. Pass --yes to accept, or raise "
        "--max-download-mb, or narrow the run with --samples / --limit."
    )
    return False


class _Meter:
    """Measures what actually crossed the network, by watching the cache grow.

    Range reads on zarr chunks are not cached, so they do not show up here. The plan
    accounts for them; this only reports what landed on disk.
    """

    def __init__(self) -> None:
        self.before = self._size()

    @staticmethod
    def _size() -> int:
        try:
            return cache.cache_size_bytes()
        except OSError:
            return 0

    def report(self) -> str:
        after = self._size()
        grew = max(0, after - self.before)
        return (
            f"cache grew by {report.human_bytes(grew)} "
            f"(now {report.human_bytes(after)} at {cache.CACHE_ROOT})"
        )


def _write(path: str | None, text: str, label: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    _say(f"wrote {label}: {p} ({len(text):,} chars)")


def _emit(floors, args: argparse.Namespace) -> None:
    md = report.to_markdown(floors)
    js = report.to_json(floors)
    _write(args.out_md, md, "markdown")
    _write(args.out_json, js, "json")
    if args.json:
        _say(js)
    elif not args.out_md:
        _say(md)


# --------------------------------------------------------------------------- census


def _census_stats() -> dict:
    """The census bookkeeping, or {} when the module does not offer any.

    `census.census` returns only the predictions, so the discard count comes from the
    module's own accounting. It is probed rather than required: a parser that guesses would
    poison every number downstream, so the count is worth printing, and an absent count is
    worth saying out loud instead of printing a zero.
    """
    from inkfloor import census

    for name in ("census_stats", "stats", "last_stats"):
        fn = getattr(census, name, None)
        if callable(fn):
            try:
                got = fn()
            except Exception:  # noqa: BLE001 - a probe must not break the command
                continue
            if isinstance(got, dict) and got:
                return got
    for name in ("SKIPPED", "LAST_SKIPPED", "skipped", "n_skipped"):
        value = getattr(census, name, None)
        if isinstance(value, int):
            return {"skipped": value}
    return {}


def _skipped_count(stats: dict) -> int | None:
    for k in ("skipped", "n_skipped", "unparsed"):
        if isinstance(stats.get(k), int):
            return stats[k]
    return None


def cmd_census(args: argparse.Namespace) -> int:
    plan = report.plan_census(args.samples)
    _print_plan(plan)
    if args.dry_run:
        _say("dry run: nothing was requested.")
        return EXIT_OK

    from inkfloor import census

    _say("listing the bucket, this is metadata only and can take a few minutes ...")
    preds = census.census(list(args.samples) if args.samples else None)
    pairs = census.pairs(preds)

    total = sum(p.size_bytes for p in preds)
    by_kind: dict[str, int] = {}
    for pair in pairs:
        by_kind[pair.kind] = by_kind.get(pair.kind, 0) + 1
    segments = {(p.sample, p.segment) for p in preds}
    floor_segments = {(pair.a.sample, pair.a.segment) for pair in pairs if pair.kind == "volume"}
    stats = _census_stats()
    skipped = _skipped_count(stats)

    if args.json:
        import json

        _say(
            json.dumps(
                {
                    "predictions": len(preds),
                    "segments": len(segments),
                    "bytes": total,
                    "skipped": skipped,
                    "skipped_by_reason": stats.get("by_reason"),
                    "pairs": by_kind,
                    "floor_segments": sorted(list(s) for s in floor_segments),
                },
                indent=2,
                default=str,
            )
        )
        return EXIT_OK

    n_samples = len({p.sample for p in preds})
    _say(f"predictions:  {len(preds):,} across {n_samples} "
         f"{'sample' if n_samples == 1 else 'samples'} and {len(segments):,} segments, "
         f"{report.human_bytes(total)} on the bucket")
    if skipped is None:
        _say("skipped keys: not reported by the census module")
    else:
        _say(f"skipped keys: {skipped:,} of "
             f"{stats.get('keys_seen', '?')} did not match the known naming scheme")
        by_reason = stats.get("by_reason")
        if isinstance(by_reason, dict):
            for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
                _say(f"  {reason}: {n:,}")
    _say(f"pairs:        {len(pairs):,} comparable")
    _say(f"  volume (the floor):  {by_kind.get('volume', 0):,}")
    _say(f"  model  (the anchor): {by_kind.get('model', 0):,}")
    _say(f"segments with a floor pair: {len(floor_segments):,}")
    if not floor_segments:
        _say("no segment has two derivations of the same scan: there is no floor to measure.")
        return EXIT_OK

    _say("")
    _say("segments that carry a floor pair:")
    for sample, segment in sorted(floor_segments):
        n = sum(1 for p in preds if p.sample == sample and p.segment == segment)
        _say(f"  {sample}  {segment}  ({n} predictions)")
    return EXIT_OK


# --------------------------------------------------------------------------- floor


def cmd_floor(args: argparse.Namespace) -> int:
    qs = tuple(args.q)
    geometry_checks = not args.no_geometry

    if args.dry_run:
        plan = report.plan_segment(
            args.sample, args.segment, geometry_checks=geometry_checks, n_chunks=args.chunks
        )
        _print_plan(plan)
        _say("dry run: nothing was requested.")
        return EXIT_OK

    from inkfloor import census

    _say(f"listing {args.sample} ... (metadata only, no payload)")
    preds = census.census([args.sample])
    seg_preds = [p for p in preds if p.segment == args.segment]
    if not seg_preds:
        _warn(f"no published prediction for {args.sample} / {args.segment}")
        _warn("run `inkfloor census --samples " + args.sample + "` to see what is published.")
        return EXIT_ERROR

    plan = report.plan_segment(
        args.sample,
        args.segment,
        preds=preds,
        geometry_checks=geometry_checks,
        n_chunks=args.chunks,
        allow_listing=True,
    )
    _print_plan(plan)
    if not _gate(plan, args):
        return EXIT_REFUSED

    meter = _Meter()
    try:
        floor = report.floor_for_segment(
            args.sample, args.segment, qs, preds=preds, geometry_checks=geometry_checks
        )
    except (ValueError, LookupError, cache.FetchError) as exc:
        _warn(f"{args.sample}/{args.segment}: {exc}")
        _warn(meter.report())
        return EXIT_ERROR
    _emit([floor], args)
    _say("")
    _say(meter.report())
    return EXIT_OK


# --------------------------------------------------------------------------- corpus


def cmd_corpus(args: argparse.Namespace) -> int:
    qs = tuple(args.q)
    geometry_checks = not args.no_geometry

    if args.dry_run:
        plan = report.plan_corpus(
            args.samples, geometry_checks=geometry_checks, n_chunks=args.chunks
        )
        _print_plan(plan)
        _say("dry run: nothing was requested.")
        return EXIT_OK

    from inkfloor import census

    target = ", ".join(args.samples) if args.samples else "the whole bucket"
    _say(f"listing {target} ... (metadata only, no payload, this can take minutes)")
    preds = census.census(list(args.samples) if args.samples else None)

    by_segment: dict[tuple[str, str], list] = {}
    for p in preds:
        by_segment.setdefault((p.sample, p.segment), []).append(p)
    segments = [seg for seg, group in sorted(by_segment.items())
                if any(pair.kind == "volume" for pair in census.pairs(group))]
    if args.limit:
        segments = segments[: args.limit]
    if not segments:
        _say("no segment has two derivations of the same scan: there is no floor to measure.")
        return EXIT_OK

    plan = report.plan_corpus(
        args.samples,
        preds=preds,
        segments=segments,
        geometry_checks=geometry_checks,
        n_chunks=args.chunks,
    )
    _print_plan(plan)
    _say(f"segments to measure: {len(segments)}")
    if not _gate(plan, args):
        return EXIT_REFUSED

    meter = _Meter()
    failures: list[tuple[str, str, Exception]] = []

    def on_segment(i: int, n: int, sample: str, segment: str) -> None:
        _say(f"[{i}/{n}] {sample} / {segment}")

    def on_error(sample: str, segment: str, exc: Exception) -> None:
        failures.append((sample, segment, exc))
        _warn(f"  skipped {sample}/{segment}: {type(exc).__name__}: {exc}")

    floors = report.corpus_floor(
        args.samples,
        qs,
        preds=preds,
        segments=segments,
        geometry_checks=geometry_checks,
        on_segment=on_segment,
        on_error=on_error,
    )
    _say("")
    _emit(floors, args)
    _say("")
    _say(f"measured {len(floors)} of {len(segments)} segments, {len(failures)} skipped")
    _say(meter.report())
    return EXIT_OK


# --------------------------------------------------------------------------- parser


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dry-run", action="store_true",
                   help="print the full plan and exit without touching the network")
    p.add_argument("--json", action="store_true", help="print JSON instead of Markdown")


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--q", type=float, nargs="+", default=list(report.DEFAULT_QS),
                   metavar="FRACTION",
                   help="positive budgets to measure at, as fractions (default: 0.01 0.05 0.2)")
    p.add_argument("--no-geometry", action="store_true",
                   help="skip the mesh and intensity checks, which saves the mesh download")
    p.add_argument("--chunks", type=int, default=5, metavar="N",
                   help="zarr chunks per volume for the intensity fit (default: 5)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="accept the download announced in the plan")
    p.add_argument("--max-download-mb", type=int, default=DEFAULT_MAX_DOWNLOAD_MB,
                   metavar="MB",
                   help=f"refuse to download more than this without --yes "
                        f"(default: {DEFAULT_MAX_DOWNLOAD_MB})")
    p.add_argument("--out-json", metavar="PATH", help="write the JSON report to a file")
    p.add_argument("--out-md", metavar="PATH", help="write the Markdown report to a file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inkfloor",
        description=(
            "Measure the reproducibility floor of the Vesuvius Challenge ink detection "
            "pipeline: how far apart two ink predictions are when the only thing that "
            "changed is which derivation of the same scan they were run on."
        ),
        epilog=(
            "Every command announces its download in MB before the first byte moves. "
            "Predictions are 30 to 50 MB each and a corpus run is tens of GB, so start "
            "with --dry-run."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_census = sub.add_parser(
        "census", help="what is published and what is comparable (listing only, no payload)"
    )
    p_census.add_argument("--samples", nargs="+", metavar="SAMPLE",
                          help="limit to these samples, e.g. PHerc0172")
    _add_common(p_census)
    p_census.set_defaults(func=cmd_census)

    p_floor = sub.add_parser("floor", help="the floor of one segment, with the confounder checks")
    p_floor.add_argument("sample", help="e.g. PHerc0172")
    p_floor.add_argument("segment", help="e.g. 20251107110950-w064_20251107110950052_flatboi")
    _add_common(p_floor)
    _add_run_flags(p_floor)
    p_floor.set_defaults(func=cmd_floor)

    p_corpus = sub.add_parser("corpus", help="the floor of every segment that has one")
    p_corpus.add_argument("--samples", nargs="+", metavar="SAMPLE",
                          help="limit to these samples")
    p_corpus.add_argument("--limit", type=int, metavar="N",
                          help="stop after N segments, in listing order")
    _add_common(p_corpus)
    _add_run_flags(p_corpus)
    p_corpus.set_defaults(func=cmd_corpus)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for q in getattr(args, "q", None) or ():
        if not 0.0 < q <= 1.0:
            parser.error(
                f"--q takes fractions in (0, 1], got {q}. 5% is 0.05, not 5."
            )
    if not getattr(args, "samples", None):
        args.samples = None
    for name in ("out_json", "out_md"):
        if not hasattr(args, name):
            setattr(args, name, None)
    if not hasattr(args, "q"):
        args.q = list(report.DEFAULT_QS)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _warn("interrupted. The cache keeps what was already downloaded.")
        return 130
    except cache.FetchError as exc:
        _warn(f"the bucket said no: {exc}")
        return EXIT_ERROR
    except ModuleNotFoundError as exc:
        _warn(f"a module of this tool is missing: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
