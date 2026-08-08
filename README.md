# inkfloor

inkfloor measures how far apart two published ink predictions are when the only thing that
changed between them is which derivation of the same scan they were run on. It puts that
distance next to the distance between two different model checkpoints on the same
derivation. The first number is the **floor**: the disagreement you get without touching the
model. The second is the **anchor**: the disagreement the community already reads as real.

It reads the public bucket over anonymous HTTP. No credentials, no GPU, no torch, no zarr
library, no local copy of the corpus.

## Why this exists

On [ScrollPrize/villa#1372](https://github.com/ScrollPrize/villa/issues/1372) I posted one
measurement. The PHerc0172 segment `20251107110950-w064_20251107110950052_flatboi` carries
four published ink predictions: two derivations of the same scan (volumes `20241024131838`
and `20241024131839`) crossed with two checkpoints. On 182.3 Mpx of common valid area, at a
matched 5% positive budget:

| what changed | Δ@5% |
| --- | --- |
| the derivation, same model | 0.620 and 0.713 |
| the model, same derivation | 0.580 and 0.750 |

The two derivations publish tifxyz meshes that decode to identical arrays, so the shift is
not a re-flattening. Their 8 bit intensities are related by an affine remap,
`A = 0.6154*B + 104.32` with `r = 0.99987`, and the pipeline clips at an absolute value of
200 without normalising per volume.

The issue also carries a correction: at q = 1% the tie bands of two of those four pairs are
wider than the difference being discussed, so only the q = 5% row supports the comparison.
This tool prints that correction by itself, which is the subject of
[Why every Δ carries an interval](#why-every-δ-carries-an-interval).

That measurement had two problems: n = 1, and nobody else could rerun it. inkfloor fixes the
second one outright. Anyone with this package and a network connection recomputes every
number from the bucket in about 90 seconds and 170 MB of download. It also turns the first
problem into something you can check rather than assume, since the census reports how many
comparable pairs exist instead of leaving the reader to guess.

### What the census says today

`inkfloor census`, whole bucket, 2026-08-08, everything after the plan header:

```
predictions:  424 across 7 samples and 255 segments, 92.6 GB on the bucket
skipped keys: 423 of 847 did not match the known naming scheme
  not-a-tif: 423
pairs:        58 comparable
  volume (the floor):  2
  model  (the anchor): 56
segments with a floor pair: 1

segments that carry a floor pair:
  PHerc0172  20251107110950-w064_20251107110950052_flatboi  (4 predictions)
```

One segment in the published corpus carries two derivations of the same scan. inkfloor does
not manufacture more data: it enumerates what exists, and today what exists supports n = 1
for the floor and n = 56 for the anchor. Every future segment published with a second
derivation raises that count with no change to this tool, and the census is the place to look
before quoting a corpus-wide claim.

### The claim this tool does not make

A floor as large as its anchor is consistent with the intensity remap driving the
difference. It is also consistent with other explanations. inkfloor measures the size of the
effect and checks the confounders it can reach. It does not run the counterfactual, so it
does not establish a cause. [What inkfloor does not do](#what-inkfloor-does-not-do) is the
full boundary.

## Install

```
python -m venv .venv
.venv/bin/pip install -e .
```

Python 3.11 or newer. Dependencies: `numpy`, `tifffile`, `numcodecs`, `imagecodecs`.

The cache can be moved anywhere:

```
export INKFLOOR_CACHE=/Volumes/BigDisk/inkfloor-cache
```

Everything downloaded lands there under the same key path as on S3 and is never fetched
twice.

## Using it

Three subcommands: `census`, `floor`, `corpus`. Each one prints what it is about to
download, in MB, before the first byte moves. `--dry-run` prints the same plan without
opening a socket.

### 1. Find out what is comparable

`census` downloads no prediction. It reads key names and object sizes.

```
$ inkfloor census --samples PHerc0172
plan: census of PHerc0172

  step  fetch  cached  what
  list    0 B       -  S3 ListObjectsV2 over PHerc0172 (object metadata, no payload)

  to download now: 0 B
  of which already in cache: 0 B
  cache: /Volumes/AppsAndFiles/dev/inkfloor/cache (9.3 MB on disk)
  note: census downloads no prediction file: it reads key names and sizes only. The listing is paginated at 1000 keys per request and a busy segment has tens of thousands of keys, so a full-bucket census takes minutes of requests.

listing the bucket, this is metadata only and can take a few minutes ...
predictions:  108 across 1 sample and 53 segments, 5.4 GB on the bucket
skipped keys: 108 of 216 did not match the known naming scheme
  not-a-tif: 108
pairs:        56 comparable
  volume (the floor):  2
  model  (the anchor): 54
segments with a floor pair: 1

segments that carry a floor pair:
  PHerc0172  20251107110950-w064_20251107110950052_flatboi  (4 predictions)
```

The skipped count is printed because a name parser that guesses would poison every number
downstream. This one refuses what it does not recognise and reports how often, broken down by
reason. Here the 108 refusals are the downsampled JPEG previews that sit next to the TIFFs.

### 2. See the bytes before spending them

```
$ inkfloor floor PHerc0172 20251107110950-w064_20251107110950052_flatboi --dry-run
plan: floor PHerc0172 / 20251107110950-w064_20251107110950052_flatboi

  step       fetch  cached  what
  list         0 B       -  S3 prefix PHerc0172/segments/20251107110950-w064_20251107110950052_flatboi/ (metadata, no payload)
  fetch    unknown       -  ink predictions: count resolved by the listing, 30.0 MB to 50.0 MB each
  fetch    unknown       -  tifxyz mesh of each derivation (x/y/z + meta.json)
  range    ~5.2 MB       -  10 surface-volume zarr chunks via HTTP Range (~528.0 KB each, not cached)
  compute        -       -  common valid mask, Delta@q per pair, null controls
  compute        -       -  mesh identity and intensity fit

  to download now: UNKNOWN until the listing. Sized steps: ~5.2 MB. Steps with no size yet: 2.
  of which already in cache: 0 B
  cache: /Volumes/AppsAndFiles/dev/inkfloor/cache (0 B on disk)
  note: Offline plan: sizes are nominal. Run without --dry-run for exact bytes, the listing phase prints them before the first fetch and downloads no payload.

dry run: nothing was requested.
```

A dry run reaches neither the network nor the analysis modules, so the size of a file nobody
has listed yet reads `unknown`. It never reads zero. Drop `--dry-run` and the listing resolves
it exactly, still before any fetch:

```
listing PHerc0172 ... (metadata only, no payload)
plan: floor PHerc0172 / 20251107110950-w064_20251107110950052_flatboi

  step        fetch  cached  what
  fetch    165.4 MB       -  ink predictions (4 published, 4 not in cache)
  fetch         0 B  9.3 MB  tifxyz mesh, 2 derivations (8 files)
  range     ~5.2 MB       -  10 surface-volume zarr chunks via HTTP Range (~528.0 KB each, not cached)
  compute         -       -  common valid mask, Delta@q per pair, null controls
  compute         -       -  mesh identity and intensity fit

  to download now: ~170.5 MB
  of which already in cache: 9.3 MB
  cache: /Volumes/AppsAndFiles/dev/inkfloor/cache (9.3 MB on disk)
```

Run it a second time and the same plan reads `0 B` to fetch against `165.4 MB` in cache. The
run ends with what actually crossed the network, measured by watching the cache grow:

```
cache grew by 165.4 MB (now 174.6 MB at /Volumes/AppsAndFiles/dev/inkfloor/cache)
```

If the announced download is over `--max-download-mb` (default 1024) the run stops before
fetching anything. The message, here on an example plan of 15.6 GB:

```
refusing to download 15.6 GB: over the 1024 MB budget. Pass --yes to accept, or raise --max-download-mb, or narrow the run with --samples / --limit.
```

### 3. Measure one segment

```
$ inkfloor floor PHerc0172 20251107110950-w064_20251107110950052_flatboi
```

About 80 seconds and 165.4 MB of download on a cold cache, peak RSS 3.0 GB. The report below
is the real output of that run, shown as it renders (the command writes the same text with
`--out-md`):

---

# inkfloor report

Δ@q = 1 - IoU between the top-q% of two ink predictions, at a matched positive budget: the same number of positives k on each side, so a difference in calibration is not read as a difference in placement. 0 means the two maps put ink in the same pixels, 1 means they are disjoint.

**Floor** = same model, two derivations of the same scan. **Anchor** = same derivation, two models. The anchor is the difference the community already treats as real, so it is the scale the floor is read against. A floor near its anchor is a measurement, not an explanation: inkfloor does not establish a cause.

**How to read a Δ cell.** Each one is `Δ [low, high]`, where the bracket is the exact interval the Δ can take over every admissible tie-break, mirrored onto Δ from the IoU bounds that `metrics.tie_bounds` returns. Published maps are 8 bit, so the top-q% cut usually lands inside a plateau of pixels that share one value, and which members of the plateau get taken is arbitrary. A narrow bracket means the Δ is a fact about the data. `!` marks a band wider than 0.050, which has to be read as an interval. `!!` marks the degenerate case: the whole budget came from one plateau on both sides, so the Δ is an artifact of a shared saturated tail and not a measurement of placement. That case can print a Δ of 0.000, which looks like the best possible result and is worth nothing.

**ρ (rank)** is Spearman's rank correlation over the valid pixels. It is invariant to any monotone rescaling of either map, so it answers a different question from Δ: ρ asks whether the two maps agree on the ordering of every pixel, Δ asks whether they agree on which pixels make the top of the list. High ρ with high Δ is informative and not a contradiction: it says the two maps rank the surface almost identically and still disagree about where the ink is.

Segments measured: 1 (1 with both a floor pair and an anchor pair)

q grid: 1%, 5%, 20%

**Chance level per q** (two independent selections of k pixels, expected IoU q/(2-q)): 1%: IoU 0.005, Δ 0.995; 5%: IoU 0.026, Δ 0.974; 20%: IoU 0.111, Δ 0.889. The floor of the metric grows with q, so a Δ is read against the chance level of its own q and never against zero. Δ values at different q are not commensurable with each other.

| segment | floor Δ@5% (median) | anchor Δ@5% (median) | floor / anchor | mesh identical | pairs floor / anchor |
| --- | --- | --- | --- | --- | --- |
| PHerc0172 / 20251107110950-w064_20251107110950052_flatboi | 0.666 | 0.665 | 1.00 | yes | 2 / 2 |

## PHerc0172 / 20251107110950-w064_20251107110950052_flatboi

### Floor: same model, different derivation of the same scan

| same model | volume A | volume B | Δ@1% [tie band] | Δ@5% [tie band] | Δ@20% [tie band] | IoU@5% | Dice@5% | ρ (rank) | k@5% | valid px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20250713185324-timesformer_scroll5_july_retreat | 20241024131838 | 20241024131839 | 0.694 [0.662, 0.716] ! | 0.620 [0.613, 0.628] | 0.519 [0.506, 0.527] | 0.380 | 0.551 | 0.745 | 9,113,897 | 182,277,946 |
| 20251222202946-timesformer_scroll5_november19 | 20241024131838 | 20241024131839 | 0.764 [0.745, 0.784] | 0.713 [0.711, 0.715] | 0.657 [0.655, 0.659] | 0.287 | 0.446 | 0.452 | 9,113,897 | 182,277,946 |

### Anchor: same derivation, different model

| same volume | model A | model B | Δ@1% [tie band] | Δ@5% [tie band] | Δ@20% [tie band] | IoU@5% | Dice@5% | ρ (rank) | k@5% | valid px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20241024131838 | 20250713185324-timesformer_scroll5_july_retreat | 20251222202946-timesformer_scroll5_november19 | 0.658 [0.592, 0.696] ! | 0.580 [0.575, 0.588] | 0.573 [0.571, 0.577] | 0.420 | 0.592 | 0.522 | 9,113,897 | 182,277,946 |
| 20241024131839 | 20250713185324-timesformer_scroll5_july_retreat | 20251222202946-timesformer_scroll5_november19 | 0.826 [0.824, 0.834] | 0.750 [0.747, 0.753] | 0.673 [0.665, 0.678] | 0.250 | 0.400 | 0.414 | 9,113,897 | 182,277,946 |

### Tie bands

**2 of 12 Δ values must be read as intervals.** Their tie band is wider than 0.050, which means the point value is largely a consequence of which pixels of a plateau of equal values happened to be taken:

- floor pair 1 (20250713185324-timesformer_scroll5_july_retreat), Δ@1% = 0.694 in [0.662, 0.716], band 0.053.
- anchor pair 1 (20241024131838), Δ@1% = 0.658 in [0.592, 0.696], band 0.104.

### Confounder checks

- mesh: **identical**, shape (671, 747) vs (671, 747), max abs diff x=0.000000 y=0.000000 z=0.000000 (coordinates bit-identical on x, y, z (671x747); meta.json differs on area_vx2, scale, uuid; [...], full note in the JSON report)
- intensity: A = 0.6153*B + 104.33 (r = 0.99988, n = 10,440,027 voxels), median 154.0 vs 80.0, at or above the clip ceiling 200: 2.26% vs 0.16%, best z offset 0, chunks (125,26,39), (123,23,28), (130,9,37), (155,10,40), (85,31,13) (5/5 chunks accepted in 10 tries (seed=0, min_nonzero=0.5); rejected: 3 absent in A, 2 present in A but absent in B, 0 too empty; scansione [...], full note in the JSON report)

### Null controls

| control | expected | Δ | IoU | Dice | k | valid px | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self | IoU = 1 | 0.000 | 1.000 | 1.000 | 9,113,897 | 182,277,946 | ok |
| shift_64px | IoU well below the pair IoU | 0.841 | 0.159 | 0.275 | 9,061,489 | 181,229,784 | ok |

---

Notes on reading that report. The four Δ@5% values reproduce villa#1372 to three decimals
from a clean checkout, which is the point of the exercise. The bracket after each Δ is the tie
band, and the two `!` markers at q = 1% are the part to read before quoting anything from that
column: see [Why every Δ carries an interval](#why-every-δ-carries-an-interval). The
diagnostic `note` strings are passed through verbatim from the module that produced them, and
the trimmed part of each is in the JSON. `--out-json report.json` writes the same content as
data: every Δ carries its own `tie` object, every segment carries `chance_iou`,
`chance_delta` and a `tie_warnings` list, and `mesh` and `intensity` stay present and `null`
when they could not be measured, so a consumer can tell "not measured" from "not in this
version of the schema".

### 4. The corpus

```
$ inkfloor corpus --dry-run
$ inkfloor corpus --samples PHerc0172 PHerc0332 --yes
$ inkfloor corpus --limit 5 --no-geometry --out-json corpus.json
```

`corpus` measures every segment that has at least one volume pair and skips the rest: with a
single derivation there is no floor to measure. A failing segment is reported and skipped
instead of aborting the run. `--no-geometry` drops the mesh and intensity checks, which saves
their download.

## The metric

`Δ@q = 1 - IoU` between the top-q% of the two maps, taking the **same number of positives k
on each side**. 0 means the two maps put ink in the same pixels. 1 means the two sets are
disjoint. Both selections have exactly k members, so `union = 2k - inter`, `dice = inter/k`
and `iou = dice/(2 - dice)`: IoU and Dice carry the same information here, and Dice is
reported because more readers have a feel for its scale.

### Why the budget is matched instead of thresholded

A fixed threshold answers a different question from the one being asked. The two derivations
of this segment are related by `A = 0.6154*B + 104.32` with `r = 0.99987`: as maps of where
the ink is they are nearly the same ranking, and as 8 bit images they sit at different
levels. Threshold both at 128 and one yields far more positives than the other, so most of
the measured difference is the remap. That is a calibration difference reported as a
localisation difference.

Matching the budget removes the part of the difference that any monotone rescaling could
explain, and what is left is placement. Two consequences:

- the pipeline clips at an absolute 200 with no per-volume normalisation, so a threshold
  chosen near the top of the range inherits that clip. `clip_frac_a` and `clip_frac_b` say how
  much of each volume sits at the ceiling: on this segment 2.26% against 0.16%, which is the
  same asymmetry the affine fit describes.
- matching the budget introduces a cost of its own, which is the subject of the next section.

### Why every Δ carries an interval

Three columns in that report exist to stop the tool from producing a confident wrong number,
and they are worth the space they take.

Published maps are `uint8`: 256 levels spread over 182 million valid pixels. The top-q% cut
lands inside a plateau of pixels that all share one value, k is matched, and which members of
the plateau get taken is decided by index order. `metrics.tie_bounds` computes the
exact minimum and maximum IoU over every admissible top-k selection, and two things came out
of running it on this corpus:

- at q = 1%, up to 19.5% of the budget k is drawn from inside a tie plateau. A Δ@1% printed to
  three decimals is precision the data does not support.
- if the two maps share a saturated tail wider than k, the whole top-k of both sides falls
  inside one plateau. Both sides can then be made to select the same pixels by index order
  alone, and the tool reports IoU 1.000, Δ 0.000. **That is the best-looking number this tool
  can print and it means nothing**: a perfect floor manufactured by saturation.

Neither case announces itself in a table of point values, so every Δ cell reads
`Δ [low, high]`, the exact interval mirrored onto Δ, with two markers for severity:

| marker | what it means |
| --- | --- |
| `!` | band wider than 0.050: read that Δ as an interval, not as a number |
| `!!` | degenerate: the whole budget came from one plateau on both sides, so the Δ says nothing about placement |

Each segment also gets a **Tie bands** section that lists the flagged cells with their widths,
or states the widest band when nothing is flagged. A passing check prints a line rather than
staying silent, because silence is indistinguishable from a check that never ran.

On the villa#1372 segment this changes what can honestly be claimed. At q = 5% the bands are
0.015 wide or narrower and the floor-against-anchor comparison holds. At q = 1% two of the
four pairs are flagged: the anchor pair on volume `20241024131838` gives Δ 0.658 inside
`[0.592, 0.696]`, a band of 0.104, and the floor pair on the July checkpoint gives Δ 0.694
inside `[0.662, 0.716]`, a band of 0.053. Both bands are wider than the gap between the floor
and the anchor at that q, so the q = 5% row is the one to quote. The report says that on its
own, without depending on whoever reads it to remember.

### Chance level: Δ at different q are not commensurable

Two independent selections of k pixels have an expected IoU of `q/(2-q)`, so the floor of the
metric grows with q:

| q | chance IoU | chance Δ |
| --- | --- | --- |
| 1% | 0.005 | 0.995 |
| 5% | 0.026 | 0.974 |
| 20% | 0.111 | 0.889 |

A Δ of 0.519 at q = 20% and a Δ of 0.694 at q = 1% cannot be ranked by size: the first sits
0.370 below its chance level, the second 0.301 below its own. Reading Δ columns across q as
one scale is the second way this tool could hand someone a wrong answer, so the chance level
is printed once per report, taken from `metrics.chance_iou(q)`.

It is a reading aid and not a significance threshold. The empirical companion is `null_shift`
at the same q, which on structured maps can land well below the chance value because real
top-k sets are spatially clustered.

### ρ: the milder story next to Δ

`metrics.spearman` is a column in both tables. It is invariant to any monotone rescaling of
either map, so it answers a different question: ρ asks whether the two maps agree on the
ordering of every valid pixel, Δ asks whether they agree on which pixels make the top of the
list.

A divergence between the two is information rather than an error. On this segment the floor
pairs give ρ 0.745 and 0.452 while the anchor pairs give 0.522 and 0.414, so the two
derivations of one scan rank the surface more alike than two models do and still disagree
about the top of the list about as much. That reading is unavailable from either number on its
own, which is why both are in every row.

### The valid mask

A pixel is valid where every map of the segment is greater than zero. In the published format
zero is the out-of-mask value and a genuine prediction of exactly zero cannot be told apart
from it. The mask is built once across all maps of the segment, so the floor and the anchor
are measured on the same pixels and their numbers can be compared. `n_valid` in every row is
that count, 182,277,946 px on the example segment.

## Null controls, and what they have to give

| control | what it is | what it must return |
| --- | --- | --- |
| `self` | the map against itself | IoU exactly 1.000 |
| `shift_64px` | the map against a 64 px translation of itself, with the wrapped strip marked invalid | IoU far below the pair IoU |

The `self` control checks that the ruler is deterministic: same input, same top-k, same
answer. Anything other than 1.000 means the measurement is unusable, and the report says
`FAIL` rather than printing a number and moving on. It does not prove that tie-breaking is
insensitive to a small change in the input, which is what `tie_bounds` is for.

The `shift` control sets the scale. If a rigid 64 px translation of one map agrees with the
original as much as the two derivations agree with each other, a Δ of 0.6 says nothing about
derivations. The report marks that case `SUSPECT`. On the example segment the shift gives
IoU 0.159 against a floor IoU of 0.380 and 0.287, so the pair difference sits well inside the
range the control leaves open. `metrics.best_shift_iou(a, b, valid, q)` goes further and
searches translations exhaustively within a radius, which answers whether a pair difference
is a plain misalignment.

## The confounders it checks

Two derivations that differ for a reason other than intensity would make the floor
meaningless. Two checks run per segment:

- **mesh**: `geometry.compare_meshes` decodes the tifxyz of both derivations and compares the
  arrays channel by channel. It compares decoded arrays and not file bytes: on this segment
  the two mesh sets are 3.5 MB and 5.7 MB on the bucket and decode to identical `float32`
  arrays of shape `(671, 747)`. A byte comparison would have reported a difference that does
  not exist.
- **intensity**: `geometry.fit_intensity` samples homologous chunks from both surface volumes,
  fits `A = slope*B + intercept` and reports `r`, both medians, the fraction of voxels at or
  above the clip ceiling of 200, the z offset that maximised the correlation, and which chunks
  it used. A low `r` is grounds to discard the fit, which is why `r` is in the report instead
  of behind it.

## What inkfloor does not do

- **It does not establish a cause.** A floor the size of its anchor is consistent with the
  affine remap driving the difference, and with other stories too. Settling it means
  re-running inference on a normalised volume, which is the pipeline's job.
- **It does not run models and does not train.** No inference, no checkpoints, no torch. If
  something in here starts importing torch, it has gone off course.
- **It does not propose a fix to the pipeline.** Measurement and report, nothing else.
- **It does not judge a prediction right or wrong.** There is no ground truth in any of these
  numbers. Δ is disagreement between two published artifacts, not error.
- **It does not rank models.** The model pairs exist to give the floor a scale, not to say
  which checkpoint is better.
- **It does not make n larger.** It reports the n the corpus has. Today that is one segment
  with a floor pair.
- **It does not verify the bucket.** Names, sizes and contents are taken as published. The
  name parser refuses what it does not recognise and counts the refusals, which is as far as
  its scepticism goes.
- **It does not pick a tie-break for you.** A reported Δ is one admissible tie-break out of
  many. The interval next to it is the range of all of them, and a Δ whose band is wide is not
  a result, whatever the point value looks like. There is no flag to turn those columns off:
  they exist so a wrong number cannot leave here looking credible.
- **It does not write to the bucket** and creates no files other than the cache and the
  reports you ask for by path.

## Bytes, cache and memory

- 1 MB here is 1024 * 1024 bytes.
- Predictions run 30 to 50 MB each. The four on the example segment are 165.4 MB together.
- All 424 predictions published across the bucket are 92.6 GB. A `corpus` run fetches only
  the segments that carry a volume pair, and the exact total is printed before the first
  fetch, so the number is never a surprise.
- Partial reads (the zarr chunks of the intensity fit) go over HTTP Range and are not cached.
  Everything else is cached under its S3 key and reused.
- Every map of a segment is held in memory at once to build the common valid mask. Budget
  roughly one byte of RSS per pixel per map. The example segment peaked at 3.0 GB.
- The tie bands and the rank correlation cost more compute than the Deltas themselves. Adding
  them took user CPU on the example segment from 15.9 s to 40.8 s, measured on the same
  machine and the same four maps. That is the price of not printing a point value the data
  does not support, and there is no switch to avoid paying it.
- `--dry-run` touches neither the network nor the analysis modules.

## Layout

| file | what it owns |
| --- | --- |
| `inkfloor/cache.py` | the only door to the network: fetch, Range reads, listings, one cache path |
| `inkfloor/census.py` | which predictions exist, which pairs are comparable, what was refused |
| `inkfloor/metrics.py` | the ruler: `delta_at_q`, `spearman`, `tie_bounds`, the null controls |
| `inkfloor/geometry.py` | the confounders: mesh identity, intensity fit |
| `inkfloor/report.py` | one record per segment, the Markdown and JSON renderers, the byte plans |
| `inkfloor/cli.py` | the three subcommands, and the only place that prints |

Module contracts are in `CONTRACTS.md`. Tests: `.venv/bin/python -m pytest`.

## License

MIT. See [LICENSE](LICENSE).
