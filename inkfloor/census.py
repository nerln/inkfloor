"""Censimento delle predizioni di inchiostro pubblicate, e classificazione delle coppie.

Il corpus pubblica le predizioni come TIFF dentro `<sample>/segments/<segment>/ink-detection/`,
con tutti i metadati nel nome del file. Questo modulo legge quei nomi e NON apre i file: non
decodifica un solo pixel, non scarica un solo TIFF. Solo elenchi S3 e parsing di stringhe.

La regola che governa tutto: `parse_prediction` restituisce None invece di indovinare. Un
nome che non corrisponde allo schema noto viene scartato e contato, mai interpretato a metà.
Un parser che tira a indovinare falsa il censimento a valle e non lascia tracce.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import cache

# ---------------------------------------------------------------------------
# Layout del bucket
# ---------------------------------------------------------------------------

SEGMENTS_DIR = "segments"
INK_DIR = "ink-detection"

#: Nomi di cartelle che, dentro un segmento, contengono artefatti e non altri segmenti.
#: Serve solo a non sprecare richieste: se qui manca un nome, il censimento resta corretto
#: ma fa qualche GET in più. Se ce n'è uno di troppo, un segmento annidato con quel nome
#: verrebbe saltato, quindi la lista tiene solo nomi visti davvero nel corpus.
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

#: Quanti elenchi S3 in parallelo. Sono GET anonime su un bucket pubblico, nessuna scrittura.
MAX_WORKERS = 8

#: Quanti nomi di key scartate tenere come campione nel referto.
MAX_DISCARDED_EXAMPLES = 12


# ---------------------------------------------------------------------------
# Lo schema dei nomi
# ---------------------------------------------------------------------------

# PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838
#   -20250713185324-timesformer_scroll5_july_retreat-tile64-stride16.tif
#
# PHerc0139-20250108000000-1.129um-0.22m-59keV-volume-20260413113053-L1
#   -20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif
#
# Pezzi opzionali verificati nel corpus: la distanza sorgente-oggetto (`-0.22m`), il livello
# piramidale (`-L1`), la coppia tile/stride. Il nome del modello può contenere trattini e
# perfino spezzoni come `1um`, quindi il voxel si ancora subito dopo il timestamp del
# segmento e il modello subito dopo l'id del volume (o dopo il livello, se c'è).
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

# `.+?` per il nome del modello è elastico: se la coda del nome è storta (per esempio
# `-tile64` senza `-stride16`, o un suffisso `-ds8` dopo lo stride) il pezzo storto finirebbe
# dentro il nome del modello e tile/stride resterebbero a None, senza che nulla segnali il
# problema. Due predizioni con stride diverso sembrerebbero lo stesso modello. Quindi: se nel
# modello resta un token `-tile<cifre>` o `-stride<cifre>`, il nome si scarta.
_LEFTOVER_RX = re.compile(r"-(?:tile|stride)\d")

# Motivi di scarto, in ordine di controllo.
R_NOT_TIF = "not-a-tif"
R_NOT_INK = "not-under-ink-detection"
R_NESTED = "nested-under-ink-detection"
R_NAME = "name-not-recognised"
R_SAMPLE = "sample-mismatch-with-path"
R_SEGMENT = "segment-mismatch-with-path"


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """Una predizione di inchiostro pubblicata, come la descrive il suo nome file.

    NON dice nulla sul contenuto: shape, dtype e frazione di pixel validi si sanno solo
    aprendo il TIFF, che qui non viene toccato. `size_bytes` è la dimensione dell'oggetto S3,
    non il numero di pixel: la compressione la rende inconfrontabile fra predizioni diverse.
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
    # Campi aggiuntivi, tutti con default: chi costruisce un Prediction con i soli campi del
    # contratto continua a funzionare. `level` è None quando il nome non lo dichiara, e in
    # quel caso resta ignoto: non viene assunto 0.
    level: int | None = None
    kev: float | None = None
    dist_m: float | None = None

    @property
    def segment_prefix(self) -> str:
        """Il prefisso S3 del segmento, con lo slash finale. Non verifica che esista."""
        return self.key.split(f"/{INK_DIR}/", 1)[0] + "/"

    @property
    def step_um(self) -> float | None:
        """Passo raster effettivo: `voxel_um * 2**level`.

        None quando il livello piramidale non è dichiarato nel nome. NON tira a indovinare
        L=0: un nome senza livello non dice che il livello sia zero, dice che non lo sappiamo.
        """
        if self.level is None:
            return None
        return self.voxel_um * float(2**self.level)

    @property
    def raster(self) -> tuple[float, int | None]:
        """Identità del raster su cui vive la mappa: (voxel_um, level).

        Due predizioni sono sovrapponibili pixel a pixel solo se questa coppia coincide.
        NON è una shape: non garantisce che i due TIFF abbiano le stesse dimensioni.
        """
        return (self.voxel_um, self.level)


@dataclass(frozen=True)
class Pair:
    """Due predizioni sullo stesso segmento, e cosa cambia fra loro."""

    a: Prediction
    b: Prediction
    kind: str  # "volume" | "model" | "both"


@dataclass(frozen=True)
class CensusReport:
    """Esito del censimento, con la contabilità di quello che è stato scartato.

    NON stampa niente: i campi sono pensati per essere formattati da `cli.py`.
    """

    predictions: list[Prediction]
    samples_scanned: list[str]
    n_keys_seen: int
    n_kept: int
    n_discarded: int
    discarded_by_reason: dict[str, int]
    discarded_examples: list[tuple[str, str]]  # (key, motivo), campione
    n_ink_dirs: int
    n_segments_with_predictions: int
    samples_with_predictions: dict[str, int]


@dataclass(frozen=True)
class PairStats:
    """Le coppie tenute e, soprattutto, quelle escluse e perché."""

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
    """Estrae i campi dal nome file. None se il nome non segue lo schema noto.

    NON indovina: se manca un pezzo obbligatorio, se il campione o il timestamp del segmento
    nel nome non combaciano con il path, o se il file non è un TIFF direttamente dentro
    `ink-detection/`, restituisce None. In particolare scarta le anteprime `downsampled/*.jpg`,
    che sono la stessa predizione ridotta 8x e non la predizione.

    NON legge il file e NON tocca la rete.
    """
    pred, _ = _parse_with_reason(key, size_bytes)
    return pred


def _parse_with_reason(key: str, size_bytes: int) -> tuple[Prediction | None, str | None]:
    """Come `parse_prediction`, ma dice anche perché ha scartato. Uso interno al referto."""
    if not key.endswith(".tif"):
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
# Enumerazione
# ---------------------------------------------------------------------------


def samples_in_corpus() -> list[str]:
    """I sample pubblicati, in ordine alfabetico.

    NON include i prefissi di servizio che iniziano con `_` (per esempio `_thumbnails/`).
    """
    return sorted(
        p.rstrip("/") for p in cache.list_prefixes("") if p and not p.startswith("_")
    )


def _ink_keys_for_sample(sample: str) -> tuple[list[tuple[str, int]], int]:
    """Tutte le key sotto gli `ink-detection/` di un sample, e quanti ne ha trovati.

    Prova prima `<sample>/segments/<candidato>/ink-detection/`. Se un candidato non ha
    predizioni, guarda un livello più sotto: nel corpus esistono contenitori intermedi
    (`segments/raw/<segmento>/`). NON scende oltre quel livello.
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
    """Come `census`, ma restituisce anche la contabilità degli scarti.

    NON stampa e NON scarica i TIFF: fa solo ListObjectsV2 sul bucket. Il conteggio delle
    key viste include tutto quello che sta sotto `ink-detection/`, anteprime comprese, così
    che `n_keys_seen == n_kept + n_discarded` sia verificabile a occhio.
    """
    names = samples_in_corpus() if samples is None else [s.rstrip("/") for s in samples]

    kept: list[Prediction] = []
    reasons: Counter[str] = Counter()
    # Il campione garantisce un nome per ogni motivo di scarto, poi si riempie fino al tetto:
    # così un motivo raro non viene mai coperto da centinaia di scarti dello stesso tipo.
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


#: Referto dell'ultimo censimento eseguito in questo processo. Serve solo a `census_stats()`,
#: perché la firma di `census()` restituisce le predizioni e non ha posto per la contabilità.
#: Nessuna logica del modulo lo legge: `census_report` e `pair_stats` restano funzioni pure.
_last_report: CensusReport | None = None


def census(samples: list[str] | None = None) -> list[Prediction]:
    """Enumera le predizioni ink di tutto il corpus (o dei soli sample indicati).

    NON restituisce gli scarti: per quelli serve `census_report`, che dà lo stesso elenco più
    la contabilità di quante key sono state scartate e perché. Chi ha in mano solo questa
    firma può leggere `census_stats()` subito dopo la chiamata.
    """
    global _last_report
    _last_report = census_report(samples)
    return _last_report.predictions


def census_stats() -> dict[str, object]:
    """Contabilità dell'ultimo `census()` di questo processo, per chi stampa.

    Dizionario vuoto se `census()` non è ancora stato chiamato: NON esegue un censimento per
    rispondere e non tocca la rete. Le chiavi `skipped`, `n_skipped` e `unparsed` portano lo
    stesso numero, cioè quante key sotto `ink-detection/` sono state rifiutate.
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
# Coppie
# ---------------------------------------------------------------------------


def comparable(a: Prediction, b: Prediction) -> bool:
    """Vero se le due mappe vivono sullo stesso raster, cioè stesso voxel e stesso livello.

    NON verifica le shape dei TIFF né l'allineamento della mesh: dice solo che confrontarle
    pixel a pixel ha senso. Il livello ignoto (None) è considerato uguale solo a un altro
    livello ignoto, perché due nomi che tacciono il livello tacciono la stessa cosa.
    """
    return abs(a.voxel_um - b.voxel_um) < 1e-9 and a.level == b.level


def _kind(a: Prediction, b: Prediction) -> str | None:
    same_volume = a.volume == b.volume
    same_model = a.model == b.model
    if same_volume and same_model:
        return None  # stesso volume e stesso modello: non è una coppia, è un duplicato
    if same_model:
        return "volume"
    if same_volume:
        return "model"
    return "both"


def pair_stats(preds: list[Prediction]) -> PairStats:
    """Come `pairs`, ma restituisce anche cosa è stato escluso e perché.

    NON stampa. NON accoppia predizioni di segmenti diversi: il pavimento si misura sulla
    stessa superficie, e due segmenti diversi hanno mesh diverse.
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
    """Tutte le coppie confrontabili, raggruppate per segmento. Esclude kind='both'.

    NON include le coppie su raster diversi (voxel o livello piramidale diversi): quelle
    mappe hanno passi diversi e sovrapporle pixel a pixel misurerebbe il ricampionamento,
    non il pavimento. Quante ne sono state escluse lo dice `pair_stats`.
    """
    return pair_stats(preds).pairs
