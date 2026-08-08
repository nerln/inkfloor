"""Il righello di inkfloor: confronto fra due mappe di predizione di inchiostro.

Ogni numero che lo strumento riporta passa da qui, quindi il criterio di scrittura di questo
modulo e' la correttezza, non l'eleganza e non la velocita'.

La metrica centrale e' Delta = 1 - IoU fra i top-q% delle due mappe **a budget di positivi
appaiato**: si prendono esattamente k = round(n*q) pixel per lato, dove n e' il numero di
pixel validi, e si confronta l'intersezione con l'unione di quei due insiemi.

Perche' non una soglia fissa: una soglia fissa confonde una differenza di calibrazione con
una differenza di localizzazione. Due mappe possono mettere l'inchiostro esattamente negli
stessi posti e avere istogrammi diversi, per esempio perche' i due volumi hanno intensita'
diverse o perche' il modello e' stato applicato con normalizzazioni diverse; con una soglia
fissa la mappa piu' "calda" ha molti piu' positivi dell'altra, l'IoU crolla, e chi legge
conclude che le due predizioni non sono d'accordo su dove sta l'inchiostro. Il budget
appaiato toglie di mezzo quel confondente: entrambi i lati spendono lo stesso numero di
positivi, e cio' che resta e' una differenza di posizione. Vedi il thread villa#191.

Convenzioni sui dati reali:
- le mappe arrivano da tifffile come uint8 2-D, shape tipo (13420, 14940);
- il pixel valido e' quello con valore > 0: lo zero e' il fuori-maschera, non un valore
  di predizione. Circa il 91% dei pixel e' valido su un segmento tipico;
- due mappe dello stesso segmento possono avere shape leggermente diverse. Qui vengono
  ritagliate alla regione comune ancorata a (0, 0), vedi `align`.

Quello che questo modulo NON fa: non legge file, non scarica niente, non stampa, non
sceglie q, non decide se un Delta e' "grande". Riceve array e restituisce dati.

Avvertenza sui pareggi, da leggere prima di fidarsi di un IoU: con mappe molto quantizzate
(uint8, 256 livelli) i pixel al confine del top-k hanno spessissimo lo stesso valore, e
quale di quei pixel entri nel top-k e' arbitrario. `delta_at_q` rompe i pareggi in modo
deterministico ma posizionale (indice piatto piu' basso per primo, cioe' le righe in alto), e
questo puo' gonfiare o sgonfiare l'IoU senza che il numero lo dia a vedere. Il caso limite e'
una mappa costante, dove l'IoU vale 1 pur non essendoci alcuna informazione condivisa; il caso
realistico e' due mappe con la stessa coda satura a 255, dove il top-k cade tutto dentro il
pareggio e l'IoU torna vicino a 1 per costruzione.
`tie_bounds` misura quanto di un IoU e' arbitrario: restituisce l'intervallo esatto
[iou_min, iou_max] su tutte le rotture di pareggio ammissibili. Un intervallo largo vuol
dire che il numero puntuale non va riportato da solo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Tolleranza sulla differenza di shape fra due mappe dello stesso segmento. Sopra questa
# soglia non si tratta piu' di arrotondamenti del bounding box della mesh e il ritaglio
# silenzioso sarebbe un errore mascherato.
MAX_SHAPE_DIFF_PX = 64
MAX_SHAPE_DIFF_FRAC = 0.01

# Limiti dei percorsi a memoria contenuta: le mappe reali hanno ~2e8 pixel e non si possono
# materializzare array float64 di quella lunghezza su un portatile.
_HIST_MAX_BINS = 1 << 22
_CHUNK = 1 << 23


class ShapeMismatch(ValueError):
    """Le shape differiscono troppo perche' il ritaglio alla regione comune sia difendibile."""


@dataclass(frozen=True)
class Delta:
    q: float
    iou: float
    dice: float
    n_valid: int
    k: int  # quanti positivi per lato (budget appaiato)


@dataclass(frozen=True)
class TieBand:
    """Quanto di un IoU e' deciso dai pareggi invece che dai dati.

    `iou` e' il valore che restituisce `delta_at_q`, cioe' una singola rottura di pareggio.
    `iou_min` e `iou_max` sono il minimo e il massimo esatti su TUTTE le selezioni top-k
    ammissibili delle due mappe. Se coincidono, il top-k e' unico e l'IoU e' un fatto; se
    sono lontani, l'IoU puntuale e' in buona parte un artefatto dell'ordine degli indici.
    """

    q: float
    k: int
    n_valid: int
    iou: float
    iou_min: float
    iou_max: float
    thr_a: float  # k-esimo valore piu' alto di a fra i pixel validi
    thr_b: float
    forced_a: int  # pixel con valore strettamente sopra la soglia: entrano per forza
    forced_b: int
    ties_needed_a: int  # quanti pixel vanno pescati dalla banda dei pareggi
    ties_needed_b: int
    tie_band_a: int  # quanti pixel sono in pareggio sulla soglia
    tie_band_b: int

    @property
    def unique(self) -> bool:
        """Vero se entrambe le selezioni top-k sono uniche, cioe' senza scelte arbitrarie."""
        return self.tie_band_a == self.ties_needed_a and self.tie_band_b == self.ties_needed_b

    @property
    def width(self) -> float:
        """Ampiezza dell'intervallo di incertezza da pareggi. 0 = nessuna arbitrarieta'."""
        return self.iou_max - self.iou_min


# --------------------------------------------------------------------------------------
# preparazione degli array
# --------------------------------------------------------------------------------------


def align(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Ritaglia tutti gli array alla shape comune minima, ancorando l'origine a (0, 0).

    Perche' questa scelta e non un padding: le mappe di un segmento sono renderizzate sulla
    stessa griglia appiattita e ancorate allo stesso angolo, e la differenza di shape fra due
    derivazioni viene dall'arrotondamento dell'estensione della mesh, cioe' e' una o poche
    righe o colonne in fondo. Ritagliare tiene i pixel omologhi sovrapposti; un padding
    inventerebbe pixel, e allineare al centro sposterebbe tutto di mezza differenza.

    NON verifica che le due mappe siano davvero ancorate allo stesso angolo: se non lo sono,
    il ritaglio confronta zone diverse senza lamentarsi. Quel dubbio si scioglie con
    `best_shift_iou`, che dice se esiste una traslazione migliore di quella nulla.

    Solleva ShapeMismatch se su qualche asse la differenza supera
    max(MAX_SHAPE_DIFF_PX, MAX_SHAPE_DIFF_FRAC * estensione), perche' a quel punto non e'
    un arrotondamento ed e' meglio fermarsi che ritagliare.
    """
    arrs = [np.asarray(x) for x in arrays]
    if not arrs:
        raise ValueError("align richiede almeno un array")
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
    """Maschera dei pixel di predizione di una singola mappa: valido se > 0.

    NON distingue un fuori-maschera da una predizione di inchiostro esattamente nulla: nel
    formato pubblicato lo zero e' il fuori-maschera e quella distinzione non esiste.
    """
    return np.asarray(a) > 0


def common_valid(maps: list[np.ndarray]) -> np.ndarray:
    """Maschera dei pixel validi in TUTTE le mappe. Un pixel e' valido se > 0 in ognuna.

    Le mappe vengono prima ritagliate alla regione comune (vedi `align`), quindi la maschera
    ha la shape comune minima, non quella della prima mappa.

    NON controlla che le mappe siano dello stesso segmento e non riporta quanto si perde
    nell'intersezione: quel conteggio si legge da `Delta.n_valid`.
    """
    if not maps:
        raise ValueError("common_valid richiede almeno una mappa")
    arrs = align(*maps)
    out = ink_valid(arrs[0])
    for x in arrs[1:]:
        out = out & (x > 0)
    return out


def shift_map(
    a: np.ndarray, valid: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trasla una mappa 2-D di (dy, dx) e restituisce (mappa traslata, validi traslati).

    La traslazione e' un `np.roll`, ma la striscia che ha fatto il giro viene marcata NON
    valida: cosi' il confronto non usa mai pixel che sono rientrati dall'altro lato.
    Convenzione: `out[y, x] = a[y - dy, x - dx]`, cioe' dy e dx positivi spostano il
    contenuto verso il basso e verso destra.

    NON interpola e non gestisce traslazioni frazionarie.
    """
    arr, m = align(a, valid)
    if arr.ndim != 2:
        raise ValueError(f"shift_map lavora su mappe 2-D, ricevuto ndim={arr.ndim}")
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
# metrica centrale
# --------------------------------------------------------------------------------------


def delta_at_q(a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float) -> Delta:
    """1 - IoU fra i top-q% di a e di b, a budget di positivi appaiato.

    Si prendono k = round(n*q) pixel per lato, con n = numero di pixel validi, e si confronta
    intersezione contro unione dei due insiemi. Poiche' i due insiemi hanno la stessa
    cardinalita' k, valgono le identita' union = 2k - inter, dice = inter/k e
    iou = dice / (2 - dice): IoU e Dice qui portano la stessa informazione, Dice e' riportato
    solo perche' e' la scala a cui molti lettori sono abituati.

    NON usa una soglia fissa: la soglia fissa confonde una differenza di calibrazione con una
    differenza di localizzazione. Vedi il thread villa#191 e il docstring del modulo.

    Altre cose che NON fa:
    - non calcola 1 - IoU al posto tuo: restituisce l'IoU, il Delta e' 1 - Delta.iou;
    - non verifica che `valid` sia coerente con le due mappe, la maschera e' quella che le
      passi (di solito `common_valid([a, b])`);
    - non tiene conto dei pareggi: se molti pixel hanno lo stesso valore sulla soglia, quali
      entrino nel top-k e' arbitrario e questa funzione non lo segnala. Usa `tie_bounds`;
    - non gestisce i NaN: in un array float un NaN finisce ordinato come il valore piu' alto e
      quindi entra nel top-k. Le mappe del corpus sono uint8, quindi il caso non si presenta,
      ma se arrivi qui con dei float ripulisci prima.

    Solleva ValueError se q non e' in (0, 1], se non ci sono pixel validi, o se
    round(n*q) = 0: su un ritaglio troppo piccolo la metrica non e' definita e restituire un
    numero comunque sarebbe peggio che fermarsi.
    """
    va, vb, n, k = _values_and_budget(a, b, valid, q)
    ma = _topk_mask(va, k)
    mb = _topk_mask(vb, k)
    return _delta_from_masks(ma, mb, q, n, k)


def spearman(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    """Correlazione di rango di Spearman fra a e b sui pixel validi.

    Invariante a ogni schiacciamento monotono dei valori, quindi racconta una storia piu'
    mite dell'IoU: due mappe che ordinano i pixel allo stesso modo danno 1 anche se le loro
    calibrazioni sono lontanissime. Vanno riportate entrambe, perche' rispondono a domande
    diverse: Spearman chiede "l'ordine e' lo stesso?", l'IoU al top-q% chiede "i pixel piu'
    caldi sono gli stessi pixel?", e su mappe quasi-ordinate ma con code diverse la prima
    resta alta mentre la seconda crolla.

    I pareggi sono trattati con i ranghi medi, che e' la correzione corretta e conta molto
    qui: su uint8 ci sono 256 livelli per milioni di pixel, quindi i gruppi di pareggio sono
    enormi. Su input interi a intervallo contenuto il calcolo passa da un istogramma
    congiunto, che da' il risultato esatto senza materializzare array di ranghi lunghi come
    la mappa.

    NON fa test di significativita' (con 1e8 pixel qualunque p-value e' zero e non vuol dire
    niente), NON gestisce NaN, e NON e' invariante a rimappature non monotone.
    Restituisce nan se una delle due mappe e' costante sui pixel validi, perche' in quel caso
    il rango non ha varianza e la correlazione non e' definita.
    """
    arr_a, arr_b, m = _prepare(a, b, valid)
    return _spearman_1d(arr_a[m], arr_b[m])


def common_valid_fraction(valid: np.ndarray) -> float:
    """Frazione di pixel validi sulla shape data. Comoda per i report, niente di piu'."""
    m = np.asarray(valid)
    return float(np.count_nonzero(m)) / float(m.size) if m.size else 0.0


def chance_iou(q: float) -> float:
    """IoU atteso fra due insiemi top-k scelti a caso e indipendenti: q / (2 - q).

    Derivazione: due sottoinsiemi indipendenti di k = n*q pixel su n hanno intersezione attesa
    k^2/n = k*q e unione attesa 2k - k*q, quindi IoU circa q/(2-q). Vale come rapporto di
    valori attesi, cioe' asintoticamente in k, che con k dell'ordine di 1e6 e' piu' che
    abbastanza.

    Serve come promemoria di lettura: il pavimento della metrica CRESCE con q. A q = 0.01 il
    caso da' 0.005, a q = 0.20 da' 0.111. Un IoU di 0.11 al 20% non e' un accordo debole, e' il
    caso. Un Delta va sempre letto contro `null_shift` allo stesso q, non contro lo zero.

    NON e' una soglia di significativita' e NON tiene conto del fatto che i top-k reali sono
    spazialmente aggregati: su mappe con struttura il valore misurato dal controllo nullo puo'
    stare parecchio sotto questo numero.
    """
    if not (0.0 < float(q) <= 1.0):
        raise ValueError(f"q fuori da (0, 1]: {q}")
    return float(q) / (2.0 - float(q))


# --------------------------------------------------------------------------------------
# controlli nulli
# --------------------------------------------------------------------------------------


def null_self(a: np.ndarray, valid: np.ndarray, q: float) -> Delta:
    """Controllo nullo che DEVE dare IoU = 1: la mappa contro se stessa.

    Non e' una tautologia: passa dalla stessa strada di `delta_at_q`, quindi se la selezione
    del top-k non fosse deterministica (per esempio se rompesse i pareggi a caso) questo
    controllo lo mostrerebbe. Un valore diverso da 1 e' un bug del righello, non un dato.

    NON dice niente sulla scala dei valori "buoni": e' solo il soffitto della metrica.
    """
    return delta_at_q(a, a, valid, q)


def null_shift(a: np.ndarray, valid: np.ndarray, q: float, px: int = 64) -> Delta:
    """Controllo nullo che DEVE dare IoU bassa: la mappa contro se stessa traslata di px in x.

    Serve a dare una scala: e' il valore che si ottiene quando le due mappe hanno la stessa
    texture e la stessa calibrazione ma le posizioni non c'entrano niente. Un Delta misurato
    vicino a questo valore vuol dire "le due predizioni non condividono localizzazione".
    Il `Delta.n_valid` restituito e' quello dell'intersezione fra validi e validi traslati,
    quindi minore di quello di `null_self`, e di conseguenza anche k e' minore.

    NON e' una garanzia. Due casi in cui questo controllo restituisce IoU alta pur essendo la
    traslazione priva di senso: una mappa costante o comunque tutta in pareggio sulla soglia
    (i pareggi vengono rotti per posizione e le due selezioni coincidono), e una mappa con
    struttura periodica in x di periodo che divide px. Per il secondo caso ripeti con un px
    non commensurato, per il primo guarda `tie_bounds`.

    NON trasla in y: se la struttura del segmento e' anisotropa, usa `shift_map` a mano.
    """
    m = np.asarray(valid).astype(bool)
    shifted, vshift = shift_map(a, m, 0, px)
    arr, m2 = align(a, m)
    return delta_at_q(arr, shifted, m2 & vshift, q)


def best_shift_iou(
    a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float, radius: int = 20
) -> tuple[int, int, float]:
    """Miglior IoU su una ricerca esaustiva di traslazioni entro radius.

    Serve a escludere che una differenza sia solo un disallineamento: se l'IoU migliore si
    trova a (0, 0) le due mappe sono allineate e il Delta misurato e' una differenza vera; se
    si trova altrove, il Delta include un pezzo di traslazione e va scartato o corretto.
    Restituisce (dy, dx, iou), con la convenzione di `shift_map`: b va traslata di (dy, dx),
    cioe' `b[y - dy, x - dx]` va confrontato con `a[y, x]`. Se b e' a spostata in basso di 3
    e a destra di 5, la risposta e' (-3, -5).

    I due insiemi top-k sono calcolati UNA volta sulla maschera comune non traslata e poi
    spostati: non vengono riselezionati per ogni traslazione. Scelta voluta, non una scorciatoia
    di velocita': riselezionare cambierebbe il budget k a ogni passo e gli IoU delle diverse
    traslazioni non sarebbero piu' confrontabili fra loro, che e' l'unica cosa che qui serve.
    Come conseguenza, per traslazioni diverse da (0, 0) il conteggio dei positivi nella
    finestra sovrapposta e' un po' minore di k e l'IoU e' calcolato sui conteggi reali della
    finestra, non su 2k - inter.

    NON cerca rotazioni, scale o deformazioni, NON interpola sotto il pixel, e costa
    (2*radius+1)^2 passate sull'intera mappa: su una mappa da 2e8 pixel e radius 20 sono
    1681 passate, misurate a circa 0.09 s ciascuna, cioe' due minuti e mezzo. Abbassa radius se
    serve solo una verifica rapida.

    Soprattutto: NON dice se la traslazione trovata sia reale. Restituisce sempre un massimo,
    anche fra due mappe che non c'entrano niente, e in quel caso la traslazione vincente e' il
    rumore piu' fortunato fra (2*radius+1)^2 tentativi. Chi legge il risultato deve confrontare
    l'IoU migliore con quello a traslazione zero: se il guadagno e' piccolo, non c'e' nessun
    disallineamento da correggere.
    """
    if radius < 0:
        raise ValueError(f"radius negativo: {radius}")
    arr_a, arr_b, m = _prepare(a, b, valid)
    if arr_a.ndim != 2:
        raise ValueError(f"best_shift_iou lavora su mappe 2-D, ricevuto ndim={arr_a.ndim}")
    va, vb = arr_a[m], arr_b[m]
    n = int(va.size)
    k = _budget(n, q)

    mask_a = np.zeros(arr_a.shape, dtype=bool)
    mask_a[m] = _topk_mask(va, k)
    mask_b = np.zeros(arr_b.shape, dtype=bool)
    mask_b[m] = _topk_mask(vb, k)

    h, w = arr_a.shape
    shifts = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    # Ordine per distanza crescente da (0, 0), e miglioramento in senso strettamente maggiore:
    # a pari IoU vince la traslazione piu' piccola, cioe' l'ipotesi meno avventurosa.
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
# incertezza da pareggi
# --------------------------------------------------------------------------------------


def tie_bounds(a: np.ndarray, b: np.ndarray, valid: np.ndarray, q: float) -> TieBand:
    """Intervallo esatto dell'IoU su tutte le rotture di pareggio ammissibili.

    Un top-k e' ammissibile se contiene tutti i pixel strettamente sopra la k-esima soglia e
    completa il budget con pixel che valgono esattamente la soglia. Quando la banda di
    pareggio e' piu' larga dei pixel che servono, ci sono molte selezioni ammissibili e l'IoU
    non e' un numero unico ma un intervallo. Questa funzione lo calcola in forma chiusa, in
    tempo costante rispetto al numero di selezioni possibili, contando i pixel nelle nove
    celle (sopra soglia / in pareggio / sotto soglia) per a incrociate con le stesse per b.

    Come leggerlo: se `width` e' vicino a zero l'IoU riportato e' un fatto; se e' grande, il
    numero puntuale e' in gran parte una conseguenza dell'ordine degli indici e va riportato
    come intervallo. Su uint8 con code sature (molti pixel a 255) `width` puo' arrivare a 1,
    cioe' il dato non vincola l'IoU per niente.

    NON dice quale rottura di pareggio sia la "giusta", perche' non ce n'e' una, e NON pesa le
    selezioni con una probabilita': e' un intervallo di ammissibilita', non un intervallo di
    confidenza.
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
    """Matrice 3x3 dei conteggi: righe = stato in a, colonne = stato in b.

    Stati: 0 = valore sopra la soglia (entra per forza), 1 = in pareggio con la soglia
    (entra se scelto), 2 = sotto la soglia (non puo' entrare).
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
    """Massima intersezione ottenibile scegliendo i pixel in pareggio.

    Sopra soglia in entrambe (cella 0,0) l'intersezione c'e' comunque. Poi ogni pixel scelto
    in a che sta sopra soglia in b (cella 1,0) aggiunge 1, simmetricamente la cella (0,1), e
    la cella (1,1) aggiunge 1 solo se scelto da entrambe le parti. L'obiettivo e' concavo e
    lineare a tratti nella quota spesa sulla cella (1,1), quindi basta valutarlo sui punti di
    rottura.
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
    """Minima intersezione ottenibile scegliendo i pixel in pareggio.

    Le scelte di a possono andare su celle innocue (1,2) o sulla cella condivisa (1,1), e solo
    quando queste finiscono cadono su (1,0), dove contano per forza. L'obiettivo e' convesso e
    lineare a tratti nelle due quote spese su (1,1), quindi il minimo sta su un vertice
    dell'arrangiamento delle rette di rottura, e le si valutano tutte.
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
    # n_tc e' un limite superiore per le scelte forzate su (1,0), quindi il minimo e' finito.
    return n_cc + int(best if best is not None else 0)


# --------------------------------------------------------------------------------------
# interni
# --------------------------------------------------------------------------------------


def _chunks(n: int, size: int | None = None) -> Iterator[slice]:
    # `size` viene letto dal modulo a ogni chiamata, non congelato come default: cosi' i test
    # possono abbassarlo e far passare i percorsi a blocchi sui casi piccoli.
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
        raise ValueError(f"q deve essere un numero, ricevuto {type(q).__name__}")
    if not (0.0 < float(q) <= 1.0):
        raise ValueError(f"q fuori da (0, 1]: {q}")
    if n <= 0:
        raise ValueError("nessun pixel valido: il confronto non e' definito")
    k = int(round(n * float(q)))
    if k == 0:
        raise ValueError(
            f"q={q} su n={n} pixel validi da' k=0: servono almeno {int(math.ceil(0.5 / q))} "
            "pixel validi, oppure un q piu' grande"
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
    """Il k-esimo valore piu' alto di un array 1-D, senza allocare array di indici.

    Su interi con intervallo contenuto (il caso del corpus, uint8) passa da un istogramma
    cumulato: esatto, e la memoria non dipende da quanti pixel ci sono. Altrimenti usa
    `np.partition`, che copia i valori una volta.
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
        tail = np.cumsum(hist[::-1])[::-1]  # tail[i] = quanti valori sono >= lo + i
        i = int(np.flatnonzero(tail >= k)[-1])  # il livello piu' alto che copre k pixel
        return values.dtype.type(lo + i)
    return np.partition(values, n - k)[n - k]


def _topk_and_threshold(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Maschera booleana dei k valori piu' alti di un array 1-D, con la soglia usata.

    Regola dei pareggi, dichiarata e non ereditata da un algoritmo di selezione: entrano tutti
    i pixel strettamente sopra la k-esima soglia, e il budget residuo si completa con i pixel
    in pareggio presi in ordine di indice piatto crescente, cioe' dalla prima riga verso il
    basso. La regola e' deterministica su qualunque versione di numpy e su qualunque layout di
    memoria, cosa che `np.argpartition` non garantisce, e non alloca un array di indici lungo
    come la mappa.

    NON e' una scelta neutrale: privilegia sistematicamente le righe in alto, e soprattutto fa
    coincidere le selezioni di due mappe che hanno la stessa banda di pareggio anche quando le
    due mappe non condividono nessuna informazione. Vedi `tie_bounds`, che misura quanto
    dell'IoU dipende da questa regola.
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
    """Rango medio (base 1) di ciascun gruppo di pareggio, dati i conteggi in ordine di valore."""
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
        raise ValueError("il percorso a istogramma vuole array interi non vuoti")
    a_min, a_len = la
    b_min, b_len = lb
    hist = np.zeros(a_len * b_len, dtype=np.int64)
    for sl in _chunks(int(va.size)):
        ia = va[sl].astype(np.int64, copy=False) - a_min
        ib = vb[sl].astype(np.int64, copy=False) - b_min
        hist += np.bincount((ia * b_len + ib).ravel(), minlength=a_len * b_len)
    return _spearman_from_hist(hist.reshape(a_len, b_len))


def _spearman_from_hist(h: np.ndarray) -> float:
    """Spearman esatto a partire dall'istogramma congiunto dei valori.

    Con i ranghi medi, Spearman e' esattamente il Pearson sui ranghi, e i ranghi dipendono
    solo dai conteggi marginali: l'istogramma congiunto contiene quindi tutto il necessario,
    e questa strada da' lo stesso numero del percorso sui ranghi pieni senza allocare array
    lunghi come la mappa.
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
