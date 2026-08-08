"""Test di inkfloor.metrics.

Criterio: ogni caso ha una risposta calcolabile a mano, e la costante calcolata a mano sta
scritta nel test. Dove la risposta esatta e' un intervallo (pareggi) c'e' una forza bruta
indipendente che enumera tutte le selezioni ammissibili e confronta con la forma chiusa.

Si esegue con la stdlib, senza pytest:
    .venv/bin/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkfloor.metrics import (  # noqa: E402
    Delta,
    ShapeMismatch,
    align,
    best_shift_iou,
    chance_iou,
    common_valid,
    delta_at_q,
    ink_valid,
    null_self,
    null_shift,
    shift_map,
    spearman,
    tie_bounds,
)
from inkfloor import metrics  # noqa: E402
from inkfloor.metrics import _kth_largest, _spearman_1d, _topk_and_threshold  # noqa: E402


def all_valid(shape) -> np.ndarray:
    return np.ones(shape, dtype=bool)


def smooth_field(shape=(256, 256), seed=7, passes=3) -> np.ndarray:
    """Campo casuale liscio a valori distinti: struttura locale, nessun pareggio."""
    rng = np.random.default_rng(seed)
    f = rng.random(shape)
    for _ in range(passes):
        f = (
            f
            + np.roll(f, 1, axis=0)
            + np.roll(f, -1, axis=0)
            + np.roll(f, 1, axis=1)
            + np.roll(f, -1, axis=1)
        ) / 5.0
    return f + 1.0  # tutto > 0, cosi' e' tutto "valido" nella convenzione del corpus


# ---------------------------------------------------------------------------------------
# la metrica sui casi a risposta nota
# ---------------------------------------------------------------------------------------


class TestExactValues(unittest.TestCase):
    def test_identical_maps_give_iou_exactly_one(self):
        a = smooth_field()
        v = all_valid(a.shape)
        for q in (0.01, 0.05, 0.20, 1.0):
            d = delta_at_q(a, a.copy(), v, q)
            self.assertEqual(d.iou, 1.0, f"q={q}")
            self.assertEqual(d.dice, 1.0, f"q={q}")
            self.assertEqual(d.k, round(a.size * q))
            self.assertEqual(d.n_valid, a.size)

    def test_disjoint_topk_gives_iou_exactly_zero(self):
        # meta' sinistra calda in a, meta' destra calda in b: qualunque rottura di pareggio,
        # i due top-10 non possono avere pixel in comune.
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 10), dtype=np.uint8)
        a[:, :5] = 200
        b[:, 5:] = 200
        d = delta_at_q(a, b, all_valid(a.shape), 0.10)
        self.assertEqual(d.k, 10)
        self.assertEqual(d.iou, 0.0)
        self.assertEqual(d.dice, 0.0)

    def test_known_overlap_k10_four_in_common(self):
        # n = 100 pixel validi, q = 0.10 -> k = 10 per lato.
        # a sceglie gli indici piatti 0..9, b sceglie 6..15: in comune 6,7,8,9 = 4 pixel.
        # inter = 4, union = 10 + 10 - 4 = 16  ->  IoU = 4/16 = 0.25 esatto
        #                                          Dice = 2*4/20 = 0.4 esatto
        a = np.ones(100, dtype=np.uint8).reshape(10, 10)
        b = np.ones(100, dtype=np.uint8).reshape(10, 10)
        a.reshape(-1)[0:10] = 255
        b.reshape(-1)[6:16] = 255
        d = delta_at_q(a, b, all_valid(a.shape), 0.10)
        self.assertEqual(d.n_valid, 100)
        self.assertEqual(d.k, 10)
        self.assertEqual(d.iou, 0.25)
        self.assertEqual(d.dice, 0.4)
        self.assertEqual(1.0 - d.iou, 0.75)  # il Delta vero e proprio

        # la selezione qui e' unica (banda di pareggio = pixel richiesti), quindi 0.25 e' un
        # fatto e non una conseguenza dell'ordine degli indici
        tb = tie_bounds(a, b, all_valid(a.shape), 0.10)
        self.assertTrue(tb.unique)
        self.assertEqual(tb.iou_min, 0.25)
        self.assertEqual(tb.iou_max, 0.25)
        self.assertEqual(tb.width, 0.0)

    def test_dice_iou_identity_under_paired_budget(self):
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 10), dtype=np.uint8)
        a.reshape(-1)[0:10] = 255
        b.reshape(-1)[6:16] = 255
        d = delta_at_q(a, b, all_valid(a.shape), 0.10)
        self.assertAlmostEqual(d.dice, 2 * d.iou / (1 + d.iou), places=15)
        self.assertAlmostEqual(d.iou, d.dice / (2 - d.dice), places=15)

    def test_budget_is_paired_where_a_fixed_threshold_would_lie(self):
        # Stessa graduatoria, calibrazioni lontane: 40 pixel caldi in a con valori 250..211,
        # gli stessi 40 pixel in b con valori 59..20. Con una soglia fissa a 128 i positivi
        # sono 40 contro 0 e l'IoU delle due mappe binarizzate e' 0: la soglia dice
        # "non sono d'accordo su niente". A budget appaiato entrambe spendono k = 10 sui pixel
        # piu' caldi, che sono gli stessi, e l'IoU e' 1.
        hot = np.arange(40)
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 10), dtype=np.uint8)
        a.reshape(-1)[hot] = 250 - hot
        b.reshape(-1)[hot] = 59 - hot

        thr_a = a >= 128
        thr_b = b >= 128
        self.assertEqual(int(thr_a.sum()), 40)
        self.assertEqual(int(thr_b.sum()), 0)
        self.assertEqual(int((thr_a & thr_b).sum()), 0)  # IoU a soglia fissa = 0

        d = delta_at_q(a, b, all_valid(a.shape), 0.10)
        self.assertEqual(d.k, 10)
        self.assertEqual(d.iou, 1.0)

    def test_monotone_recalibration_does_not_change_the_metric(self):
        # e' la ragione per cui non si usa una soglia fissa: b = a^2 e' la stessa
        # localizzazione con un'altra calibrazione, e la metrica deve dare 1.
        rng = np.random.default_rng(3)
        a = (rng.permutation(np.arange(1, 101)).reshape(10, 10)).astype(np.uint8)
        b = (a.astype(np.float64) ** 2) / 7.0 + 0.5
        v = common_valid([a, b])
        self.assertEqual(int(v.sum()), 100)
        for q in (0.05, 0.10, 0.50):
            self.assertEqual(delta_at_q(a, b, v, q).iou, 1.0, f"q={q}")
        self.assertAlmostEqual(spearman(a, b, v), 1.0, places=12)


# ---------------------------------------------------------------------------------------
# shape, maschere, input malformati
# ---------------------------------------------------------------------------------------


class TestShapesAndMasks(unittest.TestCase):
    def test_slightly_different_shapes_are_cropped_to_the_common_region(self):
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 12), dtype=np.uint8)
        a.reshape(-1)[0:10] = 255
        b[:, 10:] = 255  # la colonna in piu' e' rovente: se non fosse ritagliata, vincerebbe
        b.reshape(-1)[0:10] = 255
        aa, bb = align(a, b)
        self.assertEqual(aa.shape, (10, 10))
        self.assertEqual(bb.shape, (10, 10))
        d = delta_at_q(a, b, all_valid((10, 10)), 0.10)
        self.assertEqual(d.n_valid, 100)
        self.assertEqual(d.iou, 1.0)

    def test_shape_difference_beyond_tolerance_raises(self):
        a = np.ones((100, 100), dtype=np.uint8)
        b = np.ones((100, 200), dtype=np.uint8)
        with self.assertRaises(ShapeMismatch):
            align(a, b)
        with self.assertRaises(ShapeMismatch):
            delta_at_q(a, b, all_valid((100, 100)), 0.10)
        # entro tolleranza invece passa
        c = np.ones((100, 150), dtype=np.uint8)
        self.assertEqual(align(a, c)[0].shape, (100, 100))

    def test_zero_is_out_of_mask_not_a_prediction(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        a[2:8, 2:8] = 3
        self.assertEqual(int(ink_valid(a).sum()), 36)
        b = a.copy()
        b[9, 9] = 250  # fuori dalla maschera comune: non deve entrare nel top-k
        v = common_valid([a, b])
        self.assertEqual(int(v.sum()), 36)
        d = delta_at_q(a, b, v, 0.10)
        self.assertEqual(d.n_valid, 36)
        self.assertEqual(d.k, 4)
        self.assertEqual(d.iou, 1.0)

    def test_common_valid_intersects_and_crops(self):
        a = np.ones((10, 10), dtype=np.uint8)
        b = np.ones((10, 11), dtype=np.uint8)
        a[0, :] = 0
        b[:, 0] = 0
        v = common_valid([a, b])
        self.assertEqual(v.shape, (10, 10))
        self.assertEqual(int(v.sum()), 9 * 9)
        self.assertFalse(v[0, 5])
        self.assertFalse(v[5, 0])
        self.assertTrue(v[1, 1])

    def test_valid_mask_accepts_uint8(self):
        a = smooth_field((32, 32))
        v = np.ones((32, 32), dtype=np.uint8)
        self.assertEqual(delta_at_q(a, a, v, 0.10).iou, 1.0)

    def test_degenerate_inputs_raise_instead_of_returning_a_number(self):
        a = smooth_field((32, 32))
        v = all_valid(a.shape)
        for bad_q in (0.0, -0.1, 1.5, 2):
            with self.assertRaises(ValueError, msg=f"q={bad_q}"):
                delta_at_q(a, a, v, bad_q)
        with self.assertRaises(ValueError):
            delta_at_q(a, a, np.zeros_like(v), 0.10)  # nessun pixel valido
        # n troppo piccolo per il q chiesto: round(4 * 0.05) = 0
        small = np.ones((2, 2))
        with self.assertRaises(ValueError):
            delta_at_q(small, small, all_valid((2, 2)), 0.05)

    def test_inputs_are_not_mutated(self):
        a = (smooth_field((64, 64)) * 100).astype(np.uint8)
        b = np.roll(a, 3, axis=1)
        v = common_valid([a, b])
        a0, b0, v0 = a.copy(), b.copy(), v.copy()
        delta_at_q(a, b, v, 0.05)
        spearman(a, b, v)
        null_self(a, v, 0.05)
        null_shift(a, v, 0.05, px=8)
        best_shift_iou(a, b, v, 0.05, radius=4)
        tie_bounds(a, b, v, 0.05)
        self.assertTrue(np.array_equal(a, a0))
        self.assertTrue(np.array_equal(b, b0))
        self.assertTrue(np.array_equal(v, v0))


# ---------------------------------------------------------------------------------------
# controlli nulli
# ---------------------------------------------------------------------------------------


class TestNulls(unittest.TestCase):
    def test_null_self_is_exactly_one(self):
        for a in (smooth_field((128, 128)), (smooth_field((128, 128)) * 200).astype(np.uint8)):
            v = all_valid(a.shape)
            for q in (0.01, 0.05, 0.20):
                d = null_self(a, v, q)
                self.assertEqual(d.iou, 1.0, f"q={q}, dtype={a.dtype}")
                self.assertEqual(d.dice, 1.0)

    def test_null_shift_on_structure_is_low(self):
        a = smooth_field((256, 256), seed=11)
        v = all_valid(a.shape)
        for q in (0.01, 0.05, 0.20):
            d = null_shift(a, v, q, px=64)
            # traslazione di 64 px su una struttura con lunghezza di correlazione ~4 px:
            # i due insiemi sono in pratica indipendenti, e per insiemi indipendenti
            # E[IoU] ~ q/(2-q), cioe' 0.005 / 0.026 / 0.111 per i tre q.
            self.assertLess(d.iou, 3.0 * q / (2.0 - q), f"q={q}, iou={d.iou}")
            self.assertLess(d.n_valid, a.size)  # la striscia rientrata e' esclusa

    def test_null_shift_lands_near_the_chance_level(self):
        # Il pavimento della metrica cresce con q: il controllo nullo su una struttura
        # decorrelata deve stare nello stesso ordine di grandezza di chance_iou(q) = q/(2-q).
        a = smooth_field((256, 256), seed=11)
        v = all_valid(a.shape)
        for q in (0.01, 0.05, 0.20):
            iou = null_shift(a, v, q, px=64).iou
            chance = chance_iou(q)
            self.assertGreater(iou, 0.3 * chance, f"q={q}, iou={iou}, chance={chance}")
            self.assertLess(iou, 3.0 * chance, f"q={q}, iou={iou}, chance={chance}")
        self.assertAlmostEqual(chance_iou(0.20), 0.1111111111111111, places=15)
        self.assertAlmostEqual(chance_iou(1.0), 1.0, places=15)

    def test_null_shift_on_a_constant_map_is_one_and_means_nothing(self):
        # Il controllo nullo non e' magico. Su una mappa costante tutti i pixel sono in
        # pareggio, i pareggi vengono rotti per posizione, e le due selezioni coincidono:
        # IoU = 1 fra due mappe che non condividono nessuna informazione.
        a = np.full((128, 128), 7, dtype=np.uint8)
        v = all_valid(a.shape)
        d = null_shift(a, v, 0.05, px=64)
        self.assertEqual(d.iou, 1.0)
        # e tie_bounds lo dichiara: l'intervallo ammissibile e' tutto [0, 1]
        shifted, vs = shift_map(a, v, 0, 64)
        tb = tie_bounds(a, shifted, v & vs, 0.05)
        self.assertEqual(tb.iou, 1.0)
        self.assertEqual(tb.iou_min, 0.0)
        self.assertEqual(tb.iou_max, 1.0)
        self.assertEqual(tb.width, 1.0)
        self.assertFalse(tb.unique)

    def test_null_shift_on_a_periodic_map_is_high_and_means_nothing(self):
        # Secondo modo di ingannare il controllo nullo: struttura periodica in x di periodo
        # che divide px. La mappa traslata di 64 e' identica a se stessa.
        x = np.arange(256, dtype=np.uint8) % 64
        a = np.tile((1 + 3 * x).astype(np.uint8), (128, 1))
        v = all_valid(a.shape)
        self.assertEqual(null_shift(a, v, 0.05, px=64).iou, 1.0)
        # con un px non commensurato il controllo torna a funzionare
        self.assertLess(null_shift(a, v, 0.05, px=37).iou, 0.6)

    def test_shift_map_invalidates_the_wrapped_strip(self):
        a = np.arange(1, 26, dtype=np.uint8).reshape(5, 5)
        v = all_valid(a.shape)
        s, vs = shift_map(a, v, 0, 2)
        self.assertTrue(np.array_equal(s[:, 2:], a[:, :3]))  # out[y, x] = a[y, x - dx]
        self.assertFalse(vs[:, :2].any())
        self.assertTrue(vs[:, 2:].all())
        s, vs = shift_map(a, v, -1, 0)
        self.assertTrue(np.array_equal(s[:4], a[1:]))
        self.assertFalse(vs[4:].any())
        self.assertTrue(vs[:4].all())


# ---------------------------------------------------------------------------------------
# pareggi: determinismo e intervallo esatto
# ---------------------------------------------------------------------------------------


def brute_force_bounds(va: np.ndarray, vb: np.ndarray, k: int) -> tuple[float, float]:
    """Enumera TUTTE le selezioni top-k ammissibili delle due mappe e restituisce (min, max).

    Riferimento indipendente per `tie_bounds`: nessuna formula, solo itertools.
    """
    n = int(va.size)

    def parts(v):
        thr = np.sort(v)[::-1][k - 1]
        core = tuple(i for i in range(n) if v[i] > thr)
        tie = tuple(i for i in range(n) if v[i] == thr)
        return core, tie, k - len(core)

    core_a, tie_a, need_a = parts(va)
    core_b, tie_b, need_b = parts(vb)
    lo, hi = 2.0, -1.0
    for pa in itertools.combinations(tie_a, need_a):
        sa = set(core_a) | set(pa)
        for pb in itertools.combinations(tie_b, need_b):
            sb = set(core_b) | set(pb)
            iou = len(sa & sb) / len(sa | sb)
            lo = min(lo, iou)
            hi = max(hi, iou)
    return lo, hi


class TestTies(unittest.TestCase):
    def test_selection_is_deterministic_across_repeated_calls(self):
        # mappa uint8 con banda di pareggio enorme: 4000 pixel a 255 su 10000, k = 500
        rng = np.random.default_rng(1)
        a = rng.integers(1, 40, size=(100, 100)).astype(np.uint8)
        b = rng.integers(1, 40, size=(100, 100)).astype(np.uint8)
        a.reshape(-1)[:4000] = 255
        b.reshape(-1)[2000:6000] = 255
        v = all_valid(a.shape)
        first = delta_at_q(a, b, v, 0.05)
        for _ in range(5):
            self.assertEqual(delta_at_q(a, b, v, 0.05), first)
        # anche su copie fresche degli array, cioe' su altra memoria
        self.assertEqual(delta_at_q(a.copy(), b.copy(), v.copy(), 0.05), first)
        self.assertIsInstance(first, Delta)

    def test_quantisation_creates_the_arbitrariness(self):
        # lo stesso campo, prima in float e poi in uint8: in float il top-k e' unico, in
        # uint8 non lo e' piu' e una fetta dell'IoU diventa arbitraria.
        f = smooth_field((128, 128), seed=5)
        g = smooth_field((128, 128), seed=6)
        v = all_valid(f.shape)
        tb_float = tie_bounds(f, g, v, 0.05)
        self.assertTrue(tb_float.unique)
        self.assertEqual(tb_float.width, 0.0)

        fq = np.clip(f * 120, 1, 255).astype(np.uint8)
        gq = np.clip(g * 120, 1, 255).astype(np.uint8)
        tb_uint8 = tie_bounds(fq, gq, v, 0.05)
        self.assertFalse(tb_uint8.unique)
        self.assertGreater(tb_uint8.width, 0.0)
        self.assertLessEqual(tb_uint8.iou_min, tb_uint8.iou)
        self.assertLessEqual(tb_uint8.iou, tb_uint8.iou_max)

    def test_saturated_tails_make_the_iou_meaningless(self):
        # caso realistico: due mappe con code sature (molti pixel a 255) e insiemi saturi che
        # si sovrappongono in parte. Il top-k sta interamente dentro la banda di pareggio:
        # l'IoU riportato e' un numero, ma l'intervallo ammissibile e' tutto [0, 1].
        rng = np.random.default_rng(2)
        a = rng.integers(1, 100, size=(100, 100)).astype(np.uint8)
        b = rng.integers(1, 100, size=(100, 100)).astype(np.uint8)
        a.reshape(-1)[0:3000] = 255
        b.reshape(-1)[1500:4500] = 255
        v = all_valid(a.shape)
        tb = tie_bounds(a, b, v, 0.05)
        self.assertEqual(tb.k, 500)
        self.assertEqual(tb.forced_a, 0)
        self.assertEqual(tb.forced_b, 0)
        self.assertEqual(tb.tie_band_a, 3000)
        self.assertEqual(tb.tie_band_b, 3000)
        self.assertEqual(tb.iou_min, 0.0)
        self.assertEqual(tb.iou_max, 1.0)
        self.assertGreaterEqual(tb.iou, 0.0)
        self.assertLessEqual(tb.iou, 1.0)

    def test_tie_bounds_match_brute_force_on_random_small_cases(self):
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(400):
            n = 8
            va = rng.integers(0, 3, size=n)
            vb = rng.integers(0, 3, size=n)
            k = int(rng.integers(1, n + 1))
            q = k / n  # esatto in binario: n e' una potenza di due
            v = all_valid(n)
            tb = tie_bounds(va, vb, v, q)
            self.assertEqual(tb.k, k)
            lo, hi = brute_force_bounds(va, vb, k)
            self.assertAlmostEqual(tb.iou_min, lo, places=12, msg=f"{va} {vb} k={k}")
            self.assertAlmostEqual(tb.iou_max, hi, places=12, msg=f"{va} {vb} k={k}")
            # e il valore puntuale che riporta delta_at_q deve stare nell'intervallo
            d = delta_at_q(va, vb, v, q)
            self.assertGreaterEqual(d.iou, lo - 1e-12, msg=f"{va} {vb} k={k}")
            self.assertLessEqual(d.iou, hi + 1e-12, msg=f"{va} {vb} k={k}")
            self.assertEqual(d.iou, tb.iou)
            checked += 1
        self.assertEqual(checked, 400)

    def test_bound_candidates_match_a_dense_scan(self):
        # I due estremi sono calcolati valutando l'obiettivo su pochi punti di rottura invece
        # che su tutte le quote possibili. Qui la riduzione viene confrontata con la scansione
        # completa delle quote, su configurazioni di celle generate a caso.
        from inkfloor.metrics import _max_intersection, _min_intersection

        rng = np.random.default_rng(55)
        for _ in range(3000):
            cells = rng.integers(0, 7, size=(3, 3)).astype(np.int64)
            band_a = int(cells[1].sum())
            band_b = int(cells[:, 1].sum())
            if band_a == 0 or band_b == 0:
                continue
            need_a = int(rng.integers(1, band_a + 1))
            need_b = int(rng.integers(1, band_b + 1))
            n_ct, n_tc = int(cells[0, 1]), int(cells[1, 0])
            n_tt, n_to, n_ot = int(cells[1, 1]), int(cells[1, 2]), int(cells[2, 1])

            hi = min(n_tt, need_a, need_b)
            want_max = max(
                x + min(n_ct, need_b - x) + min(n_tc, need_a - x) for x in range(hi + 1)
            )
            self.assertEqual(
                _max_intersection(cells, need_a, need_b),
                int(cells[0, 0]) + want_max,
                msg=f"cells={cells.tolist()} need=({need_a},{need_b})",
            )

            want_min = min(
                max(0, need_a - n_to - p) + max(0, need_b - n_ot - r) + max(0, p + r - n_tt)
                for p in range(min(n_tt, need_a) + 1)
                for r in range(min(n_tt, need_b) + 1)
            )
            self.assertEqual(
                _min_intersection(cells, need_a, need_b),
                int(cells[0, 0]) + want_min,
                msg=f"cells={cells.tolist()} need=({need_a},{need_b})",
            )

    def test_tie_bounds_match_brute_force_with_wider_value_range(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            n = 8
            va = rng.integers(0, 6, size=n)
            vb = rng.integers(0, 6, size=n)
            k = int(rng.integers(1, n + 1))
            tb = tie_bounds(va, vb, all_valid(n), k / n)
            lo, hi = brute_force_bounds(va, vb, k)
            self.assertAlmostEqual(tb.iou_min, lo, places=12, msg=f"{va} {vb} k={k}")
            self.assertAlmostEqual(tb.iou_max, hi, places=12, msg=f"{va} {vb} k={k}")


# ---------------------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------------------


class TestSelectionRule(unittest.TestCase):
    """La regola dichiarata: sopra soglia tutti, poi i pareggi in ordine di indice crescente."""

    @staticmethod
    def reference_iou(a, b, valid, q) -> float:
        """Implementazione di riferimento, lenta e ovvia, della stessa regola."""
        m = np.asarray(valid).astype(bool)
        va = np.asarray(a)[m].ravel()
        vb = np.asarray(b)[m].ravel()
        n = va.size
        k = int(round(n * q))

        def sel(v):
            # valore decrescente, e a parita' di valore indice crescente (sort stabile)
            order = np.argsort(-v.astype(np.float64), kind="stable")
            return set(order[:k].tolist())

        sa, sb = sel(va), sel(vb)
        return len(sa & sb) / len(sa | sb)

    def test_matches_reference_on_random_tie_heavy_maps(self):
        rng = np.random.default_rng(77)
        for levels in (2, 4, 16, 256):
            for q in (0.01, 0.05, 0.2, 0.5):
                a = rng.integers(1, levels + 1, size=(64, 64)).astype(np.uint8)
                b = rng.integers(1, levels + 1, size=(64, 64)).astype(np.uint8)
                v = rng.random((64, 64)) > 0.1  # maschera irregolare, come nel corpus
                got = delta_at_q(a, b, v, q).iou
                want = self.reference_iou(a, b, v, q)
                self.assertAlmostEqual(got, want, places=12, msg=f"levels={levels} q={q}")

    def test_matches_reference_with_tiny_chunks(self):
        # stessa verifica con i blocchi interni ridotti a 7 elementi: se la selezione a blocchi
        # sbagliasse un confine, qui salterebbe fuori.
        rng = np.random.default_rng(78)
        a = rng.integers(1, 5, size=(40, 40)).astype(np.uint8)
        b = rng.integers(1, 5, size=(40, 40)).astype(np.uint8)
        v = rng.random((40, 40)) > 0.2
        saved = metrics._CHUNK
        try:
            metrics._CHUNK = 7
            for q in (0.01, 0.05, 0.2, 0.5, 1.0):
                self.assertAlmostEqual(
                    delta_at_q(a, b, v, q).iou,
                    self.reference_iou(a, b, v, q),
                    places=12,
                    msg=f"q={q}",
                )
            tb = tie_bounds(a, b, v, 0.05)
            self.assertLessEqual(tb.iou_min, tb.iou)
            self.assertLessEqual(tb.iou, tb.iou_max)
        finally:
            metrics._CHUNK = saved

    def test_kth_largest_matches_sorting(self):
        rng = np.random.default_rng(79)
        for values in (
            rng.integers(0, 256, size=1000).astype(np.uint8),
            rng.integers(-5, 6, size=1000),
            rng.random(1000),
            np.full(50, 3, dtype=np.uint8),
        ):
            for k in (1, 2, 17, 50):
                if k > values.size:
                    continue
                want = np.sort(values)[::-1][k - 1]
                self.assertEqual(float(_kth_largest(values, k)), float(want), f"k={k}")

    def test_tie_break_takes_the_lowest_indices(self):
        v = np.full(10, 5, dtype=np.uint8)
        mask, thr = _topk_and_threshold(v, 3)
        self.assertEqual(int(thr), 5)
        self.assertTrue(np.array_equal(mask, np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)))

    def test_identical_saturated_tail_forces_iou_one(self):
        # Il caso che puo' ingannare chi legge: due mappe che condividono SOLO la regione
        # satura a 255 e sotto quella regione non hanno niente in comune. Il top-1% cade tutto
        # dentro il pareggio, la regola posizionale sceglie gli stessi pixel in entrambe, e
        # l'IoU riportato e' 1: pavimento perfetto per due mappe che non concordano su niente.
        rng = np.random.default_rng(101)
        a = rng.integers(1, 200, size=(100, 100)).astype(np.uint8)
        b = rng.integers(1, 200, size=(100, 100)).astype(np.uint8)
        sat = np.zeros((100, 100), dtype=bool)
        sat[10:40, 10:40] = True  # 900 pixel su 10000, saturi in entrambe
        a[sat] = 255
        b[sat] = 255
        v = all_valid(a.shape)
        d = delta_at_q(a, b, v, 0.05)  # k = 500 < 900 pixel saturi
        self.assertEqual(d.k, 500)
        self.assertEqual(d.iou, 1.0)
        tb = tie_bounds(a, b, v, 0.05)
        self.assertEqual(tb.tie_band_a, 900)
        self.assertEqual(tb.tie_band_b, 900)
        self.assertEqual(tb.forced_a, 0)
        self.assertEqual(tb.iou_max, 1.0)
        # il minimo non e' zero: le due selezioni pescano 500 pixel ciascuna dalla STESSA banda
        # di 900, quindi per il principio dei cassetti almeno 500+500-900 = 100 coincidono, e
        # 100/(1000-100) = 1/9. Cioe' anche il fondo dell'intervallo e' un artefatto della
        # saturazione, non un accordo fra le due mappe.
        self.assertAlmostEqual(tb.iou_min, 1.0 / 9.0, places=15)


class TestSpearman(unittest.TestCase):
    def test_known_value_without_ties(self):
        # a = [1,2,3,4,5], b = [2,1,4,3,5]: d = [-1,1,-1,1,0], sum d^2 = 4
        # rho = 1 - 6*4 / (5*(25-1)) = 1 - 24/120 = 0.8 esatto
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 1, 4, 3, 5])
        self.assertAlmostEqual(spearman(a, b, all_valid(5)), 0.8, places=12)

    def test_known_value_with_ties(self):
        # a = [1,1,2,2] -> ranghi medi [1.5,1.5,3.5,3.5]; b = [1,2,3,4] -> [1,2,3,4]
        # cov*n = 4, sd = 2 e sqrt(5) -> rho = 4/(2*sqrt(5)) = 1/sqrt(1.25) = 0.8944271909999159
        a = np.array([1, 1, 2, 2])
        b = np.array([1, 2, 3, 4])
        expected = 1.0 / math.sqrt(1.25)
        self.assertAlmostEqual(spearman(a, b, all_valid(4)), expected, places=12)
        self.assertAlmostEqual(expected, 0.8944271909999159, places=15)

    def test_monotone_and_reversed(self):
        rng = np.random.default_rng(9)
        a = rng.permutation(np.arange(1, 257)).astype(np.uint8)
        v = all_valid(a.size)
        self.assertAlmostEqual(spearman(a, a.copy(), v), 1.0, places=12)
        self.assertAlmostEqual(spearman(a, (255 - a).astype(np.uint8), v), -1.0, places=12)
        # invariante a schiacciamenti monotoni: la radice non cambia l'ordine
        self.assertAlmostEqual(spearman(a, np.sqrt(a.astype(np.float64)), v), 1.0, places=12)

    def test_constant_map_is_nan(self):
        a = np.full(100, 5, dtype=np.uint8)
        b = np.arange(100, dtype=np.uint8)
        self.assertTrue(math.isnan(spearman(a, b, all_valid(100))))
        self.assertTrue(math.isnan(spearman(a, a, all_valid(100))))

    def test_histogram_path_and_rank_path_agree(self):
        # due implementazioni indipendenti dello stesso numero: l'istogramma congiunto
        # (usato sui dati veri, per memoria) e i ranghi pieni.
        rng = np.random.default_rng(13)
        for levels in (2, 8, 64, 256):
            va = rng.integers(0, levels, size=5000).astype(np.uint8)
            noise = rng.integers(0, levels, size=5000)
            vb = ((va // 2 + noise) % 256).astype(np.uint8)  # un po' di correlazione
            hist = _spearman_1d(va, vb, method="hist")
            rank = _spearman_1d(va, vb, method="rank")
            self.assertFalse(math.isnan(hist), f"levels={levels}")
            self.assertAlmostEqual(hist, rank, places=12, msg=f"levels={levels}")

    def test_spearman_stays_high_where_iou_collapses(self):
        # la ragione per cui vanno riportate entrambe: due mappe con lo stesso ordine
        # complessivo ma code diverse danno Spearman alto e IoU al top-1% basso.
        rng = np.random.default_rng(21)
        base = smooth_field((128, 128), seed=17)
        base = (base - base.mean()) / base.std()
        a = base + 5.0
        b = base + 0.4 * rng.standard_normal(base.shape) + 5.0
        v = all_valid(a.shape)
        rho = spearman(a, b, v)
        iou = delta_at_q(a, b, v, 0.01).iou
        # numeri misurati su questo seed: rho = 0.9247, IoU al top-1% = 0.3782.
        self.assertGreater(rho, 0.90)
        self.assertLess(iou, 0.50)
        self.assertLess(iou, rho / 2.0)


# ---------------------------------------------------------------------------------------
# ricerca della traslazione migliore
# ---------------------------------------------------------------------------------------


class TestBestShift(unittest.TestCase):
    def test_identical_maps_prefer_no_shift(self):
        a = smooth_field((96, 96), seed=31)
        v = all_valid(a.shape)
        dy, dx, iou = best_shift_iou(a, a.copy(), v, 0.05, radius=4)
        self.assertEqual((dy, dx), (0, 0))
        self.assertEqual(iou, 1.0)

    def test_recovers_a_known_shift(self):
        # b e' a spostata in basso di 3 e a destra di 5: la risposta e' (-3, -5),
        # cioe' "per far combaciare b con a, spostala indietro di 3 e 5".
        a = smooth_field((96, 96), seed=33)
        a[:6, :] = 1.0  # bordo piatto, cosi' il wrap non crea falsi massimi
        a[-6:, :] = 1.0
        a[:, :6] = 1.0
        a[:, -6:] = 1.0
        b = np.roll(a, (3, 5), axis=(0, 1))
        v = all_valid(a.shape)
        dy, dx, iou = best_shift_iou(a, b, v, 0.05, radius=8)
        self.assertEqual((dy, dx), (-3, -5))
        self.assertEqual(iou, 1.0)
        # e la metrica senza correzione, a traslazione zero, e' molto peggio
        self.assertLess(delta_at_q(a, b, v, 0.05).iou, 0.5)

    def test_radius_zero_is_the_plain_comparison(self):
        a = smooth_field((64, 64), seed=35)
        b = smooth_field((64, 64), seed=36)
        v = all_valid(a.shape)
        dy, dx, iou = best_shift_iou(a, b, v, 0.05, radius=0)
        self.assertEqual((dy, dx), (0, 0))
        self.assertAlmostEqual(iou, delta_at_q(a, b, v, 0.05).iou, places=15)

    def test_unrelated_maps_still_return_a_winner(self):
        # Trappola per chi legge: la funzione restituisce sempre un massimo. Su due mappe
        # indipendenti il massimo su 81 traslazioni e' rumore, e la traslazione "vincente" non
        # significa niente. L'unico modo di leggerla e' confrontarla con la traslazione zero.
        a = smooth_field((128, 128), seed=41)
        b = smooth_field((128, 128), seed=42)
        v = all_valid(a.shape)
        dy, dx, best = best_shift_iou(a, b, v, 0.05, radius=4)
        zero = delta_at_q(a, b, v, 0.05).iou
        self.assertNotEqual((dy, dx), (0, 0))  # il massimo cade fuori dallo zero
        self.assertGreater(best, zero)
        self.assertLess(best - zero, 0.10)  # ma il guadagno e' minimo: e' rumore, non allineamento
        self.assertLess(best, 0.20)

    def test_negative_radius_raises(self):
        a = smooth_field((32, 32))
        with self.assertRaises(ValueError):
            best_shift_iou(a, a, all_valid(a.shape), 0.05, radius=-1)


if __name__ == "__main__":
    unittest.main()
