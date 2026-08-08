"""Test di `inkfloor.census`, senza rete.

Ogni stringa qui sotto è una key reale, copiata da un ListObjectsV2 sul bucket pubblico il
2026-08-08. Niente nomi inventati: un test su nomi inventati verifica il parser contro se
stesso. Le dimensioni in byte sono quelle vere degli oggetti, così i Prediction costruiti nei
test sono identici a quelli che produrrebbe `census()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkfloor.census import (  # noqa: E402
    Pair,
    Prediction,
    comparable,
    pair_stats,
    pairs,
    parse_prediction,
)

# ---------------------------------------------------------------------------
# Key reali
# ---------------------------------------------------------------------------

SEG_2X2 = "PHerc0172/segments/20251107110950-w064_20251107110950052_flatboi"
INK_2X2 = f"{SEG_2X2}/ink-detection"

# Il 2x2 completo: due volumi (…838, …839) per due modelli (july_retreat, november19).
K_838_JULY = (
    f"{INK_2X2}/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838"
    "-20250713185324-timesformer_scroll5_july_retreat-tile64-stride16.tif"
)
K_838_NOV = (
    f"{INK_2X2}/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838"
    "-20251222202946-timesformer_scroll5_november19-tile64-stride16.tif"
)
K_839_JULY = (
    f"{INK_2X2}/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131839"
    "-20250713185324-timesformer_scroll5_july_retreat-tile64-stride16.tif"
)
K_839_NOV = (
    f"{INK_2X2}/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131839"
    "-20251222202946-timesformer_scroll5_november19-tile64-stride16.tif"
)
SIZE_838_JULY = 44942590
SIZE_838_NOV = 46337592
SIZE_839_JULY = 33840118
SIZE_839_NOV = 48283061

# Con distanza sorgente-oggetto E livello piramidale, e un nome di modello pieno di trattini.
K_L1 = (
    "PHerc0139/segments/20250108000000-w025_2025010863/ink-detection/"
    "PHerc0139-20250108000000-1.129um-0.22m-59keV-volume-20260413113053-L1"
    "-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif"
)
SIZE_L1 = 36667210

# Lo stesso modello `mrg20736-1um-s1z2` senza livello nel nome, accanto a `new_canon`:
# stesso volume, stesso raster, modelli diversi. È una coppia kind="model".
SEG_500P2 = "PHerc0500P2/segments/20250628074500-500P2_front"
K_500P2_CANON = (
    f"{SEG_500P2}/ink-detection/PHerc0500P2-20250628074500-2.215um-0.4m-111keV"
    "-volume-20250526151718-20260417190342-new_canon_autoresearch_recipe"
    "-tile256-stride128.tif"
)
K_500P2_MRG = (
    f"{SEG_500P2}/ink-detection/PHerc0500P2-20250628074500-2.215um-0.4m-111keV"
    "-volume-20250526151718-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif"
)
SIZE_500P2_CANON = 20507064
SIZE_500P2_MRG = 43698867

# Segmento con due predizioni che differiscono in tutto: volume, modello e raster.
SEG_PARIS4 = "PHercParis4/segments/20230702185753"
K_PARIS4_L1 = (
    f"{SEG_PARIS4}/ink-detection/PHercParis4-20230702185753-1.129um-0.23m-78keV"
    "-volume-20260608103018-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif"
)
K_PARIS4_24 = (
    f"{SEG_PARIS4}/ink-detection/PHercParis4-20230702185753-2.4um-0.22m-78keV"
    "-volume-20260411134726-20260417190342-new_canon_autoresearch_recipe"
    "-tile256-stride128.tif"
)
SIZE_PARIS4_L1 = 233574753
SIZE_PARIS4_24 = 166927520

# Key reali che DEVONO dare None.
K_DOWNSAMPLED = (
    f"{INK_2X2}/downsampled/PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838"
    "-20250713185324-timesformer_scroll5_july_retreat-tile64-stride16-ds8.jpg"
)
K_LAYERS_INK = "PHerc1447/segments/raw/auto_grown_20250702235910292/layers_ink/00.tif"
K_TIFXYZ = "PHerc1451/segments/raw/z_dbg_gen_00250/x.tif"
K_MESH = f"{SEG_2X2}/mesh/20251107110950-w064_20251107110950052_flatboi.obj"


def _p(key: str, size: int) -> Prediction:
    pred = parse_prediction(key, size)
    assert pred is not None, key
    return pred


# ---------------------------------------------------------------------------
# parse_prediction: i campi giusti
# ---------------------------------------------------------------------------


def test_parse_schema_senza_distanza_ne_livello() -> None:
    p = _p(K_838_JULY, SIZE_838_JULY)
    assert p.key == K_838_JULY
    assert p.sample == "PHerc0172"
    assert p.segment == "20251107110950-w064_20251107110950052_flatboi"
    assert p.volume == "20241024131838"
    assert p.model == "20250713185324-timesformer_scroll5_july_retreat"
    assert p.voxel_um == 7.91
    assert p.tile == 64
    assert p.stride == 16
    assert p.size_bytes == SIZE_838_JULY
    assert p.kev == 53.0
    assert p.dist_m is None
    # Il livello non è nel nome: resta ignoto, non diventa 0.
    assert p.level is None
    assert p.step_um is None
    assert p.segment_prefix == SEG_2X2 + "/"


def test_parse_schema_con_distanza_e_livello() -> None:
    p = _p(K_L1, SIZE_L1)
    assert p.sample == "PHerc0139"
    assert p.segment == "20250108000000-w025_2025010863"
    assert p.volume == "20260413113053"
    # Il livello sta fra volume e modello e non deve finire nel nome del modello.
    assert p.level == 1
    assert p.model == "20260709123958-mrg20736-1um-s1z2"
    assert p.voxel_um == 1.129
    assert p.dist_m == 0.22
    assert p.kev == 59.0
    assert p.tile == 256
    assert p.stride == 128
    # Passo raster effettivo: voxel * 2^L.
    assert p.step_um == 2.258


def test_parse_modello_con_trattini_senza_livello() -> None:
    """Lo stesso modello del test precedente, ma in un nome che non dichiara il livello."""
    p = _p(K_500P2_MRG, SIZE_500P2_MRG)
    assert p.model == "20260709123958-mrg20736-1um-s1z2"
    assert p.level is None
    assert p.volume == "20250526151718"
    assert p.voxel_um == 2.215
    assert p.dist_m == 0.4
    assert p.tile == 256 and p.stride == 128


def test_parse_voxel_non_confuso_con_1um_del_nome_modello() -> None:
    """`mrg20736-1um-s1z2` contiene `1um`: il voxel deve restare quello del nome file."""
    assert _p(K_PARIS4_L1, SIZE_PARIS4_L1).voxel_um == 1.129
    assert _p(K_PARIS4_24, SIZE_PARIS4_24).voxel_um == 2.4


def test_parse_tutte_e_quattro_le_predizioni_del_2x2() -> None:
    preds = [
        _p(K_838_JULY, SIZE_838_JULY),
        _p(K_838_NOV, SIZE_838_NOV),
        _p(K_839_JULY, SIZE_839_JULY),
        _p(K_839_NOV, SIZE_839_NOV),
    ]
    assert {p.volume for p in preds} == {"20241024131838", "20241024131839"}
    assert {p.model for p in preds} == {
        "20250713185324-timesformer_scroll5_july_retreat",
        "20251222202946-timesformer_scroll5_november19",
    }
    assert len({p.segment for p in preds}) == 1


# ---------------------------------------------------------------------------
# parse_prediction: i None
# ---------------------------------------------------------------------------


def test_anteprima_downsampled_da_none() -> None:
    """L'anteprima ridotta 8x ha lo stesso nome più `-ds8.jpg`: non è la predizione."""
    assert parse_prediction(K_DOWNSAMPLED, 595229) is None


def test_key_non_ink_danno_none() -> None:
    assert parse_prediction(K_LAYERS_INK, 3445270) is None
    assert parse_prediction(K_TIFXYZ, 453976) is None
    assert parse_prediction(K_MESH, 1) is None


def test_nomi_incompleti_danno_none() -> None:
    base = f"{INK_2X2}/"
    casi = [
        # senza il blocco `volume-<id>`
        "PHerc0172-20251107110950-7.91um-53keV-20250713185324-timesformer.tif",
        # id volume troppo corto
        "PHerc0172-20251107110950-7.91um-53keV-volume-2024102413-20250713185324-t.tif",
        # modello senza timestamp davanti
        "PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838-timesformer.tif",
        # senza voxel
        "PHerc0172-20251107110950-53keV-volume-20241024131838-20250713185324-t.tif",
        # tile senza stride
        "PHerc0172-20251107110950-7.91um-53keV-volume-20241024131838"
        "-20250713185324-timesformer-tile64.tif",
        # senza timestamp del segmento
        "PHerc0172-7.91um-53keV-volume-20241024131838-20250713185324-timesformer.tif",
    ]
    for nome in casi:
        assert parse_prediction(base + nome, 1) is None, nome


def test_coda_storta_non_viene_assorbita_nel_modello() -> None:
    """`-tile64` senza stride, o un suffisso dopo lo stride: il pezzo storto non deve finire
    dentro il nome del modello lasciando tile/stride a None."""
    troncato = K_838_JULY.replace("-tile64-stride16.tif", "-tile64.tif")
    assert parse_prediction(troncato, SIZE_838_JULY) is None

    con_suffisso = K_838_JULY.replace("-stride16.tif", "-stride16-ds8.tif")
    assert parse_prediction(con_suffisso, SIZE_838_JULY) is None


def test_incoerenza_col_path_da_none() -> None:
    """Il nome dice un sample o un segmento diverso da quello del path: si scarta."""
    sample_sbagliato = K_838_JULY.replace(
        "ink-detection/PHerc0172-", "ink-detection/PHerc0332-"
    )
    assert parse_prediction(sample_sbagliato, SIZE_838_JULY) is None

    segmento_sbagliato = K_838_JULY.replace(
        "ink-detection/PHerc0172-20251107110950-",
        "ink-detection/PHerc0172-20251107110999-",
    )
    assert parse_prediction(segmento_sbagliato, SIZE_838_JULY) is None


# ---------------------------------------------------------------------------
# comparable e pairs
# ---------------------------------------------------------------------------


def test_comparable_solo_su_stesso_raster() -> None:
    a = _p(K_838_JULY, SIZE_838_JULY)
    b = _p(K_839_NOV, SIZE_839_NOV)
    assert comparable(a, b)

    c = _p(K_PARIS4_L1, SIZE_PARIS4_L1)  # 1.129um, L1  -> passo 2.258
    d = _p(K_PARIS4_24, SIZE_PARIS4_24)  # 2.4um, livello ignoto
    assert not comparable(c, d)


def test_pairs_sul_2x2_da_due_volume_e_due_model() -> None:
    preds = [
        _p(K_838_JULY, SIZE_838_JULY),
        _p(K_838_NOV, SIZE_838_NOV),
        _p(K_839_JULY, SIZE_839_JULY),
        _p(K_839_NOV, SIZE_839_NOV),
    ]
    got = pairs(preds)
    kinds = sorted(p.kind for p in got)
    assert kinds == ["model", "model", "volume", "volume"]

    # kind="volume": stesso modello, volume diverso.
    for p in got:
        if p.kind == "volume":
            assert p.a.model == p.b.model and p.a.volume != p.b.volume
        else:
            assert p.a.volume == p.b.volume and p.a.model != p.b.model

    st = pair_stats(preds)
    assert st.n_candidate_pairs == 6
    assert st.n_excluded_both == 2  # le due diagonali del 2x2
    assert st.n_excluded_raster == 0
    assert st.n_excluded_duplicate == 0
    assert st.by_kind == {"volume": 2, "model": 2}
    assert st.segments_with_volume_pair == [
        ("PHerc0172", "20251107110950-w064_20251107110950052_flatboi")
    ]


def test_pairs_stesso_volume_modelli_diversi_e_una_sola_coppia() -> None:
    preds = [_p(K_500P2_CANON, SIZE_500P2_CANON), _p(K_500P2_MRG, SIZE_500P2_MRG)]
    got = pairs(preds)
    assert len(got) == 1
    assert got[0].kind == "model"
    assert got[0].a.volume == got[0].b.volume


def test_pairs_esclude_both_e_raster_diverso() -> None:
    preds = [_p(K_PARIS4_L1, SIZE_PARIS4_L1), _p(K_PARIS4_24, SIZE_PARIS4_24)]
    st = pair_stats(preds)
    assert st.pairs == []
    # Volume e modello sono entrambi diversi: il motivo di esclusione è "both", e la coppia
    # non viene contata due volte anche se il raster differisce.
    assert st.n_excluded_both == 1
    assert st.n_excluded_raster == 0


def test_pairs_non_accoppia_segmenti_diversi() -> None:
    preds = [_p(K_838_JULY, SIZE_838_JULY), _p(K_500P2_CANON, SIZE_500P2_CANON)]
    assert pairs(preds) == []


def test_pairs_ignora_un_segmento_con_una_sola_predizione() -> None:
    st = pair_stats([_p(K_L1, SIZE_L1)])
    assert st.pairs == []
    assert st.n_candidate_pairs == 0
    assert st.n_segments_single == 1


def test_pair_e_deterministico_nell_ordine() -> None:
    preds = [
        _p(K_839_NOV, SIZE_839_NOV),
        _p(K_838_JULY, SIZE_838_JULY),
        _p(K_839_JULY, SIZE_839_JULY),
        _p(K_838_NOV, SIZE_838_NOV),
    ]
    a = pairs(preds)
    b = pairs(list(reversed(preds)))
    assert [(p.a.key, p.b.key, p.kind) for p in a] == [
        (p.a.key, p.b.key, p.kind) for p in b
    ]
    assert all(isinstance(p, Pair) for p in a)


def test_census_stats_non_tocca_la_rete() -> None:
    """`census_stats()` è un referto, non un censimento: senza una corsa alle spalle è vuoto."""
    from inkfloor import census as mod

    got = mod.census_stats()
    assert isinstance(got, dict)
    if got:  # se qualcuno ha già girato un census in questo processo, i conti tornano
        assert got["keys_seen"] == got["kept"] + got["skipped"]


if __name__ == "__main__":  # esecuzione senza pytest
    import traceback

    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                fails += 1
                traceback.print_exc()
    raise SystemExit(1 if fails else 0)
