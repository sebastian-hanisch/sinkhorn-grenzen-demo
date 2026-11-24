"""Szenario: Formen, Summen, Normierung, Determinismus (auch in einem zweiten Prozess mit anderem PYTHONHASHSEED), Instanzen der Experimente, Kopier-Tests der übernommenen Module."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sgz_constants as C
import sgz_scenario as SC

ROOT = Path(__file__).resolve().parent.parent


def test_make_instanz_formen_summen_und_normierung():
    inst = SC.make_instanz(7, 9, 3, C.ZWECK_ANSICHT, 0)
    assert inst.a.shape == (7,) and inst.b.shape == (9,) and inst.kosten.shape == (7, 9)
    assert inst.a.sum() == pytest.approx(1.0) and inst.b.sum() == pytest.approx(1.0) and (inst.a > 0).all() and (inst.b > 0).all()
    assert inst.kosten.max() == pytest.approx(1.0) and inst.kosten.min() >= 0 and inst.lager.shape == (7, 2)
    with pytest.raises(ValueError):
        SC.make_instanz(0, 3, 1, 1, 0)


def test_kosten_sind_quadrierter_abstand_geteilt_durch_den_groessten_wert():
    inst = SC.make_instanz(5, 6, 9, C.ZWECK_ANSICHT, 2)
    q = ((inst.lager[:, None, :] - inst.kunden[None, :, :]) ** 2).sum(-1)
    assert inst.kosten == pytest.approx(q / q.max(), rel=1e-12)


def test_massen_dirichlet_und_mischung():
    g = SC.instanz_rng(1, 1, 1)
    m = SC.massen(g, 10, 1.0)
    assert m.sum() == pytest.approx(1.0) and (m > 0).all()
    assert SC.massen(SC.instanz_rng(1, 1, 1), 5, 0.0) == pytest.approx(np.full(5, 0.2))
    g = SC.instanz_rng(1, 1, 2)
    gemischt = SC.massen(g, 10, 0.5)
    assert gemischt.min() >= 0.5 / 10 and gemischt.sum() == pytest.approx(1.0)       # Boden (1 − s)/n


def test_gleicher_seed_gleiche_instanz_und_verschiedene_zwecke_und_indizes_sind_verschieden():
    a = SC.make_instanz(6, 6, 5, C.ZWECK_UNTERLAUF, 3)
    b = SC.make_instanz(6, 6, 5, C.ZWECK_UNTERLAUF, 3)
    assert (a.kosten == b.kosten).all() and (a.a == b.a).all()
    assert not (SC.make_instanz(6, 6, 5, C.ZWECK_ABL, 3).kosten == a.kosten).all()
    assert not (SC.make_instanz(6, 6, 5, C.ZWECK_UNTERLAUF, 4).kosten == a.kosten).all()


def test_zweiter_prozess_mit_anderem_hashseed_rechnet_dieselbe_instanz_und_denselben_plan():
    code = ("import sys; sys.path.insert(0, %r); import sgz_scenario as S, sgz_evaluation as E, sgz_constants as C; "
            "i = S.make_instanz(8, 8, 7, C.ZWECK_ABL, 1); lo = E.referenz(i); print(repr(float(i.kosten.sum())), repr(lo.opt))" % str(ROOT))
    ausgaben = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, OMP_NUM_THREADS="1")
        ausgaben.append(subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, env=env, timeout=120, check=True).stdout.strip())
    assert ausgaben[0] == ausgaben[1] and ausgaben[0]


def test_gleichstand_instanzen_massen_eins_durch_n_ganzzahlige_kosten_und_alle_gleich():
    for art, R in ((C.KOSTEN3[1], 2), (C.KOSTEN3[2], 10)):
        gesehen = set()
        for i in range(20):
            inst = SC.make_gleichstand(art, 10, 3, i)
            assert inst.a == pytest.approx(np.full(10, 0.1)) and (inst.kosten == np.round(inst.kosten)).all()
            assert 0 <= inst.kosten.min() and inst.kosten.max() <= R
            gesehen |= set(np.unique(inst.kosten).astype(int))
        assert gesehen == set(range(R + 1))                         # jeder Wert 0 bis R kommt vor
    stetig = SC.make_gleichstand(C.KOSTEN3[0], 10, 3, 0)
    assert stetig.kosten.max() == pytest.approx(1.0) and stetig.lager.shape == (10, 2)
    gleich = SC.make_gleichstand(C.KOSTEN3[3], 10, 3, 0)
    assert (gleich.kosten == 0.5).all()
    with pytest.raises(ValueError):
        SC.make_gleichstand("unbekannt", 10, 3, 0)


def test_handbeispiele():
    h = SC.handbeispiel_weit()
    assert h.kosten.tolist() == [[0.0, 0.5], [1.0, 1.0]] and h.a.tolist() == [0.5, 0.5] and h.name == C.QUELLE_HAND
    a, b, Cm, P = SC.handbeispiel_runden()
    assert a.tolist() == [0.75, 0.25] and b.tolist() == [0.25, 0.75] and Cm.tolist() == [[0.0, 1.0], [1.0, 0.0]] and P.sum() == pytest.approx(1.0)


def test_werte_und_raenge_kosten_opt_ist_die_sortierte_zuordnung_und_gleich_dem_transportsimplex():
    from sgz_simplex import loese_transport

    for i in range(5):
        x = SC.make_werte(12, 7, i)
        assert ((x > 0) & (x < 1)).all()
        a, K, opt, rang = SC.raenge_kosten(x)
        assert sorted(rang.tolist()) == list(range(1, 13)) and a.sum() == pytest.approx(1.0)
        assert opt == pytest.approx(loese_transport(a, a, K, "steilste").kosten, rel=1e-9)


def test_rng_ist_eine_bytegleiche_kopie_der_vorlage_im_website_repo_falls_vorhanden():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / "sgz_rng.py").read_bytes() == vorlage.read_bytes()


@pytest.mark.parametrize("name", ["simplex", "eindeutigkeit", "sinkhorn", "newton"])
def test_uebernommene_module_sind_kopien_der_quelle_in_sinkhorn_demo_falls_vorhanden(name):
    """Kopie mit eigenem Präfix: der Quelltext ist bis auf `snk_` → `sgz_` gleich (Simplex und Eindeutigkeitstest stammen ursprünglich aus der Demo zum Zuordnen und Transportieren)."""
    quelle = ROOT.parent / "sinkhorn-demo" / f"snk_{name}.py"
    if not quelle.exists():
        pytest.skip("Quelle nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / f"sgz_{name}.py").read_text(encoding="utf-8") == quelle.read_text(encoding="utf-8").replace("snk_", "sgz_")


@pytest.mark.parametrize("name", ["simplex", "eindeutigkeit"])
def test_simplex_und_eindeutigkeitstest_sind_auch_kopien_aus_zuordnen_transportieren_demo_falls_vorhanden(name):
    quelle = ROOT.parent / "zuordnen-transportieren-demo" / f"ztr_{name}.py"
    if not quelle.exists():
        pytest.skip("Quelle nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / f"sgz_{name}.py").read_text(encoding="utf-8") == quelle.read_text(encoding="utf-8").replace("ztr_", "sgz_")


def test_splitmix64_bekannte_ausgaben_fuer_seed_0():
    from sgz_rng import SplitMix64

    g = SplitMix64(0)
    assert [g.next_u64() for _ in range(4)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]
