"""Gleichstand (alle Kosten gleich, zwei optimale Permutationen), dev, Eindeutigkeit und die weichen Ränge (kleines Beispiel von Hand, Eigenschaften, Gleichstand, Auswahlverzerrung)."""

import numpy as np
import pytest

import sgz_constants as C
import sgz_evaluation as E
from sgz_eindeutigkeit import ist_eindeutig
from sgz_newton import loese_gibbs
from sgz_scenario import make_gleichstand, raenge_kosten
from sgz_simplex import loese_transport
from sgz_sinkhorn import loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale


def sinkhorn_plan(a, b, Cm, eps, tol=1e-12, max_iter=5000):
    erg = loese_sinkhorn(a, b, Cm, eps, tol=tol, max_iter=max_iter)
    assert erg.konvergiert
    return plan_aus_potentialen(a, b, Cm, eps, erg.f, erg.g)


@pytest.mark.parametrize("eps", [1.0, 0.1, 0.001])
@pytest.mark.parametrize("massen", ["gleich", "ungleich"])
def test_alle_kosten_gleich_der_plan_ist_das_produktmass_fuer_jedes_eps(eps, massen):
    a, b = (np.array([0.5, 0.5]),) * 2 if massen == "gleich" else (np.array([0.75, 0.25]), np.array([0.25, 0.75]))
    P = sinkhorn_plan(a, b, np.full((2, 2), 0.5), eps, max_iter=5)
    assert P == pytest.approx(np.outer(a, b), abs=1e-14)
    lo = loese_transport(a, b, np.full((2, 2), 0.5), "steilste")
    assert not ist_eindeutig(np.full((2, 2), 0.5), lo.plan, lo.u, lo.v)           # jede Kopplung ist optimal
    if massen == "gleich":                                                          # Ecken sind die zwei Permutationen: das Produktmaß hat zu jeder den L1-Abstand 1
        assert [np.abs(P - e).sum() for e in (np.eye(2) / 2, np.eye(2)[::-1] / 2)] == pytest.approx([1.0, 1.0], abs=1e-12)


def test_produktmass_hat_l1_abstand_1_zu_jeder_der_zwei_permutationen():
    assert np.abs(np.outer([0.5, 0.5], [0.5, 0.5]) - np.array([[0.5, 0.0], [0.0, 0.5]])).sum() == pytest.approx(1.0)
    assert E.dev_von(np.full((2, 2), 0.25), 2) == pytest.approx(0.5)


def test_drei_mal_drei_zwei_optimale_permutationen_der_plan_teilt_den_block_haelftig():
    """a = b = 1/3, C = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]: zwei optimale Permutationen, OPT = 0; bei ε = 0,05 hat der Plan im 2 × 2-Block überall 1/6 (dev = 1/2), L1-Abstand 2/3 zu jeder der
    beiden Ecken, Kosten 2e-9: die Kostenlücke ist klein, der Plan weit weg."""
    a = np.full(3, 1 / 3)
    Cm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    lo = loese_transport(a, a, Cm, "steilste")
    assert lo.kosten == pytest.approx(0.0, abs=1e-12) and not ist_eindeutig(Cm, lo.plan, lo.u, lo.v)
    _, f, g = verlauf_potentiale(a, a, Cm, 0.05, [400])[0]
    P = plan_aus_potentialen(a, a, Cm, 0.05, f, g)
    assert E.dev_von(P, 3) == pytest.approx(0.5, abs=1e-6) and P[:2, :2] == pytest.approx(np.full((2, 2), 1 / 6), abs=1e-8)
    ecken = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 3, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) / 3]
    assert [np.abs(P - e).sum() for e in ecken] == pytest.approx([2 / 3, 2 / 3], abs=1e-6)          # für jede der zwei Ecken (HiGHS und der Transportsimplex wählen eine davon)
    assert np.abs(P - lo.plan).sum() == pytest.approx(2 / 3, abs=1e-6)
    assert float((P * Cm).sum()) < 1e-7


def test_dev_null_bei_permutation_halb_bei_eintrag_halb_halb():
    assert E.dev_von(np.eye(4) / 4, 4) == 0.0
    P = np.array([[0.25, 0.25], [0.25, 0.25]])
    assert E.dev_von(P, 2) == pytest.approx(0.5)


def test_aggregiere_gleichstand_zaehlt_an_handgesetzten_listen():
    r = dict(kosten="x", eps=0.02, n=10, anzahl=5, eindeutig=[True, True, False, False, False], dev=[0.0, 0.01, 0.48, 0.02, 0.5],
             l1=[0.0, 0.0, 1.0, 0.4, 1.2], konvergiert=[True, False, True, False, False], iterationen=[1, 2, 3, 4, 5])
    z = E.aggregiere_gleichstand(r)
    assert (z["mehrere"], z["mehrere_nicht_ganzzahlig"], z["eindeutig"], z["eindeutig_ganzzahlig"], z["konvergiert"]) == (3, 2, 2, 2, 2)
    assert z["konvergiert_mehrere"] == 1 and z["konvergiert_eindeutig"] == 1
    assert z["dev_median_mehrere"] == pytest.approx(0.48) and z["l1_median_mehrere"] == pytest.approx(1.0)


def test_eindeutigkeit_der_stetigen_instanzen_und_mehrere_optima_bei_ganzzahligen_kosten():
    for i in range(10):
        inst = make_gleichstand(C.KOSTEN3[0], 10, C.SERIEN_SEED, i)
        lo = loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        assert ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v)
    mehrere = 0
    for i in range(10):
        inst = make_gleichstand(C.KOSTEN3[1], 10, C.SERIEN_SEED, i)
        lo = loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        mehrere += not ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v)
    assert mehrere >= 8


def test_eindeutiges_optimum_ganzzahlige_kosten_der_plan_ist_ganzzahlig_bei_kleinem_eps():
    """Bei ganzzahligen Kosten mit eindeutigem Optimum ist der Abstand des weichen Plans zur Ecke von der Größenordnung exp(−1/ε): bei ε = 0,02 liegt dev unter 1e-3 (der Plan des Newton-Verfahrens ist bei so kleinen Einträgen nur auf etwa 1e-5 genau)."""
    gefunden = 0
    for i in range(30):
        inst = make_gleichstand(C.KOSTEN3[2], 10, C.SERIEN_SEED, i)
        lo = loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        if not ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v):
            continue
        P = loese_gibbs(inst.a, inst.b, inst.kosten, 0.02)           # die Iteration selbst bräuchte Zehntausende Durchläufe (Experiment 3); der genaue Plan kommt aus dem Newton-Verfahren
        if not P.ok:                                                    # das Newton-Verfahren scheitert bei einzelnen Instanzen (Auswahlverzerrung, siehe Absatz zu den Rängen)
            continue
        assert E.dev_von(P.plan, 10) < 1e-3, i
        gefunden += 1
    assert gefunden >= 4


# ---- weiche Ränge ---------------------------------------------------------------------------------------------------------------

def weiche_raenge(x, eps_rel):
    a, K, opt, rang = raenge_kosten(x)
    P = loese_gibbs(a, a, K, eps_rel * opt)
    assert P.ok
    n = len(x)
    return P.plan, n * (P.plan * np.arange(1, n + 1)[None, :]).sum(axis=1), rang


X5 = np.array([0.40, 0.45, 0.50, 0.55, 0.60])


def test_weiche_raenge_n_5_eps_1_von_hand_gerechnete_werte():
    """n = 5, x = (0,40; 0,45; 0,50; 0,55; 0,60), ε/OPT = 1: weiche Ränge (1,806; 2,319; 3; 3,681; 4,194); Rang 1 zu hoch, Rang 5 zu niedrig, gleich stark; die Summe ist n(n+1)/2 = 15."""
    P, r, rang = weiche_raenge(X5, 1.0)
    assert rang.tolist() == [1, 2, 3, 4, 5]
    assert r == pytest.approx([1.8062, 2.3186, 3.0, 3.6814, 4.1938], abs=5e-5)
    assert r[0] > 1 and r[-1] < 5 and (r[0] - 1) == pytest.approx(5 - r[-1], abs=1e-9)
    assert r.sum() == pytest.approx(15.0, abs=1e-9)
    assert P.sum(axis=1) == pytest.approx(np.full(5, 0.2), abs=1e-12) and P.sum(axis=0) == pytest.approx(np.full(5, 0.2), abs=1e-12)


def test_weiche_raenge_die_verzerrung_faellt_mit_eps_und_waechst_mit_n_bei_festem_eps():
    fehler = [weiche_raenge(X5, e)[1][0] - 1 for e in (1.0, 0.3, 0.1, 0.03)]
    assert fehler[0] > fehler[1] > fehler[2] > fehler[3] and fehler[3] < 1e-3
    assert fehler == pytest.approx([0.8062, 0.3357, 0.0942, 0.0006], abs=1e-3)
    vorher = 0.0
    for n in (5, 10, 20, 40):                                                       # künstliche Familie gleichabständiger Werte in [0,4; 0,6]
        x = 0.4 + 0.2 * (np.arange(n) + 0.5) / n
        f = weiche_raenge(x, 1.0)[1][0] - 1
        assert f > vorher
        vorher = f


def test_gleiche_werte_bekommen_identische_zeilen_und_den_mittelrang():
    x = np.array([0.40, 0.40, 0.50, 0.55, 0.60])
    abw = []
    for e in (1.0, 0.3, 0.1):
        P, r, rang = weiche_raenge(x, e)
        assert np.abs(P[0] - P[1]).max() < 1e-12 and abs(r[0] - r[1]) < 1e-12
        abw.append(abs(r[0] - 1.5))
    assert abw[0] > abw[1] > abw[2] and abw[2] < 0.005 and abw[0] == pytest.approx(0.410, abs=1e-3)


def test_tabelle_raenge_an_handgesetzter_reihe():
    d = dict(konv=[True, True, False], dev=[0.0, 0.3, None], betrag=[0.5, 1.5, None], rang1=[1.0, 3.0, None], rangn=[-1.0, -3.0, None], richtig=[1.0, 0.5, None], summe=[15.0, 15.0, None])
    r = dict(n=5, anzahl=3, eps={f"{e:g}": d for e in C.EPS_RAENGE})
    zeilen = E.tabelle_raenge(r)
    z = zeilen[0]
    assert (z["konvergiert"], z["fast_ganz"]) == (2, 1) and z["anteil_fast_ganz"] == pytest.approx(0.5) and z["betrag"] == pytest.approx(1.0) and z["rang1"] == pytest.approx(2.0)
    assert E.zeile_raenge(z)[:3] == [E.eps_text(3.0), "2 von 3", "1 von 2"]
    leer = dict(d, konv=[False] * 3)
    assert E.zeile_raenge(E.tabelle_raenge(dict(r, eps={f"{e:g}": leer for e in C.EPS_RAENGE}))[0])[1:3] == ["0 von 3", "–"]
