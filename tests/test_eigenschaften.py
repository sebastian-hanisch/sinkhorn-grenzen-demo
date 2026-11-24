"""Eigenschaften (feste Zufalls-Seeds, 10 bis 20 Instanzen): Skalierung von Kosten und ε, Umnummerieren, Monotonie des Zeilenfehlers, Massen unter 1e-7 (der eigene Transportsimplex hat das
Problem der Bibliotheks-LPs nicht), Randfälle der Reihen."""

import numpy as np
import pytest

import sgz_constants as C
import sgz_evaluation as E
from sgz_naiv import naive_iteration
from sgz_scenario import make_instanz
from sgz_simplex import loese_transport, pruefe_loesung
from sgz_sinkhorn import loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale, zeilenfehler


def _inst(i, n=8):
    return make_instanz(n, n, 5, C.ZWECK_ANSICHT, i)


def _plan(inst, eps, k=40):
    _, f, g = verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, [k])[0]
    return plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g)


@pytest.mark.parametrize("faktor", [0.1, 7.0])
def test_kosten_und_eps_mit_demselben_faktor_geben_denselben_plan(faktor):
    for i in range(10):
        inst = _inst(i)
        eps = 0.2 * 0.05
        P = _plan(inst, eps)
        skaliert = make_instanz(8, 8, 5, C.ZWECK_ANSICHT, i)
        _, f, g = verlauf_potentiale(skaliert.a, skaliert.b, faktor * skaliert.kosten, faktor * eps, [40])[0]
        assert plan_aus_potentialen(skaliert.a, skaliert.b, faktor * skaliert.kosten, faktor * eps, f, g) == pytest.approx(P, rel=1e-9, abs=1e-14), i
        naiv = naive_iteration(inst.a, inst.b, inst.kosten, eps, np.float64, 0.0, 40, "float64").plan
        naiv_skaliert = naive_iteration(inst.a, inst.b, faktor * inst.kosten, faktor * eps, np.float64, 0.0, 40, "float64").plan
        assert naiv_skaliert == pytest.approx(naiv, rel=1e-9, abs=1e-14), i


def test_umnummerieren_permutiert_den_plan():
    for i in range(10):
        inst = _inst(i)
        g = np.random.default_rng(i)
        p, q = g.permutation(8), g.permutation(8)
        eps = 0.01
        P = _plan(inst, eps)
        _, f, gg = verlauf_potentiale(inst.a[p], inst.b[q], inst.kosten[np.ix_(p, q)], eps, [40])[0]
        assert plan_aus_potentialen(inst.a[p], inst.b[q], inst.kosten[np.ix_(p, q)], eps, f, gg) == pytest.approx(P[np.ix_(p, q)], rel=1e-9, abs=1e-14), i


def test_zeilenfehler_faellt_mit_k_an_den_zweierpotenzen():
    for i in range(10):
        inst = _inst(i)
        eps = 0.1 * E.referenz(inst).opt
        fehler = []
        for k, f, g in verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, [2 ** j for j in range(0, 9)]):
            fehler.append(zeilenfehler(plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g), inst.a))
        assert all(b <= a * (1 + 1e-9) for a, b in zip(fehler, fehler[1:])), (i, fehler)


def test_log_domaene_und_naive_float64_iteration_brauchen_dieselbe_zahl_von_iterationen_bis_auf_zwei():
    for i in range(10):
        inst = _inst(i, 10)
        eps = 0.3 * E.referenz(inst).opt
        a = loese_sinkhorn(inst.a, inst.b, inst.kosten, eps, tol=1e-8, max_iter=C.MAX_ITER1)
        b = naive_iteration(inst.a, inst.b, inst.kosten, eps, np.float64, 1e-8, C.MAX_ITER1, "float64")
        assert a.konvergiert and b.status == "ok" and abs(a.iterationen - b.iterationen) <= 2, i


def test_transportsimplex_mit_massen_unter_1e_minus_7_ist_zulaessig_und_optimal():
    """HiGHS meldet bei Massen unter etwa 1e-7 „infeasible“; der eigene Transportsimplex rechnet sie, der unabhängige Prüfer bestätigt."""
    a = np.array([1e-9, 0.5, 0.5 - 1e-9])
    b = np.array([0.25, 0.25, 0.5])
    Cm = np.array([[0.1, 0.9, 0.5], [0.4, 0.2, 0.7], [0.8, 0.3, 0.0]])
    lo = loese_transport(a, b, Cm, "steilste")
    assert not pruefe_loesung(a, b, Cm, lo) and lo.plan.sum(axis=1) == pytest.approx(a, abs=1e-12)


def test_handbeispiel_gleichstand_kosten_2e_minus_9():
    a = np.full(3, 1 / 3)
    Cm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    _, f, g = verlauf_potentiale(a, a, Cm, 0.05, [400])[0]
    kosten = float((plan_aus_potentialen(a, a, Cm, 0.05, f, g) * Cm).sum())
    assert 2.0e-9 < kosten < 2.2e-9


def test_reihen_an_der_untergrenze_k_gleich_1_und_n_gleich_6():
    z = E.ansicht2(E.Settings(n=C.N_MIN, eps2=C.EPS2_STUFEN[-1], k2=1))
    assert z["zeilenfehler"][0] > 0 and z["gerundet"][0] > z["luecke_ende"] and len(z["plaene"]) == len(C.K2_STUFEN)
    z = E.ansicht2(E.Settings(n=C.N_MAX, eps2=C.EPS2_STUFEN[0], k2=C.K2_STUFEN[-1], seed=0))
    assert z["zeilenfehler"][-1] < 1e-10
