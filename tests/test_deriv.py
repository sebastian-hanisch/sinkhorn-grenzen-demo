"""Ableitung des Plans nach den Kosten: die implizite Formel gegen finite Differenzen (3 × 3 von Hand, Zufallsinstanzen), das Durchrechnen gegen die finite Differenz der Iteration selbst,
die Eigenschaften (Randsummen von dP gleich null, Ableitung des Produktmaßes bei Gleichstand), die Ungleichung „Ableitungsfehler über Zeilenfehler“ am unfertigen Plan."""

import numpy as np
import pytest

import sgz_constants as C
from sgz_deriv import ableitung_fd, ableitung_implizit, durchrechnen, relativer_fehler
from sgz_newton import loese_gibbs
from sgz_scenario import make_instanz
from sgz_sinkhorn import plan_aus_potentialen, verlauf_potentiale, zeilenfehler

a5 = np.array([0.5, 0.3, 0.2])
b5 = np.array([0.4, 0.4, 0.2])
C5 = np.array([[0.1, 0.5, 1.0], [0.6, 0.2, 0.7], [1.0, 0.8, 0.3]])
D5 = np.array([[1.0, -1.0, 0.5], [0.2, 0.3, -0.7], [-0.4, 0.9, 0.1]])
EPS5 = 0.2


def plan_k(Cm, k, a=a5, b=b5, eps=EPS5):
    _, f, g = verlauf_potentiale(a, b, Cm, eps, [k])[0]
    return plan_aus_potentialen(a, b, Cm, eps, f, g)


def test_drei_mal_drei_implizite_formel_gleich_zentrale_finite_differenz_mit_h_1e_minus_6():
    ex = loese_gibbs(a5, b5, C5, EPS5)
    assert ex.ok and ex.fehler < 1e-12
    h = 1e-6
    fd = (loese_gibbs(a5, b5, C5 + h * D5, EPS5).plan - loese_gibbs(a5, b5, C5 - h * D5, EPS5).plan) / (2 * h)
    dP = ableitung_implizit(ex.plan, C5, EPS5, D5)
    assert relativer_fehler(dP, fd) < 1e-6
    assert np.abs(dP.sum(axis=1)).max() < 1e-12 and np.abs(dP.sum(axis=0)).max() < 1e-12      # dP behält die Randverteilungen
    assert relativer_fehler(dP, ableitung_fd(a5, b5, C5, EPS5, D5)) < 1e-6                      # Richardson-Fassung


def test_durchrechnen_ist_die_ableitung_des_plans_nach_k_durchlaeufen_gegen_finite_differenz_der_iteration():
    """Das Durchrechnen (Vorwärtsmodus) leitet die k-Durchlauf-Iteration selbst ab: Gegenprobe ist die zentrale finite Differenz der Iteration bei festem k."""
    h = 1e-6
    for k in (1, 3, 12, 40):
        (kk, P, dP), = durchrechnen(a5, b5, C5, EPS5, D5, [k])
        assert kk == k and P == pytest.approx(plan_k(C5, k), abs=1e-14)
        fd = (plan_k(C5 + h * D5, k) - plan_k(C5 - h * D5, k)) / (2 * h)
        assert relativer_fehler(dP, fd) < 1e-6, k


def test_durchrechnen_und_implizite_formel_geben_im_grenzwert_dieselbe_ableitung():
    ex = loese_gibbs(a5, b5, C5, EPS5)
    d_ref = ableitung_implizit(ex.plan, C5, EPS5, D5)
    (_, P, dP), = durchrechnen(a5, b5, C5, EPS5, D5, [400])
    assert relativer_fehler(dP, d_ref) < 1e-10 and relativer_fehler(ableitung_implizit(P, C5, EPS5, D5), d_ref) < 1e-10


def test_ableitungsfehler_liegt_am_unfertigen_plan_ueber_dem_zeilenfehler_und_ueber_dem_halben_planfehler():
    ex = loese_gibbs(a5, b5, C5, EPS5)
    d_ref = ableitung_implizit(ex.plan, C5, EPS5, D5)
    for k in (3, 6, 12):
        P = plan_k(C5, k)
        zeile = zeilenfehler(P, a5)
        ableit = relativer_fehler(ableitung_implizit(P, C5, EPS5, D5), d_ref)
        plan_f = relativer_fehler(P, ex.plan)
        assert ableit > zeile and ableit > 0.5 * plan_f, (k, ableit, zeile, plan_f)


def test_gleichstand_zwei_mal_zwei_ableitung_des_produktmasses_ist_minus_1_durch_4_eps():
    """Pfad C(t) = [[t, 1−t], [1−t, t]], a = b = (1/2, 1/2): bei t = 1/2 ist der Plan das Produktmaß, und dP11/dt = −1/(4ε) exakt (Richtung D = [[1, −1], [−1, 1]])."""
    a = np.array([0.5, 0.5])
    Cm = np.array([[0.5, 0.5], [0.5, 0.5]])
    D = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for eps in (0.05, 0.2, 1.0):
        P = np.full((2, 2), 0.25)
        dP = ableitung_implizit(P, Cm, eps, D)
        assert dP[0, 0] == pytest.approx(-1 / (4 * eps), abs=1e-12) and dP[0, 1] == pytest.approx(1 / (4 * eps), abs=1e-12)
        ex = loese_gibbs(a, a, Cm, eps)
        assert ex.plan == pytest.approx(P, abs=1e-12)


def test_zufallsinstanz_formel_gegen_finite_differenz_und_durchrechnen_gegen_iteration():
    inst = make_instanz(8, 8, C.SERIEN_SEED, C.ZWECK_ABL, 1, C.S_ABL)
    eps = 0.3 * 0.05
    g = np.random.default_rng(1)
    D = g.normal(size=inst.kosten.shape)
    D /= np.abs(D).max()
    ex = loese_gibbs(inst.a, inst.b, inst.kosten, eps)
    d_ref = ableitung_implizit(ex.plan, inst.kosten, eps, D)
    fd = ableitung_fd(inst.a, inst.b, inst.kosten, eps, D)
    assert relativer_fehler(d_ref, fd) < 1e-6
    (_, P, dP), = durchrechnen(inst.a, inst.b, inst.kosten, eps, D, [30])
    h = 1e-6 * eps

    def plan30(Cm):
        _, f, gg = verlauf_potentiale(inst.a, inst.b, Cm, eps, [30])[0]
        return plan_aus_potentialen(inst.a, inst.b, Cm, eps, f, gg)

    assert relativer_fehler(dP, (plan30(inst.kosten + h * D) - plan30(inst.kosten - h * D)) / (2 * h)) < 1e-5


def test_durchrechnen_verlangt_k_ab_1():
    with pytest.raises(ValueError):
        durchrechnen(a5, b5, C5, EPS5, D5, [0])
