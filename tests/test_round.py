"""Rundung auf zulässige Randverteilungen: Handrechnungen in Brüchen (zwei Beispiele), unabhängige Fassung in Brüchen als Gegenprobe, Eigenschaften (zulässig, nichtnegativ, Schranke,
idempotent, ein zulässiger Plan bleibt unverändert) und Randfälle."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import sgz_constants as C
from sgz_round import runden, schranke
from sgz_scenario import make_instanz
from sgz_sinkhorn import plan_aus_potentialen, verlauf_potentiale


def runden_in_bruechen(P, a, b):
    """Algorithmus 2 (Altschuler u. a.) in Brüchen, unabhängig vom numpy-Code der Demo: Zeilen herunterskalieren, Spalten herunterskalieren, Rest als Rang-1-Matrix."""
    n, m = len(a), len(b)
    r = [sum(P[i]) for i in range(n)]
    x = [min(a[i] / r[i], Fr(1)) for i in range(n)]
    F = [[x[i] * P[i][j] for j in range(m)] for i in range(n)]
    c = [sum(F[i][j] for i in range(n)) for j in range(m)]
    y = [min(b[j] / c[j], Fr(1)) for j in range(m)]
    F = [[F[i][j] * y[j] for j in range(m)] for i in range(n)]
    er = [a[i] - sum(F[i]) for i in range(n)]
    ec = [b[j] - sum(F[i][j] for i in range(n)) for j in range(m)]
    s = sum(er)
    if s > 0:
        F = [[F[i][j] + er[i] * ec[j] / s for j in range(m)] for i in range(n)]
    return F


def als_matrix(F):
    return np.array([[float(x) for x in zeile] for zeile in F])


def test_handbeispiel_zu_wenig_masse_gerundet_7_20_3_20():
    """P = [[3/10, 1/10], [1/10, 3/10]] (Gesamtmasse 4/5) auf a = b = (1/2, 1/2): R = [[7/20, 3/20], [3/20, 7/20]]; Kosten 1/2 vor, 13/20 nach der Rundung; Schranke 13/20 ≤ 1/2 + 2 (1/5 + 1/5)."""
    P = np.array([[0.3, 0.1], [0.1, 0.3]])
    a = b = np.array([0.5, 0.5])
    Cm = np.array([[0.5, 1.0], [1.0, 0.5]])
    R = runden(P, a, b)
    assert R == pytest.approx(np.array([[7 / 20, 3 / 20], [3 / 20, 7 / 20]]), abs=1e-15)
    assert (P * Cm).sum() == pytest.approx(0.5) and (R * Cm).sum() == pytest.approx(13 / 20)
    assert schranke(P, a, b, Cm) == pytest.approx(0.5 + 2 * (0.2 + 0.2)) and (R * Cm).sum() <= schranke(P, a, b, Cm)
    Pf = [[Fr(3, 10), Fr(1, 10)], [Fr(1, 10), Fr(3, 10)]]
    assert runden_in_bruechen(Pf, [Fr(1, 2)] * 2, [Fr(1, 2)] * 2) == [[Fr(7, 20), Fr(3, 20)], [Fr(3, 20), Fr(7, 20)]]


def test_handbeispiel_plan_nach_einem_durchlauf_gerundet_15_64_33_64(runden_beispiel):
    """P = [[9/40, 3/8], [1/40, 3/8]], a = (3/4, 1/4), b = (1/4, 3/4): Zeilensummen (3/5, 2/5); x = (1, 5/8), F = [[9/40, 3/8], [1/64, 15/64]], y = (1, 1); Fehlmengen (3/20, 0) und (3/320, 45/320);
    R = [[15/64, 33/64], [1/64, 15/64]]; Kosten 17/32 (roh 2/5), gerundete Lücke +1/16 = +6,25 % (roh −20 %)."""
    a, b, Cm, P = runden_beispiel
    R = runden(P, a, b)
    assert R == pytest.approx(np.array([[15 / 64, 33 / 64], [1 / 64, 15 / 64]]), abs=1e-15)
    assert R.sum(axis=1) == pytest.approx(a, abs=1e-15) and R.sum(axis=0) == pytest.approx(b, abs=1e-15)
    k_roh, k_rund = float((P * Cm).sum()), float((R * Cm).sum())
    assert k_roh == pytest.approx(2 / 5) and k_rund == pytest.approx(17 / 32)
    assert (k_roh - 0.5) / 0.5 == pytest.approx(-0.2) and (k_rund - 0.5) / 0.5 == pytest.approx(1 / 16)
    assert k_rund <= schranke(P, a, b, Cm) == pytest.approx(2 / 5 + 2 * 3 / 10)
    Pf = [[Fr(9, 40), Fr(3, 8)], [Fr(1, 40), Fr(3, 8)]]
    assert runden_in_bruechen(Pf, [Fr(3, 4), Fr(1, 4)], [Fr(1, 4), Fr(3, 4)]) == [[Fr(15, 64), Fr(33, 64)], [Fr(1, 64), Fr(15, 64)]]


def test_der_plan_aus_dem_handbeispiel_ist_der_plan_der_log_domaene_nach_einem_durchlauf(runden_beispiel):
    """Der Plan der Handrechnung ist der Plan der Iteration nach einem Durchlauf (Start u = v = 1, ε = 1/ln 3)."""
    a, b, Cm, P = runden_beispiel
    eps = 1 / np.log(3.0)
    _, f, g = verlauf_potentiale(a, b, Cm, eps, [1])[0]
    assert plan_aus_potentialen(a, b, Cm, eps, f, g) == pytest.approx(P, abs=1e-14)


def test_float_fassung_gleich_der_fassung_in_bruechen_auf_zufallsplaenen():
    g = np.random.default_rng(5)
    for _ in range(40):
        n, m = int(g.integers(2, 5)), int(g.integers(2, 5))
        P = [[Fr(int(g.integers(1, 20)), int(g.integers(20, 60))) for _ in range(m)] for _ in range(n)]
        a = [Fr(int(g.integers(1, 9))) for _ in range(n)]
        b = [Fr(int(g.integers(1, 9))) for _ in range(m)]
        s = sum(a) / sum(b)
        b = [x * s for x in b]
        gesamt = sum(sum(z) for z in P)
        P = [[x / gesamt * Fr(3, 2) for x in z] for z in P]            # Gesamtmasse 3/2 (über 1): beide Skalierungen kommen vor
        a1, b1 = [x / sum(a) for x in a], [x / sum(b) for x in b]
        exakt = runden_in_bruechen(P, a1, b1)
        assert runden(als_matrix(P), np.array([float(x) for x in a1]), np.array([float(x) for x in b1])) == pytest.approx(als_matrix(exakt), abs=1e-13)


def _verlaeufe():
    for i in range(6):
        inst = make_instanz(14, 14, C.SERIEN_SEED, C.ZWECK_SCHRANKE, i)
        for eps in (0.05, 0.01):
            for k, f, g in verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, (1, 3, 10, 50)):
                yield inst, plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g)


def test_gerundeter_plan_ist_zulaessig_nichtnegativ_und_erfuellt_die_schranke():
    anzahl = 0
    for inst, P in _verlaeufe():
        R = runden(P, inst.a, inst.b)
        assert (R >= 0).all() and R.sum(axis=1) == pytest.approx(inst.a, abs=1e-14) and R.sum(axis=0) == pytest.approx(inst.b, abs=1e-14)
        assert float((R * inst.kosten).sum()) <= schranke(P, inst.a, inst.b, inst.kosten) + 1e-12
        anzahl += 1
    assert anzahl == 48


def test_rundung_ist_idempotent_und_laesst_einen_zulaessigen_plan_unveraendert():
    for inst, P in list(_verlaeufe())[:12]:
        R = runden(P, inst.a, inst.b)
        assert runden(R, inst.a, inst.b) == pytest.approx(R, abs=1e-14)
    inst = make_instanz(10, 10, C.SERIEN_SEED, C.ZWECK_SCHRANKE, 2)
    zulaessig = np.outer(inst.a, inst.b)
    assert runden(zulaessig, inst.a, inst.b) == pytest.approx(zulaessig, abs=1e-15)


def test_randfaelle_nullzeile_nullspalte_und_ein_lager():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    # Zeile 2 ohne Masse: nicht skaliert, die Fehlmenge füllt sie auf
    R = runden(np.array([[0.5, 0.2], [0.0, 0.0]]), a, b)
    assert R.sum(axis=1) == pytest.approx(a, abs=1e-15) and R.sum(axis=0) == pytest.approx(b, abs=1e-15) and (R >= 0).all()
    R = runden(np.array([[1.0]]), np.array([1.0]), np.array([1.0]))
    assert R == pytest.approx(np.array([[1.0]]))
    # zu viel Masse: nur herunterskalieren, nichts addieren
    R = runden(np.full((2, 2), 0.5), a, b)
    assert R == pytest.approx(np.full((2, 2), 0.25), abs=1e-15)
