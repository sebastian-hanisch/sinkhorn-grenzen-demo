"""Orakel: jeweils ein ANDERER Rechenweg als der der Demo.

- Naive Iteration (float64, im Test noch einmal als Schleife nach dem Lehrbuch geschrieben) gegen die Log-Domäne und gegen `naive_iteration`, nach jedem k.
- Rundung: Algorithmus 2 in Brüchen (`tests/test_round.py`); hier zusätzlich auf Zufallsplänen: zulässig, nichtnegativ, die Schranke der Kosten und die Schranke des L1-Abstands
  ‖R − P‖₁ ≤ 2 (‖P1 − a‖₁ + ‖Pᵀ1 − b‖₁) (Lemma 7 bei Altschuler u. a.).
- Ableitung: die implizite Formel gegen eine zentrale finite Differenz, deren Pläne mit `scipy.optimize.minimize` (L-BFGS-B, eigener Gradient auf dem Dual) statt mit dem Newton-Verfahren
  der Demo gerechnet sind; das Durchrechnen gegen die finite Differenz der Iteration.
- LP-Optimum: `linprog` (HiGHS) auf Instanzen mit n ≠ m, `networkx.network_simplex` und der Mindestkostenfluss von OR-Tools auf ganzzahligen Instanzen.
- Eindeutigkeit: gegen ein zweites LP auf der optimalen Fläche.
Welche Ecke ein Löser bei Gleichstand wählt, wird nie verglichen: geprüft wird Wert, Zulässigkeit und Eindeutigkeit. Alle Läufe zusammen unter 10 Sekunden."""

import numpy as np
import pytest

import sgz_constants as C
import sgz_scenario as SC
import sgz_simplex as S
from orakel_zahlen import ORAKEL
from sgz_deriv import ableitung_implizit, durchrechnen, relativer_fehler
from sgz_eindeutigkeit import ist_eindeutig
from sgz_naiv import naive_iteration
from sgz_round import runden, schranke
from sgz_sinkhorn import loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale

sopt = pytest.importorskip("scipy.optimize")


def instanz(k, n=None, m=None):
    g = SC.instanz_rng(11, 12, k)
    n = n or 3 + g.below(4)
    m = m or 3 + g.below(4)
    a = np.array([g.expovariate(1.0) + 0.05 for _ in range(n)])
    b = np.array([g.expovariate(1.0) + 0.05 for _ in range(m)])
    Cm = np.array([[g.random() for _ in range(m)] for _ in range(n)])
    return a / a.sum(), b / b.sum(), Cm / Cm.max()


def lp(a, b, Cm, methode="highs-ds"):
    n, m = Cm.shape
    A = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    skala = 1e6                                         # HiGHS meldet bei Massen unter etwa 1e-7 „infeasible“: Massen skalieren
    r = sopt.linprog(Cm.ravel(), A_eq=A, b_eq=np.concatenate([a, b]) * skala, bounds=(0, None), method=methode)
    assert r.status == 0, r.message
    return r.fun / skala


def lehrbuch(a, b, Cm, eps, k):
    """Die Lehrbuch-Iteration u = a / (K v), v = b / (K^T u), K = exp(-C/eps), Start u = v = 1; Plan nach k Durchläufen."""
    K = np.exp(-Cm / eps)
    u, v = np.ones(len(a)), np.ones(len(b))
    for _ in range(k):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return u[:, None] * K * v[None, :]


def dual_plan_lbfgs(a, b, Cm, eps):
    """Plan aus dem Dual Phi(f, g) = a·f + b·g − eps Σ a_i b_j exp((f_i + g_j − C_ij)/eps), maximiert mit L-BFGS-B (eigene Rechnung, nicht der Code der Demo)."""
    n, m = Cm.shape

    def plan(x):
        return np.outer(a, b) * np.exp((x[:n, None] + x[None, n:] - Cm) / eps)

    def fg(x):
        P = plan(x)
        return -(a @ x[:n] + b @ x[n:] - eps * P.sum()), -np.concatenate([a - P.sum(axis=1), b - P.sum(axis=0)])

    r = sopt.minimize(fg, np.zeros(n + m), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-16, gtol=1e-14, maxcor=30))
    return plan(r.x)


def test_naive_iteration_gleich_lehrbuch_gleich_log_domaene_nach_jedem_k():
    for k_inst in range(ORAKEL["naiv"]):
        a, b, Cm = instanz(k_inst)
        eps = 0.8 * lp(a, b, Cm)
        for k, f, g in verlauf_potentiale(a, b, Cm, eps, [1, 2, 3, 5, 8, 20, 60]):
            soll = lehrbuch(a, b, Cm, eps, k)
            assert plan_aus_potentialen(a, b, Cm, eps, f, g) == pytest.approx(soll, rel=1e-9, abs=1e-13), (k_inst, k)
            assert naive_iteration(a, b, Cm, eps, np.float64, 0.0, k, "float64").plan == pytest.approx(soll, rel=1e-12, abs=1e-15), (k_inst, k)


def test_naive_float32_gleich_der_lehrbuch_iteration_in_float32():
    """Die float32-Fassung rechnet wirklich in float32: dieselbe Schleife mit float32-Arrays gibt dieselben Zahlen (Summenreihenfolge gleich, daher nicht nur im Band)."""
    for k_inst in range(4):
        a, b, Cm = instanz(k_inst)
        eps = 0.5 * lp(a, b, Cm)
        K = np.exp(-Cm / eps).astype(np.float32)
        u, v = np.ones(len(a), np.float32), np.ones(len(b), np.float32)
        for _ in range(30):
            u = a.astype(np.float32) / (K @ v)
            v = b.astype(np.float32) / (K.T @ u)
        soll = u[:, None].astype(float) * K.astype(float) * v[None, :].astype(float)
        assert naive_iteration(a, b, Cm, eps, np.float32, 0.0, 30, "float32").plan == pytest.approx(soll, rel=1e-6, abs=1e-12)


def test_rundung_zulaessig_und_kosten_und_l1_abstand_bleiben_in_den_schranken():
    """Der gerundete Plan ist zulässig, die Kosten und der L1-Abstand zu P bleiben in den Schranken der Quelle."""
    for k in range(ORAKEL["rund"]):
        a, b, Cm = instanz(k)
        eps = (0.3 + 0.2 * (k % 3)) * lp(a, b, Cm)
        for kk, f, g in verlauf_potentiale(a, b, Cm, eps, [1, 4, 20]):
            P = plan_aus_potentialen(a, b, Cm, eps, f, g)
            R = runden(P, a, b)
            assert (R >= 0).all() and R.sum(axis=1) == pytest.approx(a, abs=1e-14) and R.sum(axis=0) == pytest.approx(b, abs=1e-14)
            assert float((R * Cm).sum()) <= schranke(P, a, b, Cm) + 1e-12
            fehler = np.abs(P.sum(axis=1) - a).sum() + np.abs(P.sum(axis=0) - b).sum()
            assert np.abs(R - P).sum() <= 2 * fehler + 1e-12, (k, kk)                                 # Lemma 7: ‖R − P‖₁ ≤ 2 (‖r − a‖₁ + ‖c − b‖₁)


def test_implizite_formel_gegen_finite_differenz_mit_plaenen_aus_dem_dual_lbfgs():
    for k in range(ORAKEL["ableitung"]):
        a, b, Cm = instanz(k, 5, 5)
        eps = 0.6 * lp(a, b, Cm)
        g = np.random.default_rng(k)
        D = g.normal(size=Cm.shape)
        D /= np.abs(D).max()
        P = dual_plan_lbfgs(a, b, Cm, eps)
        h = 3e-3 * eps
        fd = (dual_plan_lbfgs(a, b, Cm + h * D, eps) - dual_plan_lbfgs(a, b, Cm - h * D, eps)) / (2 * h)
        assert relativer_fehler(ableitung_implizit(P, Cm, eps, D), fd) < 1e-4, k


def test_durchrechnen_gegen_finite_differenz_der_iteration_im_lehrbuch_k_bereich():
    for k_inst in range(3):
        a, b, Cm = instanz(k_inst, 4, 4)
        eps = 0.8 * lp(a, b, Cm)
        D = np.random.default_rng(k_inst + 10).normal(size=Cm.shape)
        D /= np.abs(D).max()
        h = 1e-6
        for k, P, dP in durchrechnen(a, b, Cm, eps, D, [2, 7, 25]):
            fd = (lehrbuch(a, b, Cm + h * D, eps, k) - lehrbuch(a, b, Cm - h * D, eps, k)) / (2 * h)
            assert relativer_fehler(dP, fd) < 1e-6, (k_inst, k)


def test_optimalwert_gleich_linprog_highs_mit_n_ungleich_m():
    for k in range(ORAKEL["lp"]):
        inst = SC.make_instanz(2 + k % 11, 2 + (k * 3) % 11, 7, C.ZWECK_ANSICHT, k)
        lo = S.loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        assert lo.kosten == pytest.approx(lp(inst.a, inst.b, inst.kosten), rel=1e-9, abs=1e-12), k
        assert not S.pruefe_loesung(inst.a, inst.b, inst.kosten, lo)


def test_ganzzahlige_instanzen_gleich_networkx_und_ortools_mit_abweichung_0():
    nx = pytest.importorskip("networkx")
    mcf_modul = pytest.importorskip("ortools.graph.python.min_cost_flow")
    for k in range(ORAKEL["netz"]):
        g = SC.instanz_rng(5, 6, k)
        n, m = 2 + g.below(5), 2 + g.below(5)
        a = [g.below(6) + 1 for _ in range(n)]
        gesamt = sum(a)
        schnitte = sorted(g.below(gesamt + 1) for _ in range(m - 1))
        b = [y - x for x, y in zip([0] + schnitte, schnitte + [gesamt])]
        Cm = np.array([[g.below(20) for _ in range(m)] for _ in range(n)], float)
        lo = S.loese_transport(a, b, Cm, "steilste")
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(("L", i), demand=-a[i])
        for j in range(m):
            G.add_node(("K", j), demand=b[j])
        for i in range(n):
            for j in range(m):
                G.add_edge(("L", i), ("K", j), weight=int(Cm[i, j]))
        assert nx.network_simplex(G)[0] == lo.kosten
        mcf = mcf_modul.SimpleMinCostFlow()
        for i in range(n):
            for j in range(m):
                mcf.add_arc_with_capacity_and_unit_cost(i, n + j, gesamt, int(Cm[i, j]))
        for i in range(n):
            mcf.set_node_supply(i, a[i])
        for j in range(m):
            mcf.set_node_supply(n + j, -b[j])
        assert mcf.solve() == mcf.OPTIMAL and mcf.optimal_cost() == lo.kosten, k


def test_eindeutigkeit_gegen_ein_zweites_lp_auf_der_optimalen_flaeche():
    """Der Plan ist eindeutig genau dann, wenn auf der optimalen Fläche jede Zelle nur einen Wert annimmt (min und max je Zelle mit linprog)."""
    n = 4
    A = np.vstack([np.kron(np.eye(n), np.ones((1, n))), np.kron(np.ones((1, n)), np.eye(n))])
    eindeutig, mehrere = 0, 0
    for k in range(ORAKEL["eind"]):
        g = SC.instanz_rng(9, 4, k)
        Cm = np.array([[float(g.below(3)) for _ in range(n)] for _ in range(n)])
        a = b = np.full(n, 1.0 / n)
        lo = S.loese_transport(a, b, Cm, "steilste")
        wert = lp(a, b, Cm)
        A_eq = np.vstack([A, Cm.ravel()[None, :]])
        b_eq = np.concatenate([a, b, [wert]])
        spanne = 0.0
        for z in range(n * n):
            c = np.zeros(n * n)
            c[z] = 1.0
            unten = sopt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
            oben = sopt.linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs-ds")
            assert unten.status == 0 and oben.status == 0
            spanne = max(spanne, -oben.fun - unten.fun)
        erwartet = spanne < 1e-9
        assert ist_eindeutig(Cm, lo.plan, lo.u, lo.v) == erwartet, k
        eindeutig += erwartet
        mehrere += not erwartet
    assert eindeutig >= 3 and mehrere >= 3          # beide Fälle kommen vor


def test_log_domaene_gleich_dual_lbfgs():
    a, b, Cm = instanz(2)
    eps = 0.5 * lp(a, b, Cm)
    erg = loese_sinkhorn(a, b, Cm, eps, tol=1e-12, max_iter=30000)
    assert erg.konvergiert and plan_aus_potentialen(a, b, Cm, eps, erg.f, erg.g) == pytest.approx(dual_plan_lbfgs(a, b, Cm, eps), abs=2e-7)
