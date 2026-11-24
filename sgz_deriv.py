"""Ableitung des Transportplans nach den Kosten (nur numpy, ohne Streamlit).

Der weiche Plan P(C) = diag(u) exp(-C/ε) diag(v) hängt glatt von den Kosten ab. Richtung D (gleiche Gestalt wie C): dP = P (df_i + dg_j − D_ij) / ε, mit Potentialen f, g
(P_ij = a_i b_j exp((f_i + g_j − C_ij)/ε)). Zwei Rechenwege für (df, dg):
  implizit   die Formel am Plan P (auch an einem unfertigen Plan P_k): M (df, dg) = (Σ_j P_ij D_ij, Σ_i P_ij D_ij) mit M = [[diag r, P], [Pᵀ, diag c]]; r, c sind die
             Zeilen- und Spaltensummen von P; eine Unbekannte wird festgehalten, weil (df + t, dg − t) denselben Plan gibt.
  durchrechnen   Vorwärtsmodus: die Iteration (Log-Domäne, Start u = v = 1) wird zusammen mit ihrer Ableitung gerechnet; df und dg nach k Durchläufen sind die Ableitung
             des Plans P_k selbst, nicht die des Grenzplans.
Als Gegenprobe dient die zentrale finite Differenz zweier genau gerechneter Pläne (`ableitung_fd`).

Einheiten:
  ableitung_implizit      dP aus (P, D)
  durchrechnen            (k, f, g, dP) nach vorgegebenen Durchläufen k in einem Lauf
  ableitung_fd            zentrale finite Differenz (P(C + hD) − P(C − hD)) / 2h mit Richardson-Verbesserung, Pläne aus dem Newton-Verfahren auf dem Dual
  relativer_fehler        Frobenius-Norm der Differenz, geteilt durch die Norm der Referenz"""

import numpy as np

from sgz_newton import loese_gibbs
from sgz_sinkhorn import plan_aus_potentialen, pruefe_massen


def ableitung_implizit(P, C, eps, D):
    """dP = P (df + dg − D)/ε; (df, dg) aus M (df, dg) = (Σ_j P_ij D_ij, Σ_i P_ij D_ij), letzte Unbekannte gleich 0 gesetzt."""
    P = np.asarray(P, float)
    n, m = P.shape
    r, c = P.sum(axis=1), P.sum(axis=0)
    M = np.block([[np.diag(r), P], [P.T, np.diag(c)]])
    rhs = np.concatenate([(P * D).sum(axis=1), (P * D).sum(axis=0)])
    try:
        sol = np.append(np.linalg.solve(M[:-1, :-1], rhs[:-1]), 0.0)
    except np.linalg.LinAlgError:
        sol = np.append(np.linalg.lstsq(M[:-1, :-1], rhs[:-1], rcond=None)[0], 0.0)
    return P * (sol[:n][:, None] + sol[n:][None, :] - D) / eps


def durchrechnen(a, b, C, eps, D, ks):
    """Plan und Ableitung nach k Durchläufen für jedes k in `ks` (aufsteigend, k >= 1), in einem Lauf (Start u = v = 1: f = −ε ln a, g = −ε ln b, Ableitung 0).
    Gibt eine Liste von (k, P_k, dP_k) zurück; P_k ist der Plan nach dem g-Schritt (Spaltensummen genau)."""
    a, b, C = pruefe_massen(a, b, C)
    D = np.asarray(D, float)
    ks = sorted(set(int(k) for k in ks))
    if not ks or ks[0] < 1:
        raise ValueError("k muss mindestens 1 sein")
    la, lb = np.log(a), np.log(b)
    CT = np.ascontiguousarray(C.T)
    DT = np.ascontiguousarray(D.T)
    f, g = -eps * la, -eps * lb
    df, dg = np.zeros(len(a)), np.zeros(len(b))
    ergebnis = []
    for k in range(1, ks[-1] + 1):
        # f-Schritt: f_i = −ε log Σ_j b_j exp((g_j − C_ij)/ε); df_i = −Σ_j w_ij (dg_j − D_ij)
        M = lb[None, :] + (g[None, :] - C) / eps
        m = M.max(axis=1)
        E = np.exp(M - m[:, None])
        s = E.sum(axis=1)
        f = -eps * (m + np.log(s))
        df = ((E / s[:, None]) * (D - dg[None, :])).sum(axis=1)
        # g-Schritt: g_j = −ε log Σ_i a_i exp((f_i − C_ij)/ε); dg_j = −Σ_i w_ji (df_i − D_ij)
        M = la[None, :] + (f[None, :] - CT) / eps
        m = M.max(axis=1)
        E = np.exp(M - m[:, None])
        s = E.sum(axis=1)
        g = -eps * (m + np.log(s))
        dg = ((E / s[:, None]) * (DT - df[None, :])).sum(axis=1)
        if k in ks:
            P = plan_aus_potentialen(a, b, C, eps, f, g)
            ergebnis.append((k, P, P * (df[:, None] + dg[None, :] - D) / eps))
    return ergebnis


def ableitung_fd(a, b, C, eps, D, h_rel=1e-2):
    """Zentrale finite Differenz der genau gerechneten Pläne (Newton-Verfahren auf dem Dual, Randfehler unter 1e-12) mit Richardson-Verbesserung:
    D1(h) = (P(C + hD) − P(C − hD)) / 2h mit h = h_rel · ε; Ergebnis (4 D1(h/2) − D1(h)) / 3 (Fehler der Ordnung h⁴ statt h²). Eine kleine Schrittweite wäre hier schlechter: der Randfehler der
    Pläne (etwa 1e-13) wird durch 2h geteilt und übersteigt bei kleinem ε/OPT, wo die Ableitung klein ist, den Wert der Ableitung. Gibt None zurück, wenn ein Newton-Verfahren nicht konvergiert."""
    C = np.asarray(C, float)
    zentral = []
    for h in (h_rel * eps, 0.5 * h_rel * eps):
        plus = loese_gibbs(a, b, C + h * D, eps)
        minus = loese_gibbs(a, b, C - h * D, eps)
        if not (plus.ok and minus.ok):
            return None
        zentral.append((plus.plan - minus.plan) / (2.0 * h))
    return (4.0 * zentral[1] - zentral[0]) / 3.0


def relativer_fehler(X, Referenz):
    """‖X − Referenz‖_F / ‖Referenz‖_F."""
    return float(np.linalg.norm(np.asarray(X) - np.asarray(Referenz)) / np.linalg.norm(Referenz))
