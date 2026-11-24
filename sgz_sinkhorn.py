"""Sinkhorn-Iteration in der Log-Domäne (nur numpy, ohne Streamlit): der Kern der Demo.

Aufgabe (entropisch regularisiert): minimiere <C, P> + eps · KL(P || a b^T) über alle Transportpläne P >= 0 mit Zeilensummen a und Spaltensummen b.
Lösung in Gibbs-Form: P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps). Die Kantorovich-Potentiale f (Lager) und g (Kunden) sind die Variablen der Iteration;
mit u_i = a_i exp(f_i / eps) und v_j = b_j exp(g_j / eps) ist das die Form P = diag(u) K diag(v) mit K = exp(-C / eps).

Eine Iteration besteht aus zwei Halbschritten (f-Aktualisierung, dann g-Aktualisierung), das sind **zwei Matrix-Vektor-Produkte** (in der Log-Domäne je ein Log-Sum-Exp über
n · m Einträge). Nach der g-Aktualisierung stimmen die Spaltensummen genau, die Zeilensummen nicht. Der Abbruchtest (Zeilenfehler des Plans, L1-Norm) benutzt den
nächsten f-Halbschritt, den die folgende Iteration wiederverwendet; beim Abbruch bleibt er übrig: bei gemeldeten 2k Produkten (`produkte`) werden 2k + 1 Log-Sum-Exp ausgewertet. Start: u = v = 1 (Lehrbuchform; der Plan nach null Durchläufen ist K = exp(-C / eps)). Die Log-Domäne löst den Unterlauf des Kerns exp(-C / eps) bei kleinem eps; sie macht die Iteration nicht schneller.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben:
  lse_zeilen            Log-Sum-Exp je Zeile
  aktualisiere_f        f-Halbschritt aus g
  aktualisiere_g        g-Halbschritt aus f
  plan_aus_potentialen  der Plan aus (f, g)
  zeilenfehler          L1-Abstand der Zeilensummen des Plans von a
  loese_sinkhorn        ruft nur der Reihe nach auf (Abbruch bei Zeilenfehler unter tol oder nach max_iter Iterationen)
  verlauf_potentiale    (f, g) nach vorgegebenen Iterationszahlen k in einem Lauf"""

from dataclasses import dataclass

import numpy as np

TOL_ZEILENFEHLER = 1e-8     # Standard-Abbruchtoleranz der Demo (float64)
MAX_ITERATIONEN = 30000     # Obergrenze der Iterationen; danach gilt der Lauf als nicht konvergiert


@dataclass(frozen=True)
class SinkhornErgebnis:
    f: np.ndarray
    g: np.ndarray
    iterationen: int
    zeilenfehler: float       # L1-Abstand der Zeilensummen von a nach `iterationen` Iterationen
    konvergiert: bool

    @property
    def produkte(self) -> int:
        """Matrix-Vektor-Produkte: zwei je Iteration (ohne das eine Produkt des Abbruchtests, siehe Kopf der Datei)."""
        return 2 * self.iterationen


def pruefe_massen(a, b, C):
    a, b, C = np.asarray(a, float), np.asarray(b, float), np.asarray(C, float)
    if a.ndim != 1 or b.ndim != 1 or C.shape != (len(a), len(b)):
        raise ValueError("a (n), b (m) und C (n x m) müssen zueinander passen")
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError("alle Massen müssen positiv sein (Log-Domäne)")
    if abs(a.sum() - b.sum()) > 1e-9 * max(1.0, a.sum()):
        raise ValueError("die Gesamtmassen von a und b müssen übereinstimmen")
    return a, b, C


def lse_zeilen(M):
    """log(sum_j exp(M_ij)) je Zeile, stabil über das Zeilenmaximum."""
    m = M.max(axis=1)
    return m + np.log(np.exp(M - m[:, None]).sum(axis=1))


def aktualisiere_f(lb, g, C, eps):
    """f_i = -eps · log sum_j b_j exp((g_j - C_ij) / eps)."""
    return -eps * lse_zeilen(lb[None, :] + (g[None, :] - C) / eps)


def aktualisiere_g(la, f, CT, eps):
    """g_j = -eps · log sum_i a_i exp((f_i - C_ij) / eps); CT ist die transponierte Kostenmatrix."""
    return -eps * lse_zeilen(la[None, :] + (f[None, :] - CT) / eps)


def plan_aus_potentialen(a, b, C, eps, f, g):
    """P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps); Unterlauf zu 0 ist hier gewollt (Einträge unter 1e-300)."""
    with np.errstate(under="ignore"):
        return np.exp(np.log(a)[:, None] + np.log(b)[None, :] + (f[:, None] + g[None, :] - C) / eps)


def zeilenfehler(P, a):
    """L1-Abstand der Zeilensummen des Plans von a."""
    return float(np.abs(P.sum(axis=1) - a).sum())


def loese_sinkhorn(a, b, C, eps, tol=TOL_ZEILENFEHLER, max_iter=MAX_ITERATIONEN, f0=None, g0=None):
    """Sinkhorn-Iteration bis Zeilenfehler < tol (oder max_iter Iterationen). Warmstart über f0, g0 (Standard: u = v = 1, also f = -eps ln a und g = -eps ln b).
    Der Fehler wird aus dem nächsten f-Halbschritt gebildet: err = sum_i a_i |exp((f_i - f_neu_i) / eps) - 1|, das ist der Zeilenfehler des Plans aus (f, g)."""
    a, b, C = pruefe_massen(a, b, C)
    if eps <= 0:
        raise ValueError("eps muss positiv sein")
    n, m = C.shape
    la, lb = np.log(a), np.log(b)
    f = -eps * la if f0 is None else np.array(f0, float)         # Start u = v = 1 (Lehrbuchform): u_i = a_i exp(f_i / eps) = 1
    g = -eps * lb if g0 is None else np.array(g0, float)
    CT = np.ascontiguousarray(C.T)
    k = 0
    fehler = np.inf
    while True:
        f_neu = aktualisiere_f(lb, g, C, eps)
        if k > 0 or f0 is not None or max_iter == 0:
            with np.errstate(over="ignore"):
                fehler = float(np.sum(a * np.abs(np.exp((f - f_neu) / eps) - 1.0)))
            if fehler < tol or k >= max_iter:
                break
        f = f_neu
        g = aktualisiere_g(la, f, CT, eps)
        k += 1
    return SinkhornErgebnis(f, g, k, fehler, bool(fehler < tol))


def verlauf_potentiale(a, b, C, eps, ks):
    """(f, g) nach k Iterationen für jedes k in `ks` (aufsteigend, k >= 1), in einem einzigen Lauf ohne Abbruch. Gibt eine Liste von (k, f, g) zurück."""
    a, b, C = pruefe_massen(a, b, C)
    ks = sorted(set(int(k) for k in ks))
    if not ks or ks[0] < 1:
        raise ValueError("k muss mindestens 1 sein")
    n, m = C.shape
    la, lb = np.log(a), np.log(b)
    f, g = -eps * la, -eps * lb                                   # Start u = v = 1
    CT = np.ascontiguousarray(C.T)
    ergebnis = []
    for k in range(1, ks[-1] + 1):
        f = aktualisiere_f(lb, g, C, eps)
        g = aktualisiere_g(la, f, CT, eps)
        if k in ks:
            ergebnis.append((k, f.copy(), g.copy()))
    return ergebnis
