"""Der regularisierte Plan als Nullstelle: Newton-Verfahren auf dem Dual (nur numpy, ohne Streamlit).

Dasselbe Problem wie in `sgz_sinkhorn.py` (P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)), anderes Verfahren: Newton auf dem konkaven Dual
Phi(f, g) = a·f + b·g - eps · sum_ij a_i b_j exp((f_i + g_j - C_ij) / eps)
mit Gradient (a - Zeilensummen, b - Spaltensummen) und Hesse-Matrix -M / eps, M = [[diag(r), P], [P^T, diag(c)]] (r, c Zeilen- und Spaltensummen von P).
Die Hesse-Matrix ist singulär (f + t, g - t gibt denselben Plan); die letzte Komponente von g wird festgehalten. Die Schrittweite folgt einer Rückverfolgung
(Armijo-Bedingung auf dem Dual oder Abnahme des Randfehlers). Kleines eps wird über eine Fortsetzung erreicht (eps absteigend, Start aus der Lösung des vorigen eps).

Wozu: Die Demo braucht den Plan zu sehr kleinem eps/OPT (Experimente 4 und 5), wo die reine Sinkhorn-Iteration zehntausende Iterationen bräuchte. Der Plan ist derselbe
(auf 1e-12 im Randfehler); gezählt werden hier keine Matrix-Vektor-Produkte, nur die Newton-Schritte."""

import math
from dataclasses import dataclass

import numpy as np

from sgz_sinkhorn import loese_sinkhorn, plan_aus_potentialen, pruefe_massen


@dataclass(frozen=True)
class NewtonErgebnis:
    f: np.ndarray
    g: np.ndarray
    plan: np.ndarray
    schritte: int
    fehler: float        # L1-Fehler beider Randverteilungen
    ok: bool


def _plan(la, lb, C, eps, f, g):
    with np.errstate(all="ignore"):
        return np.exp(la[:, None] + lb[None, :] + (f[:, None] + g[None, :] - C) / eps)


def _dual(a, b, la, lb, C, eps, f, g, P):
    return float(a @ f + b @ g - eps * P.sum())


def loese_gibbs_newton(a, b, C, eps, f0=None, g0=None, tol=1e-12, max_schritte=200):
    """Ein Newton-Lauf bei festem eps (Start f0, g0 oder Nullen). Gibt NewtonErgebnis zurück (ok = Randfehler unter tol).
    Probeschritte der Rückverfolgung dürfen überlaufen (sie werden verworfen): Warnungen von numpy sind dafür abgeschaltet."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _newton(a, b, C, eps, f0, g0, tol, max_schritte)


def _newton(a, b, C, eps, f0, g0, tol, max_schritte):
    a, b, C = pruefe_massen(a, b, C)
    n, m = C.shape
    la, lb = np.log(a), np.log(b)
    f = np.zeros(n) if f0 is None else np.array(f0, float)
    g = np.zeros(m) if g0 is None else np.array(g0, float)
    P = _plan(la, lb, C, eps, f, g)
    fehler = np.inf
    for k in range(max_schritte + 1):
        r, c = P.sum(axis=1), P.sum(axis=0)
        fehler = float(np.abs(r - a).sum() + np.abs(c - b).sum())
        if not np.isfinite(fehler):
            return NewtonErgebnis(f, g, P, k, fehler, False)
        if fehler < tol:
            return NewtonErgebnis(f, g, P, k, fehler, True)
        if k == max_schritte:
            break
        grad = np.concatenate([a - r, b - c])
        M = np.zeros((n + m, n + m))
        M[:n, :n] = np.diag(r)
        M[n:, n:] = np.diag(c)
        M[:n, n:] = P
        M[n:, :n] = P.T
        try:
            d = np.linalg.solve(M[:-1, :-1], grad[:-1]) * eps
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(M[:-1, :-1], grad[:-1], rcond=None)[0] * eps
        schritt = np.append(d, 0.0)
        gs = float(grad @ schritt)
        phi0 = _dual(a, b, la, lb, C, eps, f, g, P)
        t = 1.0
        angenommen = False
        while t > 1e-12:
            fn, gn = f + t * schritt[:n], g + t * schritt[n:]
            Pn = _plan(la, lb, C, eps, fn, gn)
            if np.all(np.isfinite(Pn)):
                phin = _dual(a, b, la, lb, C, eps, fn, gn, Pn)
                fehlern = float(np.abs(Pn.sum(axis=1) - a).sum() + np.abs(Pn.sum(axis=0) - b).sum())
                if phin >= phi0 + 1e-4 * t * gs or fehlern < fehler * (1 - 1e-4 * t):
                    angenommen = True
                    break
            t *= 0.5
        if not angenommen:
            return NewtonErgebnis(f, g, P, k, fehler, False)
        f, g, P = fn, gn, Pn
    return NewtonErgebnis(f, g, P, max_schritte, fehler, False)


def _vorwaerts(a, b, C, zustand, eps_neu, tiefe=0):
    """Von (eps, f, g) nach eps_neu laufen; bei Misserfolg den logarithmischen Mittelwert als Zwischenschritt (bis Tiefe 6)."""
    eps0, f0, g0 = zustand
    erg = loese_gibbs_newton(a, b, C, eps_neu, f0, g0)
    if erg.ok:
        return erg
    if tiefe >= 6:
        return erg
    mitte = math.sqrt(eps0 * eps_neu)
    zw = _vorwaerts(a, b, C, zustand, mitte, tiefe + 1)
    if not zw.ok:
        return erg
    return _vorwaerts(a, b, C, (mitte, zw.f, zw.g), eps_neu, tiefe + 1)


def kaltstart(a, b, C, eps, sweeps=60):
    """Startwerte für Newton aus einigen Sinkhorn-Iterationen (ohne Abbruch)."""
    erg = loese_sinkhorn(a, b, C, eps, tol=0.0, max_iter=sweeps)
    return erg.f, erg.g


def loese_gibbs_pfad(a, b, C, eps_absteigend):
    """Pläne für eine absteigende Liste von eps mit Fortsetzung. Gibt eine Liste von (eps, NewtonErgebnis) zurück."""
    e0 = eps_absteigend[0]
    f, g = kaltstart(a, b, C, e0)
    erg = loese_gibbs_newton(a, b, C, e0, f, g)
    ergebnisse = [(e0, erg)]
    for e in eps_absteigend[1:]:
        if erg.ok:
            erg = _vorwaerts(a, b, C, (ergebnisse[-1][0], erg.f, erg.g), e)
        ergebnisse.append((e, erg))
    return ergebnisse


def loese_gibbs(a, b, C, eps, faktor=3.0):
    """Der Plan bei eps. Fortsetzung von einem großen eps (halbe größte Kosten) in Schritten um `faktor` nach unten. Gibt NewtonErgebnis zurück."""
    start = 0.5 * float(np.asarray(C).max())
    stufen = [eps]
    while stufen[0] * faktor < start:
        stufen.insert(0, stufen[0] * faktor)
    return loese_gibbs_pfad(a, b, C, stufen)[-1][1]
