"""Die naive Sinkhorn-Iteration im Kern-Bereich (nur numpy, ohne Streamlit): u = a / (K v), v = b / (Kᵀ u) mit K = exp(-C / ε), in float32 oder float64.

Wozu: Die Demo zeigt, wo diese Form scheitert (Unterlauf von exp(-C/ε), Abbruchtest unter der Auflösung des Datentyps). Die Log-Domäne (`sgz_sinkhorn.py`) ist die
Gegenprobe. Beide starten mit u = v = 1 und liefern nach k Durchläufen denselben Plan, solange nichts unterläuft.

Eine Iteration besteht aus zwei Matrix-Vektor-Produkten (K v und Kᵀ u). Der Abbruchtest benutzt das Produkt K v des nächsten Halbschritts und kostet kein weiteres.
Der Zeilenfehler ist die L1-Norm von (Zeilensumme des Plans − Masse); er wird in dem Datentyp gerechnet, in dem die Iteration läuft (`test = "float32"`: die Zeilensumme
in float32 gegen die auf float32 gerundete Masse), oder in float64 gegen die exakte Masse (`test = "float64"`). Beide Wege geben verschiedene Zahlen.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben:
  kern                  K = exp(-C / ε) im Datentyp (Einträge unter der kleinsten Zahl des Datentyps werden 0)
  hat_unterlauf         Zeile oder Spalte von K ist 0, oder ein Produkt ist nicht endlich oder 0
  zeilenfehler_naiv     Zeilenfehler im gewählten Datentyp
  naive_iteration       ruft nur der Reihe nach auf (Status ok, cap oder unterlauf)
  kleinste_aufloesung   kleinster von null verschiedener Abstand einer Zahl im Datentyp bei Masse 1/4"""

from dataclasses import dataclass

import numpy as np

STATUS_OK, STATUS_CAP, STATUS_UNTERLAUF = "ok", "cap", "unterlauf"
DATENTYPEN = {"float32": np.float32, "float64": np.float64}


@dataclass(frozen=True)
class NaivErgebnis:
    status: str                    # "ok" (Toleranz erreicht), "cap" (Iterationsgrenze erreicht) oder "unterlauf"
    iterationen: int
    zeilenfehler: float            # zuletzt gemessener Zeilenfehler (nan bei Unterlauf vor dem ersten Test)
    plan: np.ndarray | None        # float64-Plan aus u, K, v (None bei Unterlauf)

    @property
    def produkte(self) -> int:
        return 2 * self.iterationen


def kern(C, eps, dtype):
    """K = exp(-C/ε), in float64 gerechnet und in den Datentyp gerundet (Unterlauf zu 0 ist hier das Thema, keine Warnung)."""
    with np.errstate(under="ignore", over="ignore"):
        return np.exp(-np.asarray(C, float) / eps).astype(dtype)


def hat_unterlauf(K) -> bool:
    """True, wenn eine Zeile oder Spalte von K nur Nullen hat (dann ist K v oder Kᵀ u an dieser Stelle 0 und die Division scheitert)."""
    return bool((K.sum(axis=1) == 0).any() or (K.sum(axis=0) == 0).any())


def zeilenfehler_naiv(u, Kv, a, test):
    """L1-Abstand der Zeilensummen u · (K v) des Plans von a; `test` = "float32" (im Datentyp der Iteration gerechnet) oder "float64" (gegen die exakte Masse)."""
    zeilensumme = u * Kv
    if test == "float64":
        return float(np.abs(zeilensumme.astype(np.float64) - np.asarray(a, float)).sum())
    return float(np.abs(zeilensumme - np.asarray(a).astype(zeilensumme.dtype)).sum())


def kleinste_aufloesung(dtype) -> float:
    """Kleinster von null verschiedener Abstand zweier Zahlen im Datentyp bei Zahlen unterhalb von 1/4 (nach unten, bei Masse 1/4): 2⁻²⁶ bei float32."""
    x = dtype(0.25)
    return float(x - np.nextafter(x, dtype(0)))


def naive_iteration(a, b, C, eps, dtype=np.float64, tol=1e-8, max_iter=3000, test="float64"):
    """Naive Iteration im Kern-Bereich bis Zeilenfehler < tol oder max_iter Iterationen; Start u = v = 1. Gibt ein NaivErgebnis zurück.
    `test` wählt den Datentyp des Abbruchtests (bei dtype float64 ohne Wirkung)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    K = kern(C, eps, dtype)
    if hat_unterlauf(K):
        return NaivErgebnis(STATUS_UNTERLAUF, 0, float("nan"), None)
    a_, b_ = a.astype(dtype), b.astype(dtype)
    with np.errstate(all="ignore"):
        v = np.ones(len(b), dtype=dtype)
        u = np.ones(len(a), dtype=dtype)
        Kv = K @ v
        fehler = float("nan")
        k = 0
        while True:
            if k > 0:
                fehler = zeilenfehler_naiv(u, Kv, a, test)
                if fehler < tol:
                    break
            if k >= max_iter:
                return NaivErgebnis(STATUS_CAP, k, fehler, _plan(u, K, v))
            if not (np.all(np.isfinite(Kv)) and np.all(Kv != 0)):
                return NaivErgebnis(STATUS_UNTERLAUF, k, fehler, None)
            u = a_ / Kv
            Ktu = K.T @ u
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(Ktu)) and np.all(Ktu != 0)):
                return NaivErgebnis(STATUS_UNTERLAUF, k, fehler, None)
            v = b_ / Ktu
            if not np.all(np.isfinite(v)):
                return NaivErgebnis(STATUS_UNTERLAUF, k, fehler, None)
            Kv = K @ v
            k += 1
    return NaivErgebnis(STATUS_OK, k, fehler, _plan(u, K, v))


def _plan(u, K, v):
    return u[:, None].astype(np.float64) * K.astype(np.float64) * v[None, :].astype(np.float64)
