"""Szenario: Instanzen aus dem Hausgenerator SplitMix64 (`sgz_rng.py`).

Alles ist deterministisch aus (Zufalls-Seed, Zweck, Index): gleiche Eingabe, gleiche Instanz, auf jeder Plattform. Gesamtmasse 1 auf beiden Seiten.
Zufallsinstanz: Lager und Kunden gleichverteilt im Einheitsquadrat, quadrierter Abstand durch den größten Wert geteilt (max C = 1), Massen a = (1 − s)/n + s · Dirichlet(1)
(Dirichlet(1) = n unabhängige Exponentialzahlen, durch ihre Summe geteilt), b ebenso. Ganzzahlige Kosten: Massen 1/n, Einträge gleichverteilt in {0, ..., R}.
Weiche Ränge: Werte x_i gleichverteilt in (0, 1), Zielstellen s_j = (j − ½)/n, Kosten (x_i − s_j)², Massen 1/n."""

from dataclasses import dataclass

import numpy as np

import sgz_constants as C
from sgz_rng import MASK, SplitMix64


@dataclass(frozen=True)
class Instanz:
    a: np.ndarray                       # Angebote der n Lager (Summe 1)
    b: np.ndarray                       # Nachfragen der m Kunden (Summe 1)
    kosten: np.ndarray                  # n x m
    lager: np.ndarray | None = None     # n x 2 Orte (nur bei geometrischen Instanzen)
    kunden: np.ndarray | None = None    # m x 2 Orte
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def m(self) -> int:
        return len(self.b)


def instanz_rng(seed: int, zweck: int, index: int) -> SplitMix64:
    """Eigener Zufallsstrom je (Seed, Zweck, Index); die ersten Zahlen werden verworfen, damit benachbarte Indizes unabhängig wirken."""
    g = SplitMix64((int(seed) * 1_000_003 + int(zweck) * 7_919 + int(index) * 104_729 + 12_345) & MASK)
    g.next_u64()
    g.next_u64()
    return g


def massen(g: SplitMix64, n: int, s: float) -> np.ndarray:
    """Angebote bzw. Nachfragen mit Summe 1: (1 - s)/n + s · Dirichlet(1)."""
    if s == 0.0:
        return np.full(n, 1.0 / n)
    e = np.array([g.expovariate(1.0) for _ in range(n)])
    return (1.0 - s) / n + s * e / e.sum()


def kosten_aus_orten(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Quadrierter Abstand der Orte, durch den größten Wert geteilt (max C = 1)."""
    q = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return q / q.max()


def make_instanz(n: int, m: int, seed: int, zweck: int, index: int, s: float = 1.0) -> Instanz:
    """Zufallsinstanz: n Lager und m Kunden im Einheitsquadrat. Reihenfolge der Ziehungen: Orte der Lager, Orte der Kunden, Massen a, Massen b."""
    if n < 1 or m < 1:
        raise ValueError("mindestens ein Lager und ein Kunde")
    g = instanz_rng(seed, zweck, index)
    X = np.array([[g.random(), g.random()] for _ in range(n)])
    Y = np.array([[g.random(), g.random()] for _ in range(m)])
    a = massen(g, n, s)
    b = massen(g, m, s)
    return Instanz(a, b, kosten_aus_orten(X, Y), X, Y, C.QUELLE_ZUFALL)


def make_gleichstand(art: str, n: int, seed: int, index: int) -> Instanz:
    """Instanz von Experiment 3 (Massen 1/n): „stetig“ (Punkte im Quadrat, quadrierter Abstand, max C = 1), „ganzzahlig 0 bis R“ (Kosten gleichverteilt in {0, ..., R}),
    „alle Kosten gleich“ (jede Kopplung ist optimal)."""
    a = np.full(n, 1.0 / n)
    if art == C.KOSTEN3[3]:
        return Instanz(a, a.copy(), np.full((n, n), 0.5), None, None, art)
    g = instanz_rng(seed, C.ZWECK_GLEICH, index)
    if art == C.KOSTEN3[0]:
        X = np.array([[g.random(), g.random()] for _ in range(n)])
        Y = np.array([[g.random(), g.random()] for _ in range(n)])
        return Instanz(a, a.copy(), kosten_aus_orten(X, Y), X, Y, art)
    if art in (C.KOSTEN3[1], C.KOSTEN3[2]):
        R = 2 if art == C.KOSTEN3[1] else 10
        K = np.array([[float(g.below(R + 1)) for _ in range(n)] for _ in range(n)])
        return Instanz(a, a.copy(), K, None, None, art)
    raise ValueError(f"unbekannte Kostenart: {art}")


def handbeispiel_weit() -> Instanz:
    """Handbeispiel „weit entferntes Lager“: Angebote und Nachfragen (1/2, 1/2), Kosten [[0, 1/2], [1, 1]]: Lager 2 hat von beiden Kunden den größten Abstand 1; OPT = 1/2."""
    kosten = np.array([[0.0, 0.5], [1.0, 1.0]])
    return Instanz(np.array([0.5, 0.5]), np.array([0.5, 0.5]), kosten, None, None, C.QUELLE_HAND)


def handbeispiel_runden() -> tuple:
    """Handbeispiel der Rundung: der Plan nach einem Durchlauf (Angebote (3/4, 1/4), Nachfragen (1/4, 3/4), Kosten [[0, 1], [1, 0]], ε = 1/ln 3, Start u = v = 1):
    P = [[9/40, 3/8], [1/40, 3/8]]. Gibt (a, b, Kosten, P) zurück."""
    return (np.array([0.75, 0.25]), np.array([0.25, 0.75]), np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[9 / 40, 3 / 8], [1 / 40, 3 / 8]]))


def make_werte(n: int, seed: int, index: int) -> np.ndarray:
    """n Werte gleichverteilt in (0, 1) (Absatz „weiche Ränge“)."""
    g = instanz_rng(seed, C.ZWECK_RAENGE, index)
    return np.array([g.random() for _ in range(n)])


def raenge_kosten(x: np.ndarray):
    """Gibt (a, C, OPT, echter Rang) zurück: Massen 1/n, Kosten (x_i − s_j)², s_j = (j − ½)/n, OPT = Kosten der sortierten Zuordnung (optimal, weil die Kosten quadratisch in x − s sind),
    echter Rang 1 bis n (bei Gleichstand nach Index)."""
    n = len(x)
    s = (np.arange(1, n + 1) - 0.5) / n
    K = (x[:, None] - s[None, :]) ** 2
    rang = np.empty(n, dtype=int)
    rang[np.argsort(x, kind="stable")] = np.arange(n)
    opt = float(K[np.arange(n), rang].sum() / n)
    return np.full(n, 1.0 / n), K, opt, rang + 1
