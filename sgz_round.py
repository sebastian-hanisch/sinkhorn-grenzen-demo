"""Rundung auf zulässige Randverteilungen (nur numpy, ohne Streamlit): Algorithmus 2 von Altschuler, Weed, Rigollet (2017).

Ein Plan nach endlich vielen Sinkhorn-Durchläufen hat nur ungefähr die Randverteilungen a und b. Die Rundung macht ihn zulässig, in drei Schritten:
  1. Zeilen herunterskalieren: jede Zeile mit Summe über a_i wird auf a_i gestutzt (Faktor min(a_i / r_i, 1)),
  2. Spalten herunterskalieren: ebenso mit b_j,
  3. den Rest (Zeilen- und Spaltenfehlmengen) als Rang-1-Matrix e_r e_cᵀ / ‖e_r‖₁ addieren.
Das Ergebnis hat die Randverteilungen exakt (bis auf Rundungsfehler), ändert einen schon zulässigen Plan nicht und erfüllt die Schranke
⟨C, R⟩ ≤ ⟨C, P⟩ + 2 · max C · (‖P 1 − a‖₁ + ‖Pᵀ 1 − b‖₁).

Einheiten:
  runden          der Algorithmus (ein Plan hinein, ein Plan heraus)
  schranke        rechte Seite der Schranke für die Kosten des gerundeten Plans"""

import numpy as np


def runden(P, a, b):
    """Algorithmus 2: gibt den gerundeten Plan R mit Zeilensummen a und Spaltensummen b zurück. Zeilen und Spalten ohne Masse (Summe 0) werden nicht skaliert."""
    P = np.asarray(P, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = P.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(r > 0, np.minimum(a / r, 1.0), 1.0)
    F = x[:, None] * P
    c = F.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(c > 0, np.minimum(b / c, 1.0), 1.0)
    F = F * y[None, :]
    er = np.maximum(a - F.sum(axis=1), 0.0)      # in exakter Rechnung nach Schritt 1 und 2 nie negativ; der Rundungsfehler (−1e-17) würde sonst negative Einträge erzeugen
    ec = np.maximum(b - F.sum(axis=0), 0.0)
    s = er.sum()
    if s > 0:
        F = F + np.outer(er, ec) / s
    return F


def schranke(P, a, b, C):
    """⟨C, P⟩ + 2 · max C · (‖P 1 − a‖₁ + ‖Pᵀ 1 − b‖₁): obere Schranke der Kosten des gerundeten Plans (Altschuler u. a., Lemma 7 in dieser Form)."""
    P = np.asarray(P, float)
    fehler = float(np.abs(P.sum(axis=1) - a).sum() + np.abs(P.sum(axis=0) - b).sum())
    return float((P * C).sum()) + 2.0 * float(np.max(C)) * fehler
