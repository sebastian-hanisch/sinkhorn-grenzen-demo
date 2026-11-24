"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (alles numpy, Zeiten auf dem Entwicklungsrechner, ein Kern; auf der Cloud wahrscheinlich 2- bis 3-mal länger):
- Die Experimente 1, 3 und 4 und der Absatz „weiche Ränge“ liegen als Reihen vor (`tools/vorgerechnet.json`, erzeugt von `tools/vorrechnen.py`): jede Einstellung der Regler
  dieser Experimente ist vorgerechnet, die App rechnet dort nichts. Die Vorrechnung braucht einige Minuten (Experiment 1 etwa 1 Minute, Experiment 3 etwa 3 Minuten).
- Experiment 2 rechnet die Instanz der Seitenleiste live: die Iteration mit ihrer Ableitung (Vorwärtsmodus) bis 20 000 Durchläufe, O(n · m) je Durchlauf (bei n = 20
  etwa 0,1 ms), dazu der genaue Plan (Newton-Verfahren auf dem Dual, O((n + m)³) je Schritt) und ein Transportsimplex (n = 20: 2,5 ms). Gesamt bei n = 20 einige Sekunden.
- Die Tabellen von Experiment 2 (n = 60 für die Rundung, n = 20 für die Ableitung, 30 Instanzen) sind ebenfalls vorgerechnet."""


QUELLE_ZUFALL = "Zufallsinstanz"
QUELLE_HAND = "Von Hand: ein weit entferntes Lager"
QUELLEN = (QUELLE_ZUFALL, QUELLE_HAND)

N_MIN, N_MAX, DEFAULT_N = 6, 20, 12               # Lager = Kunden der Instanz in Experiment 2
SEED_MAX, DEFAULT_SEED = 999999, 35
SERIEN_SEED = 2026                                # Zufalls-Seed der Reihen: nie der Regler „Zufalls-Seed“

# Experiment 1: Unterlauf und Rechengenauigkeit
N1 = 30                                           # Lager = Kunden
ANZAHL1 = 50                                      # Instanzen der Reihe (die Auswahl 10 und 20 nimmt die ersten Instanzen)
EPS1_STUFEN = (0.01, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0005)    # ε in Einheiten des größten Kostenwerts (max C = 1)
TOL1_STUFEN = (1e-8, 1e-5)
TEST1_STUFEN = ("float32", "float64")             # Datentyp, in dem der Abbruchtest der float32-Läufe gerechnet wird
INST1_STUFEN = (10, 20, 50)
MAX_ITER1 = 3000
ABWEICHUNG_SCHWELLEN = (1e-6, 1e-4, 1e-2)         # Abweichung der Kosten von der Log-Domäne, relativ
HAND_EPS = (1 / 110, 1 / 800)                     # Handbeispiel „weit entferntes Lager“: ε, bei dem exp(-C/ε) in float32 bzw. float64 unterläuft

# Experiment 2: Wie fertig ist der Plan?
EPS2_STUFEN = (1.0, 0.46, 0.3, 0.1, 0.03, 0.01)   # ε/OPT
K2_STUFEN = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000)
RUNDEN2 = ("an", "aus")
ABLEITUNG2 = ("Durchrechnen", "implizit am Plan P_k")
N_RUNDEN, EPS_RUNDEN, ANZAHL_RUNDEN = 60, 0.46, 10            # Tabelle Rundung
K_RUNDEN = (1, 3, 10, 30, 100)
N_ABL, ANZAHL_ABL = 20, 30                                    # Tabelle Ableitung
ZEILEN_ABL = ((1.0, 50), (0.3, 200), (0.1, 200), (0.03, 1000), (0.01, 1000), (0.01, 5000), (0.01, 20000))
S_ABL = 0.5                                                   # Massen der Ableitungstabelle: 0,5/n + 0,5 · Dirichlet(1)
SCHWELLE_ABL = 1e-4                                           # „reicht nicht“: Planfehler über 1e-4
RUNDEN_SCHRANKE_EPS = (0.46, 0.1, 0.03)                       # Schranke geprüft für diese ε/OPT, je Iterationszahl aus K_RUNDEN_SCHRANKE
K_RUNDEN_SCHRANKE = (1, 2, 3, 5, 10, 30, 100, 300)
ANZAHL_SCHRANKE = 10

# Experiment 3: Gleichstand
KOSTEN3 = ("stetig (Punkte im Quadrat)", "ganzzahlig 0 bis 2", "ganzzahlig 0 bis 10", "alle Kosten gleich")
EPS3_STUFEN = (0.02, 0.01, 0.002)                 # ε in Einheiten einer Kostenstufe (ganzzahlig) bzw. des größten Wertes (stetig)
N3, ANZAHL3 = 10, 60
TOL3, MAX_ITER3 = 1e-9, 20000
DEV_GRENZE = 0.02                                 # „fast ganzzahlig“: dev ≤ 0,02, dev = n · max_ij min(P_ij, 1/n − P_ij)

# Experiment 4: Aufwand
N4_STUFEN = (20, 40, 80, 160)
ANZAHL4 = {20: 10, 40: 10, 80: 10, 160: 4}
EPS4_STUFEN = (0.14, 0.07, 0.035)
ZIEL4 = ("Zeilenfehler 1e-6", "Lücke ≤ 1 %")
TOL4 = 1e-6
LUECKE4 = 0.01
RASTER4 = 10                                      # die gerundete Lücke wird alle 10 Iterationen geprüft
MAX_ITER4 = 30000

# Absatz „weiche Ränge“
N_RAENGE = (10, 30, 100)
ANZAHL_RAENGE = {10: 100, 30: 60, 100: 30}
EPS_RAENGE = (3.0, 1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005)

ZWECK_ANSICHT, ZWECK_UNTERLAUF, ZWECK_ABL, ZWECK_RUNDEN, ZWECK_GLEICH, ZWECK_AUFWAND, ZWECK_RAENGE, ZWECK_RICHTUNG, ZWECK_SCHRANKE = 1, 2, 3, 4, 5, 6, 7, 8, 9   # getrennte Zufallsströme

COLOR_F32 = "#c0504d"
COLOR_F64 = "#4c78a8"
COLOR_LOG = "#2F6B65"
COLOR_LP = "#e08a1e"
COLOR_BAND = "rgba(76,120,168,0.18)"

DEFAULTS = {
    "quelle": QUELLE_ZUFALL,
    "n_slider": DEFAULT_N,
    "seed_input": DEFAULT_SEED,
    "eps1": 0.002,
    "tol1": 1e-5,
    "test1": "float32",
    "inst1": 50,
    "eps2": 0.1,
    "k2": 100,
    "runden2": "an",
    "ableitung2": ABLEITUNG2[0],
    "kosten3": KOSTEN3[1],
    "eps3": 0.02,
    "n4": 40,
    "eps4": 0.07,
    "ziel4": ZIEL4[0],
}

# Presets: jede Zeile legt ALLE Regler fest (Schlüssel wie in DEFAULTS); `aktives_preset` vergleicht jeden Regler.
PRESETS = {
    "Von Hand: ein weit entferntes Lager": dict(DEFAULTS, quelle=QUELLE_HAND, eps2=0.01),
    "float32 bei ε = 0,002": dict(DEFAULTS),
    "Abbruch nach 10 Iterationen": dict(DEFAULTS, k2=10),
    "Gleichstand: alle Kosten gleich": dict(DEFAULTS, kosten3=KOSTEN3[3]),
    "Aufwand: n = 80": dict(DEFAULTS, n4=80),
    "Ableitung bei ε/OPT = 0,01": dict(DEFAULTS, eps2=0.01, k2=1000),
}
PRESET_HELP = {
    "Von Hand: ein weit entferntes Lager": "Zwei Lager, zwei Kunden; Lager 2 ist von beiden Kunden gleich weit entfernt. Die Ansicht oben und Experiment 2 rechnen diese Instanz bei kleinem ε; Experiment 1 und Experiment 4 zeigen dazu immer eine kleine Tabelle bzw. den Zeilenfehler 1/(2k+1).",
    "float32 bei ε = 0,002": "Die Voreinstellung: 50 Zufallsinstanzen mit 30 Lagern und Kunden, ε = 0,002 bei größtem Kostenwert 1; die float32-Iteration fällt bei diesem ε in einem Teil der Läufe aus (die Zählung steht in Experiment 1).",
    "Abbruch nach 10 Iterationen": "Experiment 2 mit k = 10 Durchläufen: der Plan ist noch nicht fertig, die Rundung macht ihn zulässig.",
    "Gleichstand: alle Kosten gleich": "Experiment 3 mit lauter gleichen Kosten: jede Kopplung ist optimal, der weiche Plan ist für jedes ε das Produktmaß.",
    "Aufwand: n = 80": "Experiment 4 mit n = 80 Lagern und Kunden: wie viele Matrix-Vektor-Produkte die Iteration bis zum Ziel braucht.",
    "Ableitung bei ε/OPT = 0,01": "Experiment 2 bei ε/OPT = 0,01 und k = 1 000 Durchläufen: Plan und Ableitung der Instanz der Seitenleiste sind dort noch nicht fertig. Die Tabelle von Experiment 2 zeigt dieselbe Rechnung für 30 Instanzen mit n = 20, auch für 5 000 und 20 000 Durchläufe.",
}
