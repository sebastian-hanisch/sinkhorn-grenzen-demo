# Wenn Sinkhorn falsch oder langsam rechnet – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sinkhorn-grenzen-demo.streamlit.app/)**

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an der Annahme an, die die Demo „Sinkhorn: weich statt scharf“ stehen lässt: *Der gezeigte Plan ist exakt gerechnet: kein Unterlauf, kein Gleichstand, und seine Ableitung ist die des Grenzplans.* Vier Experimente heben sie auf: endliche Genauigkeit und Unterlauf (Experiment 1), ein Plan vor dem Ende samt Rundung und Ableitung (Experiment 2), Gleichstand (Experiment 3) und der Aufwand in Matrix-Vektor-Produkten (Experiment 4). Dazu ein Absatz über weiche Permutationen und weiche Ränge ohne eigenes Experiment. Die Annahmen der Demo „Sinkhorn: weich statt scharf“ bleiben bestehen, soweit sie hier nicht genannt sind (kontinuierliche Kosten, gleiche Gesamtmasse, positiver Kern).

## Warum dieses Problem

Ein **Transportplan** (englisch transport plan, auch **Kopplung**, englisch coupling) sagt, wie viel von jedem Lager zu jedem Kunden fährt, sodass die **Randverteilungen** (englisch marginals) stimmen. Die **Sinkhorn-Iteration** berechnet den **weichen Plan** der **Entropie-Regularisierung** (englisch entropic regularization) mit wenigen Zeilen Code: $u = a / (K v)$, $v = b / (K^{\top} u)$ mit $K = \exp(-C/\varepsilon)$. Dass sie „das Richtige“ rechnet, ist eine Annahme. Sie kann auf vier getrennten Wegen etwas anderes liefern, als die Formel sagt: durch **Unterlauf** (englisch underflow) von $\exp(-C/\varepsilon)$, durch einen Abbruch vor der Randverteilung, bei Gleichstand und durch zu wenige Durchläufe für die Ableitung. Die Demo zählt jeden dieser Wege, nennt bei jeder Zahl die Voraussetzungen und zeigt den Aufwand in **Matrix-Vektor-Produkten**, nie in Sekunden.

## Modell

Gegeben sind Angebote $a$ ($n$ Lager, Summe 1), Nachfragen $b$ ($m$ Kunden, Summe 1) und Kosten $C$ (größter Eintrag 1). Der weiche Plan minimiert $\langle C, P\rangle + \varepsilon\,\mathrm{KL}(P \,\|\, a b^{\top})$ über alle Pläne $P \ge 0$ mit Zeilensummen $a$ und Spaltensummen $b$. Formeln und Handrechnungen stehen im Expander „📐 Mathematische Formulierung“ der App.

- **Instanzen (Objekt Ort):** Lager und Kunden gleichverteilt im Einheitsquadrat, Kosten = quadrierter Abstand, durch den größten Wert geteilt (größter Eintrag 1), Massen Dirichlet(1) (unabhängige Exponentialzahlen, durch ihre Summe geteilt). Experiment 1: n = 30 Lager und Kunden, 50 feste Instanzen. Experiment 2: die Instanz der Seitenleiste (n bis 20) und zwei Tabellen: Rundung bei n = 60, ε/OPT = 0,46, 10 Instanzen; Ableitung bei n = 20, 30 Instanzen mit den Massen $0{,}5/n + 0{,}5 \cdot \text{Dirichlet}(1)$ (kleinste Masse mindestens $0{,}5/n$). Experiment 3: n = 10, Massen $1/n$, 60 Instanzen je Zeile, Kosten gleichverteilt in $\{0, \dots, R\}$ oder Punkte im Quadrat oder alle gleich. Experiment 4: n = 20, 40, 80 (je 10 Instanzen) und 160 (4 Instanzen). Absatz über Ränge: gleichverteilte Werte, n = 10, 30, 100 (100, 60, 30 Instanzen).
- **Skalen:** ε/OPT (OPT = LP-Optimum **derselben Instanz** bei größtem Kostenwert 1) in den Experimenten 2 und 4 und im Absatz über Ränge. Der Unterlauf hängt an $C/\varepsilon$, also an ε relativ zum **größten** Kostenwert: Experiment 1 rechnet daher mit festen ε bei größtem Kostenwert 1 (ε/max C) und zeigt ε/OPT daneben. Experiment 3 misst ε in Einheiten einer Kostenstufe (ganzzahlige Kosten) bzw. des größten Wertes (Punkte).
- **Zählweise:** Eine Iteration sind **zwei Matrix-Vektor-Produkte**; der Abbruchtest benutzt das Produkt des nächsten Halbschritts und kostet keins (beim Abbruch ist dieses eine Produkt schon gerechnet und wird nicht mitgezählt); die Rundung kostet $O(nm)$ Rechenoperationen und kein Produkt. In der Log-Domäne ist die Einheit gleich, der Preis je Einheit höher ($n \cdot m$ Exponentialfunktionen je Halbschritt, in der naiven Form $n \cdot m$ Multiplikationen). In Verdikten und Tabellen stehen **Produkte**, nicht Iterationen. Start $u = v = 1$ (Lehrbuchform).
- **Abbruchtest und Toleranz** werden bei jeder Zahl genannt. Der Zeilenfehler ist die L1-Norm von Zeilensumme minus Masse; er wird in dem Datentyp gerechnet, in dem die Iteration läuft (float32: Zeilensumme in float32 gegen die auf float32 gerundete Masse), oder in float64 gegen die exakte Masse. Toleranz: 1e-8 oder 1e-5 in Experiment 1, 1e-9 in Experiment 3 (höchstens 20 000 Iterationen), 1e-6 in Experiment 4; die Zahlen der Experimente sind nur als Band vergleichbar.
- **Zähleinheit** ist die **Instanz** (Median [Minimum; Maximum] über die genannte Zahl von Instanzen, wo nichts anderes steht) beziehungsweise das Matrix-Vektor-Produkt; die Reihen benutzen feste Instanzen mit dem Zufalls-Seed 2026, unabhängig vom Regler „Zufalls-Seed“.

## Methodik

- **Naive Iteration** (`sgz_naiv.py`): im Kern-Bereich in float32 oder float64, mit den Ausgängen „fertig“, „Iterationsgrenze“ und „Ausfall“ (K, K v oder Kᵀ u hat eine Null).
- **Log-Domäne** (`sgz_sinkhorn.py`): die **Kantorovich-Potentiale** $f, g$ (englisch Kantorovich potentials), $P_{ij} = a_i b_j \exp((f_i + g_j - C_{ij})/\varepsilon)$; löst den Unterlauf, nicht die Langsamkeit.
- **Rundung** (`sgz_round.py`): Algorithmus 2 bei Altschuler u. a. (2017): Zeilen herunterskalieren, Spalten herunterskalieren, den Rest als Rang-1-Matrix addieren.
- **Ableitung** (`sgz_deriv.py`): die **implizite Differentiation** (englisch implicit differentiation) am Plan (auch am unfertigen Plan nach $k$ Durchläufen), das **Durchrechnen** der Iteration mit ihrer Ableitung (Vorwärtsmodus) und die zentrale finite Differenz als Gegenprobe.
- **Newton-Verfahren auf dem Dual** (`sgz_newton.py`): der genaue Plan (Randfehler unter 1e-12), mit Fortsetzung in ε; Referenz für die Fehler in Experiment 2 und die Pläne des Absatzes über Ränge.
- **LP-Optimum und Eindeutigkeit** (`sgz_simplex.py`, `sgz_eindeutigkeit.py`): der Transportsimplex mit Basisbaum und der exakte Eindeutigkeitstest, als Kopie mit eigenem Präfix (Kopier-Tests gegen die Quellen). Der Abstand zur LP-Ecke wird mit diesem Löser genannt: bei Gleichstand wählt jeder Löser eine Ecke, geprüft werden dort nur Wert und Eindeutigkeit.
- **Zufall:** Hausgenerator SplitMix64 (`sgz_rng.py`, auf Windows und Linux bitgleich).
- **Vorrechnung:** Die Reihen der Experimente 1, 3 und 4, die Tabellen von Experiment 2 und die Reihen des Absatzes liegen in `tools/vorgerechnet.json` (erzeugt von `tools/vorrechnen.py`, einige Minuten); jede Einstellung ihrer Regler ist vorgerechnet. Nur die Instanz der Seitenleiste (Experiment 2) rechnet live.

## Befunde (gemessen, keine Behauptungen)

Alle gemessenen Zahlen stammen aus `tools/vorgerechnet.json` (Zufalls-Seed 2026); sie gelten **in diesem Modell, bei diesen Parametern** und mit der genannten Zähleinheit und sind keine Gesetze. `tests/test_claims.py` bildet die Tabellen und die Sätze mit Messzahlen dieses Abschnitts aus der Datei; die Handrechnungen stehen in den Testdateien der Tabelle „Zahl → Test“.

### Experiment 1: Unterlauf und Rechengenauigkeit

n = 30, 50 Instanzen, höchstens 3 000 Iterationen, Toleranz 1e-5, Abbruchtest der float32-Läufe in float32 gerechnet. „Ausfall“ heißt: K, K v oder Kᵀ u hat eine Null, die Division scheitert. „Abweichung“ ist die relative Abweichung der Kosten des als fertig gemeldeten Plans von der Log-Domäne (gleiche Toleranz und Iterationsgrenze); die größte steht in Klammern.

| ε | ε/OPT (Median) | float32: fertig | Iterationsgrenze | Ausfall | Abweichung > 1e-6 / 1e-4 / 1e-2 (größte) | float64: fertig / Grenze / Ausfall | Log-Domäne: fertig / Grenze |
|---|---|---|---|---|---|---|---|
| 0,01 | 0,27 | 50 von 50 | 0 | 0 | 0 / 0 / 0 (größte 4,6e-7) | 50 / 0 / 0 | 50 / 0 |
| 0,005 | 0,13 | 50 von 50 | 0 | 0 | 0 / 0 / 0 (größte 2,9e-7) | 50 / 0 / 0 | 50 / 0 |
| 0,003 | 0,08 | 47 von 50 | 1 | 2 | 4 / 0 / 0 (größte 3e-5) | 49 / 1 / 0 | 49 / 1 |
| 0,002 | 0,053 | 30 von 50 | 3 | 17 | 3 / 1 / 0 (größte 0,00014) | 47 / 3 / 0 | 47 / 3 |
| 0,0015 | 0,04 | 14 von 50 | 2 | 34 | 2 / 1 / 0 (größte 0,0016) | 47 / 3 / 0 | 47 / 3 |
| 0,001 | 0,027 | 3 von 50 | 1 | 46 | 2 / 2 / 0 (größte 0,0062) | 43 / 7 / 0 | 43 / 7 |
| 0,0005 | 0,013 | 0 von 50 | 0 | 50 | 0 / 0 / 0 (größte 0) | 22 / 28 / 0 | 22 / 28 |

In float32 fällt bei ε = 0,003 (ε/OPT im Median 0,08) 2 von 50 Läufen aus, bei ε = 0,002 17 von 50; von den 194 als fertig gemeldeten float32-Läufen (alle ε der Tabelle, Toleranz 1e-5, Abbruchtest in float32) weichen 4 um mehr als 1e-4 relativ von der Log-Domäne ab (größte Abweichung 0,62 %). float64 fällt bei ε = 0,0005 in 0 von 50 Läufen aus. Die Log-Domäne fällt in 0 von 350 Läufen aus, aber bei ε = 0,0005 sind in 3 000 Iterationen nur 22 von 50 fertig (Toleranz 1e-5).

Ob ein float32-Lauf sich als fertig meldet, hängt am Abbruchtest und an der Toleranz, nicht nur an ε (alle sieben ε der Tabelle zusammen, 7 · 50 = 350 Läufe je Zeile):

| Toleranz | Abbruchtest der float32-Läufe in | fertig gemeldet | davon Abweichung über 1e-4 | größte Abweichung |
|---|---|---|---|---|
| 1e-8 | float32 | 122 von 350 | 2 | 0,17 % |
| 1e-8 | float64 | 0 von 350 | 0 | – |
| 1e-5 | float32 | 194 von 350 | 4 | 0,62 % |
| 1e-5 | float64 | 194 von 350 | 4 | 0,62 % |

Die Toleranz 1e-8 liegt unter dem Rundungsfehler der Massen in float32: Die Zeilensummen der float32-Iteration sind float32-Zahlen und kommen der exakten Masse $a$ nie näher als deren Rundung auf float32; die Summe $|a - \mathrm{float32}(a)|$ über die 30 Lager liegt in den 50 Instanzen bei 1,4e-8 bis 3e-8 (`tests/test_claims.py`). Mit dem Abbruchtest in float64 (gegen die exakte Masse) erreicht deshalb in dieser Messung keiner der 350 Läufe die Toleranz 1e-8 („Toleranz nicht erreicht“ ist etwas anderes als „nicht konvergiert“); beim Test in float32 (gegen die auf float32 gerundete Masse) kann der Zeilenfehler durch Rundung auf null fallen. Die Zahl der fertig gemeldeten Läufe unterscheidet sich deshalb zwischen den beiden Zeilen für 1e-8. Der Abbruchtest in float32 meldet bei Toleranz 1e-8 Läufe als fertig, die abweichen (2 von 122); bei Toleranz 1e-5 weichen 4 von 194 ab, gleich für beide Testarten: dort liegt die Abweichung an der float32-Rechnung, nicht am Test.

**Handbeispiel „weit entferntes Lager“** (zwei Lager, zwei Kunden, Angebote und Nachfragen je ½, Kosten $\begin{pmatrix} 0 & 1/2 \\ 1 & 1 \end{pmatrix}$; Lager 2 hat von beiden Kunden den größten Abstand 1; Toleranz 1e-8, Test in float64, 3 000 Iterationen):

| ε | größter Eintrag von K in Zeile 2 | float32 | float64 | Log-Domäne: Zeilenfehler nach 50 Durchläufen |
|---|---|---|---|---|
| 1/110 | 1,7e-48 | unterlauf | cap | 0,0099 |
| 1/800 | 0 | unterlauf | unterlauf | 0,0099 |

$\exp(-x)$ ist in float64 ab $x \approx 745{,}1$ genau 0 (darunter noch ein denormaler Wert), in float32 ab $x \approx 104$; für den entferntesten Eintrag ($C = 1$) heißt das $\varepsilon < 1/745 = 0{,}00134$ bzw. $\varepsilon < 1/104 = 0{,}0096$ (`tests/test_naiv.py`). Ein Ausfall tritt erst ein, wenn eine ganze Zeile oder Spalte von K null wird oder ein Produkt null wird: In float64 ist der entfernteste Eintrag bei ε = 0,0005 schon null ($\exp(-2000) = 0$), und dennoch fällt dort in der Tabelle kein Lauf aus.

### Experiment 2: Wie fertig ist der Plan?

Rundung, n = 60, ε/OPT = 0,46, 10 Instanzen, Median [Minimum; Maximum]; „Ende“: Zeilenfehler unter 1e-8.

| Durchläufe k (Produkte) | gerundete Lücke | rohe Lücke | Zeilenfehler (Median) |
|---|---|---|---|
| 1 (2) | +221,8 % [+128,8 %; +316,8 %] | −45,0 % [−51,2 %; −20,9 %] | 0,49 |
| 3 (6) | +134,4 % [+104,6 %; +198,4 %] | −24,0 % [−37,0 %; −5,0 %] | 0,27 |
| 10 (20) | +75,6 % [+47,4 %; +120,7 %] | +3,5 % [−15,8 %; +11,5 %] | 0,091 |
| 30 (60) | +30,5 % [+23,4 %; +54,1 %] | +17,3 % [+11,3 %; +25,2 %] | 0,017 |
| 100 (200) | +20,7 % [+15,0 %; +25,4 %] | +20,6 % [+14,7 %; +25,4 %] | 0,00016 |
| Ende (Zeilenfehler unter 1e-8) | +20,6 % [+14,8 %; +25,4 %] | +20,6 % [+14,8 %; +25,4 %] | – |

Die Schranke der Rundung ⟨C, R⟩ ≤ ⟨C, P⟩ + 2 · max C · (‖P1 − a‖₁ + ‖Pᵀ1 − b‖₁) gilt in 240 von 240 Fällen (10 Instanzen mit n = 20 × 3 Werte ε/OPT × 8 Iterationszahlen); die Randverteilungen des gerundeten Plans stimmen in 240 von 240 (auf 1e-12).

Nach einer Iteration (zwei Produkte) hat der gerundete Plan bei ε/OPT = 0,46 eine Lücke von +222 % (n = 60, Median über 10 Instanzen), nach 100 Iterationen +20,7 %, am Ende (Zeilenfehler unter 1e-8) +20,6 %; die Schranke der Rundung gilt in 240 von 240 Fällen (n = 20). Die Rundung macht den Plan zulässig, sie ersetzt die Iteration nicht.

Ableitung, n = 20, 30 Instanzen (Massen $0{,}5/n + 0{,}5 \cdot \text{Dirichlet}(1)$), Median über die Instanzen, Fehler relativ (Frobenius-Norm) gegen den genauen Plan bzw. dessen Ableitung; „Planfehler über 1e-4“ zählt die Instanzen:

| ε/OPT | k | Zeilenfehler | Planfehler | Ableitung: Durchrechnen | Ableitung: Formel am Plan P_k | Durchrechnen / Zeilenfehler (Median der Quotienten je Instanz) | Planfehler über 1e-4 |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 1,1e-7 | 1,7e-7 | 3,6e-7 | 2e-7 | 4 | 2 von 30 |
| 0,3 | 200 | 6,9e-9 | 1,9e-8 | 4,7e-8 | 1,8e-8 | 8 | 4 von 30 |
| 0,1 | 200 | 0,00096 | 0,0035 | 0,011 | 0,0053 | 11 | 25 von 30 |
| 0,03 | 1 000 | 7e-5 | 0,00038 | 0,0013 | 0,00065 | 25 | 19 von 30 |
| 0,01 | 1 000 | 0,017 | 0,11 | 0,22 | 0,12 | 18 | 29 von 30 |
| 0,01 | 5 000 | 1e-9 | 7,3e-9 | 3,3e-7 | 1,1e-8 | 62 | 10 von 30 |
| 0,01 | 20 000 | 1,5e-14 | 2e-13 | 3,1e-13 | 3e-13 | 17 | 4 von 30 |

Die Formel am genauen Plan stimmt mit der zentralen finiten Differenz (mit Richardson-Verbesserung, h = 1e-2 · ε und 5e-3 · ε) überein: relativer Fehler im Median 3,2e-11, größter 0,00023 (150 Vergleiche; bei kleinem ε/OPT ist die Ableitung klein gegen das Rauschen der genauen Pläne).

Bei ε/OPT = 0,03 und 1 000 Durchläufen liegt der Zeilenfehler im Median bei 7e-5, der Ableitungsfehler (Durchrechnen) bei 0,0013 (im Median der Instanzen 25-mal so groß wie der Zeilenfehler der jeweiligen Instanz); bei ε/OPT = 0,01 reichen 5 000 Durchläufe in 10 von 30 Instanzen nicht (Planfehler über 1e-4), auch 20 000 nicht in 4 von 30. Die Genauigkeit der Ableitung folgt der des Plans: Im Auslauf (Zeilen der Tabelle mit einem Median des Ableitungsfehlers (Durchrechnen) zwischen 1e-10 und 1e-2) liegt das Durchrechnen in der typischen Instanz (Median der Quotienten je Instanz) um den Faktor 1,1 bis 2,8 über dem Rechenweg „implizit am unfertigen Plan“ (n = 20, 30 Instanzen); die Mediane der Tabelle unterscheiden sich in der Zeile ε/OPT = 0,01, k = 5 000 um den Faktor 29, die Quotienten streuen zwischen den Instanzen stark.

Der Ableitungsfehler (Durchrechnen) liegt im Median in jeder Zeile der Tabelle über dem Zeilenfehler des Abbruchtests (Faktor in der vorletzten Spalte) und über dem Planfehler; er folgt der Genauigkeit des Plans, und der Rechenweg ändert das Bild nur im Auslauf um den Faktor im Satz oben.

### Experiment 3: Gleichstand

n = 10, Massen $1/n$, 60 Instanzen je Zeile; Eindeutigkeit des Optimums: exakter Test; „ganzzahlig“ heißt dev ≤ 0,02 mit $\mathrm{dev} = n \cdot \max_{ij} \min(P_{ij}, 1/n - P_{ij})$ (0 bei einer Permutationsmatrix, 0,5 bei einem Eintrag halb/halb); die LP-Ecke kommt vom eigenen Transportsimplex; Iteration in der Log-Domäne bis Zeilenfehler 1e-9, höchstens 20 000 Iterationen.

| Kosten | ε | mehrere optimale Pläne | davon nicht ganzzahlig (dev > 0,02) | dev im Median (mehrere) | eindeutiges Optimum | davon ganzzahlig | Zeilenfehler 1e-9 erreicht |
|---|---|---|---|---|---|---|---|
| stetig (Punkte im Quadrat) | 0,02 | 0 von 60 | 0 | – | 60 von 60 | 0 | 58 von 60 |
| stetig (Punkte im Quadrat) | 0,01 | 0 von 60 | 0 | – | 60 von 60 | 0 | 47 von 60 |
| stetig (Punkte im Quadrat) | 0,002 | 0 von 60 | 0 | – | 60 von 60 | 2 | 4 von 60 |
| ganzzahlig 0 bis 2 | 0,02 | 60 von 60 | 60 | 0,484 | 0 von 60 | 0 | 12 von 60 |
| ganzzahlig 0 bis 2 | 0,01 | 60 von 60 | 60 | 0,484 | 0 von 60 | 0 | 14 von 60 |
| ganzzahlig 0 bis 2 | 0,002 | 60 von 60 | 60 | 0,484 | 0 von 60 | 0 | 17 von 60 |
| ganzzahlig 0 bis 10 | 0,02 | 36 von 60 | 36 | 0,500 | 24 von 60 | 24 | 0 von 60 |
| ganzzahlig 0 bis 10 | 0,01 | 36 von 60 | 36 | 0,500 | 24 von 60 | 24 | 0 von 60 |
| ganzzahlig 0 bis 10 | 0,002 | 36 von 60 | 36 | 0,500 | 24 von 60 | 24 | 0 von 60 |
| alle Kosten gleich | 0,02 | 60 von 60 | 60 | 0,100 | 0 von 60 | 0 | 60 von 60 |
| alle Kosten gleich | 0,01 | 60 von 60 | 60 | 0,100 | 0 von 60 | 0 | 60 von 60 |
| alle Kosten gleich | 0,002 | 60 von 60 | 60 | 0,100 | 0 von 60 | 0 | 60 von 60 |

Kosten „ganzzahlig 0 bis 2“, ε = 0,02, n = 10, 60 Instanzen: 60 haben mehrere optimale Pläne; bei 60 davon ist der Sinkhorn-Plan nicht ganzzahlig (dev über 0,02, Median 0,484; L1-Abstand zur gefundenen Ecke des LP im Median 1,01). Der Zeilenfehler 1e-9 wird in 20 000 Iterationen in 12 von 60 Instanzen erreicht (davon 12 bei mehreren Optima).

Kosten „ganzzahlig 0 bis 10“, ε = 0,002, n = 10, 60 Instanzen: 36 haben mehrere optimale Pläne; bei 36 davon ist der Sinkhorn-Plan nicht ganzzahlig (dev über 0,02, Median 0,500; L1-Abstand zur gefundenen Ecke des LP im Median 0,40). Bei den 24 Instanzen mit eindeutigem Optimum ist der Plan nach höchstens 20 000 Iterationen in 24 ganzzahlig (dev höchstens 0,02). Der Zeilenfehler 1e-9 wird in 20 000 Iterationen in 0 von 60 Instanzen erreicht (davon 0 bei mehreren Optima).

Kosten „stetig (Punkte im Quadrat)“, ε = 0,002, n = 10, 60 Instanzen: keine hat mehrere optimale Pläne. Bei den 60 Instanzen mit eindeutigem Optimum ist der Plan nach höchstens 20 000 Iterationen in 2 ganzzahlig (dev höchstens 0,02). Der Zeilenfehler 1e-9 wird in 20 000 Iterationen in 4 von 60 Instanzen erreicht.

Bei „alle Kosten gleich“ ist der weiche Plan für jedes ε das Produktmaß (jede Kopplung ist optimal, dev = 1/n; `tests/test_gleichstand_und_raenge.py`); in dieser Zeile steht eine einzige Instanz (konstante Kosten 0,5, Massen 1/n), 60-mal gerechnet: „60 von 60“ bedeutet dasselbe Ergebnis einmal, keine Stichprobe (`tests/test_claims.py`). Für ε → 0 strebt der weiche Plan bei Gleichstand gegen den Plan mit größter Entropie auf der Fläche der optimalen Pläne (Sharify u. a. 2013, Satz 2.3, für Zuordnungen); bei den hier gemessenen ε ist er in jeder Zeile der Tabelle in allen Instanzen mit mehreren optimalen Plänen nicht ganzzahlig (die Spalte „davon nicht ganzzahlig“ ist gleich der Zahl der Instanzen mit mehreren Optima), seine Lage in der Fläche wird nicht geprüft. Die Zählungen zu „ganzzahlig“ gelten für den Plan nach höchstens 20 000 Iterationen, auch dort, wo der Zeilenfehler 1e-9 nicht erreicht wird. Die Kostenlücke ist dabei klein, der Plan weit weg (Handbeispiel unten).

### Experiment 4: Aufwand in Produkten, nicht in Sekunden

Log-Domäne, Abbruch bei Zeilenfehler unter 1e-6 (höchstens 30 000 Iterationen); die gerundete Lücke wird alle 10 Iterationen geprüft und nur bis zum Zeilenfehler 1e-6 gesucht (Produkte daher in Vielfachen von 20); n = 160 mit 4 Instanzen, sonst 10:

| ε/OPT | n | Produkte bis Zeilenfehler 1e-6 (Median [Minimum; Maximum]) | Instanzen mit Lücke ≤ 1 % | Produkte bis Lücke ≤ 1 % (Median [Minimum; Maximum]) | n² · Produkte bis Zeilenfehler 1e-6 (Median) |
|---|---|---|---|---|---|
| 0,14 | 20 | 275 [124; 1 714] | 0 von 10 | – | 110 000 |
| 0,14 | 40 | 644 [468; 1 418] | 0 von 10 | – | 1 030 400 |
| 0,14 | 80 | 1 426 [976; 2 926] | 0 von 10 | – | 9 126 400 |
| 0,14 | 160 | 2 601 [2 042; 2 828] | 0 von 4 | – | 66 585 600 |
| 0,07 | 20 | 669 [218; 3 988] | 7 von 10 | 400 [220; 1 580] | 267 600 |
| 0,07 | 40 | 1 189 [944; 2 752] | 6 von 10 | 810 [640; 1 480] | 1 902 400 |
| 0,07 | 80 | 2 834 [2 058; 6 030] | 5 von 10 | 3 140 [1 380; 3 400] | 18 137 600 |
| 0,07 | 160 | 5 507 [3 882; 6 496] | 3 von 4 | 3 680 [2 840; 4 500] | 140 979 200 |
| 0,035 | 20 | 1 373 [400; 6 626] | 10 von 10 | 590 [200; 3 320] | 549 200 |
| 0,035 | 40 | 2 339 [1 712; 5 014] | 10 von 10 | 1 100 [640; 2 520] | 3 742 400 |
| 0,035 | 80 | 5 884 [4 092; 18 580] | 10 von 10 | 2 700 [1 720; 5 120] | 37 657 600 |
| 0,035 | 160 | 10 879 [6 682; 11 240] | 4 von 4 | 5 440 [3 040; 5 940] | 278 502 400 |

Je Halbierung von ε/OPT wächst der Median der Produkte bis Zeilenfehler 1e-6 um den Faktor 1,85 bis 2,43 (alle n, 8 Vergleiche). Faktor der Median-Produkte bei vierfachem n (n = 20 → 80) je ε/OPT: ε/OPT = 0,14: 5,19 (Zeilenfehler 1e-6), – (Lücke ≤ 1 % nicht erreicht); ε/OPT = 0,07: 4,24 (Zeilenfehler 1e-6), 7,85 (Lücke ≤ 1 %); ε/OPT = 0,035: 4,29 (Zeilenfehler 1e-6), 4,58 (Lücke ≤ 1 %). Die Spannen in den Klammern zeigen die Streuung zwischen den Instanzen; die Faktoren des Ziels „Zeilenfehler 1e-6“ stammen aus Medianen über je 10 Instanzen, und ihre Unterschiede zwischen den drei ε/OPT liegen innerhalb dieser Streuung (die Bootstrap-Bereiche der drei Faktoren überlappen sich, `tests/test_claims.py`). Die Faktoren des Ziels „Lücke ≤ 1 %“ stammen dagegen aus Medianen über die Instanzen, die das Ziel erreichen: bei ε/OPT = 0,07 sind das 7 (n = 20) und 5 (n = 80) von je 10, bei 0,035 jeweils 10. Sie sind auswahlverzerrt (erreicht wird das Ziel nur dort, wo der Endplan schon die Lücke ≤ 1 % hat); auf derselben Teilmenge von 7 und 5 Instanzen ist der Faktor des Ziels „Zeilenfehler 1e-6“ 5,3 statt 4,2, der Faktor 7,85 ist also mit 4,24 nicht vergleichbar.

Bei ε/OPT = 0,07 braucht der Zeilenfehler 1e-6 im Median 669 / 1 189 / 2 834 / 5 507 Matrix-Vektor-Produkte für n = 20 / 40 / 80 / 160 (10 / 10 / 10 / 4 Instanzen), bei n × 4 (n = 20 → 80) den Faktor 4,2 bis 5,2 (je nach ε/OPT 5,2, 4,2, 4,3 bei ε/OPT 0,14, 0,07, 0,035; die Unterschiede zwischen den drei ε/OPT liegen innerhalb der Streuung der Instanzen); „Lücke ≤ 1 %“ erreichen bei ε/OPT = 0,07 nur 7 / 6 / 5 / 3 der Instanzen, weil bei diesem ε/OPT die Lücke des gerundeten Endplans (Zeilenfehler 1e-6) bei den übrigen schon über 1 % liegt; mehr Iterationen ändern das nicht. Verglichen wird nur, wie die Zahl der Produkte mit n und 1/ε wächst; ein Vergleich mit den Schritten eines Simplexverfahrens ist hier nicht gezeigt (Verweis auf die Demo zu PDLP und auf Dong u. a. 2020).

Der Zeilenfehler des Handbeispiels „weit entferntes Lager“ ist nach $k$ Durchläufen $1/(2k+1)$, gerechnet für ε = 1/110 und ε = 1/800 mit demselben Wert (bei ε ≥ 0,05 gilt die Formel nicht mehr, der Zeilenfehler fällt dort schneller); ein Zeilenfehler von 1e-9 bräuchte demnach etwa $5 \cdot 10^8$ Durchläufe ($1/(2k+1) = 10^{-9}$ gibt $k \approx 5 \cdot 10^8$), solange der Eintrag $K_{12}$ nicht zählt (aus der Formel gerechnet, nicht gelaufen). Zur Einordnung der Einheit verweist die Demo auf die Demo zu PDLP (dieselbe Zähleinheit, zwei Produkte je Iteration) und auf Dong u. a. (2020); sie zeigt keinen Vergleich mit einem Simplexverfahren.

### Weiche Permutation, weiche Ränge (Absatz, kein Experiment)

Eine Zuordnung von n Dingen auf n Plätze ist ein Transportplan mit Massen 1/n; ihre Ecken sind die Permutationen. n Werte $x_i$, gleichverteilt in (0, 1), werden auf feste Zielstellen $s_j = (j - \tfrac12)/n$ verteilt (Kosten $(x_i - s_j)^2$); der weiche Rang ist $\tilde r_i = n \sum_j P_{ij}\, j$. Nur **konvergierte** Instanzen (Newton-Verfahren auf dem Dual) zählen mit; die nicht konvergierten sind nicht zufällig verteilt, die Anteile bei kleinem ε/OPT sind **auswahlverzerrt**.

**n = 10** (100 Instanzen):

| ε/OPT | konvergiert | fast ganzzahlig | dev im Median | mittlerer Betrag des Rangfehlers | Rang 1 (Mittel) | Rang 10 (Mittel) | richtig gerundete Ränge |
|---|---|---|---|---|---|---|---|
| 3 | 100 von 100 | 0 von 100 | 0,472 | 0,49 | +0,78 | −0,87 | 61 % |
| 1 | 100 von 100 | 0 von 100 | 0,475 | 0,31 | +0,41 | −0,45 | 81 % |
| 0,3 | 100 von 100 | 1 von 100 | 0,463 | 0,18 | +0,18 | −0,19 | 94 % |
| 0,1 | 100 von 100 | 13 von 100 | 0,358 | 0,10 | +0,07 | −0,08 | 99 % |
| 0,05 | 100 von 100 | 26 von 100 | 0,227 | 0,06 | +0,04 | −0,05 | 100 % |
| 0,02 | 95 von 100 | 42 von 95 | 0,044 | 0,03 | +0,01 | −0,02 | 100 % |
| 0,01 | 100 von 100 | 69 von 100 | 0,001 | 0,01 | +0,01 | −0,01 | 100 % |
| 0,005 | 98 von 100 | 84 von 98 | 0,000 | 0,00 | +0,00 | −0,01 | 100 % |

**n = 30** (60 Instanzen):

| ε/OPT | konvergiert | fast ganzzahlig | dev im Median | mittlerer Betrag des Rangfehlers | Rang 1 (Mittel) | Rang 30 (Mittel) | richtig gerundete Ränge |
|---|---|---|---|---|---|---|---|
| 3 | 60 von 60 | 0 von 60 | 0,392 | 0,70 | +1,52 | −1,62 | 47 % |
| 1 | 60 von 60 | 0 von 60 | 0,472 | 0,47 | +0,84 | −0,90 | 63 % |
| 0,3 | 60 von 60 | 0 von 60 | 0,494 | 0,31 | +0,42 | −0,46 | 80 % |
| 0,1 | 60 von 60 | 0 von 60 | 0,493 | 0,21 | +0,21 | −0,25 | 91 % |
| 0,05 | 60 von 60 | 0 von 60 | 0,487 | 0,15 | +0,14 | −0,16 | 96 % |
| 0,02 | 58 von 60 | 0 von 58 | 0,455 | 0,09 | +0,07 | −0,09 | 99 % |
| 0,01 | 53 von 60 | 3 von 53 | 0,394 | 0,06 | +0,04 | −0,06 | 99 % |
| 0,005 | 37 von 60 | 8 von 37 | 0,318 | 0,04 | +0,02 | −0,03 | 100 % |

**n = 100** (30 Instanzen):

| ε/OPT | konvergiert | fast ganzzahlig | dev im Median | mittlerer Betrag des Rangfehlers | Rang 1 (Mittel) | Rang 100 (Mittel) | richtig gerundete Ränge |
|---|---|---|---|---|---|---|---|
| 3 | 30 von 30 | 0 von 30 | 0,231 | 0,95 | +2,88 | −3,13 | 36 % |
| 1 | 30 von 30 | 0 von 30 | 0,360 | 0,65 | +1,61 | −1,78 | 49 % |
| 0,3 | 30 von 30 | 0 von 30 | 0,492 | 0,44 | +0,83 | −0,91 | 65 % |
| 0,1 | 30 von 30 | 0 von 30 | 0,498 | 0,31 | +0,43 | −0,47 | 81 % |
| 0,05 | 30 von 30 | 0 von 30 | 0,498 | 0,25 | +0,28 | −0,30 | 88 % |
| 0,02 | 30 von 30 | 0 von 30 | 0,498 | 0,17 | +0,15 | −0,15 | 95 % |
| 0,01 | 30 von 30 | 0 von 30 | 0,495 | 0,12 | +0,09 | −0,08 | 98 % |
| 0,005 | 27 von 30 | 0 von 27 | 0,483 | 0,08 | +0,05 | −0,05 | 99 % |

**Gleichstände** (n = 10, zwei Paare gleicher Werte, 100 Instanzen): die Zeilen der gleichen Werte sind identisch (größter Unterschied zweier Zeilen eines Paares höchstens 0), der weiche Rang nähert sich dem Mittelrang: größte Abweichung im Median 0,35 bei ε/OPT = 1 (100 von 100 konvergiert), 0,03 bei ε/OPT = 0,1 (100 von 100 konvergiert), 1e-11 bei ε/OPT = 0,01 (84 von 100 konvergiert). Das ist die weiche Seite des Gleichstands aus Experiment 3. Die Summe der weichen Ränge ist exakt n(n+1)/2 (Spaltensummen 1/n).

Bei 30 Werten ist bis ε/OPT = 0,01 in 3 der 53 konvergierten Instanzen der Plan fast eine Permutation (dev ≤ 0,02); bei 10 Werten sind es 69 der 100 konvergierten Instanzen. Weiche Ränge liegen am Rand zur Mitte hin verschoben (Rang 1 zu hoch, Rang n zu niedrig), und der Fehler wächst mit n; gleiche Werte bekommen identische Zeilen und den Mittelrang. Gilt für gleichverteilte Werte, quadrierten Abstand, Massen 1/n und die gezählten Instanzen.

Blondel u. a. (2020) bezeichnen Sinkhorn-Sortieren wegen Kosten und Speicher als teuer ($O(T n m)$ Rechenarbeit, ab n = 1 000 mit Rückwärtsrechnung bzw. 2 000 nicht mehr im Speicher) und schlagen ein billigeres Verfahren vor; die Demo rechnet mit n bis 100 und zeigt keine Sekunden.

### Handbeispiele von Hand gerechnet

**Rundung** ($a = (\tfrac34, \tfrac14)$, $b = (\tfrac14, \tfrac34)$, $C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, der Plan nach einem Durchlauf bei $\varepsilon = 1/\ln 3$ ist $P = \begin{pmatrix} 9/40 & 3/8 \\ 1/40 & 3/8 \end{pmatrix}$ mit den Zeilensummen $(\tfrac35, \tfrac25)$): $x = (1, \tfrac58)$, $F = \begin{pmatrix} 9/40 & 3/8 \\ 1/64 & 15/64 \end{pmatrix}$, $y = (1, 1)$, Fehlmengen $(\tfrac{3}{20}, 0)$ und $(\tfrac{3}{320}, \tfrac{45}{320})$, gerundet $R = \begin{pmatrix} 15/64 & 33/64 \\ 1/64 & 15/64 \end{pmatrix}$ mit den Randverteilungen $a$ und $b$. Kosten $\tfrac{17}{32}$ gegen $\tfrac25$ vor der Rundung; das LP-Optimum ist $\tfrac12$, die rohe Lücke $-20\,\%$ wird gerundet zu $+\tfrac1{16} = +6{,}25\,\%$. Zweites Beispiel: $P = \begin{pmatrix} 3/10 & 1/10 \\ 1/10 & 3/10 \end{pmatrix}$ (Gesamtmasse $\tfrac45$) auf $a = b = (\tfrac12, \tfrac12)$ mit $C = \begin{pmatrix} 1/2 & 1 \\ 1 & 1/2 \end{pmatrix}$ gibt $R = \begin{pmatrix} 7/20 & 3/20 \\ 3/20 & 7/20 \end{pmatrix}$, Kosten $\tfrac12$ vor, $\tfrac{13}{20}$ nach der Rundung. Beide in Brüchen in `tests/test_round.py`, dort auch gegen eine unabhängige Fassung in Brüchen.

**Weit entferntes Lager** (Zeilenfehler): der Grenzkern $K = \begin{pmatrix} 1 & 0 \\ \delta & \delta \end{pmatrix}$ gibt in Brüchen bis $k = 30$ den Eintrag $P_{11} = k/(2k+1)$ und den Zeilenfehler $1/(2k+1)$ (`tests/test_naiv.py`); Franklin-Lorenz-Schranke (nach Sharify u. a. 2013, Satz 3.1, dort als Proposition bezeichnet): je Durchlauf höchstens $\kappa^2$ mit $1 - \kappa^2 = 4\exp(-0{,}25/\varepsilon)$, bei ε = 1/110 etwa $4{,}6 \cdot 10^{-12}$ – eine Schranke für positive Matrizen, keine gemessene Rate.

**Gleichstand** ($a = b = (\tfrac13, \tfrac13, \tfrac13)$, $C = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$): zwei optimale Permutationen, OPT = 0; der weiche Plan bei ε = 0,05 hat im $2 \times 2$-Block überall $\tfrac16$ (dev = ½), den L1-Abstand $\tfrac23$ zu jeder der beiden Ecken und Kosten $2 \cdot 10^{-9}$: die Kostenlücke ist klein, der Plan weit weg (`tests/test_gleichstand_und_raenge.py`). Bei lauter gleichen Kosten ist der Plan das Produktmaß, sein L1-Abstand zu jeder der zwei Permutationen ist 1.

**Ableitung** (3 × 3, $a = (0{,}5; 0{,}3; 0{,}2)$, $b = (0{,}4; 0{,}4; 0{,}2)$, ε = 0,2, $C = \begin{pmatrix} 0{,}1 & 0{,}5 & 1 \\ 0{,}6 & 0{,}2 & 0{,}7 \\ 1 & 0{,}8 & 0{,}3 \end{pmatrix}$, Richtung $D = \begin{pmatrix} 1 & -1 & 0{,}5 \\ 0{,}2 & 0{,}3 & -0{,}7 \\ -0{,}4 & 0{,}9 & 0{,}1 \end{pmatrix}$): die implizite Formel stimmt mit der zentralen finiten Differenz (h = 1e-6) auf relativ 1e-6 überein, die Zeilen- und Spaltensummen von $\mathrm{d}P$ sind null (1e-12), und nach 3, 6 und 12 Durchläufen liegt der Ableitungsfehler über dem Zeilenfehler des Abbruchtests und über dem halben Planfehler (`tests/test_deriv.py`). Beim Plan der Gleichstand-Bahn $C(t) = \begin{pmatrix} t & 1-t \\ 1-t & t \end{pmatrix}$ ist bei $t = \tfrac12$ die Ableitung $\mathrm{d}P_{11} = -1/(4\varepsilon)$ exakt.

**Weiche Ränge** (n = 5, $x = (0{,}40; 0{,}45; 0{,}50; 0{,}55; 0{,}60)$, ε/OPT = 1): weiche Ränge $(1{,}806; 2{,}319; 3; 3{,}681; 4{,}194)$, Rang 1 zu hoch und Rang 5 zu niedrig, gleich stark; die Summe ist exakt $n(n+1)/2 = 15$ (`tests/test_gleichstand_und_raenge.py`).

### Zahl → Test

| Zahl der README | Test |
|---|---|
| Tabellen und Sätze der Experimente 1 bis 4 und des Absatzes (aus `tools/vorgerechnet.json`) | `tests/test_claims.py`: `test_tabellen_und_saetze_der_readme` |
| Neu gerechnete Reihen (Band) | `test_vorgerechnete_reihen_stimmen_mit_der_neuen_rechnung_ueberein` |
| Schwellen 745,1 und 104, Auflösung von float32, Zeilenfehler 1/(2k+1) (kleines ε), Franklin-Lorenz-Schranke | `tests/test_naiv.py` |
| Rundung der Massen in float32 (1,4e-8 bis 3e-8), Zählungen zur Abbruchtest-Tabelle, Bootstrap der Faktoren, Teilmengen des Ziels „Lücke ≤ 1 %“, „alle Kosten gleich“ als eine Instanz | `tests/test_claims.py` |
| Handbeispiele der Rundung (Brüche) | `tests/test_round.py` |
| Handbeispiele zu Ableitung, Gleichstand und Rängen | `tests/test_deriv.py`, `tests/test_gleichstand_und_raenge.py` |
| Zahl der Instanzen der Orakel, Testzahl | `test_orakel_zahlen`, `test_testzahl_der_readme` |

Die Reihen werden nicht jedes Mal vollständig neu gerechnet: der Test rechnet Teilmengen (die ersten Instanzen) neu und vergleicht als Band, weil Gleitkomma-Summen (besonders in float32) und die Wahl der Ecke auf anderen Plattformen abweichen können.

## Ehrliche Grenzen

- **Gültigkeit:** Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Instanz; Matrix-Vektor-Produkte). Ein Median über 4 bis 100 Instanzen ist kein Gesetz; andere Zufalls-Seeds geben andere Zahlen im Band.
- **Annahmen, die hier stehen bleiben:** (1) *Beide Randverteilungen haben dieselbe Gesamtmasse, der Kern K = exp(−C/ε) ist überall positiv, die Kosten sind kontinuierlich verteilt (außer in Experiment 3), und ε ist ein frei gewählter Regler.* Nullen im Kern, verschiedene Gesamtmassen und ein aus Beobachtungen zurückgerechnetes ε sind hier nicht untersucht (Matrix abgleichen, Furness-Verfahren; ungleiche Mengen). (2) *Sinkhorn braucht weniger Schritte als ein exaktes Verfahren:* Diese Demo zählt nur die Produkte der Iteration und zeigt keinen Vergleich mit einem Simplexverfahren; die Zählung gibt keinen Anlass, die Aussage zu behaupten.
- **Abbruchtest und Toleranz bestimmen die Zählung fertiger float32-Läufe:** die Zahl der fertig gemeldeten Läufe und der Abweichungen gilt für die genannte Kombination (Experiment 1), für n = 30 und 50 Instanzen; ob die Schwellen mit n wandern, ist nicht untersucht.
- **Ableitung:** nur Vorwärtsmodus, nur n = 20 in den Tabellen; Rückwärtsmodus und Speicherbedarf sind nicht gemessen. Die Zahl der Instanzen, die bei ε/OPT = 0,01 nach 5 000 oder 20 000 Durchläufen nicht reichen, ist eine Zählung über 30 Instanzen, kein Median.
- **Gleichstand:** nur n = 10 und die gezeigten Kosten; „konvergiert“ heißt Zeilenfehler 1e-9 in 20 000 Iterationen; bei ganzzahligen Kosten bis 10 erreicht keine der 60 Instanzen den Zeilenfehler 1e-9 in 20 000 Iterationen, auch nicht die mit eindeutigem Optimum und ganzzahligem Plan; was die Konvergenz dort bremst, ist hier nicht untersucht.
- **Ränge:** nur konvergierte Instanzen (Auswahlverzerrung bei kleinem ε/OPT), gleichverteilte Werte, quadrierter Abstand; keine Aussage über andere Verteilungen.
- **Nicht gemessen:** Greenkhorn, ε-Skalierung und beschleunigte Verfahren; Sekunden (die Demo zählt Produkte); Verteilungen mit Atomen.
- **Nicht mit echten Mausklicks im Browser durchgeklickt:** AppTest und Tests sehen die Plotly-Abbildungen nicht.
- **Literatur** (nur so zitiert, wie gelesen): Altschuler, Weed, Rigollet (2017), arXiv 1705.09634 (Algorithmus 2, Lemma 7); Sharify, Gaubert, Grigori (2013), arXiv 1104.3830 (Satz 2.3, Satz 3.1, dort als Proposition bezeichnet); Dong u. a. (2020), arXiv 2005.01182 (Auszug gelesen); Blondel u. a. (2020) zu Kosten und Speicher des Sinkhorn-Sortierens; Peyré, Cuturi (2019), *Computational Optimal Transport* (Kapitel 1 bis 3 und Anfang von Kapitel 4 gelesen). Unterlauf, Log-Domäne, Rundung und implizite Differentiation sind Lehrstoff; die Demo macht sie mit einer Zähleinheit vorführbar.

## Verifikation

- **Kern:** Handbeispiele in Brüchen und in Zahlen (Rundung, Zeilenfehler 1/(2k+1), Schwellen des Gleitkommaformats, Ableitung, Gleichstand, weiche Ränge); Einheiten einzeln (Kern, Unterlauf-Erkennung, Abbruchtest in zwei Datentypen, Rundung, Ableitung); Randfälle (Unterlauf in der ersten Zeile mit Meldung statt NaN, alle Kosten gleich, k = 0, n = 1, Nullzeile bei der Rundung, Toleranz unter der Auflösung von float32).
- **Orakel** (`tests/test_oracle_sinkhorn_grenzen.py`, ein anderer Rechenweg): die naive Iteration als Schleife nach dem Lehrbuch gegen die Log-Domäne und gegen `naive_iteration` nach jedem k (12 Instanzen, 7 Werte von k, relative Toleranz 1e-9), die float32-Fassung gegen dieselbe Schleife in float32; die Rundung auf 40 Instanzen (zulässig, Schranken der Kosten und des L1-Abstands) und in Brüchen; die implizite Formel gegen eine finite Differenz mit Plänen aus dem Dual mit L-BFGS (6 Instanzen), das Durchrechnen gegen die finite Differenz der Iteration; das LP-Optimum gegen `linprog` (HiGHS, 30 Instanzen mit n ≠ m), `networkx` und OR-Tools (15 ganzzahlige Instanzen, Abweichung 0); die Eindeutigkeit gegen ein zweites LP auf der optimalen Fläche (25 Instanzen).
- **Eigenschaften:** Randverteilungen des gerundeten Plans, Rundung idempotent und ohne Wirkung auf einen zulässigen Plan, Summe der weichen Ränge $n(n+1)/2$, gleiche Werte geben identische Zeilen.
- **Kopier-Tests:** `sgz_simplex.py`, `sgz_eindeutigkeit.py`, `sgz_sinkhorn.py` und `sgz_newton.py` sind Kopien der Demo „Sinkhorn: weich statt scharf“ (Präfix `snk_` → `sgz_`), `sgz_rng.py` der Vorlage des Portfolios; die Tests vergleichen den Quelltext, wo die Quelle neben dieser Demo liegt.
- **Verdikte und Zahlen:** Verdikte sind reine Funktionen (`sgz_evaluation.py`) und an handgesetzten Reihen geprüft (`tests/test_evaluation.py`); die Tabellen und Messsätze dieser README bildet `tests/test_claims.py` aus der Datei, die Handrechnungen stehen in den Testdateien der Tabelle „Zahl → Test“; je Preset ist die erwartete Aussage geprüft.
- **Presets und Oberfläche:** vollständig, Werte innerhalb der Grenzen, Standard-Preset gleich Standardeinstellung; Permalink begrenzt und rastet ein; AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, alle Optionen, jedes Preset, alle Maxima zugleich), Layout und Footer.
- **Testzahl:** 216 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `requirements-dev.txt` | Laufzeit plus pytest und die Bibliotheken der Gegenproben (scipy, networkx, OR-Tools) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, Plotly |
| `sgz_naiv.py` | naive Iteration im Kern-Bereich (float32, float64), Status, Abbruchtest in zwei Datentypen |
| `sgz_sinkhorn.py` | Sinkhorn-Iteration in der Log-Domäne (Kopie, siehe Kopier-Tests) |
| `sgz_newton.py` | Newton-Verfahren auf dem Dual (Kopie) |
| `sgz_round.py` | Rundung auf zulässige Randverteilungen, Schranke |
| `sgz_deriv.py` | Ableitung: implizite Formel, Durchrechnen, zentrale finite Differenz |
| `sgz_simplex.py`, `sgz_eindeutigkeit.py` | Transportsimplex und Eindeutigkeitstest (Kopien) |
| `sgz_rng.py` | Hausgenerator SplitMix64 (Kopie der Vorlage) |
| `sgz_scenario.py` | Instanzen (Zufall, ganzzahlige Kosten, alle gleich, Werte für Ränge), Handbeispiele |
| `sgz_evaluation.py` | Reihen, Zählungen, Verdikte, Tabellenzeilen, Vorrechnung |
| `sgz_vorgerechnet.py` | liest `tools/vorgerechnet.json` |
| `sgz_constants.py`, `sgz_presets.py`, `sgz_layout.py`, `sgz_visualization.py` | Konstanten und Presets, Permalink, fester Kopf, Plotly-Abbildungen (Achsen gesperrt) |
| `tools/vorrechnen.py`, `tools/vorgerechnet.json` | Vorrechnung der Reihen (einige Minuten) und ihr Ergebnis |
| `tests/` | Handbeispiele und Einheiten, Orakel, Eigenschaften, Claims, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Keine Sekunden:** Die Demo zählt Produkte (und Multiplikationen bzw. Exponentialfunktionen); eine Wandzeitmessung fehlt absichtlich, weil sie Rechner, Bibliothek und Version mitmisst.
- **Kein Vergleich mit einem Simplexverfahren:** Die Zähleinheiten sind nicht gleichartig (ein Simplexschritt ist keine Matrix-Vektor-Multiplikation); die Demo verweist auf die Demo zu PDLP und auf Dong u. a. (2020).
- **Kein PDF-Export, keine Instanzen mit n ≠ m in der App** (der Löser kann sie, die Regler bieten n = m).
- **Keine Sinkhorn-Distanz und keine Sinkhorn-Divergenz:** Die Wörter kommen hier nicht vor; sie haben mehrere Definitionen und gehören zu einer anderen Demo der Reihe.
- **Keine Bibliothek zur Laufzeit außer NumPy und Plotly** (`scipy`, `networkx`, OR-Tools nur in den Tests).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
