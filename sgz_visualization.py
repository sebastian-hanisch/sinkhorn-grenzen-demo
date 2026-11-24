"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen von Zellen sind Annotationen (kein Text-Trace, keine texttemplate)."""

import math

import numpy as np
import plotly.graph_objects as go

import sgz_constants as C

MAX_ZELLEN_MIT_ZAHL = 144      # bis 12 x 12 steht die Zahl in der Zelle
SCHWELLE_EINTRAG = 1e-6
LOG_BODEN = 1e-17


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.25), separators=",.")
    fig.update_xaxes(exponentformat="power")             # Zehnerpotenzen statt SI-Vorsilben (1p, 10n, 100µ) an den logarithmischen Achsen
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def _titel(fig, titel):
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return fig


def zahl_text(x: float, gesamt: float) -> str:
    """Zellzahl: Anteil an der Gesamtmasse in Prozent mit Komma (unter 0,05 % als „< 0,1“)."""
    p = 100.0 * x / gesamt
    if p < 0.05:
        return "&lt;0,1"
    return f"{p:.1f}".replace(".", ",")


def build_plan_matrix(plan, a, b, titel="", hoehe=380):
    """Der Plan als Matrix (Zeilen = Lager, Spalten = Kunden, Farbe und Zahl = Anteil der Gesamtmasse in Prozent); Randverteilungen in den Achsenbeschriftungen.
    Einträge unter 1e-6 bleiben leer."""
    plan = np.asarray(plan, float)
    n, m = plan.shape
    gesamt = float(plan.sum())
    z = np.where(plan > SCHWELLE_EINTRAG, 100.0 * plan / gesamt, np.nan)
    xs = [f"K{j + 1}" for j in range(m)]
    ys = [f"L{i + 1}" for i in range(n)]
    zmax = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale=[[0, "#e3efed"], [1, C.COLOR_LOG]], showscale=False, xgap=1, ygap=1, zmin=0, zmax=zmax,
                               hovertemplate="%{y} → %{x}: %{z:.3f} %<extra></extra>"))
    if n * m <= MAX_ZELLEN_MIT_ZAHL:
        beschriftungen = []        # alle Annotationen auf einmal setzen (add_annotation prüft jedes Mal die ganze Liste: quadratisch langsam)
        for i in range(n):
            for j in range(m):
                if plan[i, j] > SCHWELLE_EINTRAG:
                    hell = 100.0 * plan[i, j] / gesamt > 0.55 * zmax
                    beschriftungen.append(dict(x=xs[j], y=ys[i], text=zahl_text(plan[i, j], gesamt), showarrow=False,
                                               font=dict(size=11 if n <= 4 else 8, color="white" if hell else "#1b3b37")))
        fig.update_layout(annotations=beschriftungen)
    fig.update_xaxes(side="top", tickmode="array", tickvals=xs, ticktext=[f"K{j + 1}<br>{100.0 * b[j]:.1f}".replace(".", ",") if m <= 12 else f"K{j + 1}" for j in range(m)],
                     tickfont=dict(size=9))
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=ys, ticktext=[f"L{i + 1} · {100.0 * a[i]:.1f}".replace(".", ",") if n <= 12 else f"L{i + 1}" for i in range(n)],
                     tickfont=dict(size=9))
    fig = _base(fig, height=hoehe, legend=False)
    if titel:
        fig.update_layout(title=dict(text=titel, font=dict(size=13)), margin=dict(l=10, r=10, t=95, b=10))
    return fig


def _log_y(fig, lo, hi, titel):
    fig.update_yaxes(type="log", range=[math.log10(lo), math.log10(hi)], title_text=titel)


def build_unterlauf(zeilen):
    """Experiment 1, links: Zahl der Ausfälle (Unterlauf) je ε für float32 und float64 und der nicht fertigen Läufe der Log-Domäne (Iterationsgrenze); rechts: die float32-Läufe nach Status."""
    xs = [f"{z['eps']:g}".replace(".", ",") for z in zeilen]
    N = zeilen[0]["anzahl"]
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=xs, y=[z["f32"]["unterlauf"] for z in zeilen], name="float32: Unterlauf", marker_color=C.COLOR_F32))
    f1.add_trace(go.Bar(x=xs, y=[z["f64"]["unterlauf"] for z in zeilen], name="float64: Unterlauf", marker_color=C.COLOR_F64))
    f1.add_trace(go.Bar(x=xs, y=[z["log"]["cap"] for z in zeilen], name="Log-Domäne: nicht fertig", marker_color=C.COLOR_LOG))
    f1.update_layout(barmode="group")
    f1.update_xaxes(title_text="ε (größter Kostenwert = 1)", type="category")
    f1.update_yaxes(title_text=f"Läufe von {N}", range=[0, N * 1.05])
    _titel(f1, "Ausfälle und nicht fertige Läufe")
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=xs, y=[z["f32"]["ok"] - z["f32"]["ueber"][1] for z in zeilen], name="fertig, Abweichung ≤ 1e-4", marker_color="#8fb8a8"))
    f2.add_trace(go.Bar(x=xs, y=[z["f32"]["ueber"][1] for z in zeilen], name="fertig, Abweichung > 1e-4", marker_color="#e08a1e"))
    f2.add_trace(go.Bar(x=xs, y=[z["f32"]["cap"] for z in zeilen], name="Iterationsgrenze", marker_color="#bbbbbb"))
    f2.add_trace(go.Bar(x=xs, y=[z["f32"]["unterlauf"] for z in zeilen], name="Unterlauf", marker_color=C.COLOR_F32))
    f2.update_layout(barmode="stack")
    f2.update_xaxes(title_text="ε (größter Kostenwert = 1)", type="category")
    f2.update_yaxes(title_text=f"float32-Läufe von {N}", range=[0, N * 1.05])
    _titel(f2, "float32: Ausgang der Läufe")
    return _base(f1), _base(f2)


def build_fehler_k(z, k_marke, runden):
    """Experiment 2, links: Fehler gegen die Zahl der Durchläufe k (beide Achsen logarithmisch): Zeilenfehler, Planfehler, Ableitungsfehler (Durchrechnen und Formel am Plan P_k);
    rechts: Lücke der Kosten gegen k (rohe Kosten des Plans, gerundeter Plan, genauer Plan als Linie)."""
    ks = z["k"]

    def nicht_null(werte):
        return [max(w, LOG_BODEN) for w in werte]

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=ks, y=nicht_null(z["zeilenfehler"]), mode="lines+markers", name="Zeilenfehler (Abbruchtest)", line=dict(color="#888888")))
    f1.add_trace(go.Scatter(x=ks, y=nicht_null(z["plan"]), mode="lines+markers", name="Planfehler", line=dict(color=C.COLOR_LOG)))
    if not z["abl_ref_null"]:                       # gegen eine Referenzableitung nahe null sagt der relative Fehler nichts
        f1.add_trace(go.Scatter(x=ks, y=nicht_null(z["durch"]), mode="lines+markers", name="Ableitung: Durchrechnen", line=dict(color=C.COLOR_F32)))
        f1.add_trace(go.Scatter(x=ks, y=nicht_null(z["impl"]), mode="lines+markers", name="Ableitung: Formel am Plan P_k", line=dict(color=C.COLOR_F64, dash="dash")))
    f1.add_vline(x=k_marke, line=dict(color="#666666", dash="dot"))
    f1.update_xaxes(type="log", title_text="Durchläufe k")
    _log_y(f1, 1e-16, 1e3, "relativer Fehler")
    _titel(f1, "Fehler gegen die Zahl der Durchläufe")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=ks, y=[100 * w for w in z["roh"]], mode="lines+markers", name="rohe Kosten des Plans P_k", line=dict(color="#888888")))
    if runden:
        f2.add_trace(go.Scatter(x=ks, y=[100 * w for w in z["gerundet"]], mode="lines+markers", name="gerundeter Plan", line=dict(color=C.COLOR_LP)))
    f2.add_hline(y=100 * z["luecke_ende"], line=dict(color=C.COLOR_LOG, dash="dash"), annotation_text="genauer Plan", annotation_position="top right")
    f2.add_vline(x=k_marke, line=dict(color="#666666", dash="dot"))
    f2.update_xaxes(type="log", title_text="Durchläufe k")
    f2.update_yaxes(title_text="Kostenlücke (⟨C, P⟩ − OPT) / OPT in %")
    _titel(f2, "Kosten gegen die Zahl der Durchläufe")
    return _base(f1, 380), _base(f2, 380)


def build_gleichstand(r):
    """Experiment 3: Verteilung von dev (Abstand des Plans von einer Permutationsmatrix) getrennt nach Instanzen mit mehreren Optima und mit eindeutigem Optimum; Linie bei 0,02."""
    eind = np.array(r["eindeutig"], bool)
    dev = np.array(r["dev"])
    fig = go.Figure()
    bins = dict(start=0.0, end=0.52, size=0.02)
    if (~eind).any():
        fig.add_trace(go.Histogram(x=dev[~eind], xbins=bins, name="mehrere optimale Pläne", marker_color=C.COLOR_F32, opacity=0.8))
    if eind.any():
        fig.add_trace(go.Histogram(x=dev[eind], xbins=bins, name="eindeutiges Optimum", marker_color=C.COLOR_F64, opacity=0.8))
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=C.DEV_GRENZE, line=dict(color="#666666", dash="dot"), annotation_text="0,02", annotation_position="top")
    fig.update_xaxes(title_text="dev = n · max min(P_ij, 1/n − P_ij): 0 = Permutation, 0,5 = Eintrag halb/halb", range=[0, 0.52])
    fig.update_yaxes(title_text=f"Instanzen von {len(dev)}")
    _titel(fig, "Wie weit ist der Plan von einer Permutation?")
    return _base(fig, 340)


def build_aufwand(zeilen_je_eps, ziel):
    """Experiment 4: Produkte bis zum Ziel gegen n, ein Linienzug je ε/OPT (Median, Spanne als Fehlerbalken), beide Achsen logarithmisch. `zeilen_je_eps`: {ε/OPT: Liste aggregiere_aufwand}."""
    fig = go.Figure()
    farben = [C.COLOR_F64, C.COLOR_LOG, C.COLOR_F32]
    for farbe, (eps, zeilen) in zip(farben, zeilen_je_eps.items()):
        z = [x for x in zeilen if x["produkte"] is not None]
        if not z:
            continue
        fig.add_trace(go.Scatter(x=[x["n"] for x in z], y=[x["produkte"][0] for x in z], mode="lines+markers", name=f"ε/OPT = {str(eps).replace('.', ',')}", line=dict(color=farbe),
                                 error_y=dict(type="data", symmetric=False, array=[x["produkte"][2] - x["produkte"][0] for x in z], arrayminus=[x["produkte"][0] - x["produkte"][1] for x in z],
                                              color="rgba(120,120,120,0.5)")))
    fig.update_xaxes(type="log", title_text="Lager = Kunden n", tickvals=list(C.N4_STUFEN), ticktext=[str(n) for n in C.N4_STUFEN])
    fig.update_yaxes(type="log", title_text="Matrix-Vektor-Produkte (Median, Spanne)")
    _titel(fig, f"Produkte bis „{ziel}“")
    return _base(fig, 380)


def build_zwei_mal_zwei(ks, fehler):
    """Handbeispiel „weit entferntes Lager“: Zeilenfehler gegen k (gerechnet) und die Formel 1/(2k+1)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=fehler, mode="markers", name="gerechnet (Log-Domäne)", marker=dict(color=C.COLOR_LOG, size=9)))
    fig.add_trace(go.Scatter(x=ks, y=[1.0 / (2 * k + 1) for k in ks], mode="lines", name="1/(2k+1)", line=dict(color="#888888", dash="dash")))
    fig.update_xaxes(type="log", title_text="Durchläufe k")
    fig.update_yaxes(type="log", title_text="Zeilenfehler")
    _titel(fig, "Zeilenfehler des Handbeispiels")
    return _base(fig, 300)


def build_raenge(tabs):
    """Absatz „weiche Ränge“: links der Anteil der konvergierten Instanzen mit dev ≤ 0,02, rechts der mittlere Betrag des Rangfehlers, je ε/OPT und n."""
    farben = {10: C.COLOR_F64, 30: C.COLOR_LOG, 100: C.COLOR_F32}
    f1, f2 = go.Figure(), go.Figure()
    for n, tab in tabs.items():
        z = [x for x in tab if x["konvergiert"] > 0]
        f1.add_trace(go.Scatter(x=[x["eps_rel"] for x in z], y=[x["anteil_fast_ganz"] for x in z], mode="lines+markers", name=f"n = {n}", line=dict(color=farben[n])))
        f2.add_trace(go.Scatter(x=[x["eps_rel"] for x in z], y=[x["betrag"] for x in z], mode="lines+markers", name=f"n = {n}", line=dict(color=farben[n])))
    for fig, titel, y in ((f1, "Anteil fast ganzzahliger Pläne (dev ≤ 0,02)", "Anteil der konvergierten Instanzen"), (f2, "Mittlerer Betrag des Rangfehlers", "Rangeinheiten")):
        fig.update_xaxes(type="log", title_text="ε/OPT", autorange="reversed")
        fig.update_yaxes(title_text=y)
        _titel(fig, titel)
    return _base(f1, 320), _base(f2, 320)
