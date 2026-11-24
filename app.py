"""Wenn Sinkhorn falsch oder langsam rechnet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Der weiche Plan der Sinkhorn-Iteration wird als exakt gerechnet gezeigt. Vier Experimente prüfen, wo das nicht stimmt: Unterlauf und Rechengenauigkeit, ein Plan vor dem Ende
(Rundung und Ableitung), Gleichstand und der Aufwand in Matrix-Vektor-Produkten. Dazu ein Absatz über weiche Permutationen und weiche Ränge.
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sgz_constants as C
import sgz_evaluation as E
from sgz_layout import dunkles_design, kopf_css
from sgz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from sgz_visualization import (
    build_aufwand,
    build_fehler_k,
    build_gleichstand,
    build_plan_matrix,
    build_raenge,
    build_unterlauf,
    build_zwei_mal_zwei,
)

st.set_page_config(page_title="Wenn Sinkhorn falsch oder langsam rechnet – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Der Lauf der Instanz wird gerechnet (Iteration mit Ableitung bis 20 000 Durchläufe) ...")
def _ansicht2(quelle, n, seed, eps2):
    return E.ansicht2(E.Settings(quelle=quelle, n=n, seed=seed, eps2=eps2))


@st.cache_data(show_spinner=False)
def _zwei_mal_zwei():
    ks = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    return ks, [E.zeilenfehler_zwei_mal_zwei(k) for k in ks]


@st.cache_data(show_spinner=False)
def _handbeispiel():
    return E.handbeispiel_unterlauf()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sgz_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')
with st.container(key="kopf"):
    st.title("🧮 Wenn Sinkhorn falsch oder langsam rechnet")
st.markdown(
    """
Die **Sinkhorn-Iteration** berechnet einen **weichen Transportplan** (englisch transport plan, auch **Kopplung**), dessen Zeilensummen die Angebote der **Lager** und dessen Spaltensummen die Nachfragen der
**Kunden** sind (die **Randverteilungen**, englisch marginals). Wie weich er ist, steuert ε; die Demo misst ε relativ zum LP-Optimum (ε/OPT), nur in Experiment 1 relativ zum größten Kostenwert, weil dort
der **Unterlauf** (englisch underflow) an C/ε hängt. Die vier Experimente prüfen eine Annahme: *Sinkhorn rechnet, was die Formel sagt.* Sie gilt nicht in jeder Rechnung: Der Plan kann
durch **Unterlauf** ausfallen oder durch den Abbruchtest falsch aussehen, vor dem Ende ist er noch unfertig (auch seine Ableitung), bei Gleichstand ist er nicht die Ecke, die ein LP-Löser liefert, und die
Zahl der **Matrix-Vektor-Produkte** wächst mit 1/ε und mit n.

Weiter unten: **❓ Kernfragen**, die vier **🔬 Experimente**, ein Absatz über weiche Permutationen und weiche Ränge, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme**
und **📐 Mathematische Formulierung** (mit Handrechnungen in Brüchen). Gezählt wird in **Matrix-Vektor-Produkten**: eine Iteration sind zwei, nie in Sekunden.
"""
)

with st.expander("So rechnet die Iteration, und wo sie falsch oder langsam wird", expanded=True):
    st.markdown(
        """
1. **Zwei Formen.** Im **Kern-Bereich** (naive Form) wird $K = \\exp(-C/\\varepsilon)$ gebildet und abwechselnd $u = a / (K v)$, $v = b / (K^{\\top} u)$ gerechnet. In der **Log-Domäne** (englisch log-domain) laufen
   statt $u, v$ die **Kantorovich-Potentiale** $f, g$ (Preise); jeder Schritt ist ein Log-Sum-Exp. Beide starten mit $u = v = 1$ und geben nach $k$ Durchläufen denselben Plan, solange nichts unterläuft (in float64 bis auf Rundungsfehler; in float32 weichen bei kleinem ε die Kosten fertig gemeldeter Läufe um bis zu 0,62 % ab, Experiment 1, Toleranz 1e-5).
2. **Unterlauf.** $\\exp(-x)$ ist in float64 ab $x \\approx 745{,}1$ genau 0 (darunter noch ein denormaler Wert), in float32 ab $x \\approx 104$. Für den entferntesten Eintrag ($C = \\max C = 1$) heißt das $\\varepsilon < 1/745 = 0{,}00134$ bzw. $\\varepsilon < 1/104 = 0{,}0096$.
   Ausfälle beginnen früher, wenn eine ganze Zeile oder Spalte von $K$ null wird oder ein Produkt $K v$ null wird. Die Log-Domäne löst den Unterlauf, nicht die Langsamkeit.
3. **Abbruch und Rundung.** Nach endlich vielen Durchläufen hat der Plan die Randverteilungen nur ungefähr. Die **Rundung** (englisch rounding; Algorithmus 2 bei Altschuler u. a.) macht ihn zulässig: Zeilen und Spalten
   herunterskalieren, den Rest als Rang-1-Matrix addieren.
4. **Ableitung.** Der weiche Plan ist glatt in den Kosten; seine Ableitung folgt aus der **impliziten Differentiation** (englisch implicit differentiation) am Plan oder durch Mitrechnen der Iteration. Beide Wege
   haben am unfertigen Plan einen Fehler.
5. **Gleichstand.** Gibt es mehrere optimale Pläne, strebt der weiche Plan für ε → 0 gegen den Plan mit größter Entropie auf der Fläche der optimalen Pläne (Sharify u. a. 2013, Satz 2.3, für Zuordnungen), nicht gegen eine Ecke; bei den hier gemessenen ε ist er nicht ganzzahlig, seine Lage in der Fläche wird nicht geprüft.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧩 Instanz der Ansicht")
    st.selectbox("Instanz", C.QUELLEN, key="quelle",
                 help="Die Ansicht „Ergebnis“ und Experiment 2 rechnen eine Zufallsinstanz oder das Handbeispiel mit einem weit entfernten Lager (Angebote und Nachfragen je ½, Kosten [[0, ½], [1, 1]]). Die Reihen der Experimente 1, 3 und 4 haben feste Instanzen.")
    st.slider("Lager = Kunden (n)", *bounds("n_slider"), key="n_slider",
              help="Zahl der Lager; ebenso viele Kunden. Wirkt auf die Zufallsinstanz der Ansicht und von Experiment 2 (das Handbeispiel hat n = 2; die Tabellen von Experiment 2 haben feste n).")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.header("🔬 Experiment 1: Unterlauf")
    st.select_slider("ε (größter Kostenwert = 1)", options=C.EPS1_STUFEN, key="eps1", format_func=lambda x: E.de_g(x, 2),
                     help="ε in Einheiten des größten Kostenwerts (max C = 1, also ε/max C). Die Tabelle zeigt alle Stufen; die Zeile dieser Stufe steht darunter einzeln, mit ε/OPT.")
    st.select_slider("Toleranz des Abbruchtests", options=C.TOL1_STUFEN, key="tol1", format_func=E.tol_text,
                     help="Zeilenfehler (L1) unter dieser Toleranz heißt fertig. 1e-8 liegt unter dem Rundungsfehler der Massen in float32.")
    st.radio("Abbruchtest der float32-Läufe gerechnet in", C.TEST1_STUFEN, key="test1",
             help="float32: die Zeilensumme in float32 gegen die auf float32 gerundete Masse (der Fehler kann durch Rundung auf null fallen). float64: in float64 gegen die exakte Masse.")
    st.select_slider("Instanzen", options=C.INST1_STUFEN, key="inst1", help="So viele der 50 festen Instanzen (n = 30) zählen mit (Zähleinheit ist die Instanz).")
    st.header("🔬 Experiment 2: Wie fertig ist der Plan?")
    st.select_slider("Weichheit ε/OPT", options=C.EPS2_STUFEN, key="eps2", format_func=E.eps_text,
                     help="ε geteilt durch das LP-Optimum der Instanz der Ansicht. Klein heißt: viele Durchläufe bis zum fertigen Plan.")
    st.select_slider("Durchläufe k", options=C.K2_STUFEN, key="k2", format_func=E.ganz,
                     help="Der Plan nach k Durchläufen (2k Matrix-Vektor-Produkte), mit Rundung und Ableitung. Die Kurven zeigen alle k.")
    st.radio("Rundung", C.RUNDEN2, key="runden2", horizontal=True, help="Der gerundete Plan erscheint in der Ansicht und im rechten Diagramm, wenn die Rundung an ist.")
    st.radio("Ableitung", C.ABLEITUNG2, key="ableitung2",
             help="Durchrechnen: die Iteration mit ihrer Ableitung (Vorwärtsmodus). Implizit: die Formel am Plan nach k Durchläufen. Der gewählte Weg steht in der Kennzahl und im Verdikt; beide stehen im Diagramm.")
    st.header("🔬 Experiment 3: Gleichstand")
    st.selectbox("Kosten", C.KOSTEN3, key="kosten3",
                 help="n = 10, Massen 1/n. Punkte im Quadrat (quadrierter Abstand, größter Wert 1), ganzzahlige Kosten gleichverteilt in 0 bis 2 bzw. 0 bis 10, oder alle Kosten gleich (dann ist jede Kopplung optimal).")
    st.select_slider("ε", options=C.EPS3_STUFEN, key="eps3", format_func=lambda x: E.de_g(x, 2),
                     help="ε in Einheiten einer Kostenstufe (ganzzahlig) bzw. des größten Kostenwerts (Punkte). Bei „alle Kosten gleich“ ist der Plan für jedes ε das Produktmaß; der Regler ändert dort nichts.")
    st.header("🔬 Experiment 4: Aufwand")
    st.select_slider("Lager = Kunden (n)", options=C.N4_STUFEN, key="n4",
                     help="Die Tabelle zeigt alle n, diese Stufe ist in der Abbildung und in der Zeile unter der Tabelle hervorgehoben (n = 160 mit 4, sonst 10 Instanzen).")
    st.select_slider("Weichheit ε/OPT", options=C.EPS4_STUFEN, key="eps4", format_func=E.eps_text, help="Die drei Stufen unterscheiden sich um den Faktor 2. Die Tabelle gilt für diese Stufe.")
    st.radio("Ziel", C.ZIEL4, key="ziel4",
             help="Zeilenfehler 1e-6 (Iteration mit Abbruchtest) oder gerundeter Plan mit höchstens 1 % Kostenlücke gegen das LP-Optimum (alle 10 Iterationen geprüft).")

sync_query_params({k: st.session_state[k] for k in C.DEFAULTS})

settings = E.settings_aus_zustand(st.session_state)
zwei = settings.quelle == C.QUELLE_HAND

z2 = _ansicht2(settings.quelle, settings.n, settings.seed, settings.eps2)
unterlauf = E.unterlauf_v()
zeilen1 = E.reihe_aggregiert1(unterlauf, settings.tol1, settings.test1, settings.inst1)
runden_r, schranke_r, abl_r = E.runden_v(), E.schranke_v(), E.ableitung_v()
abl_tab = E.tabelle_ableitung(abl_r)
gleich = {(art, e): E.gleichstand_v(art, e) for art in C.KOSTEN3 for e in C.EPS3_STUFEN}
auf = {(n, e): E.aufwand_v(n, e) for n in C.N4_STUFEN for e in C.EPS4_STUFEN}
raenge = {n: E.raenge_v(n) for n in C.N_RAENGE}
raenge_tab = {n: E.tabelle_raenge(raenge[n]) for n in C.N_RAENGE}

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: der Plan nach k Durchläufen")
inst, i = z2["inst"], z2["k"].index(settings.k2)
st.caption(f"{inst.name}: {inst.n} Lager, {inst.m} Kunden; OPT = {E.de_g(z2['opt'], 4)} (LP-Optimum bei größtem Kostenwert 1); ε = {E.de_g(z2['eps'], 3)}, also ε/OPT = {E.eps_text(settings.eps2)}. "
           f"Referenz: {z2['ref_art']}.")
durch_oder_impl = z2["durch"][i] if settings.ableitung2 == C.ABLEITUNG2[0] else z2["impl"][i]
m1, m2, m3 = st.columns(3)
m1.metric("Zeilenfehler (L1)", E.de_g(z2["zeilenfehler"][i], 2), delta=f"nach {settings.k2} Durchläufen = {E.ganz(2 * settings.k2)} Produkten", delta_color="off")
m2.metric("Planfehler (relativ)", E.de_g(z2["plan"][i], 2), delta="gegen den genauen Plan", delta_color="off")
if z2["abl_ref_null"]:
    m3.metric("Ableitungsfehler (relativ)", "–", delta="Referenzableitung praktisch null", delta_color="off")
else:
    m3.metric("Ableitungsfehler (relativ)", E.de_g(durch_oder_impl, 2), delta=f"{settings.ableitung2}; Faktor zum Zeilenfehler {E.de(durch_oder_impl / z2['zeilenfehler'][i], 1) if z2['zeilenfehler'][i] > 0 else '–'}", delta_color="off")
m4, m5, m6 = st.columns(3)
m4.metric("Lücke des gerundeten Plans", E.prozent_vz(z2["gerundet"][i], 2) if settings.runden2 == "an" else "–", delta="Rundung an" if settings.runden2 == "an" else "Rundung aus", delta_color="off")
m5.metric("Rohe Kostenlücke von P_k", E.prozent_vz(z2["roh"][i], 2), delta="(⟨C, P_k⟩ − OPT) / OPT, ohne Rundung", delta_color="off")
m6.metric("Lücke des genauen Plans", E.prozent_vz(z2["luecke_ende"], 2), delta="der Plan nach unendlich vielen Durchläufen", delta_color="off")
if not z2["ref_ok"]:
    st.warning("Das Newton-Verfahren für den genauen Plan ist bei diesem ε/OPT nicht konvergiert; die Fehler sind gegen den Plan nach 20 000 Durchläufen gerechnet.")
kind, text = E.verdikt_ansicht2(z2, settings)
getattr(st, kind)(text)
c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(build_plan_matrix(z2["plaene"][settings.k2], inst.a, inst.b, titel=f"Plan nach k = {settings.k2} Durchläufen (Zahl = Anteil der Gesamtmasse in %)"), width="stretch", key="plan_k")
with c2:
    if settings.runden2 == "an":
        st.plotly_chart(build_plan_matrix(z2["gerundete_plaene"][settings.k2], inst.a, inst.b, titel="gerundeter Plan (Randverteilungen exakt)"), width="stretch", key="plan_rund")
    else:
        st.info("Rundung ist aus: der Plan links hat die Randverteilungen erst ungefähr.")
with c3:
    st.plotly_chart(build_plan_matrix(z2["plan_ref"], inst.a, inst.b, titel="genauer Plan"), width="stretch", key="plan_ref")
st.caption("Zeilen = Lager (Angebot in % in der Beschriftung), Spalten = Kunden (Nachfrage in %). Leere Zellen: Einträge unter 1e-6.")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**Stimmt das Ergebnis?**")
kind, text = E.verdikt_unterlauf(zeilen1)
getattr(st, kind)(text)
gz = E.aggregiere_gleichstand(gleich[(settings.kosten3, settings.eps3)])
kind, text = E.verdikt_gleichstand(gz)
getattr(st, kind)(text)
st.markdown("**Wie viele Produkte kostet es?**")
kind, text = E.verdikt_aufwand(auf)
getattr(st, kind)(text)
st.markdown("**Wie genau ist die Ableitung?**")
kind, text = E.verdikt_stimmt_ableitung(abl_tab)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Unterlauf und Rechengenauigkeit")
N1 = zeilen1[0]["anzahl"]
st.caption(f"n = {C.N1} Lager und Kunden, {N1} feste Instanzen (Zufalls-Seed der Reihen {C.SERIEN_SEED}), höchstens {E.ganz(C.MAX_ITER1)} Iterationen, Toleranz {E.tol_text(settings.tol1)} (Zeilenfehler L1); "
           f"Abbruchtest der float32-Läufe in {settings.test1} gerechnet. „Ausfall“ heißt: K, K v oder Kᵀ u hat eine Null, die Division scheitert. „Abweichung“ ist die relative Abweichung der Kosten des "
           f"als fertig gemeldeten Plans von der Log-Domäne (gleiche Toleranz und Iterationsgrenze).")
if not E.toleranz_erreichbar_float32(settings.tol1) and settings.test1 == "float64":
    lo1, hi1 = E.masse_rundung_float32()
    st.warning(f"Die Toleranz 1e-8 liegt unter dem Rundungsfehler der Massen in float32 (Summe |a − float32(a)| über die {C.N1} Lager: {E.de_g(lo1, 2)} bis {E.de_g(hi1, 2)} in den {C.ANZAHL1} Instanzen): "
               "Mit dem Abbruchtest in float64 erreicht in dieser Messung kein float32-Lauf sie. „Iterationsgrenze“ heißt dort „Toleranz nicht erreicht“, nicht „nicht konvergiert“.")
st.markdown(E.md_tabelle(E.KOPF_UNTERLAUF, [E.zellen_unterlauf(z) for z in zeilen1]))
ze = E.zeile1(zeilen1, settings.eps1)
st.markdown(f"**Bei ε = {E.de_g(settings.eps1, 2)}** (ε/max C = {E.de_g(settings.eps1, 2)}, ε/OPT im Median {E.eps_text(ze['eps_opt'])}; Toleranz {E.tol_text(settings.tol1)}, Test in {settings.test1}): "
            f"float32 fällt in {E.zaehlung(ze['f32']['unterlauf'], N1)} Läufen aus, {ze['f32']['ok']} melden sich als fertig (davon {ze['f32']['ueber'][1]} mit Abweichung über 1e-4), "
            f"{ze['f32']['cap']} erreichen die Iterationsgrenze; float64 fällt in {E.zaehlung(ze['f64']['unterlauf'], N1)} Läufen aus; die Log-Domäne ist in {ze['log']['ok']} von {N1} Läufen fertig.")
g1, g2 = build_unterlauf(zeilen1)
uc1, uc2 = st.columns(2)
with uc1:
    st.plotly_chart(g1, width="stretch", key="unterlauf_ausfall")
with uc2:
    st.plotly_chart(g2, width="stretch", key="unterlauf_ausgang")
kind, text = E.verdikt_unterlauf(zeilen1)
getattr(st, kind)(text)
st.markdown("**Handbeispiel: ein weit entferntes Lager** (zwei Lager, zwei Kunden, Angebote und Nachfragen je ½, Kosten [[0, ½], [1, 1]]; Toleranz 1e-8, Test in float64). Lager 2 hat von beiden Kunden den größten Abstand:")
hb = _handbeispiel()
st.markdown(E.md_tabelle(E.KOPF_HAND, [E.zellen_hand(h) for h in hb]))
st.caption("„cap“ und „unterlauf“ wie oben. Der Zeilenfehler der Log-Domäne ist für diese beiden ε praktisch gleich 1/(2k+1) (Experiment 4); ε = 1/110 lässt exp(-110) = 1,7e-48 in float32 zu null werden.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie fertig ist der Plan?")
st.caption(f"Die Diagramme zeigen die Instanz der Seitenleiste, die Tabellen darunter feste Reihen: Rundung bei n = {runden_r['n']}, ε/OPT = {E.eps_text(runden_r['eps_rel'])}, {runden_r['anzahl']} Instanzen; Ableitung bei n = {abl_r['n']}, {abl_r['anzahl']} Instanzen "
           f"(Massen 0,5/n + 0,5 · Dirichlet(1)). Zufalls-Seed der Reihen {C.SERIEN_SEED}. Fehler relativ (Frobenius-Norm) gegen den genauen Plan bzw. dessen Ableitung.")
f1, f2 = build_fehler_k(z2, settings.k2, settings.runden2 == "an")
ec1, ec2 = st.columns(2)
with ec1:
    st.plotly_chart(f1, width="stretch", key="fehler_k")
with ec2:
    st.plotly_chart(f2, width="stretch", key="luecke_k")
st.markdown("**Rundung** (gerundete Lücke gegen das LP-Optimum, Median [Minimum; Maximum] über die Instanzen):")
st.markdown(E.md_tabelle(E.KOPF_RUNDEN, E.zeilen_runden(runden_r)))
st.caption(E.schranke_text(schranke_r))
kind, text = E.verdikt_plan(runden_r, schranke_r)
getattr(st, kind)(text)
st.markdown("**Ableitung** (n = 20; Median über die Instanzen; „über 1e-4“ zählt die Instanzen mit Planfehler über 1e-4):")
st.markdown(E.md_tabelle(E.KOPF_ABLEITUNG, [E.zellen_ableitung(z) for z in abl_tab]))
st.caption(E.fd_text(abl_r))
kind, text = E.verdikt_ableitung(abl_tab)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Gleichstand")
r3 = gleich[(settings.kosten3, settings.eps3)]
st.caption(f"n = {C.N3}, Massen 1/n, {C.ANZAHL3} feste Instanzen je Zeile (Zufalls-Seed der Reihen {C.SERIEN_SEED}); Kosten: {settings.kosten3}. Eindeutigkeit des Optimums: exakter Test (ein Pfeilgraph über die optimalen Zellen), "
           f"die LP-Ecke kommt vom eigenen Transportsimplex (Regel „steilste“; bei Gleichstand wählt jeder Löser eine). Iteration in der Log-Domäne bis Zeilenfehler {E.tol_text(C.TOL3)}, höchstens {E.ganz(C.MAX_ITER3)} Iterationen.")
if settings.kosten3 == C.KOSTEN3[3]:
    st.info("Bei „alle Kosten gleich“ ist der weiche Plan für jedes ε das Produktmaß (jede Kopplung ist optimal); der Regler ε ändert hier nichts.")
st.plotly_chart(build_gleichstand(r3), width="stretch", key="gleichstand")
zeilen3 = [E.aggregiere_gleichstand(gleich[(settings.kosten3, e)]) for e in C.EPS3_STUFEN]
st.markdown(E.md_tabelle(E.KOPF_GLEICH, [E.zellen_gleichstand(z) for z in zeilen3]))
kind, text = E.verdikt_gleichstand(gz)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Aufwand in Produkten, nicht in Sekunden")
st.caption(f"Log-Domäne, Zeilenfehler unter {E.tol_text(C.TOL4)} bzw. gerundete Lücke höchstens 1 % (alle {C.RASTER4} Iterationen geprüft, gesucht bis zum Zeilenfehler {E.tol_text(C.TOL4)}); Instanzen mit Dirichlet(1)-Massen "
           f"(n = 160: 4 Instanzen, sonst 10; Zufalls-Seed der Reihen {C.SERIEN_SEED}). Eine Iteration sind zwei Matrix-Vektor-Produkte; die Rechenarbeit n² · Produkte ist zugleich die Zahl der Multiplikationen der naiven Form "
           "und der Exponentialfunktionen der Log-Domäne (die Exponentialfunktion kostet je Auswertung mehr; die Demo zeigt keine Sekunden).")
je_eps = {e: [E.aggregiere_aufwand(auf[(n, e)], settings.ziel4) for n in C.N4_STUFEN] for e in C.EPS4_STUFEN}
st.plotly_chart(build_aufwand(je_eps, settings.ziel4), width="stretch", key="aufwand")
zeilen4 = je_eps[settings.eps4]
st.markdown(f"**Bei ε/OPT = {E.eps_text(settings.eps4)}, Ziel „{settings.ziel4}“:**")
st.markdown(E.md_tabelle(E.KOPF_AUFWAND, [E.zellen_aufwand(z, z["n"] == settings.n4) for z in zeilen4]))
fk = E.faktor_n4(zeilen4)
n20, n80 = E.anzahl_faktor_n4(zeilen4)
st.caption("Faktor der Median-Produkte bei vierfachem n (n = 20 → 80): " + (E.de(fk, 2) if fk is not None else "–")
           + f" (ε/OPT = {E.eps_text(settings.eps4)}, Ziel „{settings.ziel4}“; Median über die Instanzen, die das Ziel erreichen: {n20} bei n = 20, {n80} bei n = 80"
           + ("; das sind nur die Instanzen, deren Endplan schon die Lücke von 1 % erreicht, der Faktor ist dadurch verzerrt" if settings.ziel4 == C.ZIEL4[1] and (n20 < C.ANZAHL4[20] or n80 < C.ANZAHL4[80]) else "")
           + "; n = 160 steht in der Tabelle).")
kind, text = E.verdikt_aufwand(auf)
getattr(st, kind)(text)
st.markdown("**Das Handbeispiel „weit entferntes Lager“:** der Zeilenfehler nach k Durchläufen ist 1/(2k+1), gerechnet für ε = 1/110 und für ε = 1/800: derselbe Wert bei beiden ε. Ein Zeilenfehler von 1e-9 bräuchte demnach etwa 5 · 10⁸ Durchläufe, solange der Eintrag K₁₂ nicht zählt (aus der Formel gerechnet, nicht gelaufen).")
ks22, fe22 = _zwei_mal_zwei()
st.plotly_chart(build_zwei_mal_zwei(ks22, fe22), width="stretch", key="zwei_mal_zwei")

st.markdown("---")

# --- Absatz weiche Ränge -------------------------------------------------------------------------------------------------------------------

with st.expander("🔀 Weiche Permutation, weiche Ränge"):
    st.markdown(
        """
Eine Zuordnung von n Dingen auf n Plätze ist ein Transportplan mit Massen 1/n; seine Ecken sind die Permutationen. Der weiche Plan (englisch soft permutation) liegt im Inneren. **Weiche Ränge** (englisch soft ranks):
n Werte $x_i$ werden auf feste Zielstellen $s_j = (j - \\tfrac12)/n$ verteilt (Kosten $(x_i - s_j)^2$, Massen $1/n$); der weiche Rang ist $\\tilde r_i = n \\sum_j P_{ij}\\, j$, der echte Rang $r_i$, der Fehler $\\tilde r_i - r_i$ in Rangeinheiten.
Die Werte sind gleichverteilt in (0, 1); der Plan kommt aus dem Newton-Verfahren auf dem Dual; **nur konvergierte Instanzen** zählen mit. Die nicht konvergierten (bei kleinem ε/OPT) sind nicht zufällig verteilt:
die Anteile bei kleinem ε/OPT sind **auswahlverzerrt**.
        """
    )
    for n in C.N_RAENGE:
        st.markdown(f"**n = {n}** ({raenge[n]['anzahl']} Instanzen; „fast ganzzahlig“ heißt dev ≤ 0,02):")
        st.markdown(E.md_tabelle(E.kopf_raenge(n), [E.zeile_raenge(z) for z in raenge_tab[n]]))
    rc1, rc2 = build_raenge(raenge_tab)
    ra1, ra2 = st.columns(2)
    with ra1:
        st.plotly_chart(rc1, width="stretch", key="raenge_anteil")
    with ra2:
        st.plotly_chart(rc2, width="stretch", key="raenge_betrag")
    st.markdown(E.raenge_gleichstand_text(E.raenge_gleich_v()))
    kind, text = E.verdikt_raenge(raenge_tab)
    getattr(st, kind)(text)
    st.markdown("**Gegenstimme und Kosten.** Blondel u. a. (2020) bezeichnen Sinkhorn-Sortieren wegen Kosten und Speicher als teuer (O(T · n · m) Rechenarbeit, ab n = 1 000 mit Rückwärtsrechnung bzw. 2 000 nicht mehr im Speicher) und schlagen ein billigeres Verfahren vor. "
                "Diese Demo rechnet mit n bis 100 und zeigt keine Sekunden. Die Linie über Sortiermaschinen behandelt das Sortieren selbst.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme dieser Demo | Was passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| **Der gezeigte Plan ist exakt gerechnet: kein Unterlauf, kein Gleichstand, und seine Ableitung ist die des Grenzplans.** | Hier aufgehoben: In float32 fällt die naive Iteration im Gebiet „fast scharf“ aus, der Abbruchtest in float32 kann bei Toleranz 1e-8 Läufe als fertig melden, die abweichen (2 von 122); bei Toleranz 1e-5 weichen 4 von 194 ab, gleich für beide Testarten (Experiment 1); vor dem Ende ist der Plan unfertig und seine Ableitung auch (Experiment 2); bei Gleichstand ist der weiche Plan nicht ganzzahlig, strebt für ε → 0 gegen den Plan mit größter Entropie auf der Fläche der optimalen Pläne und konvergiert schlecht (Experiment 3). | hier behandelt |
| **Beide Randverteilungen haben dieselbe Gesamtmasse, der Kern K = exp(−C/ε) ist überall positiv, die Kosten sind kontinuierlich verteilt (außer in Experiment 3), und ε ist ein frei gewählter Regler.** | Bleibt bestehen: Nullen im Kern, verschiedene Gesamtmassen und ein aus Beobachtungen zurückgerechnetes ε sind hier nicht untersucht. | Matrix abgleichen (Furness, IPF); ungleiche Mengen |
| **Sinkhorn braucht weniger Schritte als ein exaktes Verfahren.** | Diese Demo zählt nur die Matrix-Vektor-Produkte der Iteration und zeigt keinen Vergleich mit einem Simplexverfahren; die Zählung gibt keinen Anlass, die Aussage zu behaupten. Einen Vergleich mit einem Verfahren erster Ordnung für Lineare Programme in derselben Zähleinheit zeigt die Demo zu PDLP. | PDLP-Demo |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Lager und Kunden sind Zufallspunkte im Einheitsquadrat, die Kosten der quadrierte Abstand (durch den größten Wert geteilt, größter Eintrag 1); die Massen sind Dirichlet(1)-verteilt (Experiment 2, Ableitung: 0,5/n + 0,5 · Dirichlet(1));
  ganzzahlige Kosten und Massen 1/n nur in Experiment 3 und im Absatz über Ränge. Beide Gesamtmassen sind 1.
- Das LP-Optimum OPT kommt aus einem eigenen Transportsimplex; ε/OPT ist je Instanz mit ihrem eigenen OPT gebildet.
- Zufall aus dem Hausgenerator SplitMix64; die Reihen benutzen feste Instanzen (Zufalls-Seed 2026) unabhängig vom Regler „Zufalls-Seed“.

**Grenzen.**
- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Instanz; Matrix-Vektor-Produkte). Ein Median über 4 bis 50 Instanzen ist kein Gesetz.
- Der Ausfall in float32 und float64 wurde bei n = 30 gezählt; ob die Schwellen mit n wandern, ist nicht untersucht.
- Ob ein float32-Lauf „fertig“ meldet und wie viele Läufe abweichen, hängt am Abbruchtest (Datentyp der Zeilensumme) und an der Toleranz, nicht nur an ε.
- Die Ableitung ist nur im Vorwärtsmodus und nur für n = 20 in den Tabellen gemessen; Rückwärtsmodus und Speicherbedarf nicht.
- Gleichstand ist nur für n = 10 und die gezeigten Kosten gemessen; die Zahlen des Absatzes über Ränge stehen mit der Auswahlverzerrung (nur konvergierte Instanzen).
- Verfahren wie Greenkhorn, ε-Skalierung und Beschleunigung sind nicht gemessen.

**Bekannte Probleme und Unsicherheiten.**
- Die Zahlen sind mit Zufalls-Seed 2026 gemessen; andere Zufalls-Seeds geben andere Zahlen im Band, nicht dieselben. Zählungen von float32-Läufen können je nach Rechner und Version von numpy um einzelne Läufe abweichen (Summenreihenfolge).
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Angebote $a \in \mathbb{R}^n_{>0}$ (Summe 1), Nachfragen $b \in \mathbb{R}^m_{>0}$ (Summe 1), Kosten $C \in \mathbb{R}^{n\times m}$ (größter Eintrag 1). *Gesucht:* der weiche Transportplan
$P_\varepsilon$ mit Zeilensummen $a$ und Spaltensummen $b$, der $\langle C, P\rangle + \varepsilon\,\mathrm{KL}(P \,\|\, a b^{\top})$ minimiert, und wie genau Rechnung, Rundung und Ableitung ihn treffen.

| Symbol | Bedeutung |
|---|---|
| $n$, $m$ | Zahl der Lager und der Kunden |
| $a$, $b$ | Angebote und Nachfragen (Randverteilungen), Summe 1 |
| $C$ | Kosten, größter Eintrag 1 |
| $\varepsilon$ | Gewicht der Entropie-Strafe (Einheit der Kosten); $\varepsilon/\mathrm{OPT}$ ist die Skala der Experimente 2 und 4 |
| $K$ | Kern $K_{ij} = e^{-C_{ij}/\varepsilon}$ |
| $u$, $v$ | Skalierungsvektoren, $P = \mathrm{diag}(u)\,K\,\mathrm{diag}(v)$ |
| $f$, $g$ | Kantorovich-Potentiale der Log-Domäne, $u_i = a_i e^{f_i/\varepsilon}$, $v_j = b_j e^{g_j/\varepsilon}$ |
| $r$, $c$ | Zeilen- und Spaltensummen eines Plans $P$ |
| $R$ | gerundeter Plan |
| $D$ | Richtung der Kostenänderung (gleiche Gestalt wie $C$), $\mathrm{d}P$ die Ableitung des Plans in diese Richtung |

**Iteration.** Im Kern-Bereich $u \leftarrow a \oslash (K v)$, $v \leftarrow b \oslash (K^{\top} u)$; in der Log-Domäne
$$f_i \leftarrow -\varepsilon \ln \sum_j b_j\, e^{(g_j - C_{ij})/\varepsilon}, \qquad g_j \leftarrow -\varepsilon \ln \sum_i a_i\, e^{(f_i - C_{ij})/\varepsilon}.$$
Eine Iteration sind **zwei Matrix-Vektor-Produkte**; der Abbruchtest (Zeilenfehler $\|P\mathbf 1 - a\|_1$) benutzt das Produkt des nächsten Halbschritts und kostet keins. Die Rundung kostet $O(nm)$ Rechenoperationen und kein Produkt.
Start $u = v = 1$.

**Unterlauf.** $\exp(-x) = 0$ in float64 ab $x \approx 745{,}1$ (darunter noch ein denormaler Wert), in float32 ab $x \approx 104$: für $C = 1$ heißt das $\varepsilon < 1/745$ bzw. $\varepsilon < 1/104$.

**Rundung** (Algorithmus 2 bei Altschuler u. a.). Mit $x_i = \min(a_i / r_i, 1)$, $F = \mathrm{diag}(x)\,P$, $y_j = \min(b_j / c_j(F), 1)$, $F' = F\,\mathrm{diag}(y)$ und den Fehlmengen $e_r = a - F'\mathbf 1$, $e_c = b - F'^{\top}\mathbf 1$:
$$R = F' + \frac{e_r\, e_c^{\top}}{\|e_r\|_1}, \qquad \langle C, R\rangle \le \langle C, P\rangle + 2\,\max C\,\big(\|P\mathbf 1 - a\|_1 + \|P^{\top}\mathbf 1 - b\|_1\big).$$
**Handbeispiel Rundung** ($a = (\tfrac34, \tfrac14)$, $b = (\tfrac14, \tfrac34)$, $C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $P = \begin{pmatrix} 9/40 & 3/8 \\ 1/40 & 3/8 \end{pmatrix}$, der Plan nach einem Durchlauf bei $\varepsilon = 1/\ln 3$): Zeilensummen $(\tfrac35, \tfrac25)$,
$x = (1, \tfrac58)$, $F = \begin{pmatrix} 9/40 & 3/8 \\ 1/64 & 15/64 \end{pmatrix}$, $y = (1, 1)$, Fehlmengen $e_r = (\tfrac{3}{20}, 0)$, $e_c = (\tfrac{3}{320}, \tfrac{45}{320})$,
$$R = \begin{pmatrix} 15/64 & 33/64 \\ 1/64 & 15/64 \end{pmatrix}.$$
Kosten $\tfrac{17}{32}$ gegen $\tfrac25$ vor der Rundung; das LP-Optimum ist $\tfrac12$: die rohe Lücke $-20\,\%$ wird gerundet zu $+\tfrac1{16} = +6{,}25\,\%$.

**Ableitung.** Mit $P_{ij} = a_i b_j \exp((f_i + g_j - C_{ij})/\varepsilon)$ ist $\mathrm{d}P_{ij} = P_{ij}\,(\mathrm{d}f_i + \mathrm{d}g_j - D_{ij})/\varepsilon$, wobei $(\mathrm{d}f, \mathrm{d}g)$ aus
$$\begin{pmatrix} \mathrm{diag}(r) & P \\ P^{\top} & \mathrm{diag}(c) \end{pmatrix} \begin{pmatrix} \mathrm{d}f \\ \mathrm{d}g \end{pmatrix} = \begin{pmatrix} (P \circ D)\mathbf 1 \\ (P \circ D)^{\top}\mathbf 1 \end{pmatrix}$$
folgt (eine Unbekannte wird festgehalten, weil $(f + t, g - t)$ denselben Plan gibt). **Implizit am Plan $P_k$** heißt: $P_k$, $r$, $c$ des Plans nach $k$ Durchläufen einsetzen; **Durchrechnen** heißt: die Iteration mit ihrer
Ableitung ($\mathrm{d}f_i = -\sum_j w_{ij}(\mathrm{d}g_j - D_{ij})$ mit den Softmax-Gewichten $w_{ij}$ des Log-Sum-Exp) mitrechnen; das ist die Ableitung von $P_k$, nicht des Grenzplans. Gegenprobe: zentrale finite Differenz.

**Gleichstand.** Haben mehrere Pläne die kleinsten Kosten, ist der Grenzwert $\varepsilon \to 0$ der weiche Plan mit größter Entropie auf der Fläche der optimalen Pläne (Sharify u. a. 2013, Satz 2.3, für Zuordnungen).
Handbeispiel: $a = b = (\tfrac13, \tfrac13, \tfrac13)$, $C = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$ hat zwei optimale Permutationen (Kosten 0); der weiche Plan hat im $2 \times 2$-Block überall $\tfrac16$ ($\mathrm{dev} = \tfrac12$),
L1-Abstand $\tfrac23$ zur Ecke des LP und Kosten $\approx 2 \cdot 10^{-9}$ bei $\varepsilon = 0{,}05$: die Kostenlücke ist klein, der Plan weit weg.

**Handbeispiel weit entferntes Lager** ($C = \begin{pmatrix} 0 & 1/2 \\ 1 & 1 \end{pmatrix}$, $a = b = (\tfrac12, \tfrac12)$, $\mathrm{OPT} = \tfrac12$): der Zeilenfehler nach $k$ Durchläufen ist für kleine $\varepsilon$ (hier $1/110$ und $1/800$, geprüft bis $k = 1\,000$) praktisch genau $\tfrac{1}{2k+1}$, und die rohe Kostenlücke ebenso;
für den Grenzkern $K = \begin{pmatrix} 1 & 0 \\ \delta & \delta \end{pmatrix}$ gilt es in Brüchen exakt ($P_{11} = k/(2k+1)$, geprüft bis $k = 30$). Bei $\varepsilon \ge 0{,}05$ gilt es nicht mehr. Die Franklin-Lorenz-Schranke (nach Sharify u. a. 2013, Satz 3.1, dort als Proposition bezeichnet) ist eine Schranke für positive Matrizen, keine gemessene Rate.

**Literatur** (nur so zitiert, wie gelesen): Altschuler, Weed, Rigollet (2017), *Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration*, arXiv 1705.09634 (Algorithmus 2, Lemma 7); Sharify, Gaubert, Grigori (2013),
*Solution of the optimal assignment problem by diagonal scaling algorithms*, arXiv 1104.3830 (Satz 2.3, Satz 3.1, dort als Proposition bezeichnet); Dong u. a. (2020), *A Study of Performance of Optimal Transport*, arXiv 2005.01182 (Auszug gelesen); Blondel u. a. (2020) zu Kosten und
Speicher des Sinkhorn-Sortierens; Peyré, Cuturi (2019), *Computational Optimal Transport* (Kapitel 1 bis 3 und Anfang von Kapitel 4 gelesen). Unterlauf, Log-Domäne, Rundung und implizite Differentiation sind Lehrstoff.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
