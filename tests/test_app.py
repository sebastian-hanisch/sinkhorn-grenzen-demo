"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen und prüft,
dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException`
bei Slidern mit berechneten Grenzen. Dazu: je Preset die erwartete Aussage, Permalink, Randfälle, Worst-Case am Regler-Maximum.

Laufzeit: Die Reihen der Experimente 1, 3 und 4 sind vorgerechnet (tools/vorgerechnet.json), die Seite rechnet dort nichts; nur die Instanz der Ansicht (Experiment 2) rechnet live
(Zwischenspeicher gilt im Prozess). Die Tests wählen darum bei Reglerwechseln eine kleine Ausgangslage (Adresszeile: n = 6)."""

import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import sgz_constants as C
import sgz_evaluation as E
import sgz_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio")
KLEIN = dict(n="6")                                # Adresszeile der kleinen Ausgangslage


def _app(**adresse) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in adresse.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _klein() -> AppTest:
    return _app(**KLEIN)


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _setze(at, **werte):
    for key, wert in werte.items():
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (werte, [str(e) for e in at.exception])
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in ("success", "info", "warning", "markdown", "caption") for e in getattr(at, kind))


def _metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_all_sections():
    at = _app()
    titel = " ".join(s.value for s in at.subheader)
    for abschnitt in ("Experiment 1: Unterlauf und Rechengenauigkeit", "Experiment 2: Wie fertig ist der Plan?", "Experiment 3: Gleichstand", "Experiment 4: Aufwand in Produkten", "Wo die Annahmen enden"):
        assert abschnitt in titel, abschnitt
    labels = [e.label for e in at.expander]
    assert [l for l in labels if "Mathematische Formulierung" in l] == ["📐 Mathematische Formulierung"] and any("Annahmen, Grenzen und bekannte Probleme" in l for l in labels)
    assert any("Weiche Permutation, weiche Ränge" in l for l in labels) and any("So rechnet die Iteration" in l for l in labels)
    assert len(at.metric) == 6 and len(at.info) >= 8 and not at.warning
    schluessel = [p.key for p in at.get("plotly_chart")]
    assert len(set(schluessel)) == len(schluessel) >= 12         # jedes Diagramm hat einen eigenen Schlüssel
    titel_alle = " ".join(m.value for m in at.markdown)
    assert "Kernfragen" in titel_alle and "Stimmt das Ergebnis?" in titel_alle and "Wie viele Produkte kostet es?" in titel_alle and "Wie genau ist die Ableitung?" in titel_alle


def test_every_button_click_does_not_raise():
    at = _klein()
    labels = [b.label for b in at.button]
    assert len(labels) == len(C.PRESETS) + 1, labels                      # sechs Presets und der Zufalls-Seed-Button
    for index, label in enumerate(labels):
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    at = _klein()
    schluessel = [s.key for s in at.sidebar.slider]
    assert schluessel == ["n_slider"], schluessel
    for key in schluessel:
        for edge in ("min", "max"):
            regler = _regler(at, key)
            wert = regler.min if edge == "min" else regler.max
            regler.set_value(int(wert)).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


# ----------------------------------------------------------------------------------------------------------------------------------
# Auswahlregler und Presets
# ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,optionen", [
    ("quelle", C.QUELLEN),
    ("eps1", C.EPS1_STUFEN),
    ("tol1", C.TOL1_STUFEN),
    ("test1", C.TEST1_STUFEN),
    ("inst1", C.INST1_STUFEN),
    ("eps2", C.EPS2_STUFEN),
    ("k2", (C.K2_STUFEN[0], 3, 1000, C.K2_STUFEN[-1])),
    ("runden2", C.RUNDEN2),
    ("ableitung2", C.ABLEITUNG2),
    ("kosten3", C.KOSTEN3),
    ("eps3", C.EPS3_STUFEN),
    ("n4", C.N4_STUFEN),
    ("eps4", C.EPS4_STUFEN),
    ("ziel4", C.ZIEL4),
])
def test_optionen_der_auswahlregler_laufen_ohne_fehler(key, optionen):
    at = _klein()
    for option in optionen:
        _setze(at, **{key: option})


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler_und_zeigt_seine_einstellung(name):
    at = _klein()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, wert in C.PRESETS[name].items():
        assert at.session_state[key] == wert, (name, key)


def test_preset_von_hand_zeigt_den_zeilenfehler_1_durch_2k_plus_1_und_die_instanz():
    at = _klein()
    at.sidebar.button(key="preset_Von Hand: ein weit entferntes Lager").click().run(timeout=TIMEOUT)
    m = _metrics(at)
    assert m["Zeilenfehler (L1)"] == "0,005" and m["Lücke des genauen Plans"] == "0,00 %"
    assert m["Rohe Kostenlücke von P_k"] == "+0,50 %" and "Von Hand: ein weit entferntes Lager: 2 Lager, 2 Kunden" in _texte(at)       # rohe Lücke 1/(2k+1) = 1/201
    assert m["Ableitungsfehler (relativ)"] == "–" and "Ableitungsfehler nicht angegeben (die Referenzableitung ist praktisch null" in _texte(at) and "4,2e10" not in _texte(at)


def test_preset_float32_bei_eps_0_002_nennt_die_gezaehlten_ausfaelle_aus_der_reihe():
    at = _klein()
    zeilen = E.reihe_aggregiert1(E.unterlauf_v(), C.DEFAULTS["tol1"], C.DEFAULTS["test1"], C.DEFAULTS["inst1"])
    z2, z3 = E.zeile1(zeilen, 0.002), E.zeile1(zeilen, 0.003)
    text = _texte(at)
    assert f"bei ε = 0,003 (ε/OPT im Median {E.eps_text(z3['eps_opt'])}) {z3['f32']['unterlauf']} von 50 Läufen aus, bei ε = 0,002 {z2['f32']['unterlauf']} von 50" in text
    assert f"**Bei ε = 0,002** (ε/max C = 0,002, ε/OPT im Median {E.eps_text(z2['eps_opt'])}; Toleranz 1e-5, Test in float32): float32 fällt in {z2['f32']['unterlauf']} von 50 Läufen aus" in text


def test_preset_abbruch_nach_10_iterationen_zeigt_unfertigen_plan():
    at = _klein()
    at.sidebar.button(key="preset_Abbruch nach 10 Iterationen").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["k2"] == 10
    text = _texte(at)
    assert "Nach 10 Durchläufen (20 Produkte)" in text and "nach nach" not in text
    assert float(_metrics(at)["Zeilenfehler (L1)"].replace(",", ".")) > 1e-4


def test_preset_gleichstand_alle_kosten_gleich_zeigt_das_produktmass():
    at = _klein()
    at.sidebar.button(key="preset_Gleichstand: alle Kosten gleich").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at)
    assert "Kosten „alle Kosten gleich“, ε = 0,02, n = 10, 60 Instanzen: 60 haben mehrere optimale Pläne; bei 60 davon ist der Sinkhorn-Plan nicht ganzzahlig (dev über 0,02, Median 0,100" in text
    assert "der weiche Plan für jedes ε das Produktmaß" in text


def test_preset_aufwand_n_80_hebt_die_zeile_n_80_hervor():
    at = _klein()
    at.sidebar.button(key="preset_Aufwand: n = 80").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["n4"] == 80
    z = E.aggregiere_aufwand(E.aufwand_v(80, 0.07), C.ZIEL4[0])
    text = _texte(at)
    assert "| **80** | 10 von 10 | " + E.band_ganz(z["produkte"]) in text


def test_preset_ableitung_bei_eps_0_01_laeuft_mit_k_1000_und_zeigt_den_unfertigen_plan():
    at = _klein()
    at.sidebar.button(key="preset_Ableitung bei ε/OPT = 0,01").click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["eps2"] == 0.01 and at.session_state["k2"] == 1000
    assert "Nach 1000 Durchläufen (2000 Produkte) bei ε/OPT = 0,01" in _texte(at)
    assert float(_metrics(at)["Zeilenfehler (L1)"].replace(",", ".")) > 1e-3 and _metrics(at)["Ableitungsfehler (relativ)"] != "–"


def test_rundung_aus_zeigt_den_hinweis_statt_des_gerundeten_plans():
    at = _setze(_klein(), runden2="aus")
    assert _metrics(at)["Lücke des gerundeten Plans"] == "–" and any("Rundung ist aus" in i.value for i in at.info)
    at = _setze(at, runden2="an")
    assert _metrics(at)["Lücke des gerundeten Plans"] != "–"


def test_toleranz_1e_minus_8_mit_test_in_float64_warnt_und_nennt_die_auflösung():
    at = _setze(_klein(), tol1=1e-8, test1="float64")
    assert any("liegt unter dem Rundungsfehler der Massen in float32" in w.value for w in at.warning) and not any("kleinsten Abstand" in w.value for w in at.warning)
    assert "erreicht kein float32-Lauf die Toleranz" in _texte(at)


# ----------------------------------------------------------------------------------------------------------------------------------
# Permalink, Randfälle, Worst-Case, Texte
# ----------------------------------------------------------------------------------------------------------------------------------

def test_permalink_uebernimmt_werte_begrenzt_und_eingerastet():
    at = _app(n="9999", e1="0.0031", k2="99999", q="unbekannt", t1="3e-6", i1="33", e3="0.5", n4="100", e4="0.05", z4="Lücke ≤ 1 %", k3="unbekannt")
    s = at.session_state
    assert s["n_slider"] == C.N_MAX and s["eps1"] == 0.003 and s["k2"] == 20000 and s["quelle"] == C.DEFAULTS["quelle"] and s["tol1"] == 1e-5 and s["inst1"] in (20, 50)
    assert s["eps3"] == 0.02 and s["n4"] in (80, 160) and s["eps4"] in (0.07, 0.035) and s["ziel4"] == C.ZIEL4[1] and s["kosten3"] == C.DEFAULTS["kosten3"]


def test_adresszeile_spiegelt_die_einstellungen():
    at = _setze(_klein(), n_slider=7, eps1=0.003, k2=20)
    assert at.query_params["n"] in (["7"], "7") and at.query_params["e1"] in (["0.003"], "0.003") and at.query_params["k2"] in (["20"], "20")


def test_randfaelle_alle_minima_und_seed_0():
    at = _klein()
    _setze(at, quelle=C.QUELLE_ZUFALL, n_slider=C.N_MIN, seed_input=0, eps1=C.EPS1_STUFEN[-1], tol1=C.TOL1_STUFEN[0], test1="float64", inst1=C.INST1_STUFEN[0], eps2=C.EPS2_STUFEN[-1], k2=1,
           runden2="aus", ableitung2=C.ABLEITUNG2[1], kosten3=C.KOSTEN3[0], eps3=C.EPS3_STUFEN[-1], n4=C.N4_STUFEN[0], eps4=C.EPS4_STUFEN[-1], ziel4=C.ZIEL4[1])


def test_worst_case_alle_maxima_zugleich_ohne_fehler():
    """Regler-Maximum: n = 20, ε/OPT = 0,01 mit 20 000 Durchläufen und Ableitung, alle Reihen mit der größten Auswahl. Die Zeit wird nicht geprüft (Abschnitt 1c)."""
    at = _app()
    _setze(at, n_slider=C.N_MAX, eps1=C.EPS1_STUFEN[0], tol1=C.TOL1_STUFEN[-1], test1="float32", inst1=C.INST1_STUFEN[-1], eps2=C.EPS2_STUFEN[-1], k2=C.K2_STUFEN[-1],
           kosten3=C.KOSTEN3[2], eps3=C.EPS3_STUFEN[0], n4=C.N4_STUFEN[-1], eps4=C.EPS4_STUFEN[-1], ziel4=C.ZIEL4[1])
    assert len(at.metric) == 6


def test_die_ansicht_nennt_keine_planbegriffe_und_keine_umgebungsaussagen():
    at = _app()
    gesamt = _texte(at) + " " + " ".join(e.label for e in at.expander) + " " + " ".join(s.value for s in at.subheader) + " " + " ".join(m.value for m in at.markdown)
    for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Orakel", "Saatwert", "Sinkhorn-Distanz", "Sinkhorn-Divergenz", "wasserstein", "Stück", "Vormessung", "Prüfung B"):
        assert verboten not in gesamt, verboten
    assert not re.search(r"\btr[äa]gt\b|\btragen\b", gesamt)
    assert "Zufalls-Seed" in " ".join(w.label for w in at.sidebar.number_input)


def test_jeder_regler_der_seitenleiste_hat_einen_hilfetext():
    at = _app()
    for art in REGLER_ARTEN:
        for w in getattr(at.sidebar, art):
            if w.key in P.SETTING_SPECS and w.key != "seed_input":
                assert w.help, w.key
