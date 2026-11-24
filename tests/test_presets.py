"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzen und einrasten; jedes Preset lässt sich rechnen und zeigt die erwartete Aussage."""

import re

import pytest

import sgz_constants as C
import sgz_evaluation as E
import sgz_presets as P

NAMEN = ["Von Hand: ein weit entferntes Lager", "float32 bei ε = 0,002", "Abbruch nach 10 Iterationen", "Gleichstand: alle Kosten gleich", "Aufwand: n = 80", "Ableitung bei ε/OPT = 0,01"]


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and list(C.PRESETS) == NAMEN
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.SETTING_SPECS), name          # jedes Preset legt ALLE Regler fest
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_no_two_presets_are_identical_and_the_default_is_exactly_one_preset():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)
    assert [n for n, p in C.PRESETS.items() if p == C.DEFAULTS] == ["float32 bei ε = 0,002"]
    assert E.Settings() == E.settings_aus_zustand(C.DEFAULTS)


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"q", "n", "seed", "e1", "t1", "ts1", "i1", "e2", "k2", "r2", "a2", "k3", "e3", "n4", "e4", "z4"}
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)          # keine doppelten Adressparameter


def test_preset_hilfetexte_nennen_keine_messergebnisse():
    """Die Hilfetexte der Presets nennen keine gemessenen Zahlen (jede Zahl der App steht in der README oder in einem Test): keine Prozent, keine Zählungen „x von y“."""
    for name, text in C.PRESET_HELP.items():
        assert "%" not in text and not re.search(r"\d+ von \d+", text), (name, text)


def test_jedes_preset_hat_auf_jeder_ebene_eine_vorgerechnete_reihe_und_das_verdikt_zeigt_die_erwartete_aussage():
    for name, preset in C.PRESETS.items():
        s = E.settings_aus_zustand(preset)
        gz = E.aggregiere_gleichstand(E.gleichstand_v(s.kosten3, s.eps3))
        text = E.verdikt_gleichstand(gz)[1]
        assert f"Kosten „{s.kosten3}“" in text, name
        zeilen = E.reihe_aggregiert1(E.unterlauf_v(), s.tol1, s.test1, s.inst1)
        assert E.verdikt_unterlauf(zeilen)[0] == "info" and E.zeile1(zeilen, s.eps1)["anzahl"] == s.inst1
        assert E.aggregiere_aufwand(E.aufwand_v(s.n4, s.eps4), s.ziel4)["anzahl"] == C.ANZAHL4[s.n4]
    gleich = C.PRESETS["Gleichstand: alle Kosten gleich"]
    gz = E.aggregiere_gleichstand(E.gleichstand_v(gleich["kosten3"], gleich["eps3"]))
    assert gz["mehrere"] == gz["mehrere_nicht_ganzzahlig"] == gz["anzahl"] and gz["konvergiert"] == gz["anzahl"] and gz["dev_median_mehrere"] == pytest.approx(0.1)   # Produktmaß: dev = 1/n


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laesst_sich_rechnen(name):
    s = E.settings_aus_zustand(C.PRESETS[name])
    z = E.ansicht2(s)
    assert z["plaene"][s.k2].shape == (z["inst"].n, z["inst"].m) and z["opt"] > 0
    if name == "Von Hand: ein weit entferntes Lager":
        assert z["inst"].name == C.QUELLE_HAND and z["zeilenfehler"][z["k"].index(s.k2)] == pytest.approx(1 / (2 * s.k2 + 1), abs=1e-6)
    if name == "Abbruch nach 10 Iterationen":
        assert s.k2 == 10 and z["zeilenfehler"][z["k"].index(10)] > 1e-4          # der Plan ist nach 10 Durchläufen noch unfertig
    if name == "Ableitung bei ε/OPT = 0,01":
        i = z["k"].index(s.k2)
        assert s.k2 == 1000 and z["eps"] == pytest.approx(0.01 * z["opt"])
        assert min(z["zeilenfehler"][i], z["plan"][i], z["durch"][i], z["impl"][i]) > 1e-3          # „Plan und Ableitung sind dort noch nicht fertig“ (bei k = 5 000 ist diese Instanz fertig)
        assert not z["abl_ref_null"] and "noch nicht fertig" in C.PRESET_HELP[name] and "5 000" in C.PRESET_HELP[name]
        text = E.verdikt_ansicht2(z, s)[1]
        assert f"Nach 1000 Durchläufen (2000 Produkte) bei ε/OPT = 0,01 (12 Lager, 12 Kunden): Zeilenfehler {E.de_g(z['zeilenfehler'][i], 2)}," in text and "Ableitungsfehler nicht angegeben" not in text
    if name == "Von Hand: ein weit entferntes Lager":
        i = z["k"].index(s.k2)
        assert z["abl_ref_null"]                                                                   # die Referenzableitung ist praktisch null: relative Fehler (4e10) sagen nichts
        text = E.verdikt_ansicht2(z, s)[1]
        assert "Ableitungsfehler nicht angegeben (die Referenzableitung ist praktisch null" in text and "e10" not in text and "Zeilenfehler 0,005" in text and "rohe Lücke +0,50 %" in text
    if name == "float32 bei ε = 0,002":
        assert "oft" not in C.PRESET_HELP[name] and "Experiment 1" in C.PRESET_HELP[name]


def test_einrasten_und_begrenzen_des_permalinks():
    sp = P.SETTING_SPECS
    assert P._eingerastet(sp["n_slider"], 500) == C.N_MAX and P._eingerastet(sp["n_slider"], -3) == C.N_MIN
    assert P._eingerastet(sp["eps1"], 0.0031) == 0.003 and P._eingerastet(sp["eps1"], 1e9) == 0.01 and P._eingerastet(sp["eps1"], 1e-9) == 0.0005
    assert P._eingerastet(sp["tol1"], 3e-6) == 1e-5 and P._eingerastet(sp["inst1"], 33) in (20, 50)
    assert P._eingerastet(sp["k2"], 4) in (3, 5) and P._eingerastet(sp["k2"], 99999) == 20000 and P._eingerastet(sp["k2"], 0) == 1
    assert P._eingerastet(sp["quelle"], C.QUELLE_HAND) == C.QUELLE_HAND and P._eingerastet(sp["quelle"], "unbekannt") is None
    assert P._eingerastet(sp["eps2"], 0.5) == 0.46 and P._eingerastet(sp["n4"], 100) in (80, 160) and P._eingerastet(sp["eps4"], 0.05) in (0.07, 0.035)
    assert P._eingerastet(sp["kosten3"], C.KOSTEN3[2]) == C.KOSTEN3[2] and P._eingerastet(sp["eps3"], 0.5) == 0.02
    assert P._eingerastet(sp["seed_input"], -5) == 0 and P._eingerastet(sp["eps1"], float("nan")) is None
