"""Auswertung: Formatfunktionen, Zählungen an handgesetzten Listen, Verdikte (Schwellen, jede Zahl im Text, Randfälle), die Lauf-Einheiten (Aufwand, Ansicht von Experiment 2) gegen eine zweite Rechnung."""

import json

import numpy as np
import pytest

import sgz_constants as C
import sgz_evaluation as E
from sgz_scenario import make_instanz
from sgz_sinkhorn import loese_sinkhorn
from sgz_vorgerechnet import DATEI, hole, schluessel


# ---- Formatierung ---------------------------------------------------------------------------------------------------------------

def test_formatfunktionen():
    assert E.de(1.234, 2) == "1,23" and E.de_g(0.00123456, 3) == "0,00123" and E.de_g(1.5e-8, 2) == "1,5e-8" and E.de_g(0, 3) == "0"
    assert E.ganz(1296) == "1 296" and E.band_ganz([1296.0, 398, 4240]) == "1 296 [398; 4 240]"
    assert E.prozent(0.2075, 2) == "20,75 %" and E.prozent_vz(0.0625, 2) == "+6,25 %" and E.prozent_vz(-0.2, 0) == "−20 %" and E.prozent_vz(-1e-17, 2) == "0,00 %"
    assert E.tol_text(1e-8) == "1e-8" and E.tol_text(1e-5) == "1e-5" and E.eps_text(0.035) == "0,035" and E.eps_text(1.0) == "1"
    assert E.zaehlung(3, 50) == "3 von 50" and E.de_vz(-0.5, 2) == "−0,50" and E.de_vz(0.5, 1) == "+0,5"
    assert E.band([3, 1, 2]) == [2.0, 1.0, 3.0] and E.band_prozent([0.1, 0.2, 0.3], 0) == "+20 % [+10 %; +30 %]"


def test_jsonbar_macht_numpy_zu_python_typen():
    x = E.jsonbar({"a": np.array([1.5, 2.5]), "b": (np.int64(3), np.float64(0.5)), 4: np.bool_(True)})
    assert x == {"a": [1.5, 2.5], "b": [3, 0.5], "4": True} and type(x["b"][0]) is int and type(x["4"]) is bool


def test_settings_aus_zustand_nimmt_standardwerte_und_wandelt_typen():
    s = E.settings_aus_zustand({"n_slider": 8.0, "eps1": 1, "k2": 10.0, "tol1": 1e-8})
    assert E.settings_aus_zustand({}) == E.Settings() and s.n == 8 and isinstance(s.n, int) and isinstance(s.eps1, float) and s.k2 == 10 and s.tol1 == 1e-8
    assert E.settings_aus_zustand(C.DEFAULTS) == E.Settings() and set(C.DEFAULTS) == {"quelle", "n_slider", "seed_input", "eps1", "tol1", "test1", "inst1", "eps2", "k2", "runden2", "ableitung2",
                                                                                      "kosten3", "eps3", "n4", "eps4", "ziel4"}


# ---- Experiment 1 -----------------------------------------------------------------------------------------------------------------

def _daten1(f32t32, f32t64, f64, log, opt=None):
    N = len(f64)
    return {"opt": opt or [0.04] * N, "eps": {f"{e:g}": {f"{t:g}": {"f32t32": f32t32, "f32t64": f32t64, "f64": f64, "log": log} for t in C.TOL1_STUFEN} for e in C.EPS1_STUFEN}}


def test_aggregiere_unterlauf_zaehlt_status_und_abweichungen_an_handgesetzten_laeufen():
    f32 = [["ok", 10, 0.0], ["ok", 11, 2e-6], ["ok", 12, 3e-4], ["ok", 9, 2e-2], ["cap", 3000, 0.1], ["unterlauf", 4, None]]
    f64 = [["ok", 10, 1e-15]] * 5 + [["cap", 3000, None]]
    log = [["ok", 10, None]] * 4 + [["cap", 3000, None]] * 2
    d = _daten1(f32, f32, f64, log, [0.05] * 6)
    z = E.aggregiere_unterlauf(d, 0.002, 1e-5, "float32", 6)
    assert z["f32"] == {"ok": 4, "cap": 1, "unterlauf": 1, "ueber": [3, 2, 1], "max_abweichung": 2e-2}
    assert z["f64"]["ok"] == 5 and z["f64"]["ueber"] == [0, 0, 0] and z["log"] == {"ok": 4, "cap": 2, "unterlauf": 0}
    assert z["eps_opt"] == pytest.approx(0.04) and z["anzahl"] == 6
    z3 = E.aggregiere_unterlauf(d, 0.002, 1e-5, "float32", 3)         # die ersten Instanzen
    assert z3["anzahl"] == 3 and z3["f32"]["ok"] == 3 and z3["f32"]["ueber"] == [2, 1, 0]


def test_verdikt_unterlauf_nennt_jede_zahl_aus_der_reihe_und_den_abbruchtest():
    f32 = [["ok", 10, 3e-4]] * 3 + [["unterlauf", 4, None]] * 2
    f64 = [["ok", 10, 1e-15]] * 5
    log = [["ok", 10, None]] * 4 + [["cap", 3000, None]]
    zeilen = E.reihe_aggregiert1(_daten1(f32, f32, f64, log), 1e-5, "float32", 5)
    art, text = E.verdikt_unterlauf(zeilen)
    assert art == "info"
    assert "ε/OPT im Median 0,075" in text                                                   # ε = 0,003 bei OPT 0,04
    assert "2 von 5 Läufen aus, bei ε = 0,002 2 von 5" in text and "von den 21 als fertig gemeldeten float32-Läufen" in text
    assert "weichen 21 um mehr als 1e-4" in text and "Toleranz 1e-5, Abbruchtest in float32" in text and "0,03 %" in text
    assert "float64 fällt bei ε = 0,0005 in 0 von 5 Läufen aus" in text and "0 von 35 Läufen aus" in text and "nur 4 von 5 fertig" in text


def test_verdikt_unterlauf_toleranz_nicht_erreichbar_und_kein_fertiger_lauf():
    f32 = [["cap", 3000, 0.0]] * 5
    ok64 = [["ok", 10, 0.0]] * 5
    zeilen = E.reihe_aggregiert1(_daten1(f32, f32, ok64, ok64), 1e-8, "float64", 5)
    text = E.verdikt_unterlauf(zeilen)[1]
    lo, hi = E.masse_rundung_float32()
    assert f"erreicht kein float32-Lauf die Toleranz; sie liegt unter dem Rundungsfehler der Massen in float32 (Summe |a − float32(a)| über die 30 Lager: {E.de_g(lo, 2)} bis {E.de_g(hi, 2)} in den 50 Instanzen)" in text
    assert "kleinsten Abstand" not in text
    zeilen = E.reihe_aggregiert1(_daten1(f32, f32, ok64, ok64), 1e-5, "float32", 5)
    assert "kein float32-Lauf meldet sich als fertig (Toleranz 1e-5, Abbruchtest in float32)" in E.verdikt_unterlauf(zeilen)[1]


def test_toleranz_erreichbar_float32():
    assert not E.toleranz_erreichbar_float32(1e-8) and E.toleranz_erreichbar_float32(1e-5)


def test_lauf_unterlauf_an_einer_instanz_stimmt_mit_den_einzelnen_laeufen_ueberein():
    inst = make_instanz(C.N1, C.N1, C.SERIEN_SEED, C.ZWECK_UNTERLAUF, 0)
    lauf = E.lauf_unterlauf(inst, 0.005, 1e-5)
    assert set(lauf) == {"log", "f32t32", "f32t64", "f64"} and lauf["log"][0] == "ok" and lauf["f64"][0] == "ok" and lauf["f64"][2] < 1e-9
    aus = E.lauf_unterlauf(inst, 0.0005, 1e-5)
    assert aus["f32t32"][0] == "unterlauf" and aus["f32t32"][2] is None


def test_handbeispiel_unterlauf_tabelle():
    z = E.handbeispiel_unterlauf(50)
    assert [r["nenner"] for r in z] == [110, 800] and z[0]["float32"] == "unterlauf" and z[0]["float64"] == "cap" and z[1]["float64"] == "unterlauf"
    assert z[0]["eintrag"] == pytest.approx(np.exp(-110.0)) and z[1]["eintrag"] == 0.0
    assert all(r["zeilenfehler"] == pytest.approx(1 / 101, abs=1e-9) for r in z)


# ---- Experiment 2 -----------------------------------------------------------------------------------------------------------------

def test_ansicht2_handbeispiel_zeilenfehler_1_durch_2k_plus_1_und_genauer_plan():
    z = E.ansicht2(E.Settings(quelle=C.QUELLE_HAND, eps2=0.01))
    assert z["k"] == list(C.K2_STUFEN) and z["opt"] == pytest.approx(0.5) and z["ref_ok"]
    for k, fehler in zip(z["k"], z["zeilenfehler"]):
        assert fehler == pytest.approx(1 / (2 * k + 1), abs=1e-9), k
    assert z["luecke_ende"] == pytest.approx(0.0, abs=1e-3) and z["roh"][0] == pytest.approx(1 / 3, abs=1e-9)         # rohe Lücke nach einem Durchlauf 1/(2k+1) = 1/3
    assert z["plan"][-1] < 1e-2 and set(z["plaene"]) == set(C.K2_STUFEN)


def test_ansicht2_zufallsinstanz_gerundeter_plan_ist_zulaessig_fehler_fallen_mit_k_und_verdikt():
    s = E.Settings(n=8, seed=3, eps2=0.3, k2=100)
    z = E.ansicht2(s)
    inst = z["inst"]
    for k in z["k"]:
        R = z["gerundete_plaene"][k]
        assert R.sum(axis=1) == pytest.approx(inst.a, abs=1e-14) and R.sum(axis=0) == pytest.approx(inst.b, abs=1e-14)
    assert z["zeilenfehler"][-1] < 1e-10 and z["plan"][-1] < 1e-10 and z["durch"][-1] < 1e-8
    assert z["gerundet"][0] > z["luecke_ende"] >= 0 and all(g >= -1e-12 for g in z["gerundet"])
    art, text = E.verdikt_ansicht2(z, s)
    assert art == "info" and "Nach 100 Durchläufen (200 Produkte) bei ε/OPT = 0,3 (8 Lager, 8 Kunden)" in text and "(Durchrechnen)" in text
    art, text = E.verdikt_ansicht2(z, E.Settings(n=8, seed=3, eps2=0.3, k2=10, ableitung2=C.ABLEITUNG2[1]))
    assert "Nach 10 Durchläufen (20 Produkte)" in text and "(implizit am Plan P_k)" in text


def test_verdikt_plan_nennt_die_gemessenen_zahlen():
    r = dict(n=60, eps_rel=0.46, anzahl=3, k=[1, 3, 10, 30, 100], gerundet=[[2.0, 2.2, 2.1], [1.4] * 3, [0.8] * 3, [0.3] * 3, [0.2, 0.21, 0.205]], ende=[0.2, 0.21, 0.2075])
    s = dict(faelle=240, halten=240, n=20)
    art, text = E.verdikt_plan(r, s)
    assert "+210 %" in text and "nach 100 Iterationen +20,5 %" in text and "am Ende (Zeilenfehler unter 1e-8) +20,8 %" in text and "in 240 von 240 Fällen" in text and "n = 60, Median über 3 Instanzen" in text


def _abl_tab(plan_ueber=(9, 1)):
    zeilen = []
    for er, k in C.ZEILEN_ABL:
        zeilen.append(dict(eps_rel=er, k=k, anzahl=30, plan=1e-3, zeile=1e-4, durch=5e-3, impl=2e-3, faktor_zeile=50.0, faktor_plan=5.0, faktor_impl=2.5, ueber=0))
    for z in zeilen:
        if (z["eps_rel"], z["k"]) == (0.01, 5000):
            z["ueber"] = plan_ueber[0]
        if (z["eps_rel"], z["k"]) == (0.01, 20000):
            z["ueber"] = plan_ueber[1]
    return zeilen


def test_verdikt_ableitung_zaehlung_statt_median():
    art, text = E.verdikt_ableitung(_abl_tab())
    assert "bei ε/OPT = 0,01 reichen 5 000 Durchläufe in 9 von 30 Instanzen nicht (Planfehler über 1e-4), auch 20 000 nicht in 1 von 30" in text
    assert "Zeilenfehler im Median bei 0,0001, der Ableitungsfehler (Durchrechnen) bei 0,005 (im Median der Instanzen 50-mal so groß wie der Zeilenfehler der jeweiligen Instanz)" in text
    assert "in der typischen Instanz (Median der Quotienten je Instanz) um den Faktor 2,5 bis 2,5" in text and "zwischen 1e-10 und 1e-2" in text
    assert "die Mediane der Tabelle unterscheiden sich in der Zeile ε/OPT = 0,01, k = 5 000 um den Faktor 2, die Quotienten streuen" in text          # 5e-3 / 2e-3 = 2,5 (Median-Verhältnis)
    assert "(n = 20, 30 Instanzen" in text
    kurz = E.verdikt_stimmt_ableitung(_abl_tab())[1]
    assert "im Median der Instanzen 50-mal so groß wie der Zeilenfehler des Abbruchtests der jeweiligen Instanz" in kurz and "in 9 von 30 Instanzen nicht" in kurz


def test_tabelle_ableitung_an_handgesetzten_fehlern():
    r = dict(n=20, anzahl=2, zeilen=[dict(eps_rel=0.01, k=5000, plan=[1e-3, 1e-5], zeile=[1e-4, 1e-6], durch=[1e-2, 1e-4], impl=[5e-3, 5e-5])], fd={})
    z = E.tabelle_ableitung(r)[0]
    assert z["ueber"] == 1 and z["anzahl"] == 2 and z["plan"] == pytest.approx(5.05e-4) and z["faktor_zeile"] == pytest.approx(100.0) and z["faktor_impl"] == pytest.approx(2.0)


# ---- Experiment 3 -----------------------------------------------------------------------------------------------------------------

def _agg3(**kw):
    z = dict(anzahl=60, mehrere=58, mehrere_nicht_ganzzahlig=58, eindeutig=2, eindeutig_ganzzahlig=2, konvergiert=13, konvergiert_mehrere=12, konvergiert_eindeutig=1,
             dev_median_mehrere=0.485, dev_median_eindeutig=0.0, l1_median_mehrere=1.01, kosten="ganzzahlig 0 bis 2", eps=0.02, n=10)
    z.update(kw)
    return z


def test_verdikt_gleichstand_nennt_die_zaehlungen_und_unterscheidet_die_faelle():
    text = E.verdikt_gleichstand(_agg3())[1]
    assert "58 haben mehrere optimale Pläne; bei 58 davon ist der Sinkhorn-Plan nicht ganzzahlig (dev über 0,02, Median 0,485" in text
    assert "Bei den 2 Instanzen mit eindeutigem Optimum ist der Plan nach höchstens 20 000 Iterationen in 2 ganzzahlig" in text and "in 20 000 Iterationen in 13 von 60 Instanzen erreicht (davon 12 bei mehreren Optima)" in text
    ohne = E.verdikt_gleichstand(_agg3(mehrere=0, mehrere_nicht_ganzzahlig=0, eindeutig=60, eindeutig_ganzzahlig=0, dev_median_mehrere=None, l1_median_mehrere=None))[1]
    assert "keine hat mehrere optimale Pläne." in ohne and "ist der Plan nach höchstens 20 000 Iterationen in 0 ganzzahlig" in ohne and "(davon" not in ohne
    assert "einzige Instanz" not in text and "einzige Instanz" not in ohne


def test_verdikt_gleichstand_alle_kosten_gleich_nennt_die_eine_instanz_60_mal_gerechnet():
    z = _agg3(kosten=C.KOSTEN3[3], mehrere=60, mehrere_nicht_ganzzahlig=60, eindeutig=0, eindeutig_ganzzahlig=0, konvergiert=60, konvergiert_mehrere=60, konvergiert_eindeutig=0, dev_median_mehrere=0.1)
    text = E.verdikt_gleichstand(z)[1]
    assert "In dieser Zeile steht eine einzige Instanz (konstante Kosten, Massen 1/n), 60-mal gerechnet: die Zählungen bedeuten dasselbe Ergebnis einmal, keine Stichprobe." in text


# ---- Experiment 4 -----------------------------------------------------------------------------------------------------------------

def test_lauf_aufwand_gleich_loese_sinkhorn_mit_toleranz_1e_minus_6():
    for i in range(3):
        inst = make_instanz(20, 20, C.SERIEN_SEED, C.ZWECK_AUFWAND, i)
        opt = E.referenz(inst).opt
        r = E.lauf_aufwand(inst, opt, 0.07 * opt)
        sk = loese_sinkhorn(inst.a, inst.b, inst.kosten, 0.07 * opt, tol=C.TOL4, max_iter=C.MAX_ITER4)
        assert sk.konvergiert and r["zeile"] == sk.produkte and r["iterationen"] == sk.iterationen
        if r["luecke"] is not None:                                                         # nur alle 10 Iterationen (20 Produkte) oder am Ende geprüft, nie nach dem Ende
            assert r["luecke"] % (2 * C.RASTER4) == 0 or r["luecke"] == r["zeile"]
            assert r["luecke"] <= r["zeile"]


def test_aggregiere_aufwand_faktor_und_verdikt_an_handgesetzten_reihen():
    def reihe(n, eps, p):
        return dict(n=n, eps_rel=eps, anzahl=len(p), zeile=p, luecke=[x if x < 1000 else None for x in p], luecke_ende=[0.0] * len(p), opt=[0.1] * len(p))

    rs = {}
    for e in C.EPS4_STUFEN:
        for n, p in zip(C.N4_STUFEN, ([100, 200, 300], [200, 300, 400], [400, 500, 600], [1000, 1100, 1200])):
            rs[(n, e)] = reihe(n, e, p)
    z = E.aggregiere_aufwand(rs[(20, 0.07)], C.ZIEL4[0])
    assert z["produkte"] == [200.0, 100.0, 300.0] and z["erreicht"] == 3 and z["multiplikationen"] == 20 * 20 * 200.0
    assert E.aggregiere_aufwand(rs[(160, 0.07)], C.ZIEL4[1])["produkte"] is None
    zs = [E.aggregiere_aufwand(rs[(n, 0.07)], C.ZIEL4[0]) for n in C.N4_STUFEN]
    assert E.faktor_n4(zs) == pytest.approx(2.5)
    text = E.verdikt_aufwand(rs)[1]
    assert "im Median 200 / 300 / 500 / 1 100 Matrix-Vektor-Produkte für n = 20 / 40 / 80 / 160 (3 / 3 / 3 / 3 Instanzen)" in text and "den Faktor 2,5 bis 2,5" in text
    assert "nur 3 / 3 / 3 / 0 der Instanzen" in text and "ein Vergleich mit den Schritten eines Simplexverfahrens ist hier nicht gezeigt" in text
    assert E.faktor_n4([dict(n=20, produkte=None), dict(n=80, produkte=[1, 1, 1])]) is None


def test_luecke_ist_eigenschaft_des_endplans_nur_wenn_jede_reihe_so_ausfaellt():
    gut = dict(luecke=[40, None, 60], luecke_ende=[0.005, 0.02, 0.009])
    assert E.luecke_ist_endplan_eigenschaft({(20, 0.07): gut})
    assert not E.luecke_ist_endplan_eigenschaft({(20, 0.07): dict(luecke=[40, None], luecke_ende=[0.005, 0.0])})        # zweite Instanz hat Endlücke ≤ 1 %, erreicht aber nicht
    assert E.anzahl_faktor_n4([dict(n=20, erreicht=7), dict(n=80, erreicht=5)]) == (7, 5)


def test_zeilenfehler_zwei_mal_zwei_ist_1_durch_2k_plus_1():
    for k in (1, 5, 50, 200):
        assert E.zeilenfehler_zwei_mal_zwei(k) == pytest.approx(1 / (2 * k + 1), abs=1e-9)


# ---- Vorrechnung ------------------------------------------------------------------------------------------------------------------

def test_vorgerechnete_datei_enthaelt_jeden_eintrag_und_die_schluessel_sind_eindeutig():
    liste = E.vorzurechnen()
    schluessel_liste = [schluessel(n, **p) for n, p, _ in liste]
    assert len(set(schluessel_liste)) == len(schluessel_liste) == 4 + 4 * 3 + 4 * 3 + 3 + 1
    daten = json.loads(DATEI.read_text(encoding="utf-8"))
    assert set(schluessel_liste) == set(daten)
    assert hole("nichtvorhanden") is None


def test_die_form_der_vorgerechneten_reihen():
    u = E.unterlauf_v()
    assert set(u) == {"opt", "eps"} and len(u["opt"]) == C.ANZAHL1 and set(u["eps"]) == {f"{e:g}" for e in C.EPS1_STUFEN}
    assert all(len(l) == C.ANZAHL1 for l in u["eps"]["0.002"]["1e-05"].values())
    r = E.runden_v()
    assert len(r["gerundet"]) == len(C.K_RUNDEN) and len(r["gerundet"][0]) == C.ANZAHL_RUNDEN and len(r["ende"]) == C.ANZAHL_RUNDEN
    a = E.ableitung_v()
    assert [(z["eps_rel"], z["k"]) for z in a["zeilen"]] == sorted(C.ZEILEN_ABL, key=lambda t: (-t[0], t[1])) and all(len(z["plan"]) == C.ANZAHL_ABL for z in a["zeilen"])
    for art in C.KOSTEN3:
        for e in C.EPS3_STUFEN:
            assert len(E.gleichstand_v(art, e)["dev"]) == C.ANZAHL3
    for n in C.N4_STUFEN:
        for e in C.EPS4_STUFEN:
            assert E.aufwand_v(n, e)["anzahl"] == C.ANZAHL4[n]
    for n in C.N_RAENGE:
        assert E.raenge_v(n)["anzahl"] == C.ANZAHL_RAENGE[n] and set(E.raenge_v(n)["eps"]) == {f"{e:g}" for e in C.EPS_RAENGE}
    assert E.raenge_gleich_v()["anzahl"] == 100


def test_ohne_vorgerechnete_datei_rechnet_die_reihe_selbst(monkeypatch):
    import sgz_vorgerechnet as V

    monkeypatch.setattr(V, "hole", lambda name, **params: None)
    r = E._vorgerechnet_oder("test", {}, lambda: {"x": np.array([1.0])})
    assert r == {"x": [1.0]}
