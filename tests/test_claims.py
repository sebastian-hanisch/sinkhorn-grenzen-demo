"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4 und §11a).

Die Reihen liegen in `tools/vorgerechnet.json` (erzeugt von `tools/vorrechnen.py` mit dem Code der Demo). Dieser Test macht drei Dinge:
(1) die Tabellen und Sätze der README werden aus der Datei mit denselben Formatfunktionen wie in der App gebildet und müssen wörtlich in der README stehen (so kann keine Zahl der
    README von der Messung abweichen);
(2) die Aussagen, die die README über diese Zahlen macht („in jeder Zeile“, „wächst mit n“), werden an der ganzen Reihe gezählt;
(3) Teilmengen der Reihen (die ersten Instanzen) werden neu gerechnet und als Band mit der Datei verglichen (Zählungen auf 2, Gleitkommazahlen relativ auf 1e-6), weil Gleitkomma-Summen
    (besonders in float32) und die Wahl der Ecke auf anderen Plattformen abweichen können."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sgz_constants as C
import sgz_evaluation as E
import readme_zeilen as R
from orakel_zahlen import ORAKEL

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
V = json.loads((ROOT / "tools" / "vorgerechnet.json").read_text(encoding="utf-8"))


def test_tabellen_und_saetze_der_readme():
    stuecke = [R.unterlauf_tabelle(1e-5, "float32"), R.unterlauf_verdikt(1e-5, "float32"), R.fertig_tabelle(), R.handbeispiel_tabelle(), R.runden_tabelle(), R.schranke_satz(), R.plan_verdikt(),
               R.ableitung_tabelle(), R.fd_satz(), R.ableitung_verdikt(), R.gleichstand_tabelle(), R.aufwand_tabelle(), R.eps_faktor_satz(), R.faktoren_satz(), R.aufwand_verdikt(),
               R.raenge_gleich_satz(), R.raenge_verdikt()]
    stuecke += [R.gleichstand_verdikt(C.KOSTEN3[1], 0.02), R.gleichstand_verdikt(C.KOSTEN3[2], 0.002), R.gleichstand_verdikt(C.KOSTEN3[0], 0.002)]
    stuecke += [R.raenge_tabelle(n) for n in C.N_RAENGE]
    for text in stuecke:
        assert text in README, text[:200]


def test_konstanten_der_readme():
    for text in (f"n = {C.N1} Lager und Kunden, {C.ANZAHL1} feste Instanzen", f"Rundung bei n = {C.N_RUNDEN}, ε/OPT = {E.eps_text(C.EPS_RUNDEN)}, {C.ANZAHL_RUNDEN} Instanzen",
                 f"Ableitung bei n = {C.N_ABL}, {C.ANZAHL_ABL} Instanzen", f"Experiment 3: n = {C.N3}", f"{C.ANZAHL3} Instanzen je Zeile", f"Zufalls-Seed {C.SERIEN_SEED}",
                 "Experiment 4: n = 20, 40, 80 (je 10 Instanzen) und 160 (4 Instanzen)", "n = 10, 30, 100 (100, 60, 30 Instanzen)", f"n bis {C.N_MAX}"):
        assert text in README, text
    assert C.ANZAHL4 == {20: 10, 40: 10, 80: 10, 160: 4} and C.ANZAHL_RAENGE == {10: 100, 30: 60, 100: 30} and C.MAX_ITER1 == 3000 and C.TOL3 == 1e-9 and C.MAX_ITER3 == 20000 and C.TOL4 == 1e-6


def test_orakel_zahlen():
    d = ORAKEL
    for text in (f"({d['naiv']} Instanzen, 7 Werte von k, relative Toleranz 1e-9)", f"Rundung auf {d['rund']} Instanzen", f"({d['ableitung']} Instanzen)",
                 f"(HiGHS, {d['lp']} Instanzen mit n ≠ m)", f"({d['netz']} ganzzahlige Instanzen, Abweichung 0)", f"({d['eind']} Instanzen)"):
        assert text in README, text


# ---- Aussagen über die Reihen -------------------------------------------------------------------------------------------------------

def test_ableitungsfehler_im_median_in_jeder_zeile_ueber_zeilenfehler_und_planfehler():
    tab = E.tabelle_ableitung(E.ableitung_v())
    assert len(tab) == len(C.ZEILEN_ABL)
    for z in tab:
        assert z["durch"] > z["zeile"] and z["durch"] > z["plan"], z
    assert "im Median in jeder Zeile der Tabelle über dem Zeilenfehler" in README


def test_zaehlungen_der_ableitung_5000_und_20000_durchlaeufe():
    tab = E.tabelle_ableitung(E.ableitung_v())
    z5 = next(z for z in tab if (z["eps_rel"], z["k"]) == (0.01, 5000))
    z20 = next(z for z in tab if (z["eps_rel"], z["k"]) == (0.01, 20000))
    assert z20["ueber"] < z5["ueber"] < z5["anzahl"] and z5["anzahl"] == C.ANZAHL_ABL            # mehr Durchläufe, weniger Instanzen mit Planfehler über 1e-4
    assert f"in {z5['ueber']} von {C.ANZAHL_ABL} Instanzen nicht" in README and f"auch 20 000 nicht in {z20['ueber']} von {C.ANZAHL_ABL}" in README


def test_formel_gegen_finite_differenz_im_median_sehr_genau():
    med, gross, n = E.fd_zusammenfassung(E.ableitung_v())
    assert med < 1e-8 and gross < 1e-2 and n == 5 * C.ANZAHL_ABL
    assert all(v["nicht_ok"] == 0 for v in E.ableitung_v()["fd"].values())


def test_gleichstand_in_jeder_zeile_alle_instanzen_mit_mehreren_optima_nicht_ganzzahlig():
    for art in C.KOSTEN3:
        for e in C.EPS3_STUFEN:
            z = E.aggregiere_gleichstand(E.gleichstand_v(art, e))
            assert z["mehrere"] == z["mehrere_nicht_ganzzahlig"], (art, e)
            if art == C.KOSTEN3[3]:                                                           # alle Kosten gleich: Produktmaß, dev = 1/n, Konvergenz in jeder Instanz
                assert z["konvergiert"] == z["anzahl"] and z["dev_median_mehrere"] == pytest.approx(1 / C.N3) and z["mehrere"] == z["anzahl"]
    assert "ist gleich der Zahl der Instanzen mit mehreren Optima" in README
    z = E.aggregiere_gleichstand(E.gleichstand_v(C.KOSTEN3[2], 0.002))
    assert z["konvergiert"] == 0 and z["eindeutig"] > 0 and z["eindeutig_ganzzahlig"] == z["eindeutig"]          # „keine der 60 Instanzen ... auch nicht die mit eindeutigem Optimum“
    for e in C.EPS3_STUFEN:
        assert E.aggregiere_gleichstand(E.gleichstand_v(C.KOSTEN3[2], e))["konvergiert"] == 0
    assert "erreicht keine der 60 Instanzen den Zeilenfehler 1e-9 in 20 000 Iterationen" in README and C.ANZAHL3 == 60


def test_unterlauf_aussagen_der_readme():
    u = E.unterlauf_v()
    z = E.reihe_aggregiert1(u, 1e-5, "float32", 50)
    z05 = E.zeile1(z, 0.0005)
    assert np.exp(-2000.0) == 0.0 and z05["f64"]["unterlauf"] == 0 and sum(x["log"]["unterlauf"] for x in z) == 0              # „entfernteste Eintrag schon null, dennoch kein Ausfall“, Log-Domäne ohne Ausfall
    assert z05["f32"]["unterlauf"] == 50 and E.zeile1(z, 0.01)["f32"]["unterlauf"] == 0                                         # float32 fällt im kleinen ε aus, im großen nicht
    fertig = {(t, d): sum(x["f32"]["ok"] for x in E.reihe_aggregiert1(u, t, d, 50)) for t in C.TOL1_STUFEN for d in C.TEST1_STUFEN}
    assert fertig[(1e-8, "float64")] == 0 and fertig[(1e-8, "float32")] > 0 and fertig[(1e-5, "float32")] == fertig[(1e-5, "float64")]      # „unterscheidet sich zwischen den beiden Zeilen für 1e-8“
    assert "dennoch fällt dort in der Tabelle kein Lauf aus" in README
    assert abs(1 / (2 * 5e8 + 1) - 1e-9) < 1e-12 and "$k \\approx 5 \\cdot 10^8$" in README                  # Zeilenfehler 1e-9 im Handbeispiel erst nach etwa 5 · 10^8 Durchläufen
    assert sum(x["f32"]["ok"] for x in E.reihe_aggregiert1(u, 1e-8, "float64", 50)) == 0 and "keiner der 350 Läufe die Toleranz 1e-8" in README and C.ANZAHL1 * len(C.EPS1_STUFEN) == 350


def test_raenge_aussagen_der_readme_wachsen_mit_n_und_rang_1_zu_hoch_rang_n_zu_niedrig():
    tabs = {n: E.tabelle_raenge(E.raenge_v(n)) for n in C.N_RAENGE}
    for i, e in enumerate(sorted(C.EPS_RAENGE, reverse=True)):
        betraege = [tabs[n][i]["betrag"] for n in C.N_RAENGE]
        assert betraege[0] < betraege[1] < betraege[2], (e, betraege)                              # der Fehler wächst mit n bei festem ε/OPT
        for n in C.N_RAENGE:
            z = tabs[n][i]
            assert z["konvergiert"] > 0 and z["rang1"] > 0 and z["rangn"] < 0, (n, e)             # Rang 1 zu hoch, Rang n zu niedrig (Mittel über die konvergierten Instanzen)
    assert "und der Fehler wächst mit n" in README and "Rang 1 zu hoch, Rang n zu niedrig" in README


def test_summe_der_weichen_raenge_ist_n_mal_n_plus_1_halbe_in_jeder_konvergierten_instanz():
    for n in C.N_RAENGE:
        r = E.raenge_v(n)
        for d in r["eps"].values():
            for s in d["summe"]:
                assert s is None or abs(s - n * (n + 1) / 2) < 1e-7, (n, s)


def test_auswahlverzerrung_bei_kleinem_eps_mehr_nicht_konvergierte_instanzen_als_bei_grossem():
    for n in C.N_RAENGE:
        tab = E.tabelle_raenge(E.raenge_v(n))
        assert tab[-1]["konvergiert"] < tab[0]["konvergiert"] <= C.ANZAHL_RAENGE[n]


def test_gleichstaende_der_raenge_zeilen_identisch():
    rg = E.raenge_gleich_v()
    assert max(x for d in rg["eps"].values() for x in d["zeilen"] if x is not None) < 1e-12


def test_aufwand_wachstum_mit_n_und_1_durch_eps():
    rs = {(n, e): E.aufwand_v(n, e) for n in C.N4_STUFEN for e in C.EPS4_STUFEN}
    for e in C.EPS4_STUFEN:
        med = [E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0])["produkte"][0] for n in C.N4_STUFEN]
        assert med == sorted(med), (e, med)                                                         # mehr n, mehr Produkte
    for n in C.N4_STUFEN:
        med = [E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0])["produkte"][0] for e in C.EPS4_STUFEN]
        assert med == sorted(med), (n, med)                                                         # kleineres ε/OPT, mehr Produkte
    assert all(E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0])["erreicht"] == C.ANZAHL4[n] for n in C.N4_STUFEN for e in C.EPS4_STUFEN)    # alle Instanzen erreichen den Zeilenfehler 1e-6
    for n in C.N4_STUFEN:
        assert E.aggregiere_aufwand(rs[(n, 0.14)], C.ZIEL4[1])["erreicht"] == 0                    # bei ε/OPT = 0,14 erreicht keine Instanz die Lücke von 1 %
    assert all(p % 20 == 0 for r in rs.values() for p in r["luecke"] if p is not None)           # Raster von 20 Produkten


def test_rundung_in_der_readme_gerundete_luecke_faellt_mit_k_und_liegt_ueber_dem_ende():
    r = E.runden_v()
    med = [float(np.median(x)) for x in r["gerundet"]]
    assert med == sorted(med, reverse=True) and med[0] > 2.0 > med[-1] and float(np.median(r["ende"])) <= med[-1] + 1e-3
    assert all(w > 0 for x in r["gerundet"] for w in x)                                           # ein zulässiger Plan hat nie Kosten unter dem LP-Optimum
    assert min(r["roh"][0]) < 0                                                                    # die rohe Lücke ist nach einer Iteration negativ (der Plan ist nicht zulässig)


# ---- Texte --------------------------------------------------------------------------------------------------------------------------

def test_testzahl_der_readme():
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    zahl = int(re.search(r"(\d+) tests? collected", r.stdout).group(1))
    assert f"- **Testzahl:** {zahl} Tests (`pytest --collect-only -q tests`)." in README, zahl


def test_readme_und_app_nennen_keine_planbegriffe_und_keine_umgebungsaussagen():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for text in (README, app):
        for verboten in ("Phase 0", "Pilot", "Korrekturrunde", "läuft lokal", "geplant", "Saatwert", "trägt", "tragen", "Vormessung", "Mutationstest", "Stück", "V1", "O6", "TODO ", "Plan-", "PLAN-"):
            if verboten == "TODO " and text is README:
                continue                                                # der Live-Link-Kommentar bleibt bis zum Deploy
            assert verboten not in text, verboten
    assert "Orakel" not in app


def test_readme_nennt_die_kante_wortlaut_und_keine_simplex_zahlen():
    assert "*Der gezeigte Plan ist exakt gerechnet: kein Unterlauf, kein Gleichstand, und seine Ableitung ist die des Grenzplans.*" in README
    assert "weniger Schritte als ein exaktes Verfahren" in README and "schneller als der Simplex" not in README
    assert not re.search(r"\bnit\b|Iterationen des Simplex|HiGHS-Iterationen", README)


def test_handwerte_der_readme_stehen_in_den_tests():
    for text in ("$R = \\begin{pmatrix} 15/64 & 33/64 \\\\ 1/64 & 15/64 \\end{pmatrix}$", "$R = \\begin{pmatrix} 7/20 & 3/20 \\\\ 3/20 & 7/20 \\end{pmatrix}$", "$P_{11} = k/(2k+1)$",
                 "$\\mathrm{d}P_{11} = -1/(4\\varepsilon)$", "$+\\tfrac1{16} = +6{,}25\\,\\%$", "$\\varepsilon < 1/745 = 0{,}00134$", "$\\varepsilon < 1/104 = 0{,}0096$"):
        assert text in README, text
    belege = {"test_round.py": ("Fr(15, 64), Fr(33, 64)", "Fr(7, 20), Fr(3, 20)", "1 / 16"), "test_naiv.py": ("Fr(k, 2 * k + 1)", "np.exp(-745.0) > 0", "np.exp(np.float32(-103.0)) > 0"),
              "test_deriv.py": ("-1 / (4 * eps)", "h = 1e-6"), "test_gleichstand_und_raenge.py": ("[1.8062, 2.3186, 3.0, 3.6814, 4.1938]", "2 / 3")}
    for datei, stuecke in belege.items():
        inhalt = (ROOT / "tests" / datei).read_text(encoding="utf-8")
        for beleg in stuecke:
            assert beleg in inhalt, (datei, beleg)


# ---- Neu gerechnet (Band) ------------------------------------------------------------------------------------------------------------

ZAEHLFELDER = {"iterationen", "produkte", "zeile", "luecke_prod"}


def _band(a, b, rel=1e-6, abs_=1e-12):
    assert a == pytest.approx(b, rel=rel, abs=abs_)


def test_unterlauf_neu_gerechnet_erste_zehn_instanzen_als_band():
    neu = E.reihe_unterlauf(anzahl=10, eps_stufen=(0.003, 0.002, 0.0005), tol_stufen=(1e-5,))
    alt = V["unterlauf|"]
    assert neu["opt"] == pytest.approx(alt["opt"][:10], rel=1e-9)
    for eps in ("0.003", "0.002", "0.0005"):
        for lauf in ("f32t32", "f32t64", "f64", "log"):
            n, a = neu["eps"][eps]["1e-05"][lauf], alt["eps"][eps]["1e-05"][lauf][:10]
            unterschiede = sum(1 for x, y in zip(n, a) if x[0] != y[0])
            assert unterschiede <= 1, (eps, lauf)                                                      # float32-Summen können auf anderen Plattformen um einen Lauf abweichen
            for x, y in zip(n, a):
                if x[0] == y[0] == "ok" and lauf != "f32t32" and lauf != "f32t64":
                    assert abs(x[1] - y[1]) <= 2 and (x[2] is None or x[2] == pytest.approx(y[2], abs=1e-9))


def test_runden_neu_gerechnet_erste_instanzen(monkeypatch):
    monkeypatch.setattr(C, "ANZAHL_RUNDEN", 2)
    neu, alt = E.reihe_runden(), V["runden|"]
    for j in range(len(C.K_RUNDEN)):
        _band(neu["gerundet"][j], alt["gerundet"][j][:2])
        _band(neu["roh"][j], alt["roh"][j][:2])
        _band(neu["zeilenfehler"][j], alt["zeilenfehler"][j][:2])
    _band(neu["ende"], alt["ende"][:2])
    assert all(abs(x - y) <= 2 for x, y in zip(neu["iterationen_ende"], alt["iterationen_ende"][:2]))


def test_schranke_neu_gerechnet_240_von_240():
    assert E.reihe_schranke() == pytest.approx(V["schranke|"], rel=1e-6, abs=1e-12)


def test_ableitung_neu_gerechnet_erste_drei_instanzen_als_band(monkeypatch):
    monkeypatch.setattr(C, "ANZAHL_ABL", 3)
    neu, alt = E.reihe_ableitung(), V["ableitung|"]
    assert len(neu["zeilen"]) == len(alt["zeilen"])
    for zn, za in zip(neu["zeilen"], alt["zeilen"]):
        assert (zn["eps_rel"], zn["k"]) == (za["eps_rel"], za["k"])
        for feld in ("plan", "zeile", "durch", "impl"):
            _band(zn[feld], za[feld][:3], rel=1e-4, abs_=1e-11)                                          # Fehler unter 1e-10 hängen am Rauschen des genauen Plans


def test_gleichstand_neu_gerechnet_erste_fuenf_instanzen(monkeypatch):
    monkeypatch.setattr(C, "ANZAHL3", 5)
    for art, eps in ((C.KOSTEN3[1], 0.02), (C.KOSTEN3[3], 0.002), (C.KOSTEN3[0], 0.02)):
        neu, alt = E.reihe_gleichstand(art, eps), V[f"gleichstand|eps={eps!r},kosten={art!r}"]
        assert neu["eindeutig"] == alt["eindeutig"][:5]
        _band(neu["dev"], alt["dev"][:5], rel=1e-5, abs_=1e-9)
        assert sum(a != b for a, b in zip(neu["konvergiert"], alt["konvergiert"][:5])) <= 1


def test_aufwand_neu_gerechnet_n_20_und_40_als_band():
    for n, e in ((20, 0.07), (40, 0.035)):
        neu, alt = E.reihe_aufwand(n, e), V[f"aufwand|eps={e!r},n={n!r}"]
        assert neu["opt"] == pytest.approx(alt["opt"], rel=1e-9)
        for feld in ("zeile", "luecke"):
            for x, y in zip(neu[feld], alt[feld]):
                assert (x is None) == (y is None) and (x is None or abs(x - y) <= 40)                       # höchstens zwei Prüfpunkte Unterschied
        _band(neu["luecke_ende"], alt["luecke_ende"], rel=1e-4, abs_=1e-9)


def test_raenge_neu_gerechnet_erste_zehn_instanzen(monkeypatch):
    monkeypatch.setattr(C, "ANZAHL_RAENGE", {10: 10, 30: 60, 100: 30})
    neu, alt = E.reihe_raenge(10), V["raenge|n=10"]
    for e, d in neu["eps"].items():
        assert sum(a != b for a, b in zip(d["konv"], alt["eps"][e]["konv"][:10])) <= 1, e
        # Eine Instanz darf auf einer anderen Plattform (letztes Bit der Summen) anders konvergieren: nur dort vergleichen,
        # wo beide Läufe gleich ausgehen (Linux-Lauf 2026-11-24: ε = 0,02, eine Instanz 0,0015 gegen „nicht konvergiert“).
        for k_neu, k_alt, a, b in zip(d["konv"], alt["eps"][e]["konv"][:10], d["betrag"], alt["eps"][e]["betrag"][:10]):
            if k_neu != k_alt:
                continue
            assert (a is None) == (b is None) and (a is None or a == pytest.approx(b, rel=1e-5, abs=1e-8)), e


# ---- Befunde der Aussagenprüfung (2026-11-24) ----------------------------------------------------------------------------------------

APP = (ROOT / "app.py").read_text(encoding="utf-8")


def test_toleranz_1e_minus_8_liegt_unter_dem_rundungsfehler_der_massen_in_float32():
    """Die Summe |a − float32(a)| über die 30 Lager liegt in allen 50 Instanzen über 1e-8: die Zeilensummen der float32-Iteration (float32-Zahlen) kommen a nie näher."""
    lo, hi = E.masse_rundung_float32()
    assert 1e-8 < lo < 1.5e-8 and 2.5e-8 < hi < 3.5e-8                                              # 1,4e-8 bis 3e-8
    assert not E.toleranz_erreichbar_float32(1e-8) and E.toleranz_erreichbar_float32(1e-5)
    from sgz_naiv import naive_iteration
    from sgz_scenario import make_instanz

    for i in (0, 3, 7):
        inst = make_instanz(C.N1, C.N1, C.SERIEN_SEED, C.ZWECK_UNTERLAUF, i)
        boden = float(np.abs(inst.a - inst.a.astype(np.float32).astype(np.float64)).sum())
        r = naive_iteration(inst.a, inst.b, inst.kosten, 0.01, np.float32, 1e-12, 300, "float64")
        assert r.zeilenfehler >= boden - 1e-12, (i, r.zeilenfehler, boden)                       # der Zeilenfehler (Test in float64) fällt nie unter diese Summe
    text = f"{E.de_g(lo, 2)} bis {E.de_g(hi, 2)}"
    assert text in README and text in E.verdikt_unterlauf(E.reihe_aggregiert1(E.unterlauf_v(), 1e-8, "float64", 50))[1]
    assert "kleinsten Abstand" not in README and "kleinsten Abstand" not in APP


def test_zaehlungen_zur_tabelle_abbruchtest_und_toleranz():
    u = E.unterlauf_v()
    z = {(t, d): E.reihe_aggregiert1(u, t, d, 50) for t in C.TOL1_STUFEN for d in C.TEST1_STUFEN}
    fertig = {k: sum(x["f32"]["ok"] for x in v) for k, v in z.items()}
    ueber = {k: sum(x["f32"]["ueber"][1] for x in v) for k, v in z.items()}
    assert (fertig[(1e-8, "float32")], ueber[(1e-8, "float32")]) == (122, 2) and fertig[(1e-8, "float64")] == 0
    assert (fertig[(1e-5, "float32")], ueber[(1e-5, "float32")]) == (194, 4) == (fertig[(1e-5, "float64")], ueber[(1e-5, "float64")])
    def zaehl(zs):
        return [(x["f32"]["ok"], x["f32"]["cap"], x["f32"]["unterlauf"], x["f32"]["ueber"]) for x in zs]

    assert zaehl(z[(1e-5, "float32")]) == zaehl(z[(1e-5, "float64")])                                  # bei 1e-5 gleich für beide Testarten, Zeile für Zeile
    assert E.prozent(max(x["f32"]["max_abweichung"] for x in z[(1e-5, "float32")]), 2) == "0,62 %"
    for text in ("(2 von 122)", "4 von 194"):
        assert text in README and text in APP
    assert "bis zu 0,62 %" in APP


def test_ableitung_verdikt_median_der_quotienten_gegen_quotient_der_mediane():
    tab = E.tabelle_ableitung(E.ableitung_v())
    z03 = next(z for z in tab if (z["eps_rel"], z["k"]) == (0.03, 1000))
    assert round(z03["durch"] / z03["zeile"]) == 18 and round(z03["faktor_zeile"]) == 25            # Quotient der Mediane und Median der Quotienten unterscheiden sich
    z5 = next(z for z in tab if (z["eps_rel"], z["k"]) == (0.01, 5000))
    assert round(z5["durch"] / z5["impl"]) == 29 and round(z5["faktor_impl"], 1) == 1.1
    auslauf = [z for z in tab if 1e-10 < z["durch"] < 1e-2]
    assert len(auslauf) == 4 and (0.01, 5000) in [(z["eps_rel"], z["k"]) for z in auslauf]
    assert "im Median der Instanzen 25-mal so groß wie der Zeilenfehler der jeweiligen Instanz" in README
    assert "um den Faktor 29, die Quotienten streuen" in README and "(Median der Quotienten je Instanz)" in README


def test_alle_kosten_gleich_ist_eine_einzige_instanz():
    from sgz_scenario import make_gleichstand

    erste = make_gleichstand(C.KOSTEN3[3], C.N3, C.SERIEN_SEED, 0)
    for i in range(1, C.ANZAHL3):
        weitere = make_gleichstand(C.KOSTEN3[3], C.N3, C.SERIEN_SEED, i)
        assert np.array_equal(weitere.kosten, erste.kosten) and np.array_equal(weitere.a, erste.a) and np.array_equal(weitere.b, erste.b)
    for e in C.EPS3_STUFEN:
        r = E.gleichstand_v(C.KOSTEN3[3], e)
        assert max(r["dev"]) - min(r["dev"]) < 1e-9 and len(set(r["eindeutig"])) == 1
    assert "in dieser Zeile steht eine einzige Instanz" in README and "dasselbe Ergebnis einmal" in README


def test_luecke_1_prozent_ist_eigenschaft_des_endplans_und_die_faktoren_haben_verschiedene_teilmengen():
    rs = {(n, e): E.aufwand_v(n, e) for n in C.N4_STUFEN for e in C.EPS4_STUFEN}
    assert E.luecke_ist_endplan_eigenschaft(rs)                                                       # erreicht genau, wer am Ende (Zeilenfehler 1e-6) Lücke ≤ 1 % hat
    z = {e: [E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[1]) for n in (20, 80)] for e in C.EPS4_STUFEN}
    assert (z[0.07][0]["erreicht"], z[0.07][1]["erreicht"]) == (7, 5) and (z[0.035][0]["erreicht"], z[0.035][1]["erreicht"]) == (10, 10) and z[0.14][0]["erreicht"] == 0
    # Faktor des Ziels „Zeilenfehler 1e-6“ auf derselben Teilmenge der Instanzen, die das Ziel „Lücke ≤ 1 %“ erreichen (7 und 5 Instanzen)
    r20, r80 = rs[(20, 0.07)], rs[(80, 0.07)]
    teil20 = np.median([x for x, y in zip(r20["zeile"], r20["luecke"]) if y is not None])
    teil80 = np.median([x for x, y in zip(r80["zeile"], r80["luecke"]) if y is not None])
    assert round(teil80 / teil20, 1) == 5.3 and round(np.median(r80["zeile"]) / np.median(r20["zeile"]), 1) == 4.2
    assert "5,3 statt 4,2" in README and "7 (n = 20) und 5 (n = 80) von je 10" in README


def test_unterschiede_der_faktoren_zwischen_den_drei_eps_liegen_innerhalb_der_streuung_bootstrap():
    rs = {(n, e): E.aufwand_v(n, e) for n in (20, 80) for e in C.EPS4_STUFEN}
    g = np.random.default_rng(1)
    faktor, bereich = {}, {}
    for e in C.EPS4_STUFEN:
        a, b = np.array(rs[(20, e)]["zeile"], float), np.array(rs[(80, e)]["zeile"], float)
        faktor[e] = float(np.median(b) / np.median(a))
        f = [np.median(g.choice(b, len(b))) / np.median(g.choice(a, len(a))) for _ in range(2000)]
        bereich[e] = (np.percentile(f, 5), np.percentile(f, 95))
    for e, (lo, hi) in bereich.items():
        assert all(lo < faktor[x] < hi for x in C.EPS4_STUFEN), (e, bereich, faktor)              # jeder Bereich (90 %) enthält alle drei Faktoren
    assert max(faktor.values()) / min(faktor.values()) < 1.3


def test_matrizen_des_3_mal_3_beispiels_stehen_in_der_readme():
    from test_deriv import C5, D5

    def latex(M):
        def zahl(x):
            return f"{x:g}".replace(".", "{,}")
        return "\\begin{pmatrix} " + " \\\\ ".join(" & ".join(zahl(x) for x in zeile) for zeile in M) + " \\end{pmatrix}"

    assert f"$C = {latex(C5)}$" in README and f"Richtung $D = {latex(D5)}$" in README


def test_franklin_lorenz_schranke_4_6_e_minus_12_und_proposition_3_1_in_readme_und_app():
    assert round(4 * np.exp(-27.5) * 1e12, 1) == 4.6
    assert "bei ε = 1/110 etwa $4{,}6 \\cdot 10^{-12}$" in README and "4{,}4" not in README
    assert "Satz 3.1, dort als Proposition bezeichnet" in README and "Satz 3.1, dort als Proposition bezeichnet" in APP and "Proposition 3.1" not in README


def test_ueberarbeitete_aussagen_stehen_in_readme_und_app_und_die_alten_nicht_mehr():
    app = APP.replace("\\\\", "\\")                                                                    # Quelltext von app.py hat in normalen Strings doppelte Rückstriche
    for text in (README, app):
        assert "wählt der weiche Plan die Mitte" not in text and "verwerfen Sinkhorn-Sortieren" not in text and "unabhängig von $\\varepsilon$" not in text and "genau 0 ab $x > 745$" not in text
        assert "bezeichnen Sinkhorn-Sortieren wegen Kosten und Speicher als teuer" in text and "x \\approx 745{,}1$" in text
    assert "praktisch genau $\\tfrac{1}{2k+1}$" in app and "Bei $\\varepsilon \\ge 0{,}05$ gilt es nicht mehr" in app
