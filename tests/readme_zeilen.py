"""Tabellen und Sätze der README, aus `tools/vorgerechnet.json` mit denselben Formatfunktionen wie in der App gebildet (sgz_evaluation).

Die README muss jedes dieser Stücke wörtlich enthalten (`tests/test_claims.py`); `python tests/readme_zeilen.py` druckt sie zum Einfügen."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sgz_constants as C  # noqa: E402
import sgz_evaluation as E  # noqa: E402


def unterlauf_tabelle(tol: float, test: str) -> str:
    zeilen = E.reihe_aggregiert1(E.unterlauf_v(), tol, test, C.ANZAHL1)
    return E.md_tabelle(E.KOPF_UNTERLAUF, [E.zellen_unterlauf(z) for z in zeilen])


def unterlauf_verdikt(tol: float, test: str) -> str:
    return E.verdikt_unterlauf(E.reihe_aggregiert1(E.unterlauf_v(), tol, test, C.ANZAHL1))[1]


def fertig_tabelle() -> str:
    """Fertig gemeldete float32-Läufe über alle sieben ε (7 · 50 = 350 Läufe) je Toleranz und Abbruchtest."""
    zeilen = []
    for tol in C.TOL1_STUFEN:
        for test in C.TEST1_STUFEN:
            z = E.reihe_aggregiert1(E.unterlauf_v(), tol, test, C.ANZAHL1)
            fertig = sum(x["f32"]["ok"] for x in z)
            ueber = sum(x["f32"]["ueber"][1] for x in z)
            groesste = max(x["f32"]["max_abweichung"] for x in z)
            zeilen.append([E.tol_text(tol), test, E.zaehlung(fertig, len(z) * C.ANZAHL1), str(ueber), E.prozent(groesste, 2) if fertig else "–"])
    return E.md_tabelle(["Toleranz", "Abbruchtest der float32-Läufe in", "fertig gemeldet", "davon Abweichung über 1e-4", "größte Abweichung"], zeilen)


def handbeispiel_tabelle() -> str:
    return E.md_tabelle(E.KOPF_HAND, [E.zellen_hand(h) for h in E.handbeispiel_unterlauf(50)])


def runden_tabelle() -> str:
    return E.md_tabelle(E.KOPF_RUNDEN, E.zeilen_runden(E.runden_v()))


def schranke_satz() -> str:
    return E.schranke_text(E.schranke_v())


def plan_verdikt() -> str:
    return E.verdikt_plan(E.runden_v(), E.schranke_v())[1]


def ableitung_tabelle() -> str:
    tab = E.tabelle_ableitung(E.ableitung_v())
    return E.md_tabelle(E.KOPF_ABLEITUNG, [E.zellen_ableitung(z) for z in tab])


def fd_satz() -> str:
    return E.fd_text(E.ableitung_v())


def ableitung_verdikt() -> str:
    return E.verdikt_ableitung(E.tabelle_ableitung(E.ableitung_v()))[1]


def gleichstand_tabelle() -> str:
    zeilen = []
    for art in C.KOSTEN3:
        for e in C.EPS3_STUFEN:
            zeilen.append([art] + E.zellen_gleichstand(E.aggregiere_gleichstand(E.gleichstand_v(art, e))))
    return E.md_tabelle(["Kosten"] + E.KOPF_GLEICH, zeilen)


def gleichstand_verdikt(art: str, eps: float) -> str:
    return E.verdikt_gleichstand(E.aggregiere_gleichstand(E.gleichstand_v(art, eps)))[1]


def aufwand_rs() -> dict:
    return {(n, e): E.aufwand_v(n, e) for n in C.N4_STUFEN for e in C.EPS4_STUFEN}


def aufwand_tabelle() -> str:
    rs = aufwand_rs()
    zeilen = []
    for e in C.EPS4_STUFEN:
        for n in C.N4_STUFEN:
            zs = E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0])
            lu = E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[1])
            zeilen.append([E.eps_text(e), str(n), E.band_ganz(zs["produkte"]), E.zaehlung(lu["erreicht"], lu["anzahl"]), E.band_ganz(lu["produkte"]) if lu["produkte"] else "–",
                           E.ganz(zs["multiplikationen"])])
    return E.md_tabelle(["ε/OPT", "n", "Produkte bis Zeilenfehler 1e-6 (Median [Minimum; Maximum])", "Instanzen mit Lücke ≤ 1 %", "Produkte bis Lücke ≤ 1 % (Median [Minimum; Maximum])",
                         "n² · Produkte bis Zeilenfehler 1e-6 (Median)"], zeilen)


def faktoren_satz() -> str:
    rs = aufwand_rs()
    teile = []
    for e in C.EPS4_STUFEN:
        f = E.faktor_n4([E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0]) for n in C.N4_STUFEN])
        g = E.faktor_n4([E.aggregiere_aufwand(rs[(n, e)], C.ZIEL4[1]) for n in C.N4_STUFEN])
        teile.append(f"ε/OPT = {E.eps_text(e)}: {E.de(f, 2)} (Zeilenfehler 1e-6)" + (f", {E.de(g, 2)} (Lücke ≤ 1 %)" if g is not None else ", – (Lücke ≤ 1 % nicht erreicht)"))
    return "; ".join(teile)


def eps_faktor_satz() -> str:
    return E.eps_faktor_text(aufwand_rs())


def aufwand_verdikt() -> str:
    return E.verdikt_aufwand(aufwand_rs())[1]


def raenge_tabelle(n: int) -> str:
    return E.md_tabelle(E.kopf_raenge(n), [E.zeile_raenge(z) for z in E.tabelle_raenge(E.raenge_v(n))])


def raenge_gleich_satz() -> str:
    return E.raenge_gleichstand_text(E.raenge_gleich_v())


def raenge_verdikt() -> str:
    return E.verdikt_raenge({n: E.tabelle_raenge(E.raenge_v(n)) for n in C.N_RAENGE})[1]


if __name__ == "__main__":
    print("## unterlauf 1e-5 float32\n" + unterlauf_tabelle(1e-5, "float32"))
    print("\n" + unterlauf_verdikt(1e-5, "float32"))
    print("\n## fertig\n" + fertig_tabelle())
    print("\n## hand\n" + handbeispiel_tabelle())
    print("\n## runden\n" + runden_tabelle() + "\n" + schranke_satz() + "\n" + plan_verdikt())
    print("\n## ableitung\n" + ableitung_tabelle() + "\n" + fd_satz() + "\n" + ableitung_verdikt())
    print("\n## gleichstand\n" + gleichstand_tabelle())
    for art, e in ((C.KOSTEN3[1], 0.02), (C.KOSTEN3[2], 0.002), (C.KOSTEN3[0], 0.002), (C.KOSTEN3[3], 0.02)):
        print("\n" + gleichstand_verdikt(art, e))
    print("\n## aufwand\n" + aufwand_tabelle() + "\n" + faktoren_satz() + "\n" + aufwand_verdikt())
    for n in C.N_RAENGE:
        print(f"\n## raenge {n}\n" + raenge_tabelle(n))
    print("\n" + raenge_gleich_satz() + "\n" + raenge_verdikt())
