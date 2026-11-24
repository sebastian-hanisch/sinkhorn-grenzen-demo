"""Kennzahlen, Reihen und Verdikte der Demo (reine Funktionen, ohne Streamlit).

Aufbau: Einstellungen und Formatierung; `referenz` (LP-Optimum mit dem eigenen Transportsimplex); je Experiment die Reihe (`reihe_*`, rechnet; Ergebnis als reine Python-Typen),
die Auswertung der Reihe (`aggregiere_*`, `tabelle_*`) und das Verdikt `verdikt_*` als Funktion (Art, Text), die die App mit `getattr(st, Art)(Text)` anzeigt.
Jede Zahl eines Verdikts stammt aus der Reihe, die es bekommt.

Zählweise: eine Iteration = zwei Matrix-Vektor-Produkte. Skalen: ε/OPT (OPT = LP-Optimum derselben Instanz bei max C = 1) in den Experimenten 2 und 4 und im Absatz „weiche
Ränge“; ε in Einheiten des größten Kostenwerts (max C = 1, also ε/max C) in Experiment 1, weil der Unterlauf an C/ε hängt; ε in Einheiten einer Kostenstufe in Experiment 3.
Start der Iteration überall u = v = 1 (der Plan nach null Durchläufen ist K = exp(-C/ε))."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import sgz_constants as C
from sgz_deriv import ableitung_fd, ableitung_implizit, durchrechnen, relativer_fehler
from sgz_eindeutigkeit import ist_eindeutig
from sgz_naiv import STATUS_CAP, STATUS_OK, STATUS_UNTERLAUF, naive_iteration
from sgz_newton import _vorwaerts, kaltstart, loese_gibbs, loese_gibbs_newton
from sgz_round import runden, schranke
from sgz_scenario import Instanz, handbeispiel_weit, instanz_rng, make_gleichstand, make_instanz, make_werte, raenge_kosten
from sgz_simplex import loese_transport, pruefe_loesung
from sgz_sinkhorn import aktualisiere_f, aktualisiere_g, loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale, zeilenfehler


# ----------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen und Formatierung
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    quelle: str = C.DEFAULTS["quelle"]
    n: int = C.DEFAULTS["n_slider"]
    seed: int = C.DEFAULTS["seed_input"]
    eps1: float = C.DEFAULTS["eps1"]
    tol1: float = C.DEFAULTS["tol1"]
    test1: str = C.DEFAULTS["test1"]
    inst1: int = C.DEFAULTS["inst1"]
    eps2: float = C.DEFAULTS["eps2"]
    k2: int = C.DEFAULTS["k2"]
    runden2: str = C.DEFAULTS["runden2"]
    ableitung2: str = C.DEFAULTS["ableitung2"]
    kosten3: str = C.DEFAULTS["kosten3"]
    eps3: float = C.DEFAULTS["eps3"]
    n4: int = C.DEFAULTS["n4"]
    eps4: float = C.DEFAULTS["eps4"]
    ziel4: str = C.DEFAULTS["ziel4"]


_ZUSTAND_ZU_FELD = {"n_slider": "n", "seed_input": "seed"}


def settings_aus_zustand(zustand) -> Settings:
    """Settings aus dem Session State (Schlüssel der Regler); fehlende Schlüssel nehmen den Standardwert."""
    felder = {_ZUSTAND_ZU_FELD.get(key, key): zustand[key] for key in C.DEFAULTS if key in zustand}
    for feld in ("n", "seed", "inst1", "k2", "n4"):
        if feld in felder:
            felder[feld] = int(felder[feld])
    for feld in ("eps1", "tol1", "eps2", "eps3", "eps4"):
        if feld in felder:
            felder[feld] = float(felder[feld])
    return Settings(**felder)


def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{stellen}f}".replace(".", ",")


def de_g(x: float, signifikant: int = 3) -> str:
    """Zahl mit Dezimalkomma und höchstens `signifikant` gültigen Stellen (ohne Exponent, wo es lesbar bleibt)."""
    if x == 0:
        return "0"
    text = f"{x:.{signifikant}g}"
    if "e" in text:
        mant, exp = text.split("e")
        return f"{mant.replace('.', ',')}e{int(exp)}"
    return text.replace(".", ",")


def ganz(x: float) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (für Zählungen wie Produkte)."""
    return f"{int(round(x)):,}".replace(",", " ")


def band_ganz(b) -> str:
    """Median [Minimum; Maximum] einer Zählung als ganze Zahlen."""
    return f"{ganz(b[0])} [{ganz(b[1])}; {ganz(b[2])}]"


def prozent(x: float, stellen: int = 1) -> str:
    return de(100.0 * x, stellen) + " %"


def prozent_vz(x: float, stellen: int = 1) -> str:
    """Prozent mit Vorzeichen (+ oder −), für Lücken."""
    text = de(100.0 * abs(x), stellen)
    if float(text.replace(",", ".")) == 0.0:                       # ein Wert, der auf 0 rundet, bekommt kein Vorzeichen
        return text + " %"
    return ("+" if x >= 0 else "−") + text + " %"


def zaehlung(x, N) -> str:
    """„x von N“."""
    return f"{x} von {N}"


def band_prozent(w, stellen: int = 1) -> str:
    """Median [Minimum; Maximum] von Lücken als Prozent mit Vorzeichen."""
    b = band(w)
    return f"{prozent_vz(b[0], stellen)} [{prozent_vz(b[1], stellen)}; {prozent_vz(b[2], stellen)}]"


def de_vz(x: float, stellen: int = 2) -> str:
    """Zahl mit Vorzeichen (+ oder −) und Dezimalkomma."""
    return ("+" if x >= 0 else "−") + de(abs(x), stellen)


def eps_text(e: float) -> str:
    """ε für die Anzeige: höchstens zwei gültige Stellen."""
    return de_g(e, 2)


def tol_text(t: float) -> str:
    """Toleranz als 1e-8 (ohne führende Null im Exponenten)."""
    return f"1e-{int(round(-math.log10(t)))}"


def band(werte):
    """[Median, Minimum, Maximum] einer Liste (als Liste, damit die Werte unverändert durch JSON gehen)."""
    v = np.asarray(list(werte), float)
    return [float(np.median(v)), float(v.min()), float(v.max())]


def jsonbar(x):
    """Wandelt Ergebnisse in reine Python-Typen um (Listen, Zahlen, Texte): dieselbe Gestalt wie nach dem Laden aus der JSON-Datei."""
    if isinstance(x, dict):
        return {str(k): jsonbar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonbar(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonbar(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


# ----------------------------------------------------------------------------------------------------------------------------------------
# Referenz
# ----------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Referenz:
    opt: float                 # LP-Optimum
    plan_lp: np.ndarray        # eine optimale Ecke (eigener Transportsimplex, Regel „steilste“)
    loesung: object            # sgz_simplex.Loesung (Preise für den Eindeutigkeitstest)


def referenz(inst: Instanz) -> Referenz:
    """LP-Optimum mit dem eigenen Transportsimplex; der unabhängige Prüfer läuft mit und meldet eine Verletzung als Fehler."""
    lo = loese_transport(inst.a, inst.b, inst.kosten, "steilste")
    fehler = pruefe_loesung(inst.a, inst.b, inst.kosten, lo)
    if fehler:
        raise RuntimeError("Transportsimplex: " + "; ".join(fehler))
    return Referenz(lo.kosten, lo.plan, lo)


def luecke_von(P, inst: Instanz, opt: float) -> float:
    """(⟨C, P⟩ − OPT) / OPT."""
    return (float((P * inst.kosten).sum()) - opt) / opt


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Unterlauf und Rechengenauigkeit
# ----------------------------------------------------------------------------------------------------------------------------------------

LAEUFE1 = ("f32t32", "f32t64", "f64", "log")      # float32 mit Test in float32, float32 mit Test in float64, float64, Log-Domäne


def lauf_unterlauf(inst: Instanz, eps: float, tol: float, max_iter: int = C.MAX_ITER1) -> dict:
    """Die vier Läufe einer Instanz bei einem ε (größter Kostenwert 1) und einer Toleranz: Liste [Status, Iterationen, relative Abweichung der Kosten von der Log-Domäne] je Lauf.
    Die Log-Domäne läuft mit derselben Toleranz und Iterationsgrenze und ist die Vergleichsgröße; ihre eigene Abweichung ist leer."""
    a, b, K = inst.a, inst.b, inst.kosten
    sk = loese_sinkhorn(a, b, K, eps, tol=tol, max_iter=max_iter)
    kosten_log = float((plan_aus_potentialen(a, b, K, eps, sk.f, sk.g) * K).sum())
    ergebnis = {"log": [STATUS_OK if sk.konvergiert else STATUS_CAP, sk.iterationen, None]}
    for name, dtype, test in (("f32t32", np.float32, "float32"), ("f32t64", np.float32, "float64"), ("f64", np.float64, "float64")):
        r = naive_iteration(a, b, K, eps, dtype, tol, max_iter, test)
        dev = abs(float((r.plan * K).sum()) - kosten_log) / kosten_log if r.plan is not None else None
        ergebnis[name] = [r.status, r.iterationen, dev]
    return ergebnis


def reihe_unterlauf(anzahl: int = C.ANZAHL1, eps_stufen=C.EPS1_STUFEN, tol_stufen=C.TOL1_STUFEN) -> dict:
    """Experiment 1: {"opt": [...], "eps": {ε: {Toleranz: {Lauf: [[Status, Iterationen, Abweichung] je Instanz]}}}} (Schlüssel als Text, `f"{x:g}"`)."""
    insts = [make_instanz(C.N1, C.N1, C.SERIEN_SEED, C.ZWECK_UNTERLAUF, i) for i in range(anzahl)]
    daten = {"opt": [referenz(inst).opt for inst in insts], "eps": {}}
    for eps in eps_stufen:
        daten["eps"][f"{eps:g}"] = {}
        for tol in tol_stufen:
            laeufe = [lauf_unterlauf(inst, eps, tol) for inst in insts]
            daten["eps"][f"{eps:g}"][f"{tol:g}"] = {name: [l[name] for l in laeufe] for name in LAEUFE1}
    return jsonbar(daten)


def _zaehle(laeufe, mit_abweichung: bool) -> dict:
    ok = [l for l in laeufe if l[0] == STATUS_OK]
    z = {"ok": len(ok), "cap": sum(1 for l in laeufe if l[0] == STATUS_CAP), "unterlauf": sum(1 for l in laeufe if l[0] == STATUS_UNTERLAUF)}
    if mit_abweichung:
        devs = [l[2] for l in ok if l[2] is not None]
        z["ueber"] = [sum(1 for d in devs if d > s) for s in C.ABWEICHUNG_SCHWELLEN]
        z["max_abweichung"] = max(devs) if devs else 0.0
    return z


def aggregiere_unterlauf(daten: dict, eps: float, tol: float, test: str, anzahl: int) -> dict:
    """Zählung für ein ε: je Rechengenauigkeit ok / cap / Unterlauf, bei float32 und float64 die Zahl der ok-Läufe mit Abweichung über 1e-6, 1e-4, 1e-2 und die größte Abweichung.
    `test` wählt, welcher float32-Lauf gemeint ist (Abbruchtest in float32 oder in float64 gerechnet)."""
    e = daten["eps"][f"{eps:g}"][f"{tol:g}"]
    f32 = e["f32t32" if test == "float32" else "f32t64"][:anzahl]
    opt = daten["opt"][:anzahl]
    return {"eps": eps, "tol": tol, "test": test, "anzahl": len(f32), "eps_opt": float(np.median([eps / o for o in opt])),
            "f32": _zaehle(f32, True), "f64": _zaehle(e["f64"][:anzahl], True), "log": _zaehle(e["log"][:anzahl], False)}


def reihe_aggregiert1(daten: dict, tol: float, test: str, anzahl: int) -> list:
    return [aggregiere_unterlauf(daten, eps, tol, test, anzahl) for eps in C.EPS1_STUFEN]


@lru_cache(maxsize=1)
def masse_rundung_float32() -> tuple:
    """(Minimum, Maximum) von Σ_i |a_i − float32(a_i)| über die Lager der festen Instanzen von Experiment 1: der Rundungsfehler der Massen in float32.
    Die Zeilensummen der float32-Iteration sind float32-Zahlen und kommen der exakten Masse a nie näher als diese Summe."""
    summen = []
    for i in range(C.ANZAHL1):
        a = make_instanz(C.N1, C.N1, C.SERIEN_SEED, C.ZWECK_UNTERLAUF, i).a
        summen.append(float(np.abs(a - a.astype(np.float32).astype(np.float64)).sum()))
    return min(summen), max(summen)


def toleranz_erreichbar_float32(tol: float) -> bool:
    """False, wenn die Toleranz unter dem Rundungsfehler der Massen in float32 liegt (in allen festen Instanzen): dann erreicht der Abbruchtest in float64 sie nie."""
    return tol >= masse_rundung_float32()[0]


def zeile1(zeilen, eps):
    for z in zeilen:
        if abs(z["eps"] - eps) < 1e-12:
            return z
    return None


def verdikt_unterlauf(zeilen: list):
    """Verdikt zu Experiment 1 (Art, Text) aus `reihe_aggregiert1` (alle ε-Stufen, eine Toleranz, ein Abbruchtest)."""
    z3, z2, z0 = zeile1(zeilen, 0.003), zeile1(zeilen, 0.002), zeile1(zeilen, 0.0005)
    N, tol, test = zeilen[0]["anzahl"], zeilen[0]["tol"], zeilen[0]["test"]
    fertig = sum(z["f32"]["ok"] for z in zeilen)
    ueber4 = sum(z["f32"]["ueber"][1] for z in zeilen)
    groesste = max(z["f32"]["max_abweichung"] for z in zeilen)
    if fertig:
        ab = (f"von den {fertig} als fertig gemeldeten float32-Läufen (alle ε der Tabelle, Toleranz {tol_text(tol)}, Abbruchtest in {test}) weichen {ueber4} um mehr als 1e-4 relativ "
              f"von der Log-Domäne ab (größte Abweichung {de_g(100 * groesste, 2)} %)")
    elif not toleranz_erreichbar_float32(tol) and test == "float64":
        lo, hi = masse_rundung_float32()
        ab = (f"mit Toleranz {tol_text(tol)} und Abbruchtest in float64 erreicht kein float32-Lauf die Toleranz; sie liegt unter dem Rundungsfehler der Massen in float32 "
              f"(Summe |a − float32(a)| über die {C.N1} Lager: {de_g(lo, 2)} bis {de_g(hi, 2)} in den {C.ANZAHL1} Instanzen)")
    else:
        ab = f"kein float32-Lauf meldet sich als fertig (Toleranz {tol_text(tol)}, Abbruchtest in {test})"
    log_aus = sum(z["log"]["unterlauf"] for z in zeilen)
    return "info", (f"In float32 fällt bei ε = 0,003 (ε/OPT im Median {eps_text(z3['eps_opt'])}) {z3['f32']['unterlauf']} von {N} Läufen aus, bei ε = 0,002 {z2['f32']['unterlauf']} von {N}; {ab}. "
                    f"float64 fällt bei ε = 0,0005 in {z0['f64']['unterlauf']} von {N} Läufen aus. Die Log-Domäne fällt in {log_aus} von {N * len(zeilen)} Läufen aus, aber bei ε = 0,0005 "
                    f"sind in {ganz(C.MAX_ITER1)} Iterationen nur {z0['log']['ok']} von {N} fertig (Toleranz {tol_text(tol)}).")


def handbeispiel_unterlauf(k: int = 50) -> list:
    """Das Handbeispiel „weit entferntes Lager“ bei ε = 1/110 und 1/800: Status von float32 und float64 (Toleranz 1e-8, 3 000 Iterationen) und Zeilenfehler der Log-Domäne nach k Durchläufen.
    Gibt eine Liste von Dicts zurück (je ε)."""
    inst = handbeispiel_weit()
    zeilen = []
    for eps in C.HAND_EPS:
        r32 = naive_iteration(inst.a, inst.b, inst.kosten, eps, np.float32, 1e-8, C.MAX_ITER1, "float64")
        r64 = naive_iteration(inst.a, inst.b, inst.kosten, eps, np.float64, 1e-8, C.MAX_ITER1, "float64")
        _, f, g = verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, [k])[0]
        fehler = zeilenfehler(plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g), inst.a)
        zeilen.append(dict(eps=eps, nenner=round(1 / eps), eintrag=math.exp(-1.0 / eps), float32=r32.status, float64=r64.status, k=k, zeilenfehler=fehler))
    return zeilen


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Wie fertig ist der Plan?
# ----------------------------------------------------------------------------------------------------------------------------------------

def richtung(n: int, m: int, seed: int, index: int) -> np.ndarray:
    """Richtung D der Ableitung: normalverteilte Einträge, durch den größten Betrag geteilt (max |D| = 1)."""
    g = instanz_rng(seed, C.ZWECK_RICHTUNG, index)
    D = np.array([[g.normal() for _ in range(m)] for _ in range(n)])
    return D / np.abs(D).max()


def instanz_der_ansicht(s: Settings) -> Instanz:
    if s.quelle == C.QUELLE_HAND:
        return handbeispiel_weit()
    return make_instanz(s.n, s.n, s.seed, C.ZWECK_ANSICHT, 0)


ABL_REF_NULL = 1e-9        # ‖dP_ref‖ unter diesem Anteil von ‖P_ref‖/ε (bei max |D| = 1): die Referenzableitung ist praktisch null


def _zeilenfehler_plan(P, a):
    return float(np.abs(P.sum(axis=1) - a).sum())


def ansicht2(s: Settings) -> dict:
    """Experiment 2 an der Instanz der Seitenleiste: Plan, gerundeter Plan und Ableitung nach jedem k aus K2_STUFEN gegen den genauen Plan (Newton-Verfahren auf dem Dual).
    Gibt ein Dict mit Listen über K2_STUFEN zurück; `ref_art` nennt, woher die Referenz stammt."""
    inst = instanz_der_ansicht(s)
    ref = referenz(inst)
    a, b, K = inst.a, inst.b, inst.kosten
    eps = s.eps2 * ref.opt
    D = richtung(inst.n, inst.m, s.seed, 0)
    lauf = durchrechnen(a, b, K, eps, D, C.K2_STUFEN)
    ex = loese_gibbs(a, b, K, eps)
    if ex.ok:
        P_ref, ref_art = ex.plan, "genauer Plan (Newton-Verfahren auf dem Dual, Randfehler unter 1e-12)"
    else:
        P_ref, ref_art = lauf[-1][1], f"Plan nach {ganz(C.K2_STUFEN[-1])} Durchläufen (das Newton-Verfahren ist nicht konvergiert)"
    d_ref = ableitung_implizit(P_ref, K, eps, D)
    abl_ref_null = bool(np.linalg.norm(d_ref) < ABL_REF_NULL * np.linalg.norm(P_ref) / eps)   # relativer Fehler gegen eine Referenz nahe null sagt nichts
    z = dict(k=list(C.K2_STUFEN), zeilenfehler=[], roh=[], gerundet=[], plan=[], durch=[], impl=[], plaene={}, gerundete_plaene={})
    for k, P, dP in lauf:
        R = runden(P, a, b)
        z["zeilenfehler"].append(_zeilenfehler_plan(P, a))
        z["roh"].append(luecke_von(P, inst, ref.opt))
        z["gerundet"].append(luecke_von(R, inst, ref.opt))
        z["plan"].append(relativer_fehler(P, P_ref))
        z["durch"].append(relativer_fehler(dP, d_ref))
        z["impl"].append(relativer_fehler(ableitung_implizit(P, K, eps, D), d_ref))
        z["plaene"][k], z["gerundete_plaene"][k] = P, R
    z.update(inst=inst, opt=ref.opt, eps=eps, ref_art=ref_art, plan_ref=P_ref, abl_ref_null=abl_ref_null, plan_lp=ref.plan_lp, luecke_ende=luecke_von(P_ref, inst, ref.opt), ref_ok=bool(ex.ok))
    return z


def reihe_runden() -> dict:
    """Tabelle Rundung: n = 60, ε/OPT = 0,46, 10 Instanzen. Je k aus K_RUNDEN und am Ende (Zeilenfehler unter 1e-8) die gerundete und die rohe Lücke und der Zeilenfehler je Instanz."""
    n, eps_rel = C.N_RUNDEN, C.EPS_RUNDEN
    ks = list(C.K_RUNDEN)
    rund, roh, zeile = [[] for _ in ks], [[] for _ in ks], [[] for _ in ks]
    ende, ende_roh, iter_ende = [], [], []
    for i in range(C.ANZAHL_RUNDEN):
        inst = make_instanz(n, n, C.SERIEN_SEED, C.ZWECK_RUNDEN, i)
        ref = referenz(inst)
        eps = eps_rel * ref.opt
        for j, (k, f, g) in enumerate(verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, ks)):
            P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g)
            rund[j].append(luecke_von(runden(P, inst.a, inst.b), inst, ref.opt))
            roh[j].append(luecke_von(P, inst, ref.opt))
            zeile[j].append(_zeilenfehler_plan(P, inst.a))
        sk = loese_sinkhorn(inst.a, inst.b, inst.kosten, eps, tol=1e-8, max_iter=30000)
        P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, sk.f, sk.g)
        ende.append(luecke_von(runden(P, inst.a, inst.b), inst, ref.opt))
        ende_roh.append(luecke_von(P, inst, ref.opt))
        iter_ende.append(sk.iterationen)
    return jsonbar(dict(n=n, eps_rel=eps_rel, anzahl=C.ANZAHL_RUNDEN, k=ks, gerundet=rund, roh=roh, zeilenfehler=zeile, ende=ende, ende_roh=ende_roh, iterationen_ende=iter_ende))


def reihe_schranke() -> dict:
    """Die Schranke der Rundung ⟨C, R⟩ ≤ ⟨C, P⟩ + 2 max C (‖P1 − a‖₁ + ‖Pᵀ1 − b‖₁) an Instanz (n = 20, 10 Instanzen) × 3 ε/OPT × 8 Iterationszahlen; dazu die Randverteilungen von R."""
    faelle = halten = rand_ok = 0
    kleinster = float("inf")
    for i in range(C.ANZAHL_SCHRANKE):
        inst = make_instanz(20, 20, C.SERIEN_SEED, C.ZWECK_SCHRANKE, i)
        ref = referenz(inst)
        for er in C.RUNDEN_SCHRANKE_EPS:
            eps = er * ref.opt
            for k, f, g in verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, C.K_RUNDEN_SCHRANKE):
                P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g)
                R = runden(P, inst.a, inst.b)
                abstand = schranke(P, inst.a, inst.b, inst.kosten) - float((R * inst.kosten).sum())
                faelle += 1
                halten += int(abstand >= -1e-12)
                kleinster = min(kleinster, abstand)
                rand_ok += int(np.abs(R.sum(axis=1) - inst.a).max() < 1e-12 and np.abs(R.sum(axis=0) - inst.b).max() < 1e-12 and (R >= 0).all())
    return jsonbar(dict(faelle=faelle, halten=halten, rand_ok=rand_ok, kleinster_abstand=kleinster, n=20, anzahl=C.ANZAHL_SCHRANKE))


def reihe_ableitung() -> dict:
    """Tabelle Ableitung: n = 20, 30 Instanzen (Massen 0,5/n + 0,5 · Dirichlet(1)), die Zeilen aus ZEILEN_ABL. Je Zeile und Instanz: Planfehler, Zeilenfehler, Fehler des Durchrechnens
    und der Formel am Plan P_k (alle relativ, Frobenius-Norm, gegen den genauen Plan bzw. dessen Ableitung); je ε/OPT der Fehler der Formel gegen die zentrale finite Differenz."""
    ergebnis = dict(n=C.N_ABL, anzahl=C.ANZAHL_ABL, zeilen=[], fd={})
    for er in sorted({e for e, _ in C.ZEILEN_ABL}, reverse=True):
        ks = sorted({k for e, k in C.ZEILEN_ABL if e == er})
        je_k = {k: dict(plan=[], zeile=[], durch=[], impl=[]) for k in ks}
        fd, nicht_ok = [], 0
        for i in range(C.ANZAHL_ABL):
            inst = make_instanz(C.N_ABL, C.N_ABL, C.SERIEN_SEED, C.ZWECK_ABL, i, C.S_ABL)
            ref = referenz(inst)
            eps = er * ref.opt
            D = richtung(C.N_ABL, C.N_ABL, C.SERIEN_SEED, i)
            ex = loese_gibbs(inst.a, inst.b, inst.kosten, eps)
            if not ex.ok:
                nicht_ok += 1
                continue
            d_ref = ableitung_implizit(ex.plan, inst.kosten, eps, D)
            d_fd = ableitung_fd(inst.a, inst.b, inst.kosten, eps, D)
            if d_fd is not None:
                fd.append(relativer_fehler(d_ref, d_fd))
            for k, P, dP in durchrechnen(inst.a, inst.b, inst.kosten, eps, D, ks):
                je_k[k]["plan"].append(relativer_fehler(P, ex.plan))
                je_k[k]["zeile"].append(_zeilenfehler_plan(P, inst.a))
                je_k[k]["durch"].append(relativer_fehler(dP, d_ref))
                je_k[k]["impl"].append(relativer_fehler(ableitung_implizit(P, inst.kosten, eps, D), d_ref))
        for k in ks:
            ergebnis["zeilen"].append(dict(eps_rel=er, k=k, **je_k[k]))
        ergebnis["fd"][f"{er:g}"] = dict(fehler=fd, nicht_ok=nicht_ok)
    return jsonbar(ergebnis)


def tabelle_ableitung(r: dict) -> list:
    """Je Zeile von ZEILEN_ABL die Mediane (und Faktoren, Zählungen) der Fehler. Gibt eine Liste von Dicts zurück."""
    zeilen = []
    for z in r["zeilen"]:
        plan, zeile, durch, impl = (np.array(z[k]) for k in ("plan", "zeile", "durch", "impl"))
        zeilen.append(dict(eps_rel=z["eps_rel"], k=z["k"], anzahl=len(plan), plan=float(np.median(plan)), zeile=float(np.median(zeile)), durch=float(np.median(durch)), impl=float(np.median(impl)),
                           faktor_zeile=float(np.median(durch / zeile)), faktor_plan=float(np.median(durch / plan)), faktor_impl=float(np.median(durch / impl)),
                           ueber=int((plan > C.SCHWELLE_ABL).sum())))
    return zeilen


def _zeile_abl(tab, eps_rel, k):
    for z in tab:
        if abs(z["eps_rel"] - eps_rel) < 1e-12 and z["k"] == k:
            return z
    return None


def verdikt_plan(runden_r: dict, schranke_r: dict):
    """Verdikt zu Experiment 2, Teil Plan (Art, Text), aus `reihe_runden` und `reihe_schranke`."""
    mi = {k: float(np.median(runden_r["gerundet"][j])) for j, k in enumerate(runden_r["k"])}
    ende = float(np.median(runden_r["ende"]))
    return "info", (f"Nach einer Iteration (zwei Produkte) hat der gerundete Plan bei ε/OPT = {eps_text(runden_r['eps_rel'])} eine Lücke von {prozent_vz(mi[1], 0)} (n = {runden_r['n']}, Median über "
                    f"{runden_r['anzahl']} Instanzen), nach 100 Iterationen {prozent_vz(mi[100], 1)}, am Ende (Zeilenfehler unter 1e-8) {prozent_vz(ende, 1)}; die Schranke der Rundung gilt in "
                    f"{schranke_r['halten']} von {schranke_r['faelle']} Fällen (n = {schranke_r['n']}). Die Rundung macht den Plan zulässig, sie ersetzt die Iteration nicht.")


def verdikt_ableitung(tab: list):
    """Verdikt zu Experiment 2, Teil Ableitung (Art, Text) aus `tabelle_ableitung`: Zählung statt Median für „reicht nicht“."""
    z03, z5, z20 = _zeile_abl(tab, 0.03, 1000), _zeile_abl(tab, 0.01, 5000), _zeile_abl(tab, 0.01, 20000)
    N = z5["anzahl"]
    faktoren = [z["faktor_impl"] for z in tab if z["durch"] < 1e-2 and z["durch"] > 1e-10]
    lo, hi = (min(faktoren), max(faktoren)) if faktoren else (float("nan"), float("nan"))
    return "info", (f"Bei ε/OPT = 0,03 und 1 000 Durchläufen liegt der Zeilenfehler im Median bei {de_g(z03['zeile'], 2)}, der Ableitungsfehler (Durchrechnen) bei {de_g(z03['durch'], 2)} "
                    f"(im Median der Instanzen {de(z03['faktor_zeile'], 0)}-mal so groß wie der Zeilenfehler der jeweiligen Instanz); bei ε/OPT = 0,01 reichen 5 000 Durchläufe in {z5['ueber']} von {N} Instanzen nicht "
                    f"(Planfehler über 1e-4), auch 20 000 nicht in {z20['ueber']} von {N}. "
                    f"Die Genauigkeit der Ableitung folgt der des Plans: Im Auslauf (Zeilen der Tabelle mit einem Median des Ableitungsfehlers (Durchrechnen) zwischen 1e-10 und 1e-2) liegt das Durchrechnen "
                    f"in der typischen Instanz (Median der Quotienten je Instanz) um den Faktor {de(lo, 1)} bis {de(hi, 1)} über dem Rechenweg „implizit am unfertigen Plan“ (n = {C.N_ABL}, {N} Instanzen); "
                    f"die Mediane der Tabelle unterscheiden sich in der Zeile ε/OPT = 0,01, k = 5 000 um den Faktor {de(z5['durch'] / z5['impl'], 0)}, die Quotienten streuen zwischen den Instanzen stark.")


def verdikt_stimmt_ableitung(tab: list):
    """Kernfrage „Wie genau ist die Ableitung?“ (Art, Text), kurz."""
    z5 = _zeile_abl(tab, 0.01, 5000)
    z03 = _zeile_abl(tab, 0.03, 1000)
    return "info", (f"Bei ε/OPT = 0,03 und 1 000 Durchläufen ist der Ableitungsfehler im Median der Instanzen {de(z03['faktor_zeile'], 0)}-mal so groß wie der Zeilenfehler des Abbruchtests der jeweiligen Instanz; bei ε/OPT = 0,01 "
                    f"reichen 5 000 Durchläufe in {z5['ueber']} von {z5['anzahl']} Instanzen nicht (Planfehler über 1e-4; n = {C.N_ABL}, Experiment 2).")


def verdikt_ansicht2(z: dict, s: Settings):
    """Zusammenfassung des Laufs der Seitenleiste (Art, Text) bei k = s.k2."""
    i = z["k"].index(s.k2)
    durch, impl = z["durch"][i], z["impl"][i]
    gewaehlt = durch if s.ableitung2 == C.ABLEITUNG2[0] else impl
    abl = (f"Ableitungsfehler {de_g(gewaehlt, 2)} ({s.ableitung2})" if not z["abl_ref_null"]
           else "Ableitungsfehler nicht angegeben (die Referenzableitung ist praktisch null, ein relativer Fehler sagt hier nichts)")
    return "info", (f"Nach {s.k2} Durchläufen ({2 * s.k2} Produkte) bei ε/OPT = {eps_text(s.eps2)} ({z['inst'].n} Lager, {z['inst'].m} Kunden): Zeilenfehler {de_g(z['zeilenfehler'][i], 2)}, "
                    f"Planfehler {de_g(z['plan'][i], 2)}, {abl}; gerundete Lücke {prozent_vz(z['gerundet'][i], 2)}, rohe Lücke {prozent_vz(z['roh'][i], 2)}, "
                    f"Lücke des genauen Plans {prozent_vz(z['luecke_ende'], 2)}.")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Gleichstand
# ----------------------------------------------------------------------------------------------------------------------------------------

def dev_von(P, n: int) -> float:
    """dev = n · max_ij min(P_ij, 1/n − P_ij): 0 bei einer Permutationsmatrix, ½ bei einem Eintrag halb/halb (Massen 1/n)."""
    return float(n * np.max(np.minimum(P, 1.0 / n - P)))


def reihe_gleichstand(art: str, eps: float) -> dict:
    """Experiment 3: n = 10, 60 Instanzen, Massen 1/n. Je Instanz: eindeutiges Optimum (exakter Test), dev des weichen Plans, L1-Abstand zur LP-Ecke (eigener Transportsimplex),
    Zeilenfehler 1e-9 in 20 000 Iterationen erreicht, Iterationen."""
    n = C.N3
    eindeutig, dev, l1, konv, iters = [], [], [], [], []
    for i in range(C.ANZAHL3):
        inst = make_gleichstand(art, n, C.SERIEN_SEED, i)
        lo = loese_transport(inst.a, inst.b, inst.kosten, "steilste")
        eindeutig.append(ist_eindeutig(inst.kosten, lo.plan, lo.u, lo.v))
        sk = loese_sinkhorn(inst.a, inst.b, inst.kosten, eps, tol=C.TOL3, max_iter=C.MAX_ITER3)
        P = plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, sk.f, sk.g)
        dev.append(dev_von(P, n))
        l1.append(float(np.abs(P - lo.plan).sum()))
        konv.append(bool(sk.konvergiert))
        iters.append(sk.iterationen)
    return jsonbar(dict(kosten=art, eps=eps, n=n, anzahl=C.ANZAHL3, eindeutig=eindeutig, dev=dev, l1=l1, konvergiert=konv, iterationen=iters))


def aggregiere_gleichstand(r: dict) -> dict:
    """Zählungen: mehrere Optima, davon nicht ganzzahlig (dev über 0,02); eindeutige Instanzen, davon ganzzahlig; konvergiert (insgesamt, je Gruppe); Mediane."""
    eind = np.array(r["eindeutig"], bool)
    dev, l1, konv = np.array(r["dev"]), np.array(r["l1"]), np.array(r["konvergiert"], bool)
    mehr = ~eind
    nicht_ganz = dev > C.DEV_GRENZE
    return dict(anzahl=len(dev), mehrere=int(mehr.sum()), mehrere_nicht_ganzzahlig=int((mehr & nicht_ganz).sum()), eindeutig=int(eind.sum()), eindeutig_ganzzahlig=int((eind & ~nicht_ganz).sum()),
                konvergiert=int(konv.sum()), konvergiert_mehrere=int((konv & mehr).sum()), konvergiert_eindeutig=int((konv & eind).sum()),
                dev_median_mehrere=float(np.median(dev[mehr])) if mehr.any() else None, dev_median_eindeutig=float(np.median(dev[eind])) if eind.any() else None,
                l1_median_mehrere=float(np.median(l1[mehr])) if mehr.any() else None, kosten=r["kosten"], eps=r["eps"], n=r["n"])


def verdikt_gleichstand(z: dict):
    """Verdikt zu Experiment 3 (Art, Text) für die gewählte Kostenart und ε aus `aggregiere_gleichstand`."""
    N = z["anzahl"]
    teile = [f"Kosten „{z['kosten']}“, ε = {de_g(z['eps'], 2)}, n = {z['n']}, {N} Instanzen:"]
    if z["mehrere"]:
        teile.append(f"{z['mehrere']} haben mehrere optimale Pläne; bei {z['mehrere_nicht_ganzzahlig']} davon ist der Sinkhorn-Plan nicht ganzzahlig (dev über 0,02, Median {de(z['dev_median_mehrere'], 3)}; "
                     f"L1-Abstand zur gefundenen Ecke des LP im Median {de(z['l1_median_mehrere'], 2)}).")
    else:
        teile.append("keine hat mehrere optimale Pläne.")
    if z["eindeutig"]:
        teile.append(f"Bei den {z['eindeutig']} Instanzen mit eindeutigem Optimum ist der Plan nach höchstens {ganz(C.MAX_ITER3)} Iterationen in {z['eindeutig_ganzzahlig']} ganzzahlig (dev höchstens 0,02).")
    teile.append(f"Der Zeilenfehler {tol_text(C.TOL3)} wird in {ganz(C.MAX_ITER3)} Iterationen in {z['konvergiert']} von {N} Instanzen erreicht"
                 + (f" (davon {z['konvergiert_mehrere']} bei mehreren Optima)." if z["mehrere"] else "."))
    if z["kosten"] == C.KOSTEN3[3]:
        teile.append(f"In dieser Zeile steht eine einzige Instanz (konstante Kosten, Massen 1/n), {N}-mal gerechnet: die Zählungen bedeuten dasselbe Ergebnis einmal, keine Stichprobe.")
    return "info", " ".join(teile)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Aufwand in Produkten
# ----------------------------------------------------------------------------------------------------------------------------------------

def lauf_aufwand(inst: Instanz, opt: float, eps: float) -> dict:
    """Iteration der Log-Domäne bis Zeilenfehler unter 1e-6 (höchstens 30 000 Iterationen); dabei alle 10 Iterationen die Lücke des gerundeten Plans.
    Gibt die Produkte (2 × Iterationen) bis zum Zeilenfehler 1e-6 und bis zur ersten gerundeten Lücke ≤ 1 % (None: nicht erreicht) und die gerundete Lücke am Ende zurück."""
    a, b, K = inst.a, inst.b, inst.kosten
    la, lb = np.log(a), np.log(b)
    CT = np.ascontiguousarray(K.T)
    f, g = -eps * la, -eps * lb                       # Start u = v = 1
    k = 0
    p_zeile = p_luecke = None
    while True:
        f_neu = aktualisiere_f(lb, g, K, eps)
        if k > 0:
            with np.errstate(over="ignore"):
                fehler = float(np.sum(a * np.abs(np.exp((f - f_neu) / eps) - 1.0)))
            if (k % C.RASTER4 == 0 or fehler < C.TOL4) and p_luecke is None:
                P = plan_aus_potentialen(a, b, K, eps, f, g)
                if luecke_von(runden(P, a, b), inst, opt) <= C.LUECKE4:
                    p_luecke = 2 * k
            if fehler < C.TOL4:
                p_zeile = 2 * k
                break
            if k >= C.MAX_ITER4:
                break
        f = f_neu
        g = aktualisiere_g(la, f, CT, eps)
        k += 1
    P = plan_aus_potentialen(a, b, K, eps, f, g)
    return dict(zeile=p_zeile, luecke=p_luecke, luecke_ende=luecke_von(runden(P, a, b), inst, opt), iterationen=k)


def reihe_aufwand(n: int, eps_rel: float) -> dict:
    """Experiment 4: je Instanz (Dirichlet(1)-Massen) die Produkte bis zum Ziel; Instanzzahl nach ANZAHL4."""
    zeile, luecke, ende, opts = [], [], [], []
    for i in range(C.ANZAHL4[n]):
        inst = make_instanz(n, n, C.SERIEN_SEED, C.ZWECK_AUFWAND, i)
        opt = referenz(inst).opt
        r = lauf_aufwand(inst, opt, eps_rel * opt)
        zeile.append(r["zeile"])
        luecke.append(r["luecke"])
        ende.append(r["luecke_ende"])
        opts.append(opt)
    return jsonbar(dict(n=n, eps_rel=eps_rel, anzahl=len(zeile), zeile=zeile, luecke=luecke, luecke_ende=ende, opt=opts))


def aggregiere_aufwand(r: dict, ziel: str) -> dict:
    """Median [Minimum; Maximum] der Produkte der Instanzen, die das Ziel erreichen, und die Zahl der Instanzen, die es erreichen; Multiplikationen 2 n² k = n² · Produkte (Median)."""
    werte = [w for w in r["zeile" if ziel == C.ZIEL4[0] else "luecke"] if w is not None]
    n = r["n"]
    z = dict(n=n, eps_rel=r["eps_rel"], anzahl=r["anzahl"], erreicht=len(werte), produkte=band(werte) if werte else None)
    z["multiplikationen"] = n * n * z["produkte"][0] if werte else None
    return z


def faktor_n4(zeilen: list, n_von: int = 20, n_nach: int = 80):
    """Faktor der Median-Produkte bei vierfachem n (n = 20 → 80); None, wenn eine der beiden Zeilen kein Ziel erreicht."""
    a = next((z for z in zeilen if z["n"] == n_von), None)
    b = next((z for z in zeilen if z["n"] == n_nach), None)
    if not a or not b or a["produkte"] is None or b["produkte"] is None:
        return None
    return b["produkte"][0] / a["produkte"][0]


def luecke_ist_endplan_eigenschaft(rs: dict) -> bool:
    """True, wenn in jeder Reihe genau die Instanzen die Lücke ≤ 1 % erreichen, deren gerundeter Endplan (Zeilenfehler 1e-6) eine Lücke ≤ 1 % hat: dann hängt die Zählung an ε/OPT, nicht an der Iterationszahl."""
    return all([x is not None for x in r["luecke"]] == [e <= C.LUECKE4 for e in r["luecke_ende"]] for r in rs.values())


def anzahl_faktor_n4(zeilen: list, n_von: int = 20, n_nach: int = 80) -> tuple:
    """Zahl der Instanzen, die das Ziel erreichen und in den Median des Faktors `faktor_n4` eingehen (bei n_von und bei n_nach)."""
    a = next(z for z in zeilen if z["n"] == n_von)
    b = next(z for z in zeilen if z["n"] == n_nach)
    return a["erreicht"], b["erreicht"]


def verdikt_aufwand(rs: dict):
    """Verdikt zu Experiment 4 (Art, Text): Produkte bis Zeilenfehler 1e-6 bei ε/OPT = 0,07 für alle n, Faktor bei n × 4 (für alle drei ε), und wie viele Instanzen die Lücke ≤ 1 % erreichen.
    `rs`: {(n, ε/OPT): reihe_aufwand}."""
    e = 0.07
    z = [aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0]) for n in C.N4_STUFEN]
    l = [aggregiere_aufwand(rs[(n, e)], C.ZIEL4[1]) for n in C.N4_STUFEN]
    faktoren = [faktor_n4([aggregiere_aufwand(rs[(n, er)], C.ZIEL4[0]) for n in C.N4_STUFEN]) for er in C.EPS4_STUFEN]
    return "info", (f"Bei ε/OPT = {de_g(e, 2)} braucht der Zeilenfehler 1e-6 im Median " + " / ".join(ganz(x["produkte"][0]) for x in z) + " Matrix-Vektor-Produkte für n = "
                    + " / ".join(str(n) for n in C.N4_STUFEN) + " (" + " / ".join(str(x["anzahl"]) for x in z) + f" Instanzen), bei n × 4 (n = 20 → 80) den Faktor {de(min(faktoren), 1)} bis {de(max(faktoren), 1)} "
                    f"(je nach ε/OPT {', '.join(de(f, 1) for f in faktoren)} bei ε/OPT {', '.join(de_g(er, 2) for er in C.EPS4_STUFEN)}; die Unterschiede zwischen den drei ε/OPT liegen innerhalb der Streuung der Instanzen); „Lücke ≤ 1 %“ erreichen bei ε/OPT = {de_g(e, 2)} nur "
                    + " / ".join(str(x["erreicht"]) for x in l) + " der Instanzen"
                    + (", weil bei diesem ε/OPT die Lücke des gerundeten Endplans (Zeilenfehler 1e-6) bei den übrigen schon über 1 % liegt; mehr Iterationen ändern das nicht" if luecke_ist_endplan_eigenschaft(rs) else "")
                    + ". Verglichen wird nur, wie die Zahl der Produkte mit n und 1/ε wächst; ein Vergleich mit den Schritten eines "
                    "Simplexverfahrens ist hier nicht gezeigt (Verweis auf die Demo zu PDLP und auf Dong u. a. 2020).")


def zeilenfehler_zwei_mal_zwei(k: int) -> float:
    """Handbeispiel „weit entferntes Lager“: Zeilenfehler der Log-Domäne nach k Durchläufen (ε = 1/110); er ist 1/(2k+1) und hängt für kleine ε nicht von ε ab (bei ε ≥ 0,05 nicht mehr)."""
    inst = handbeispiel_weit()
    _, f, g = verlauf_potentiale(inst.a, inst.b, inst.kosten, 1 / 110, [k])[0]
    return zeilenfehler(plan_aus_potentialen(inst.a, inst.b, inst.kosten, 1 / 110, f, g), inst.a)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Absatz: weiche Permutation, weiche Ränge
# ----------------------------------------------------------------------------------------------------------------------------------------

def genaue_plaene(a, K, eps_absteigend):
    """Genaue Pläne (Newton-Verfahren auf dem Dual, Randfehler unter 1e-12) zu einer absteigenden Liste von ε mit Fortsetzung: jeder Plan startet von dem des vorigen ε, bei Misserfolg erst mit
    Zwischenstufen (`sgz_newton._vorwaerts`), dann nach 20 000 Iterationen der Log-Domäne von diesem Stand. Scheitert eine Stufe, startet die nächste vom letzten gelungenen Stand.
    Gibt eine Liste von (ε, NewtonErgebnis) zurück; `ok` False heißt: nicht konvergiert."""
    ergebnisse = []
    letzter = None                                     # (ε, f, g) der letzten gelungenen Stufe
    for e in eps_absteigend:
        if letzter is None:
            f0, g0 = kaltstart(a, a, K, e)
            erg = loese_gibbs_newton(a, a, K, e, f0, g0)
        else:
            erg = _vorwaerts(a, a, K, letzter, e)
            if not erg.ok:
                sk = loese_sinkhorn(a, a, K, e, tol=1e-7, max_iter=20000, f0=letzter[1], g0=letzter[2])
                erg = loese_gibbs_newton(a, a, K, e, sk.f, sk.g)
        if erg.ok:
            letzter = (e, erg.f, erg.g)
        ergebnisse.append((e, erg))
    return ergebnisse


def reihe_raenge(n: int) -> dict:
    """Weiche Ränge: n Werte gleichverteilt in (0, 1), je ε/OPT aus EPS_RAENGE der genaue Plan (Newton-Verfahren mit Fortsetzung); nur konvergierte Instanzen zählen mit.
    Je (Instanz, ε/OPT): konvergiert, dev, Betrag des Rangfehlers (Mittel über die Ränge), Fehler bei Rang 1 und Rang n, Anteil richtig gerundeter Ränge."""
    stufen = sorted(C.EPS_RAENGE, reverse=True)
    eintraege = {f"{e:g}": dict(konv=[], dev=[], betrag=[], rang1=[], rangn=[], richtig=[], summe=[]) for e in stufen}
    for i in range(C.ANZAHL_RAENGE[n]):
        x = make_werte(n, C.SERIEN_SEED, i)
        a, K, opt, rang = raenge_kosten(x)
        pfad = genaue_plaene(a, K, [e * opt for e in stufen])
        for e, (_, erg) in zip(stufen, pfad):
            d = eintraege[f"{e:g}"]
            d["konv"].append(bool(erg.ok))
            if not erg.ok:
                d["dev"].append(None), d["betrag"].append(None), d["rang1"].append(None), d["rangn"].append(None), d["richtig"].append(None), d["summe"].append(None)
                continue
            P = erg.plan
            weich = n * (P * np.arange(1, n + 1)[None, :]).sum(axis=1)
            fehler = weich - rang
            d["dev"].append(dev_von(P, n))
            d["betrag"].append(float(np.abs(fehler).mean()))
            d["rang1"].append(float(fehler[rang == 1][0]))
            d["rangn"].append(float(fehler[rang == n][0]))
            d["richtig"].append(float((np.round(weich) == rang).mean()))
            d["summe"].append(float(weich.sum()))
    return jsonbar(dict(n=n, anzahl=C.ANZAHL_RAENGE[n], eps=eintraege))


def reihe_raenge_gleichstand(anzahl: int = 100, n: int = 10) -> dict:
    """Gleichstände: n = 10 Werte mit zwei Paaren gleicher Werte, je ε/OPT 1, 0,1, 0,01: größter Unterschied der Zeilen eines Paares und größte Abweichung des weichen Rangs vom Mittelrang."""
    stufen = (1.0, 0.1, 0.01)
    erg = {f"{e:g}": dict(konv=[], zeilen=[], abw=[]) for e in stufen}
    for i in range(anzahl):
        x = make_werte(n - 2, C.SERIEN_SEED, 1000 + i)
        x = np.concatenate([[x[0], x[0], x[1], x[1]], x[2:]])
        a, K, opt, rang = raenge_kosten(x)
        pfad = genaue_plaene(a, K, [e * opt for e in sorted(stufen, reverse=True)])
        for e, (_, r) in zip(sorted(stufen, reverse=True), pfad):
            d = erg[f"{e:g}"]
            d["konv"].append(bool(r.ok))
            if not r.ok:
                d["zeilen"].append(None), d["abw"].append(None)
                continue
            P = r.plan
            weich = n * (P * np.arange(1, n + 1)[None, :]).sum(axis=1)
            d["zeilen"].append(float(max(np.abs(P[0] - P[1]).max(), np.abs(P[2] - P[3]).max())))
            d["abw"].append(float(max(abs(weich[0] - (rang[0] + rang[1]) / 2), abs(weich[2] - (rang[2] + rang[3]) / 2))))
    return jsonbar(dict(n=n, anzahl=anzahl, eps=erg))


def tabelle_raenge(r: dict) -> list:
    """Je ε/OPT: Zahl konvergiert, Anteil fast ganzzahlig (dev ≤ 0,02) unter den konvergierten, Median dev, mittlerer Betrag des Rangfehlers, Fehler bei Rang 1 und Rang n (Mittel), Anteil richtig gerundet."""
    zeilen = []
    for e in sorted(C.EPS_RAENGE, reverse=True):
        d = r["eps"][f"{e:g}"]
        ok = [i for i, k in enumerate(d["konv"]) if k]
        dev = np.array([d["dev"][i] for i in ok])
        zeilen.append(dict(eps_rel=e, n=r["n"], anzahl=r["anzahl"], konvergiert=len(ok),
                           fast_ganz=int((dev <= C.DEV_GRENZE).sum()) if ok else 0, anteil_fast_ganz=float((dev <= C.DEV_GRENZE).mean()) if ok else None,
                           dev_median=float(np.median(dev)) if ok else None,
                           betrag=float(np.mean([d["betrag"][i] for i in ok])) if ok else None, rang1=float(np.mean([d["rang1"][i] for i in ok])) if ok else None,
                           rangn=float(np.mean([d["rangn"][i] for i in ok])) if ok else None, richtig=float(np.mean([d["richtig"][i] for i in ok])) if ok else None,
                           summe_abw=float(max(abs(d["summe"][i] - r["n"] * (r["n"] + 1) / 2) for i in ok)) if ok else None))
    return zeilen


def zeile_raenge(z: dict) -> list:
    """Zellen einer Zeile der Rangtabelle (ε/OPT, konvergiert, fast ganzzahlig, dev, Betrag, Fehler bei Rang 1 und Rang n, richtig gerundet); „–“, wenn nichts konvergiert."""
    if not z["konvergiert"]:
        return [eps_text(z["eps_rel"]), zaehlung(0, z["anzahl"])] + ["–"] * 6
    return [eps_text(z["eps_rel"]), zaehlung(z["konvergiert"], z["anzahl"]), zaehlung(z["fast_ganz"], z["konvergiert"]), de(z["dev_median"], 3), de(z["betrag"], 2),
            de_vz(z["rang1"], 2), de_vz(z["rangn"], 2), prozent(z["richtig"], 0)]


def _raenge_zeile(tab, e):
    return next(z for z in tab if abs(z["eps_rel"] - e) < 1e-12)


def verdikt_raenge(tabs: dict):
    """Verdikt des Absatzes (Art, Text) aus `tabelle_raenge` je n (Schlüssel 10, 30, 100); ε/OPT = 0,01 als Vergleichsstelle."""
    z30, z10 = _raenge_zeile(tabs[30], 0.01), _raenge_zeile(tabs[10], 0.01)
    return "info", (f"Bei 30 Werten ist bis ε/OPT = 0,01 in {z30['fast_ganz']} der {z30['konvergiert']} konvergierten Instanzen der Plan fast eine Permutation (dev ≤ 0,02); bei 10 Werten sind es "
                    f"{z10['fast_ganz']} der {z10['konvergiert']} konvergierten Instanzen. Weiche Ränge liegen am Rand zur Mitte hin verschoben (Rang 1 zu hoch, Rang n zu niedrig), und der Fehler "
                    "wächst mit n; gleiche Werte bekommen identische Zeilen und den Mittelrang. Gilt für gleichverteilte Werte, quadrierten Abstand, Massen 1/n und die gezählten Instanzen.")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Vorrechnung
# ----------------------------------------------------------------------------------------------------------------------------------------

def vorzurechnen():
    """Liste (Name, Parameter, Funktion) aller vorgerechneten Reihen; `tools/vorrechnen.py` schreibt sie nach `tools/vorgerechnet.json`."""
    liste = [("unterlauf", {}, reihe_unterlauf), ("runden", {}, reihe_runden), ("schranke", {}, reihe_schranke), ("ableitung", {}, reihe_ableitung)]
    for art in C.KOSTEN3:
        for eps in C.EPS3_STUFEN:
            liste.append(("gleichstand", dict(kosten=art, eps=eps), lambda art=art, eps=eps: reihe_gleichstand(art, eps)))
    for n in C.N4_STUFEN:
        for eps in C.EPS4_STUFEN:
            liste.append(("aufwand", dict(n=n, eps=eps), lambda n=n, eps=eps: reihe_aufwand(n, eps)))
    for n in C.N_RAENGE:
        liste.append(("raenge", dict(n=n), lambda n=n: reihe_raenge(n)))
    liste.append(("raenge_gleich", {}, reihe_raenge_gleichstand))
    return liste


def _vorgerechnet_oder(name: str, params: dict, rechne):
    """Der vorgerechnete Eintrag aus `tools/vorgerechnet.json` oder (wenn er fehlt) die Rechnung selbst."""
    from sgz_vorgerechnet import hole

    d = hole(name, **params)
    return d if d is not None else jsonbar(rechne())


def unterlauf_v():
    return _vorgerechnet_oder("unterlauf", {}, reihe_unterlauf)


def runden_v():
    return _vorgerechnet_oder("runden", {}, reihe_runden)


def schranke_v():
    return _vorgerechnet_oder("schranke", {}, reihe_schranke)


def ableitung_v():
    return _vorgerechnet_oder("ableitung", {}, reihe_ableitung)


def gleichstand_v(art: str, eps: float):
    return _vorgerechnet_oder("gleichstand", dict(kosten=art, eps=eps), lambda: reihe_gleichstand(art, eps))


def aufwand_v(n: int, eps: float):
    return _vorgerechnet_oder("aufwand", dict(n=n, eps=eps), lambda: reihe_aufwand(n, eps))


def raenge_v(n: int):
    return _vorgerechnet_oder("raenge", dict(n=n), lambda: reihe_raenge(n))


def raenge_gleich_v():
    return _vorgerechnet_oder("raenge_gleich", {}, reihe_raenge_gleichstand)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Tabellen (die App und die Tests der README bilden ihre Zeilen mit denselben Funktionen)
# ----------------------------------------------------------------------------------------------------------------------------------------

def md_tabelle(kopf, zeilen) -> str:
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


def md_zeile(zellen) -> str:
    """Eine Tabellenzeile, wie sie in `md_tabelle` und in der README steht."""
    return "| " + " | ".join(zellen) + " |"


KOPF_UNTERLAUF = ["ε", "ε/OPT (Median)", "float32: fertig", "Iterationsgrenze", "Ausfall", "Abweichung > 1e-6 / 1e-4 / 1e-2 (größte)", "float64: fertig / Grenze / Ausfall", "Log-Domäne: fertig / Grenze"]


def zellen_unterlauf(z: dict) -> list:
    N = z["anzahl"]
    return [de_g(z["eps"], 2), eps_text(z["eps_opt"]), zaehlung(z["f32"]["ok"], N), str(z["f32"]["cap"]), str(z["f32"]["unterlauf"]),
            f"{z['f32']['ueber'][0]} / {z['f32']['ueber'][1]} / {z['f32']['ueber'][2]} (größte {de_g(z['f32']['max_abweichung'], 2)})",
            f"{z['f64']['ok']} / {z['f64']['cap']} / {z['f64']['unterlauf']}", f"{z['log']['ok']} / {z['log']['cap']}"]


KOPF_HAND = ["ε", "größter Eintrag von K in Zeile 2", "float32", "float64", "Log-Domäne: Zeilenfehler nach 50 Durchläufen"]


def zellen_hand(h: dict) -> list:
    return [f"1/{h['nenner']}", de_g(h["eintrag"], 2), h["float32"], h["float64"], de_g(h["zeilenfehler"], 3)]


KOPF_RUNDEN = ["Durchläufe k (Produkte)", "gerundete Lücke", "rohe Lücke", "Zeilenfehler (Median)"]


def zeilen_runden(r: dict) -> list:
    """Zellen der Zeilen der Rundungstabelle (je k aus K_RUNDEN und die Zeile „Ende“)."""
    zeilen = [[f"{k} ({2 * k})", band_prozent(r["gerundet"][j]), band_prozent(r["roh"][j]), de_g(band(r["zeilenfehler"][j])[0], 2)] for j, k in enumerate(r["k"])]
    return zeilen + [["Ende (Zeilenfehler unter 1e-8)", band_prozent(r["ende"]), band_prozent(r["ende_roh"]), "–"]]


KOPF_ABLEITUNG = ["ε/OPT", "k", "Zeilenfehler", "Planfehler", "Ableitung: Durchrechnen", "Ableitung: Formel am Plan P_k", "Durchrechnen / Zeilenfehler (Median der Quotienten je Instanz)", "Planfehler über 1e-4"]


def zellen_ableitung(z: dict) -> list:
    return [eps_text(z["eps_rel"]), ganz(z["k"]), de_g(z["zeile"], 2), de_g(z["plan"], 2), de_g(z["durch"], 2), de_g(z["impl"], 2), de(z["faktor_zeile"], 0), zaehlung(z["ueber"], z["anzahl"])]


def fd_zusammenfassung(abl: dict) -> tuple:
    """(Median, Maximum, Zahl) der relativen Fehler der Formel gegen die finite Differenz über alle Instanzen und ε/OPT."""
    fd = sorted(x for v in abl["fd"].values() for x in v["fehler"])
    return float(np.median(fd)), float(fd[-1]), len(fd)


KOPF_GLEICH = ["ε", "mehrere optimale Pläne", "davon nicht ganzzahlig (dev > 0,02)", "dev im Median (mehrere)", "eindeutiges Optimum", "davon ganzzahlig", f"Zeilenfehler {tol_text(C.TOL3)} erreicht"]


def zellen_gleichstand(z: dict) -> list:
    return [de_g(z["eps"], 2), zaehlung(z["mehrere"], z["anzahl"]), str(z["mehrere_nicht_ganzzahlig"]), de(z["dev_median_mehrere"], 3) if z["dev_median_mehrere"] is not None else "–",
            zaehlung(z["eindeutig"], z["anzahl"]), str(z["eindeutig_ganzzahlig"]), zaehlung(z["konvergiert"], z["anzahl"])]


KOPF_AUFWAND = ["n", "Instanzen, die das Ziel erreichen", "Produkte (Median [Minimum; Maximum])", "n² · Produkte (Median)"]


def zellen_aufwand(z: dict, hervorgehoben: bool = False) -> list:
    return [("**" + str(z["n"]) + "**") if hervorgehoben else str(z["n"]), zaehlung(z["erreicht"], z["anzahl"]), band_ganz(z["produkte"]) if z["produkte"] else "–",
            ganz(z["multiplikationen"]) if z["multiplikationen"] else "–"]


def kopf_raenge(n: int) -> list:
    return ["ε/OPT", "konvergiert", "fast ganzzahlig", "dev im Median", "mittlerer Betrag des Rangfehlers", "Rang 1 (Mittel)", f"Rang {n} (Mittel)", "richtig gerundete Ränge"]


def raenge_gleichstand_text(rg: dict) -> str:
    """Der Satz zu den Gleichständen (App und README): identische Zeilen, Abweichung vom Mittelrang im Median je ε/OPT."""
    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])

    zeilen = max(x for d in rg["eps"].values() for x in d["zeilen"] if x is not None)
    abw = ", ".join(f"{de_g(med(d['abw']), 2)} bei ε/OPT = {eps_text(float(e))} ({sum(d['konv'])} von {rg['anzahl']} konvergiert)" for e, d in rg["eps"].items())
    return (f"**Gleichstände** (n = {rg['n']}, zwei Paare gleicher Werte, {rg['anzahl']} Instanzen): die Zeilen der gleichen Werte sind identisch (größter Unterschied zweier Zeilen eines Paares höchstens "
            f"{de_g(zeilen, 2)}), der weiche Rang nähert sich dem Mittelrang: größte Abweichung im Median {abw}. Das ist die weiche Seite des Gleichstands aus Experiment 3. "
            "Die Summe der weichen Ränge ist exakt n(n+1)/2 (Spaltensummen 1/n).")


def schranke_text(s: dict) -> str:
    """Satz zur Schranke der Rundung (App und README)."""
    return (f"Die Schranke der Rundung ⟨C, R⟩ ≤ ⟨C, P⟩ + 2 · max C · (‖P1 − a‖₁ + ‖Pᵀ1 − b‖₁) gilt in {s['halten']} von {s['faelle']} Fällen "
            f"({s['anzahl']} Instanzen mit n = {s['n']} × 3 Werte ε/OPT × 8 Iterationszahlen); die Randverteilungen des gerundeten Plans stimmen in {s['rand_ok']} von {s['faelle']} (auf 1e-12).")


def fd_text(abl: dict) -> str:
    """Satz zur Formel gegen die finite Differenz (App und README)."""
    med, gross, n = fd_zusammenfassung(abl)
    return (f"Die Formel am genauen Plan stimmt mit der zentralen finiten Differenz (mit Richardson-Verbesserung, h = 1e-2 · ε und 5e-3 · ε) überein: relativer Fehler im Median {de_g(med, 2)}, "
            f"größter {de_g(gross, 2)} ({n} Vergleiche; bei kleinem ε/OPT ist die Ableitung klein gegen das Rauschen der genauen Pläne).")


def eps_faktor_text(rs: dict) -> str:
    """Satz zum Wachstum mit 1/ε: je Halbierung von ε/OPT der Faktor der Median-Produkte (Zeilenfehler 1e-6), kleinster und größter Wert über alle n."""
    faktoren = []
    for n in C.N4_STUFEN:
        med = [aggregiere_aufwand(rs[(n, e)], C.ZIEL4[0])["produkte"][0] for e in C.EPS4_STUFEN]
        faktoren += [med[i + 1] / med[i] for i in range(len(med) - 1)]
    return f"Je Halbierung von ε/OPT wächst der Median der Produkte bis Zeilenfehler 1e-6 um den Faktor {de(min(faktoren), 2)} bis {de(max(faktoren), 2)} (alle n, {len(faktoren)} Vergleiche)."
