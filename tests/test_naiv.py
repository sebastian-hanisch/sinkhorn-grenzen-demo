"""Die naive Iteration im Kern-Bereich und der Unterlauf: Handrechnungen (Schwellen des Gleitkommaformats, das weit entfernte Lager, die Auflösung von float32), die Einheiten einzeln, Randfälle."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import sgz_constants as C
from sgz_naiv import STATUS_CAP, STATUS_OK, STATUS_UNTERLAUF, hat_unterlauf, kern, kleinste_aufloesung, naive_iteration, zeilenfehler_naiv
from sgz_scenario import handbeispiel_weit
from sgz_sinkhorn import loese_sinkhorn, plan_aus_potentialen, verlauf_potentiale, zeilenfehler


def test_schwellen_des_gleitkommaformats_exp_minus_745_und_minus_104():
    """exp(−x) ist in float64 exakt null ab x ≈ 745,13 (darunter noch ein denormaler Wert), in float32 ab x ≈ 104; für C = 1 heißt das ε < 1/745 = 0,00134 bzw. ε < 1/104 = 0,0096."""
    assert np.exp(-745.0) > 0 and np.exp(-746.0) == 0.0
    assert np.exp(-745.13) == 5e-324 and np.exp(-745.14) == 0.0                 # denormal bis x ≈ 745,133
    assert np.exp(np.float32(-103.0)) > 0 and np.exp(np.float32(-104.0)) == 0.0
    C1 = np.array([[1.0]])
    assert kern(C1, 0.0014, np.float64)[0, 0] > 0 and kern(C1, 0.0013, np.float64)[0, 0] == 0
    assert kern(C1, 0.0098, np.float32)[0, 0] > 0 and kern(C1, 0.0096, np.float32)[0, 0] == 0
    assert abs(1 / 745.13 - 0.001342) < 1e-6 and abs(1 / 103.97 - 0.009619) < 1e-6


def test_weit_entferntes_lager_float32_scheitert_bei_eps_1_durch_110_float64_nicht(weit):
    a, b, K = weit.a, weit.b, weit.kosten
    r32 = naive_iteration(a, b, K, 1 / 110, np.float32, 1e-8, 50, "float64")
    r64 = naive_iteration(a, b, K, 1 / 110, np.float64, 1e-8, 50, "float64")
    assert r32.status == STATUS_UNTERLAUF and r32.plan is None                  # exp(-110) = 1,7e-48 ist in float32 null: Zeile 2 von K ist null
    assert np.exp(-110.0) == pytest.approx(1.7e-48, rel=0.02)
    assert r64.status == STATUS_CAP and r64.plan is not None                    # float64 rechnet, aber der Zeilenfehler fällt nur wie 1/(2k+1)
    assert np.abs(r64.plan.sum(axis=0) - b).sum() < 1e-12
    r64b = naive_iteration(a, b, K, 1 / 800, np.float64, 1e-8, 50, "float64")
    assert r64b.status == STATUS_UNTERLAUF                                       # exp(-800) = 0 auch in float64


@pytest.mark.parametrize("eps", [1 / 110, 1 / 800])
@pytest.mark.parametrize("k", [1, 10, 200])
def test_zeilenfehler_und_rohe_luecke_der_log_domaene_sind_1_durch_2k_plus_1(weit, eps, k):
    """Die Log-Domäne rechnet weiter, aber nicht schneller: Zeilenfehler und rohe Kostenlücke nach k Durchläufen sind genau 1/(2k+1), für beide ε."""
    _, f, g = verlauf_potentiale(weit.a, weit.b, weit.kosten, eps, [k])[0]
    P = plan_aus_potentialen(weit.a, weit.b, weit.kosten, eps, f, g)
    assert zeilenfehler(P, weit.a) == pytest.approx(1 / (2 * k + 1), abs=1e-9)
    assert (float((P * weit.kosten).sum()) - 0.5) / 0.5 == pytest.approx(1 / (2 * k + 1), abs=1e-9)
    assert np.abs(P.sum(axis=0) - weit.b).sum() < 1e-12


def test_zeilenfehler_1_durch_2k_plus_1_gilt_nur_fuer_kleine_eps():
    """Bis ε = 0,02 stimmt der Zeilenfehler bis k = 100 auf 1e-4 (relativ) mit 1/(2k+1); bei ε = 0,05 ist er bei k = 100 schon weniger als halb so groß; bei ε = 0,1 fällt er auf Rundungsniveau."""
    inst = handbeispiel_weit()

    def verhaeltnis(eps, k):
        _, f, g = verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, [k])[0]
        return zeilenfehler(plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g), inst.a) * (2 * k + 1)

    for k in (1, 10, 100):
        assert verhaeltnis(0.02, k) == pytest.approx(1.0, rel=1e-4)
    assert verhaeltnis(0.05, 100) < 0.5 and verhaeltnis(0.1, 100) < 1e-6


def test_grenzkern_in_bruechen_p11_ist_k_durch_2k_plus_1():
    """Der Grenzkern K = [[1, 0], [δ, δ]] (K12 = exp(−0,5/ε) und K21 = K22 = exp(−1/ε) sind für kleines ε kleiner): P11 = k/(2k+1), Zeilenfehler 1/(2k+1), in Brüchen bis k = 30."""
    d = Fr(1, 10 ** 6)
    Kg = [[Fr(1), Fr(0)], [d, d]]
    v = [Fr(1), Fr(1)]
    half = Fr(1, 2)
    for k in range(1, 31):
        Kv = [Kg[i][0] * v[0] + Kg[i][1] * v[1] for i in range(2)]
        u = [half / Kv[0], half / Kv[1]]
        Ktu = [Kg[0][j] * u[0] + Kg[1][j] * u[1] for j in range(2)]
        v = [half / Ktu[0], half / Ktu[1]]
        P11, P21, P22 = u[0] * Kg[0][0] * v[0], u[1] * Kg[1][0] * v[0], u[1] * Kg[1][1] * v[1]
        assert P11 == Fr(k, 2 * k + 1)
        assert abs(half - P11) + abs(P21 + P22 - half) == Fr(1, 2 * k + 1)


def test_franklin_lorenz_schranke_ist_4_exp_minus_viertel_durch_eps_bei_eps_1_durch_110():
    assert abs(1 / math.cosh(0.125 * 110) ** 2 - 4 * math.exp(-0.25 * 110)) < 1e-15
    assert 4.55e-12 < 4 * math.exp(-0.25 * 110) < 4.57e-12                      # etwa 4,6e-12 (die README nennt 4,6 · 10⁻¹²)


def test_auflösung_von_float32_bei_masse_viertel_und_die_masse_0_3():
    """Bei Masse 1/4 ist der kleinste Abstand einer float32-Zahl nach unten 2⁻²⁶ = 1,49e-8 (über der Toleranz 1e-8), nach oben 2⁻²⁵; schon die gerundete Masse 0,3 weicht um 1,19e-8 ab."""
    f32 = np.float32
    assert kleinste_aufloesung(f32) == 2.0 ** -26 and kleinste_aufloesung(f32) > 1e-8
    assert float(np.nextafter(f32(0.25), f32(1)) - f32(0.25)) == 2.0 ** -25
    assert 1.19e-8 < abs(float(f32(0.3)) - 0.3) < 1.20e-8
    assert kleinste_aufloesung(np.float64) < 1e-16


def test_abbruchtest_in_float32_und_in_float64_geben_verschiedene_zahlen():
    a = np.array([0.3, 0.7])
    u = np.ones(2, dtype=np.float32)
    Kv = np.array([0.3, 0.7], dtype=np.float32)           # float32(0,3) weicht von 0,3 um 1,19e-8 ab
    assert zeilenfehler_naiv(u, Kv, a, "float32") == 0.0 and zeilenfehler_naiv(u, Kv, a, "float64") > 1e-8


def test_toleranz_unter_der_aufloesung_wird_in_float64_getest_nie_erreicht_bei_float32_test_aber_durch_rundung():
    from sgz_scenario import make_instanz

    inst = make_instanz(30, 30, C.SERIEN_SEED, C.ZWECK_UNTERLAUF, 3)
    r64test = naive_iteration(inst.a, inst.b, inst.kosten, 0.01, np.float32, 1e-8, 400, "float64")
    assert r64test.status == STATUS_CAP and r64test.zeilenfehler > 1e-8        # Toleranz nicht erreichbar, nicht „nicht konvergiert“
    r32test = naive_iteration(inst.a, inst.b, inst.kosten, 0.01, np.float32, 1e-8, 400, "float32")
    assert r32test.status in (STATUS_OK, STATUS_CAP)                           # der Fehler kann durch Rundung auf null fallen (Zahl hängt an der Plattform: nur ein Zweig-Test)
    ok5 = naive_iteration(inst.a, inst.b, inst.kosten, 0.01, np.float32, 1e-5, 3000, "float64")
    assert ok5.status == STATUS_OK


def test_naive_float64_gleich_log_domaene_nach_k_durchlaeufen_und_gleiche_iterationszahl(weit):
    from sgz_scenario import make_instanz

    inst = make_instanz(12, 12, C.SERIEN_SEED, C.ZWECK_ANSICHT, 1)
    eps = 0.3 * 0.05
    for k in (1, 3, 10, 40):
        naiv = naive_iteration(inst.a, inst.b, inst.kosten, eps, np.float64, 0.0, k, "float64")
        _, f, g = verlauf_potentiale(inst.a, inst.b, inst.kosten, eps, [k])[0]
        assert naiv.iterationen == k
        assert naiv.plan == pytest.approx(plan_aus_potentialen(inst.a, inst.b, inst.kosten, eps, f, g), rel=1e-9, abs=1e-14)
    n = naive_iteration(inst.a, inst.b, inst.kosten, eps, np.float64, 1e-8, 5000, "float64")
    lg = loese_sinkhorn(inst.a, inst.b, inst.kosten, eps, tol=1e-8, max_iter=5000)
    assert n.status == STATUS_OK and lg.konvergiert and abs(n.iterationen - lg.iterationen) <= 2


def test_randfaelle_unterlauf_in_der_ersten_zeile_meldung_statt_nan_alle_kosten_gleich_k_null_n_gleich_1():
    a = b = np.array([0.5, 0.5])
    # Unterlauf schon im Kern: Zeile 2 ist null
    Cm = np.array([[0.0, 0.5], [1.0, 1.0]])
    r = naive_iteration(a, b, Cm, 1 / 800, np.float64, 1e-8, 10, "float64")
    assert r.status == STATUS_UNTERLAUF and r.plan is None and r.iterationen == 0 and math.isnan(r.zeilenfehler)
    assert hat_unterlauf(kern(Cm, 1 / 800, np.float64)) and not hat_unterlauf(kern(Cm, 1 / 110, np.float64))
    # alle Kosten gleich: das Produktmaß nach einem Durchlauf, der Zeilenfehler ist danach 0
    Cg = np.full((2, 2), 0.5)
    r = naive_iteration(a, b, Cg, 0.1, np.float64, 1e-8, 10, "float64")
    assert r.status == STATUS_OK and r.iterationen == 1 and r.plan == pytest.approx(np.outer(a, b), abs=1e-15)
    # k = 0: der Plan ist K selbst, der Zeilenfehler wird nie gemessen
    r = naive_iteration(a, b, Cg, 0.1, np.float64, 1e-8, 0, "float64")
    assert r.status == STATUS_CAP and r.iterationen == 0 and r.produkte == 0
    # n = 1: ein Lager, ein Kunde
    r = naive_iteration(np.array([1.0]), np.array([1.0]), np.array([[0.3]]), 0.1, np.float32, 1e-5, 10, "float64")
    assert r.status == STATUS_OK and r.plan[0, 0] == pytest.approx(1.0, rel=1e-6)


def test_produkte_sind_zwei_je_iteration():
    r = naive_iteration(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([[0.5, 1.0], [1.0, 0.5]]), 0.9102392266268373, np.float64, 1e-8, 100, "float64")
    assert r.produkte == 2 * r.iterationen > 0
