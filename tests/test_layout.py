"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Die Tests stellen sicher, dass wirklich JEDER Regler der
Seitenleiste dort steht (sonst bliebe das Preset trotz Änderung markiert) und dass die Änderung jedes Reglers die Markierung aufhebt: für jeden Regler direkt an der Funktion
(ohne die Reihen neu zu rechnen), für einige Regler zusätzlich in der laufenden App."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import sgz_constants as C
import sgz_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")
STANDARD = ["float32 bei ε = 0,002"]


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _primaer(at) -> list:
    return [n for n, typ in _typen(at).items() if typ == "primary"]


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    standard = [name for name, preset in C.PRESETS.items() if preset == C.DEFAULTS]
    assert standard == STANDARD and _primaer(_app()) == STANDARD


def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf():
    at = _app()
    for name in C.PRESETS:
        at.sidebar.button(key=f"preset_{name}").click().run()
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert _primaer(at) == [name], name
        at.run()
        assert _primaer(at) == [name], name


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, "ein Eintrag in SETTING_SPECS hat keinen Regler"


def _anderer_wert(spec, state_key, aktuell, ohne_preset=True):
    """Ein erlaubter Wert, der vom aktuellen abweicht und (wenn `ohne_preset`) den Stand nicht zufällig zu einem anderen Preset macht (sonst wäre dieses zu Recht hervorgehoben)."""
    kandidaten = list(spec.options) if spec.options is not None else [aktuell + 1, aktuell - 1, spec.hi, spec.lo]
    for wert in kandidaten:
        if wert == aktuell or (spec.lo is not None and not spec.lo <= wert <= spec.hi):
            continue
        stand = {k: P.SETTING_SPECS[k].default for k in P.SETTING_SPECS}
        stand[state_key] = wert
        if not ohne_preset or stand not in C.PRESETS.values():
            return wert
    raise AssertionError("kein passender Wert")


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf_an_der_funktion(state_key, monkeypatch):
    """Ändert ein Regler den Stand, ist danach höchstens das Preset hervorgehoben, dessen Werte jetzt ALLE stimmen (der Regler „Instanz“ macht aus dem Standard
    das Preset „Von Hand“); sonst keines."""
    spec = P.SETTING_SPECS[state_key]
    zustand = {k: s.default for k, s in P.SETTING_SPECS.items()}
    monkeypatch.setattr(P, "st", SimpleNamespace(session_state=zustand))
    assert P.aktives_preset() == STANDARD[0]
    zustand[state_key] = _anderer_wert(spec, state_key, zustand[state_key], ohne_preset=False)
    passend = [n for n, p in C.PRESETS.items() if p == zustand]
    assert P.aktives_preset() == (passend[0] if passend else None), state_key
    assert P.aktives_preset() != STANDARD[0]
    for name, preset in C.PRESETS.items():                     # jedes Preset ist hervorgehoben, wenn alle Regler seinen Werten entsprechen
        zustand.update(preset)
        assert P.aktives_preset() == name


@pytest.mark.parametrize("state_key", ["k2", "kosten3", "eps4", "tol1"])
def test_aenderung_eines_reglers_hebt_die_hervorhebung_in_der_app_auf_und_erneutes_anklicken_markiert_wieder(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(spec, state_key, regler.value)).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not _primaer(at)
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _primaer(at) == [name]


def test_breite_markdown_tabellen_scrollen_auf_dem_telefon_in_der_tabelle_statt_den_hauptbereich_zu_verschieben():
    quelle = Path(APP).read_text(encoding="utf-8")
    zeile = ("""st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')""")
    assert zeile in quelle and quelle.index("kopf_css(") < quelle.index(zeile) < quelle.index("with st.container(key=\"kopf\")")
