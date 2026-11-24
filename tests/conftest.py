"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sgz_scenario import handbeispiel_runden, handbeispiel_weit  # noqa: E402


@pytest.fixture
def weit():
    """Handbeispiel „weit entferntes Lager“: Angebote und Nachfragen (1/2, 1/2), Kosten Zeilen (0, 1/2), (1, 1); OPT = 1/2 (der Einheitsplan, eindeutig).
    Der Zeilenfehler der Log-Domäne nach k Durchläufen ist für kleine ε (bis etwa 0,02) 1/(2k+1), unabhängig von ε."""
    return handbeispiel_weit()


@pytest.fixture
def runden_beispiel():
    """Handbeispiel der Rundung: Angebote (3/4, 1/4), Nachfragen (1/4, 3/4), Kosten Zeilen (0, 1), (1, 0), Plan P = [[9/40, 3/8], [1/40, 3/8]] (nach einem Durchlauf, ε = 1/ln 3);
    gerundet [[15/64, 33/64], [1/64, 15/64]] mit den Kosten 17/32."""
    return handbeispiel_runden()
