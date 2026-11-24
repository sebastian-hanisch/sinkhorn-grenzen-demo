"""Vorgerechnete Reihen: `tools/vorgerechnet.json` enthält die Ergebnisse der Reihen der Experimente 1, 3 und 4, die Tabellen von Experiment 2 und die Reihen des Absatzes
„weiche Ränge“ (erzeugt von `tools/vorrechnen.py` mit dem Code der Demo).

Wozu: Diese Reihen brauchen auf dem Entwicklungsrechner zusammen einige Minuten (Experiment 1 etwa 1 Minute, Experiment 3 etwa 3 Minuten, Experiment 4 und die Ränge je einige
Minuten). Jede Einstellung der Regler dieser Experimente ist vorgerechnet; die Seite rechnet dort nichts. `tests/test_claims.py` rechnet Stichproben neu und vergleicht sie als Band."""

import json
from functools import lru_cache
from pathlib import Path

DATEI = Path(__file__).resolve().parent / "tools" / "vorgerechnet.json"


def schluessel(name: str, **params) -> str:
    """Eindeutiger Schlüssel aus Name und Parametern (Zahlen als int bzw. float, Texte als Texte)."""
    return name + "|" + ",".join(f"{k}={params[k]!r}" for k in sorted(params))


@lru_cache(maxsize=1)
def _lade() -> dict:
    try:
        return json.loads(DATEI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def hole(name: str, **params):
    """Der vorgerechnete Eintrag oder None."""
    return _lade().get(schluessel(name, **params))
