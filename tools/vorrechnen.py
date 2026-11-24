"""Rechnet die Reihen der Experimente vor und schreibt (ergänzt) `tools/vorgerechnet.json` (siehe sgz_vorgerechnet.py).

Aufruf (im Ordner der Demo): <Test-venv>/Scripts/python.exe -X utf8 tools/vorrechnen.py [Name ...]
Ohne Namen werden alle Reihen gerechnet (mehrere Minuten); mit Namen (unterlauf, runden, schranke, ableitung, gleichstand, aufwand, raenge, raenge_gleich) nur diese; bereits
vorhandene Einträge anderer Reihen bleiben stehen. `OMP_NUM_THREADS=1` setzen, damit die Zeiten vergleichbar bleiben.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sgz_evaluation as E  # noqa: E402
from sgz_vorgerechnet import DATEI, schluessel  # noqa: E402


def main(namen):
    daten = json.loads(DATEI.read_text(encoding="utf-8")) if DATEI.exists() else {}
    for name, params, rechne in E.vorzurechnen():
        if namen and name not in namen:
            continue
        t = time.time()
        daten[schluessel(name, **params)] = E.jsonbar(rechne())
        print(f"{name} {params}: {time.time() - t:.1f} s", flush=True)
        DATEI.write_text(json.dumps(daten, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size} Byte)")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
