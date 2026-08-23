"""
run_lref_validation_compare.py
==============================
Robi CALE porownanie walidacji przed/po poprawce LREF jednym poleceniem:

    python run_lref_validation_compare.py

Dlaczego to istnieje: procedura recznie sklada sie z czterech krokow i dwoch
zmiennych srodowiskowych, a te ustawia sie INACZEJ w kazdym shellu (`set` w
cmd.exe, `$env:` w PowerShell, `export` w bashu). W PowerShell `set` jest
aliasem Set-Variable i NIE tworzy zmiennej srodowiskowej — po cichu, bez
bledu. Efekt: oba przebiegi liczyly sie tym samym modelem, a porownanie
pokazywalo identyczne slupki.

Tutaj zmienne ustawia Python (os.environ przekazany do podprocesu), wiec shell
nie ma nic do rzeczy i pomylka jest niemozliwa.

Przebieg:
  1. walidacja ze STARA normalizacja  (LREF = dlugosc kadluba, 1.285 m)
  2. zapis migawki 'przed_LREF'
  3. walidacja z NOWA normalizacja    (LREF = srednica, 0.070 m)
  4. porownanie + wykres

Kazdy przebieg to pelny DATCOM per lot, wiec calosc trwa dlugo.

Opcje:
    --name NAZWA     nazwa migawki (domyslnie przed_LREF)
    --skip-before    pomin krok 1-2 (migawka juz istnieje i jest wiarygodna)
    --no-plot        bez wykresu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYZE = ROOT / "telemetry_analysis" / "analyze_per_flight_6dof.py"
COMPARE = ROOT / "compare_validation.py"


def run(script: Path, args: list[str], extra_env: dict) -> int:
    """Uruchamia skrypt w podprocesie z JAWNIE podanym srodowiskiem."""
    env = os.environ.copy()
    # Kluczowe: klucz nieobecny musi byc USUNIETY, a nie ustawiony na "".
    # Generator sprawdza wartosc, ale aero.py porownuje z "1" — zostawienie
    # pustego stringa dzialaloby, natomiast usuniecie jest jednoznaczne i
    # odporne na przyszle zmiany warunkow.
    for k, v in extra_env.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    env["PYTHONIOENCODING"] = "utf-8"      # polskie znaki na konsoli Windows
    cmd = [sys.executable, str(script)] + args
    shown = " ".join(f"{k}={v}" for k, v in extra_env.items() if v is not None)
    print(f"\n>>> {' '.join(cmd)}" + (f"   [{shown}]" if shown else "   [bez LREF_MODE]"))
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


BEFORE = {"FLIGHTSIM_LREF_MODE": "length", "FLIGHTSIM_ALLOW_STALE_LREF": "1"}
AFTER = {"FLIGHTSIM_LREF_MODE": None, "FLIGHTSIM_ALLOW_STALE_LREF": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="przed_LREF")
    ap.add_argument("--skip-before", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print("POROWNANIE WALIDACJI: LREF = dlugosc kadluba  vs  LREF = srednica")
    print("=" * 78)
    print("Zmienne srodowiskowe ustawia ten skrypt — shell nie ma znaczenia.")

    if not a.skip_before:
        print("\n--- KROK 1/4: walidacja ze STARA normalizacja (przed) ---")
        if run(ANALYZE, [], BEFORE) != 0:
            print("\nBLAD: walidacja 'przed' nie przeszla. Przerywam."); return 1

        print("\n--- KROK 2/4: zapis migawki ---")
        if run(COMPARE, ["--save-baseline", a.name], BEFORE) != 0:
            print("\nBLAD: nie udalo sie zapisac migawki. Przerywam."); return 1
    else:
        print(f"\n--- KROK 1-2/4 pominiete (--skip-before), uzywam '{a.name}' ---")

    print("\n--- KROK 3/4: walidacja z NOWA normalizacja (po) ---")
    if run(ANALYZE, [], AFTER) != 0:
        print("\nBLAD: walidacja 'po' nie przeszla. Przerywam."); return 1

    print("\n--- KROK 4/4: porownanie ---")
    cmp_args = ["--vs", a.name] + ([] if a.no_plot else ["--plot"])
    rc = run(COMPARE, cmp_args, AFTER)
    if rc == 2:
        # compare_validation zwraca 2, gdy oba pliki sa identyczne. Tutaj to juz
        # NIE moze byc wina shella, wiec problem lezy gdzie indziej.
        print("\nOba przebiegi daly identyczny wynik MIMO ustawienia zmiennych")
        print("przez Pythona. To wyklucza shell — sprawdz, czy walidacja")
        print("faktycznie przelicza DATCOM: python check_lref_mode.py")
        print("(sekcja 3b: katalogi '_cant...' powinny miec rozne LREF)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
