"""
run_control_tests.py
====================
Uruchamia wszystkie testy naraz i wypisuje zbiorcze podsumowanie.

Testy sterowania sa oznaczone osobno; regresyjne (istniejace wczesniej)
sluza jako kontrola, ze dolozenie modulu sterowania niczego nie zepsulo.

Uzycie:
    python run_control_tests.py            # wszystkie
    python run_control_tests.py --control  # tylko testy sterowania
    python run_control_tests.py -v         # pelne wyjscie kazdego testu
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"

CONTROL_TESTS = [
    ("test_control_step.py",        "lancuch command->actuator->moment->6DOF"),
    ("test_control_derivatives.py", "pochodne sterowania ze sweepa DATCOM"),
    ("test_gam_baseline_compare.py", "A/B usuniecia GAM"),
]
REGRESSION_TESTS = [
    ("test_gyroscopic.py",   "efekty zyroskopowe"),
    ("test_3dof_vs_6dof.py", "zgodnosc 3DOF/6DOF"),
    ("test_energy.py",       "bilans energii"),
    ("test_launcher.py",     "model szyny"),
    ("test_quaternion.py",   "kwaterniony"),
]

RE_RESULT = re.compile(r"Wynik:\s*(\d+)\s*/\s*(\d+)")
RE_SKIP   = re.compile(r"\[SKIP\]")


def run_one(name: str, verbose: bool):
    path = TESTS / name
    if not path.exists():
        return None, f"brak pliku {name}"
    # Windows: domyslne kodowanie konsoli (cp1250/cp1252) nie obsluguje znakow
    # uzywanych w testach (Delta, approx, indeks gorny 2, znak tensora), wiec
    # proces potomny wywracal sie na UnicodeEncodeError przy samym print().
    # Wymuszamy UTF-8 po obu stronach: PYTHONIOENCODING w potomku i jawne
    # encoding/errors przy odczycie strumieni.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(path)],
                          capture_output=True, text=True, cwd=str(ROOT),
                          env=env, encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    if verbose:
        print(out)
    m = RE_RESULT.search(out)
    n_skip = len(RE_SKIP.findall(out))
    if not m:
        # Bez linii "Wynik:" test sie wywrocil. Pokazujemy linie z typem
        # wyjatku (ostatnia linia tracebacku), a nie ostatnia linie wyjscia —
        # przy wieloliniowym komunikacie bledu ta ostatnia bywa najmniej istotna.
        lines = [l.rstrip() for l in out.strip().splitlines() if l.strip()]
        exc = next((l for l in reversed(lines)
                    if re.match(r"^\w+(\.\w+)*(Error|Exception)\b", l.strip())), None)
        return None, (exc or (lines[-1] if lines else "brak wyniku"))[:100]
    return (int(m.group(1)), int(m.group(2)), n_skip, proc.returncode), None


def section(title, tests, verbose):
    print(f"\n{title}")
    print("-" * 74)
    tot_p = tot_n = tot_s = 0
    failed = []
    for name, desc in tests:
        res, err = run_one(name, verbose)
        if res is None:
            print(f"  {'BLAD':>6}  {name:<30} {err}")
            failed.append(name)
            continue
        p, n, s, rc = res
        tot_p += p; tot_n += n; tot_s += s
        status = "OK" if rc == 0 and p == n else "FAIL"
        if status == "FAIL":
            failed.append(name)
        extra = f"  ({s} pominietych)" if s else ""
        print(f"  {status:>6}  {name:<30} {p:>3}/{n:<3} {desc}{extra}")
    return tot_p, tot_n, tot_s, failed


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    only_ctrl = "--control" in sys.argv

    print("=" * 74)
    print("TESTY — modul sterowania i regresja")
    print("=" * 74)

    P = N = S = 0
    failed = []

    p, n, s, f = section("STEROWANIE", CONTROL_TESTS, verbose)
    P += p; N += n; S += s; failed += f

    if not only_ctrl:
        p, n, s, f = section("REGRESJA (musza przechodzic bez zmian)",
                             REGRESSION_TESTS, verbose)
        P += p; N += n; S += s; failed += f

    print("\n" + "=" * 74)
    print(f"  RAZEM: {P}/{N} testow zaliczonych" +
          (f", {S} pominietych" if S else ""))
    if S:
        print("  Pominiete = czekaja na pliki .out z DATCOM "
              "(uruchom: python run_control_datcom.py all --run)")
    if failed:
        print(f"  NIEUDANE: {', '.join(failed)}")
        print("  STATUS: FAIL")
    else:
        print("  STATUS: OK")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
