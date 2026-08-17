"""
tests/test_control_derivatives.py
==================================
Test wyciagania pochodnych sterowania z DATCOM + weryfikacja znakow.

Trzy czesci:

  A) MATEMATYKA DOPASOWANIA — zawsze, bez DATCOM.
     fit_derivative_from_sweep() na danych syntetycznych: czysto liniowych
     (musi trafic nachylenie i dac R^2=1) oraz z dodana nieliniowoscia
     (musi ja WYKRYC przez max_dev, zamiast po cichu usrednic).

  B) ETYKIETY SWEEPA — zawsze.
     Karty CASEID w wygenerowanym decku daja sie odczytac i pokrywaja
     dokladnie iloczyn (kanaly x wychylenia). To one wiaza przypadek DATCOM
     z kanalem sterowania — bez nich przypisanie byloby zgadywaniem.

  C) PRAWDZIWE POCHODNE — uruchamia sie dopiero gdy istnieje
     datcom_runs/rocket_70mm_canards/datcom_ctrl.out
     (wygeneruj na Windows: python run_control_datcom.py ctrl --run).
     Sprawdza znaki, liniowosc i spojnosc fizyczna.

Uruchomienie:  python3 tests/test_control_derivatives.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control.datcom_control import parse_sweep_labels
from control.derivatives import PANEL_PATTERNS, fit_derivative_from_sweep

PASS = FAIL = SKIP = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name} — {why}")


print("=" * 68)
print("TEST POCHODNYCH STEROWANIA (sweep wychylen DATCOM)")
print("=" * 68)

SWEEP = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.])

# --- A. Matematyka dopasowania -------------------------------------------
print("\nA. Dopasowanie pochodnej ze sweepa")

slope_per_deg = 0.02
lin = slope_per_deg * SWEEP + 0.001
s, info = fit_derivative_from_sweep(SWEEP, lin, linear_range_deg=6.0)
check("liniowe dane: nachylenie [1/rad] poprawne",
      abs(s - slope_per_deg * 180.0 / np.pi) < 1e-9, f"(s={s:.6f})")
check("liniowe dane: R^2 == 1", abs(info["r2"] - 1.0) < 1e-12)
check("liniowe dane: zerowe odchylenie na calym sweepie",
      info["max_dev_full_sweep"] < 1e-12)
check("wyraz wolny odtworzony", abs(info["intercept"] - 0.001) < 1e-9)

# Nieliniowosc poza zakresem dopasowania musi zostac ZAUWAZONA.
nonlin = slope_per_deg * SWEEP + 0.0004 * SWEEP**2
s2, info2 = fit_derivative_from_sweep(SWEEP, nonlin, linear_range_deg=6.0)
check("nieliniowosc wykryta przez max_dev", info2["max_dev_full_sweep"] > 1e-3,
      f"(max_dev={info2['max_dev_full_sweep']:.5f})")
check("dopasowanie liczone tylko w zakresie liniowym",
      info2["n_fit"] == int(np.sum(np.abs(SWEEP) <= 6.0)),
      f"(n_fit={info2['n_fit']})")
# Skladowa kwadratowa jest parzysta => nie psuje nachylenia wokol zera.
check("nachylenie odporne na skladowa parzysta",
      abs(s2 - slope_per_deg * 180.0 / np.pi) < 1e-6, f"(s2={s2:.6f})")

# --- B. Etykiety sweepa ---------------------------------------------------
print("\nB. Etykiety CASEID w decku")
deck = ROOT / "datcom_runs" / "rocket_70mm_canards" / "for005_ctrl.dat"
if not deck.exists():
    skip("odczyt etykiet", f"brak {deck} (uruchom run_control_datcom.py ctrl)")
else:
    labels = parse_sweep_labels(deck)
    check("etykiety odczytane", labels is not None and len(labels) > 0)
    if labels:
        chans = sorted({c for c, _ in labels})
        check("kanaly zgodne z PANEL_PATTERNS", chans == sorted(PANEL_PATTERNS),
              f"({chans})")
        expected = {(c, float(d)) for c in PANEL_PATTERNS for d in SWEEP}
        check("pokrycie: kazdy kanal x kazde wychylenie",
              set(labels) == expected,
              f"(brakuje {sorted(expected - set(labels))[:3]})")
        check("brak duplikatow", len(labels) == len(set(labels)))

# --- C. Prawdziwe pochodne z DATCOM --------------------------------------
print("\nC. Pochodne z prawdziwego wyjscia DATCOM")
out = ROOT / "datcom_runs" / "rocket_70mm_canards" / "datcom_ctrl.out"
if not out.exists():
    skip("budowa i weryfikacja tablic pochodnych",
         f"brak {out.name} (uruchom na Windows: "
         f"python run_control_datcom.py ctrl --run)")
else:
    from control.datcom_control import build_control_derivatives
    from datcom_io.config_reader import load_config
    from datcom_io.missile_datcom_reader import (missile_datcom_to_table_aero,
                                                 parse_missile_datcom_output)

    cfg = load_config(str(ROOT / "configurations" / "rocket_70mm_canards.yaml"))
    tab = build_control_derivatives(out, sweep_deg=list(SWEEP),
                                    linear_range_deg=6.0)

    check("ksztalty tablic zgodne z siatka",
          tab.Cm_delta.shape == (len(tab.alpha_rad), len(tab.mach)))
    check("pochodne niezerowe (canardy dzialaja)",
          np.any(tab.Cm_delta != 0.0) and np.any(tab.Cl_delta != 0.0))

    # 1. Znak: canardy PRZED srodkiem ciezkosci => destabilizujace.
    #    Cm_delta musi miec znak PRZECIWNY do Cm_alpha.
    base = missile_datcom_to_table_aero(parse_missile_datcom_output(out))
    al = np.asarray(base["alpha_deg"], float)
    CM = np.asarray(base["CM"], float)
    i0 = int(np.argmin(np.abs(al)))
    i1 = min(i0 + 1, len(al) - 1)
    Cm_alpha = float(np.mean((CM[i1] - CM[i0]) / max(al[i1] - al[i0], 1e-9)))
    Cm_delta0 = float(np.mean(tab.Cm_delta[i0]))
    print(f"    Cm_alpha ~ {Cm_alpha:+.5f} /deg,  Cm_delta ~ {Cm_delta0:+.5f} /rad")
    check("Cm_delta ma znak PRZECIWNY do Cm_alpha (canard destabilizuje)",
          Cm_alpha * Cm_delta0 < 0.0,
          "— jesli nie, przypisanie paneli do plaszczyzn w PANEL_PATTERNS "
          "jest bledne (popraw wzorce, NIE znak w efektorze)")

    # 2. Liniowosc — sweep pozwala to SPRAWDZIC, nie zakladac.
    for field, info in (tab.fit_info or {}).items():
        r2 = info.get("worst_r2", 1.0)
        print(f"    {field:9s} R^2(min)={r2:.4f}  "
              f"max_dev={info.get('max_dev_full_sweep', 0.0):.5f}")
        check(f"{field}: liniowe w zakresie +/-6 deg (R^2>0.98)", r2 > 0.98,
              f"(R^2={r2:.4f})")

    # 3. Symetria krzyzowa — kontrola, nie zrodlo danych.
    cm = float(np.mean(np.abs(tab.Cm_delta)))
    cn = float(np.mean(np.abs(tab.Cn_delta)))
    if cm > 1e-9 and cn > 1e-9:
        rel = abs(cn - cm) / cm
        print(f"    |Cn_delta|/|Cm_delta| = {cn/cm:.3f}")
        check("Cn_delta ~ Cm_delta co do modulu (uklad krzyzowy)", rel < 0.30,
              f"(rozjazd {rel*100:.1f}% — sprawdz wzorce paneli)")
    else:
        skip("symetria Cn/Cm", "jedna z pochodnych ~0")

    # 4. Sanity: efektor liczy niezerowy moment przy realnych danych.
    from control import AeroSurfaceEffector
    from control.types import FlightState
    eff = AeroSurfaceEffector(tab)
    W = eff.wrench(np.array([2.0, 0.0, 0.0]),
                   FlightState(t=0.0, q_dyn=1.0e4, mach=0.5, speed=200.0))
    check("efektor daje niezerowy moment pitch na realnych pochodnych",
          abs(W.My) > 0.0, f"(My={W.My:.4g} N*m)")

print("\n" + "=" * 68)
total = PASS + FAIL
print(f"  Wynik: {PASS}/{total} testow zaliczonych" +
      (f", {SKIP} pominietych" if SKIP else ""))
print("  STATUS: OK" if FAIL == 0 else f"  STATUS: {FAIL} FAIL")
print("=" * 68)
sys.exit(1 if FAIL else 0)
