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
decks = sorted((ROOT / "datcom_runs" / "rocket_70mm_canards").glob("for005_ctrl*.dat"))
if not decks:
    skip("odczyt etykiet", "brak for005_ctrl*.dat (uruchom run_control_datcom.py ctrl)")
else:
    labels = []
    for dk in decks:                      # sweep podzielony na kanaly
        labels += parse_sweep_labels(dk) or []
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
RUN_DIR = ROOT / "datcom_runs" / "rocket_70mm_canards"
# Sweep jest dzielony na kanaly (patrz run_control_datcom.py) — zbieramy
# wszystkie pliki czastkowe; stary pojedynczy plik dziala jako fallback.
outs = sorted(RUN_DIR.glob("datcom_ctrl_*.out"))
if not outs and (RUN_DIR / "datcom_ctrl.out").exists():
    outs = [RUN_DIR / "datcom_ctrl.out"]
out = outs[0] if outs else RUN_DIR / "datcom_ctrl.out"
if not outs:
    skip("budowa i weryfikacja tablic pochodnych",
         f"brak {out.name} (uruchom na Windows: "
         f"python run_control_datcom.py ctrl --run)")
else:
    from control.datcom_control import build_control_derivatives
    from datcom_io.config_reader import load_config
    from datcom_io.missile_datcom_reader import (missile_datcom_to_table_aero,
                                                 parse_missile_datcom_cases,
                                                 parse_missile_datcom_output)

    cfg = load_config(str(ROOT / "configurations" / "rocket_70mm_canards.yaml"))
    print(f"    pliki: {[o.name for o in outs]}")
    tab = build_control_derivatives(outs, sweep_deg=list(SWEEP),
                                    linear_range_deg=6.0)

    al = np.degrees(tab.alpha_rad)
    i0 = int(np.argmin(np.abs(al)))

    check("ksztalty tablic zgodne z siatka",
          tab.Cm_delta.shape == (len(tab.alpha_rad), len(tab.mach)))
    check("pochodne pitch/yaw niezerowe (canardy dzialaja)",
          np.any(tab.Cm_delta != 0.0) and np.any(tab.Cn_delta != 0.0))
    if not np.any(tab.Cl_delta != 0.0):
        skip("pochodna roll (Cl_delta)",
             "brak przypadkow d_roll w wyniku — przebieg DATCOM byl urwany; "
             "uruchom ponownie run_control_datcom.py ctrl --run")
    else:
        check("pochodna roll niezerowa", np.any(tab.Cl_delta != 0.0))

    # 1. ZNAK — niezmiennik jednoznaczny dla powierzchni PRZED srodkiem
    #    ciezkosci: dodatnie wychylenie daje dodatnia sile normalna na
    #    canardzie, a ta na ramieniu przed xcg daje moment na nos w gore.
    #    Czyli sign(Cm_delta) == sign(CN_delta).
    #    (Nie porownujemy ze znakiem Cm_alpha: konfiguracja z canardami bywa
    #    stabilna poddzwiekowo i niestabilna transonicznie, wiec "przeciwny do
    #    Cm_alpha" nie jest niezmiennikiem — usredniony po Machu Cm_alpha nie
    #    ma nawet dobrze okreslonego znaku.)
    cm0 = float(np.mean(tab.Cm_delta[i0]))
    cn0 = float(np.mean(tab.CN_delta[i0]))
    print(f"    Cm_delta ~ {cm0:+.4f} /rad,  CN_delta ~ {cn0:+.4f} /rad")
    check("sign(Cm_delta) == sign(CN_delta) — canard przed xcg",
          np.sign(cm0) == np.sign(cn0) and abs(cm0) > 1e-9,
          "— jesli nie, przypisanie paneli do plaszczyzn w PANEL_PATTERNS "
          "jest bledne (popraw wzorce, NIE znak w efektorze)")

    # Informacyjnie: stabilnosc statyczna konfiguracji z canardami,
    # per Mach (bez usredniania — patrz komentarz wyzej).
    gbase = parse_missile_datcom_cases(outs[0])[0]
    machs = sorted(c.mach for c in gbase.cases)
    CMb = np.array([gbase.get_case(m).CM for m in machs]).T
    ab = np.asarray(gbase.cases[0].alpha, float)
    j0 = int(np.argmin(np.abs(ab)))
    cma = (CMb[j0 + 1] - CMb[j0 - 1]) / (ab[j0 + 1] - ab[j0 - 1])
    print("    Cm_alpha wg Macha [1/deg]: " +
          "  ".join(f"M{m:.1f}:{v:+.4f}" for m, v in zip(machs, cma)))
    print(f"    (ujemne = stabilny; konfiguracja z canardami zmienia znak "
          f"z Machem — dlatego nie jest to dobry test znaku)")

    # 2. Liniowosc — sweep pozwala to SPRAWDZIC, nie zakladac.
    #    Raportujemy najgorsza komorke ORAZ mediane: pojedyncza komorka przy
    #    skrajnej alfa/Machu potrafi byc wyraznie nieliniowa, co nie znaczy,
    #    ze tablica jest bezuzyteczna.
    for field, info in (tab.fit_info or {}).items():
        r2 = info.get("worst_r2", 1.0)
        med = info.get("median_r2", r2)
        nb = info.get("n_below_resolution", 0)
        nc = info.get("n_cells", 0)
        extra = f"  [{nb}/{nc} ponizej rozdzielczosci]" if nb else ""
        print(f"    {field:9s} R^2: mediana={med:.4f}  najgorsza={r2:.4f}  "
              f"max_dev(pelny sweep)={info.get('max_dev_full_sweep', 0.0):.5f}{extra}")
        check(f"{field}: typowa komorka liniowa (mediana R^2>0.99)", med > 0.99,
              f"(mediana={med:.4f})")
        # R^2 sprawdzamy TYLKO na komorkach powyzej rozdzielczosci wydruku
        # DATCOM — tam gdzie caly sweep miesci sie w 1-2 cyfrach wydruku,
        # R^2 mierzy kwantyzacje, nie fizyke (patrz fit_derivative_from_sweep).
        check(f"{field}: najgorsza ROZROZNIALNA komorka sensowna (R^2>0.95)",
              r2 > 0.95, f"(najgorsza={r2:.4f})")
        # Wiekszosc tablicy musi byc rozroznialna, inaczej pochodna jest
        # w praktyce nieznana — to warto wiedziec, a nie przemilczec.
        if nc:
            check(f"{field}: wiekszosc komorek rozroznialna (<50% ponizej rozdz.)",
                  nb < 0.5 * nc, f"({nb}/{nc} ponizej rozdzielczosci)")

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
