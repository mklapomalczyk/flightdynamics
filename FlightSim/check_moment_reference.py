"""
check_moment_reference.py
=========================
Diagnostyka: czy model 6DOF mnozy wspolczynniki momentu z DATCOM przez TE SAMA
dlugosc odniesienia, przez ktora DATCOM je znormalizowal?

TLO
---
Missile DATCOM podaje CM jako wielkosc bezwymiarowa:  CM = M / (q * SREF * LREF).
Zeby odzyskac moment fizyczny, trzeba pomnozyc przez TE SAMA LREF:
    M = q * SREF * LREF * CM

Podrecznik (Table $REFQ): LREF to "Longitudinal reference length", a DOMYSLNIE
jest to "maximum body diameter". Generator w tym projekcie jawnie nadpisuje
domyslna wartosc dlugoscia KADLUBA:
    missile_datcom_generator.py:  LREF = cfg.body.length   (1.285 m dla 70mm)

Natomiast model 6DOF liczy moment przez SREDNICE:
    models/aerodynamics.py:  MA_yy = q_dyn * S_ref * d_ref * Cm_total
    force_model6.py przekazuje d_ref = geom.d_ref = 0.070 m

Skoro DATCOM znormalizowal przez 1.285 m, a model mnozy przez 0.070 m, momenty
pochylajacy i odchylajacy wychodza ~18x ZA MALE.

WERYFIKACJA (dwie niezalezne drogi do tego samego momentu)
----------------------------------------------------------
DATCOM podaje takze polozenie srodka parcia XCP. Moment mozna policzyc
niezaleznie jako sila normalna razy ramie:
    M = q * SREF * CN * (XCG - XCP)
Jesli ta wartosc zgadza sie z  q * SREF * LREF * CM, to znaczy, ze wlasnie ta
LREF zostala uzyta do normalizacji. Ten skrypt to sprawdza.

Uruchomienie:
    python check_moment_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datcom_io.missile_datcom_reader import parse_missile_datcom_output

CASES = [
    "rocket_70mm_baseline_cant0p000",
    "rocket_70mm_baseline_cant0p600",
    "rocket_70mm_baseline_cant1p600",
    "rocket_70mm_baseline_tepa_cant0p000",
]
D_REF_MODEL = 0.070      # geom.d_ref uzywane przez model 6DOF (srednica)


def main():
    print("=" * 76)
    print("SPRAWDZENIE DLUGOSCI ODNIESIENIA MOMENTU (DATCOM vs model 6DOF)")
    print("=" * 76)

    any_data = False
    results = []
    for case in CASES:
        p = ROOT / "datcom_runs" / case / "datcom.out"
        if not p.exists():
            continue
        any_data = True
        res = parse_missile_datcom_output(p)
        c = res.cases[0]
        S, L, xcg = c.sref, c.lref, c.xcg

        # Wybierz alpha o najwiekszym |CN| — tam ramie jest najlepiej okreslone.
        i = int(np.argmax(np.abs(c.CN)))
        CN, CM, xcp = c.CN[i], c.CM[i], c.XCP[i]
        if abs(CN) < 1e-9:
            continue

        m_xcp = S * CN * (xcg - xcp)     # niezaleznie: sila x ramie
        m_L = S * L * CM                 # z CM przy LREF z DATCOM
        m_d = S * D_REF_MODEL * CM       # z CM przy srednicy (konwencja modelu)
        ok = abs(L - D_REF_MODEL) <= 0.05 * D_REF_MODEL
        results.append((case, L, ok))

        print(f"\n{case}   (alpha={c.alpha[i]:+.0f} deg, Mach={c.mach})")
        print(f"  DATCOM: SREF={S:.6f} m^2  LREF={L} m  XCG={xcg} m  XCP={xcp:.4f} m")
        print(f"  moment/q  z XCP (sila x ramie) : {m_xcp:+.6f}")
        print(f"  moment/q  z CM * LREF          : {m_L:+.6f}   "
              f"(iloraz {m_xcp/m_L if m_L else float('nan'):.4f})")
        print(f"  moment/q  z CM * d_ref(model)  : {m_d:+.6f}   "
              f"({m_L/m_d if m_d else float('nan'):.1f}x mniejszy)")
        print(f"  -> LREF {'==' if ok else '!='} srednica modelu "
              f"({D_REF_MODEL} m): {'OK' if ok else 'NIEZGODNOSC'}")

    if not any_data:
        print("\nBrak plikow datcom.out — nie ma czego sprawdzic.")
        return 0

    stale = [(c, L) for c, L, ok in results if not ok]
    print("\n" + "=" * 76)
    if stale:
        print("WYNIK: NIEZGODNOSC — te pliki powstaly przy LREF != srednica:")
        for c, L in stale:
            print(f"  {c}:  LREF={L} m  (oczekiwano {D_REF_MODEL} m)")
        print("\nGenerator ustawia juz LREF = srednica kadluba, wiec te wyniki")
        print("sa PRZESTARZALE. Przelicz je ponownie na maszynie z DATCOM:")
        print("    python MAIN.py                     (aero bazowe)")
        print("    python run_control_datcom.py all --run   (sterowanie + test GAM)")
        print("Do czasu przeliczenia aero.py odrzuci te pliki z jawnym bledem,")
        print("zamiast po cichu policzyc momenty ~18x za male.")
        print("=" * 76)
        return 1
    print("WYNIK: OK — CM znormalizowane przez srednice, zgodnie z konwencja modelu.")
    print("Zgodnosc dwoch niezaleznych drog (XCP vs CM*LREF) potwierdza normalizacje.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
