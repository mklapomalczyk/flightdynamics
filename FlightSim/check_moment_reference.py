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

        print(f"\n{case}   (alpha={c.alpha[i]:+.0f} deg, Mach={c.mach})")
        print(f"  DATCOM: SREF={S:.6f} m^2  LREF={L} m  XCG={xcg} m  XCP={xcp:.4f} m")
        print(f"  moment/q  z XCP (sila x ramie) : {m_xcp:+.6f}")
        print(f"  moment/q  z CM * LREF          : {m_L:+.6f}   "
              f"(iloraz {m_xcp/m_L if m_L else float('nan'):.4f})")
        print(f"  moment/q  z CM * d_ref(model)  : {m_d:+.6f}   "
              f"({m_L/m_d if m_d else float('nan'):.1f}x mniejszy)")

    if not any_data:
        print("\nBrak plikow datcom.out — nie ma czego sprawdzic.")
        return 0

    print("\n" + "=" * 76)
    print("WNIOSEK")
    print("=" * 76)
    print("""
Iloraz "z XCP" / "z CM * LREF" bliski 1.000 potwierdza, ze DATCOM znormalizowal
CM wlasnie przez LREF podana w decku (dlugosc kadluba), a nie przez srednice.

Model 6DOF mnozy natomiast przez d_ref = srednica, wiec momenty pochylajacy
i odchylajacy sa okolo 18x za male.

CO Z TYM ZROBIC — do decyzji, bo dotyka ZWALIDOWANYCH wynikow:

  Opcja A (zalecana): ustawic w generatorze LREF = srednica kadluba, czyli
    wartosc DOMYSLNA wg podrecznika i standardowa konwencje pociskowa.
    Wymaga ponownego przebiegu DATCOM dla wszystkich przypadkow. Model
    pozostaje bez zmian. Uwaga: XCP jest podawany W JEDNOSTKACH LREF, wiec
    czytnik automatycznie dostanie spojne wartosci.

  Opcja B: zostawic LREF = dlugosc kadluba i mnozyc w modelu przez lref
    z tablicy zamiast przez geom.d_ref. Nie wymaga DATCOM, ale rozjezdza sie
    z konwencja, w ktorej liczone sa czlony tlumiace (q*d/2V) i moment toczacy.

Wplyw zmierzony na locie 19 (wiatr staly, pochodne sterowania z DATCOM):
    crossrange:  99 m  ->  991 m     (telemetria: 1410 m)
    downrange : 5319 m -> 4935 m     (telemetria: 4233 m)
    apogeum   : 2080 m -> 2163 m     (telemetria: 1537 m)

Czyli poprawka podnosi crossrange z ~7% do ~70% wartosci zmierzonej. To jest
prawdopodobnie glowna przyczyna niedoszacowania crossrange, ktore w tej sesji
probowalismy tlumaczyc kolejno rollem, kierunkiem wiatru, sila wiatru, profilem
wiatru z wysokoscia i sztywnoscia pitch/yaw. Dopasowanie sztywnosci pitch/yaw
"nie zbiegalo" wlasnie dlatego, ze testowany zakres siegal x3, a potrzebne bylo
okolo x18.

Apogeum nadal przestrzelone — to osobny watek (deficyt oporu w fazie balistycznej).
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
