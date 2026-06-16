"""
debug_cant_datcom.py — pokaz surowe wartosci CLL z datcom.out
Uruchom z katalogu FlightSim: python debug_cant_datcom.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

CASE_NAME = "rocket_36mm_malarakieta_base"
CANT_VALS = [0.1, 0.25, 0.4, 0.5, 0.7, 1.0]

for cant in CANT_VALS:
    case_tag = f"{CASE_NAME}_cant{str(cant).replace('.','p')}"
    out_path = Path("datcom_runs") / case_tag / "datcom.out"
    if not out_path.exists():
        print(f"cant={cant}°: brak {out_path}")
        continue

    # Znajdz linie z CLL w datcom.out
    lines = open(out_path, encoding='utf-8', errors='replace').readlines()
    print(f"\ncant={cant}° — {out_path.name}:")
    in_table = False
    for i, l in enumerate(lines):
        if 'ALPHA' in l and 'CLL' in l:
            in_table = True
            print(l.rstrip())
            continue
        if in_table:
            if l.strip() == '' or '****' in l:
                in_table = False
                continue
            # Pokaz tylko linie z alpha=0
            parts = l.split()
            if parts and len(parts) >= 7:
                try:
                    alpha = float(parts[0])
                    cll   = float(parts[6])
                    if abs(alpha) < 1.0:
                        print(f"  alpha={alpha:6.2f}  CLL={cll:.6f}  (raw: '{parts[6]}')")
                except ValueError:
                    pass
