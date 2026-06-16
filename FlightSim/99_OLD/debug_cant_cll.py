"""
debug_cant_cll.py — sprawdz CLL dla roznych cant_angle z cache pkl
Uruchom z katalogu FlightSim: python debug_cant_cll.py
"""
import sys, numpy as np, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

CASE_NAME = "rocket_36mm_malarakieta_base"
CANT_VALS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0]

print(f"{'cant':>6} {'CLL(a=0,Ma=0.3)':>18} {'Clp(a=0,Ma=0.3)':>18} {'p_eq@V=80m/s':>15}")
print("-"*62)

for cant in CANT_VALS:
    case_tag = f"{CASE_NAME}_cant{str(cant).replace('.','p')}"
    pkl_path = Path("datcom_runs") / case_tag / "aero_table_missile.pkl"

    if not pkl_path.exists():
        print(f"{cant:>6}°  brak cache: {pkl_path}")
        continue

    with open(pkl_path, "rb") as f:
        aero = pickle.load(f)

    idx_a0 = np.argmin(np.abs(np.degrees(aero.alpha_table)))

    CLL = float(aero.CLL_table[idx_a0, 0]) if aero.CLL_table is not None else 0.0
    Clp = float(aero.Clp_table[idx_a0, 0]) if aero.Clp_table is not None else 0.0

    d = 0.036
    V = 80.
    if abs(Clp) > 1e-10 and abs(CLL) > 1e-10:
        p_eq_degs = float(np.degrees(-CLL / (Clp * d/(2*V))))
    else:
        p_eq_degs = 0.0

    print(f"{cant:>6}°  {CLL:>18.6f}  {Clp:>18.4f}  {p_eq_degs:>15.1f} °/s")
