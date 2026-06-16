"""
debug_cant_cll2.py — sprawdz pelne tabele CLL z cache pkl
Uruchom z katalogu FlightSim: python debug_cant_cll2.py
"""
import sys, numpy as np, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

CASE_NAME = "rocket_36mm_malarakieta_base"
CANT_VALS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0]

for cant in CANT_VALS:
    case_tag = f"{CASE_NAME}_cant{str(cant).replace('.','p')}"
    pkl_path = Path("datcom_runs") / case_tag / "aero_table_missile.pkl"
    if not pkl_path.exists():
        print(f"cant={cant}°: brak {pkl_path}")
        continue

    with open(pkl_path, "rb") as f:
        aero = pickle.load(f)

    idx_a0 = np.argmin(np.abs(np.degrees(aero.alpha_table)))

    print(f"\ncant={cant}°:")
    print(f"  CLL_table is None: {aero.CLL_table is None}")
    if aero.CLL_table is not None:
        print(f"  CLL shape: {aero.CLL_table.shape}")
        print(f"  CLL @ alpha=0, wszystkie Mach: {aero.CLL_table[idx_a0, :]}")
        print(f"  CLL min={aero.CLL_table.min():.6f}  max={aero.CLL_table.max():.6f}")
    print(f"  Clp_table is None: {aero.Clp_table is None}")
    if aero.Clp_table is not None:
        print(f"  Clp @ alpha=0, Ma[0]: {aero.Clp_table[idx_a0, 0]:.4f}")
