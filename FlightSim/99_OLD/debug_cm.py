"""Sprawdz Cm przed i po korekcji — uruchom z katalogu FlightSim"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from aero import get_aero_model

CASE_NAME = "rocket_70mm_baseline"
aero = get_aero_model(CASE_NAME, method="missile_datcom")

print(f"xcg_ref = {aero.xcg_ref*1000:.1f}mm")
print(f"d = 36mm")
print()

# Przy alpha=2 deg, Ma=0.3
alpha_deg = 2.0
alpha_rad = np.deg2rad(alpha_deg)
mach = 0.3
xcg_ref = aero.xcg_ref

Cm_raw = aero._interp(aero.Cm_table, alpha_rad, mach)
CN     = aero._interp(aero.CN_table, alpha_rad, mach)
xcp    = aero._interp(aero.xcp_table, alpha_rad, mach)
d      = 0.070

print(f"alpha={alpha_deg}°, Ma={mach}")
print(f"CN     = {CN:.4f}")
print(f"Cm_raw = {Cm_raw:.4f}  (z DATCOM dla xcg_ref={xcg_ref*1000:.1f}mm)")
print(f"xcp    = {xcp*1000:.1f}mm")
print()

# Korekcja dla roznych xcg
print("Korekcja Cm dla roznych xcg:")
print(f"{'xcg[mm]':>10} {'Cm_raw':>10} {'korekta':>10} {'Cm_corr':>10} {'arm_eff[mm]':>12} {'stabilny':>10}")
for xcg in [0.12, 0.20, 0.232, 0.28, 0.36, 0.44]:
    korekta  = CN * (xcg - xcg_ref) / d
    Cm_corr  = Cm_raw + korekta
    arm_eff  = Cm_corr * d / CN if abs(CN) > 0.01 else 0.
    stable   = Cm_corr < 0  # ujemne Cm = stabilizujace
    print(f"{xcg*1000:>10.1f} {Cm_raw:>10.4f} {korekta:>10.4f} {Cm_corr:>10.4f} {arm_eff*1000:>12.1f} {'TAK' if stable else 'NIE':>10}")

print()
print("Cm powinno byc ujemne dla stabilnej rakiety przy alpha>0")
print("arm_eff = xcg_eff - xcp, ujemne = xcp za xcg = stabilne")
