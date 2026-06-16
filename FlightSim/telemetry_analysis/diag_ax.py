"""
diag_ax.py — analiza kolumny 'Predkosc obrotowa AX' vs gyro_X.
Sprawdza czy AX to high-range czujnik rolla ktory moglby zastapic
wysycony gyro_X w lotach 15-21.
Uzycie: python telemetry_analysis/diag_ax.py 16
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import parse_telemetry, resolve_data_file
from imu_reconstruction import detect_ignition

fno = int(sys.argv[1])
tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)
ax = tel.raw_columns.get("gyro_ax")
gx = tel.gyro_x

if ax is None:
    print("Brak kolumny gyro_AX w tym pliku")
    sys.exit(0)

t_ign = detect_ignition(tel)
print(f"=== Lot {fno} (zaplon t={t_ign:.2f}s) ===")

rest = tel.time < (t_ign - 1)
fl = (tel.time >= t_ign) & (tel.time <= tel.time[-1])

print(f"\nSpoczynek:")
print(f"  gyro_AX: srednia={np.nanmean(ax[rest]):.1f}  std={np.nanstd(ax[rest]):.1f}")
print(f"  gyro_X:  srednia={np.nanmean(gx[rest]):.2f}  std={np.nanstd(gx[rest]):.2f}")

print(f"\nW locie:")
print(f"  gyro_AX: srednia={np.nanmean(ax[fl]):.0f}  zakres=[{np.nanmin(ax[fl]):.0f}, {np.nanmax(ax[fl]):.0f}]")
print(f"  gyro_X:  srednia={np.nanmean(gx[fl]):.0f}  zakres=[{np.nanmin(gx[fl]):.0f}, {np.nanmax(gx[fl]):.0f}]")

print(f"\nSaturacja w locie:")
sat_ax = np.sum(np.abs(ax[fl]) > 1999)
sat_gx = np.sum(np.abs(gx[fl]) > 1999)
print(f"  gyro_AX wysycony (>1999): {sat_ax} ({100*sat_ax/np.sum(fl):.1f}%)")
print(f"  gyro_X  wysycony (>1999): {sat_gx} ({100*sat_gx/np.sum(fl):.1f}%)")

# Czy AX jest zawsze dodatni? (= moduł predkosci)
neg = np.sum(ax[fl] < -1)
print(f"\n  gyro_AX wartosci ujemne (<-1): {neg}")
print(f"  -> jesli ~0, AX to MODUL |roll rate| (bez znaku)")

# Tam gdzie gyro_X NIE wysycony — porownaj |gyro_X| z AX
notsat = (np.abs(gx[fl]) < 1990) & (np.abs(gx[fl]) > 50)
if np.sum(notsat) > 20:
    ratio = np.nanmedian(ax[fl][notsat] / np.abs(gx[fl][notsat]))
    print(f"\n  Gdzie gyro_X niewysycony: mediana AX/|gyro_X| = {ratio:.2f}")
    print(f"  (jesli ~1.0 to ta sama wielkosc, AX moze zastapic wysycony gyro_X)")

# Co pokazuje AX tam gdzie gyro_X JEST wysycony?
sat_mask = np.abs(gx[fl]) > 1999
if np.sum(sat_mask) > 10:
    print(f"\n  Tam gdzie gyro_X wysycony (±2000):")
    print(f"    gyro_AX srednia={np.nanmean(ax[fl][sat_mask]):.0f}  max={np.nanmax(ax[fl][sat_mask]):.0f}")
    print(f"    -> jesli >2000, AX widzi prawdziwy roll ktorego gyro_X nie zlapal!")
