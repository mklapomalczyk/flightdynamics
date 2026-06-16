"""
diag_sat.py — wysycenie KAZDEJ osi zyroskopu osobno w fazie lotu.
Uzycie: python telemetry_analysis/diag_sat.py 16
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import parse_telemetry, resolve_data_file
from imu_reconstruction import detect_ignition

fno = int(sys.argv[1])
tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)
t_ign = detect_ignition(tel)
fl = (tel.time >= t_ign) & (tel.time <= tel.time[-1])
n = int(np.sum(fl))

print(f"=== Lot {fno}: wysycenie osi zyroskopu (zaplon t={t_ign:.2f}s) ===")
for name, g in [("gyro_X (roll)", tel.gyro_x[fl]),
                ("gyro_Y (pitch)", tel.gyro_y[fl]),
                ("gyro_Z (yaw)", tel.gyro_z[fl])]:
    sat = np.sum(np.abs(g) > 1999)
    print(f"  {name}: wysycony {sat}/{n} ({100*sat/n:.1f}%)  zakres=[{np.nanmin(g):.0f},{np.nanmax(g):.0f}]")

# Tylko faza napedowa (pierwsze 2s) vs balistyczna
print(f"\n  Rozbicie pitch/yaw na fazy:")
for ph_name, lo, hi in [("naped (0-2s)", t_ign, t_ign+2),
                         ("balistyka (2s+)", t_ign+2, tel.time[-1])]:
    m = (tel.time >= lo) & (tel.time <= hi)
    nn = int(np.sum(m))
    if nn == 0: continue
    sy = np.sum(np.abs(tel.gyro_y[m]) > 1999)
    sz = np.sum(np.abs(tel.gyro_z[m]) > 1999)
    print(f"    {ph_name}: pitch wysycony {100*sy/nn:.1f}%  yaw wysycony {100*sz/nn:.1f}%")
