"""
diag_roll_strategy.py — testuje strategie obslugi wysyconego rolla.
Porownuje rekonstrukcje wysokosci IMU vs GPS dla roznych podejsc do rolla.
Uzycie: python telemetry_analysis/diag_roll_strategy.py 16
"""
import sys
import copy
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import parse_telemetry, resolve_data_file
from imu_reconstruction import (calibrate, initial_orientation, reconstruct,
                                detect_ignition)

fno = int(sys.argv[1])
# azymut/elewacja z configs
import csv
cfg = {}
cpath = resolve_data_file("configs.txt")
with open(cpath, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
hdr = lines[0].strip().split("\t")
for ln in lines[1:]:
    p = ln.strip().split("\t")
    if len(p) >= len(hdr):
        row = dict(zip(hdr, p))
        try:
            if int(row.get("flight no",-1)) == fno:
                cfg = row; break
        except: pass
az = float(cfg.get("azimuth", 92)); el = float(cfg.get("elevation", 45))

tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)
cal = calibrate(tel, t_rest_end=tel.time[np.argmax(tel.flag_flight>0)]-0.5 if tel.flag_flight is not None else -1.0)
q0 = initial_orientation(cal, az, el, use_gravity=False)
t_ign = detect_ignition(tel)
mask = (tel.time >= t_ign)
alt_gps = tel.alt_onboard[mask] - tel.alt_onboard[mask][0]
t_gps = tel.time[mask]
apo_gps = np.nanmax(alt_gps)

def rms_vs_gps(traj):
    ai = np.interp(t_gps, traj.time, traj.alt)
    return np.sqrt(np.nanmean((ai - alt_gps)**2))

print(f"=== Lot {fno} (az={az} el={el}) — strategie rolla ===")
print(f"GPS apogeum: {apo_gps:.0f}m\n")

# A. Roll jak jest (wysycony)
trajA = reconstruct(tel, cal, q0, t_start=t_ign, t_end=tel.time[-1])
print(f"A. roll wysycony (baseline): apogeum={trajA.alt.max():.0f}m  RMS={rms_vs_gps(trajA):.0f}m")

# B. Roll = 0 (pominiety calkowicie)
telB = copy.deepcopy(tel); telB.gyro_x = np.zeros_like(telB.gyro_x)
trajB = reconstruct(telB, cal, q0, t_start=t_ign, t_end=tel.time[-1])
print(f"B. roll=0 (pominiety):       apogeum={trajB.alt.max():.0f}m  RMS={rms_vs_gps(trajB):.0f}m")

# C. Roll z AX (przeskalowany, znak z gyro_X)
ax = tel.raw_columns.get("gyro_ax")
if ax is not None:
    telC = copy.deepcopy(tel)
    # AX to |roll rate|, offset ~150 w spoczynku, skala ~4.3x wzgledem gyro_X
    # Kalibracja: odejmij offset spoczynkowy, podziel przez skale
    rest = tel.time < (t_ign - 0.5)
    ax_offset = np.nanmean(ax[rest])
    ax_cal = (ax - ax_offset)  # teraz w jednostkach AX
    # Znak z gyro_X (gdzie niewysycony), inaczej zachowaj poprzedni
    sign = np.sign(tel.gyro_x)
    sign[np.abs(tel.gyro_x) > 1999] = 0  # nieznany znak gdy wysycony
    # Wypelnij nieznane znaki ostatnim znanym (roll zwykle nie zmienia kierunku)
    last = 1
    sign_filled = sign.copy()
    for i in range(len(sign)):
        if sign[i] == 0: sign_filled[i] = last
        else: last = sign[i]
    # Skala: dopasuj AX do gyro_X tam gdzie gyro_X niewysycony
    ns = (np.abs(tel.gyro_x) < 1990) & (np.abs(tel.gyro_x) > 100)
    if np.sum(ns) > 20:
        scale = np.nanmedian(np.abs(tel.gyro_x[ns]) / (np.abs(ax_cal[ns])+1e-9))
    else:
        scale = 1.0
    telC.gyro_x = ax_cal * scale * sign_filled
    trajC = reconstruct(telC, cal, q0, t_start=t_ign, t_end=tel.time[-1])
    print(f"C. roll z AX (skala={scale:.3f}):  apogeum={trajC.alt.max():.0f}m  RMS={rms_vs_gps(trajC):.0f}m")
    print(f"   AX roll max: {np.nanmax(np.abs(telC.gyro_x[mask])):.0f} st/s")

print(f"\nNajlepsza strategia = najmniejszy RMS")
