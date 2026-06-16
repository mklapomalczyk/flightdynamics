"""
diag_gps_jumps.py — analiza skokow GPS i detekcji zaplonu.
Uzycie: python telemetry_analysis/diag_gps_jumps.py 16
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import parse_telemetry, resolve_data_file
from gps_fusion import latlon_to_enu

fno = int(sys.argv[1])
tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)

print(f"=== Lot {fno} ===")

# --- DETEKCJA ZAPLONU: porownanie metod ---
print("\n--- Detekcja zaplonu ---")
# Metoda 1: cisnienie
if tel.press_cham is not None and np.any(tel.press_cham > 2):
    t_press = tel.time[np.argmax(tel.press_cham > 2.0)]
    print(f"  press_cham>2bar:  t={t_press:.2f}s")
# Metoda 2: przyspieszenie osiowe
for thr in [3, 5, 8, 10]:
    if np.any(tel.acc_x > thr):
        t_acc = tel.time[np.argmax(tel.acc_x > thr)]
        print(f"  acc_X>{thr}g:        t={t_acc:.2f}s")
# Metoda 3: FLAGA LOT
if tel.flag_flight is not None and np.any(tel.flag_flight > 0):
    t_flag = tel.time[np.argmax(tel.flag_flight > 0)]
    print(f"  FLAGA_LOT:        t={t_flag:.2f}s")

# --- SKOKI GPS ---
print("\n--- Skoki pozycji GPS ---")
mask = tel.time >= 0
t = tel.time[mask]
lat = tel.lat[mask]; lon = tel.lon[mask]
alt = tel.alt_onboard[mask]

# Punkty gdzie GPS sie zmienia
e, n = latlon_to_enu(lat, lon, lat[0], lon[0])
gps_change = np.where((np.abs(np.diff(e)) > 1e-6) | (np.abs(np.diff(n)) > 1e-6))[0]
if len(gps_change) > 1:
    t_ch = t[gps_change]
    e_ch = e[gps_change]; n_ch = n[gps_change]; a_ch = alt[gps_change]
    dt_ch = np.diff(t_ch)
    de = np.diff(e_ch); dn = np.diff(n_ch); da = np.diff(a_ch)
    dist = np.sqrt(de**2 + dn**2)
    speed_implied = dist / dt_ch   # predkosc pozioma implikowana skokiem
    print(f"  liczba aktualizacji GPS: {len(gps_change)}")
    print(f"  typowy odstep: {np.median(dt_ch):.3f}s")
    print(f"  skok poziomy: mediana={np.median(dist):.1f}m  max={dist.max():.1f}m")
    print(f"  predkosc implikowana skokiem: mediana={np.median(speed_implied):.0f}m/s  max={speed_implied.max():.0f}m/s")
    # Ile skokow implikuje predkosc > 400 m/s (niefizyczna)
    bad = np.sum(speed_implied > 400)
    print(f"  skoki implikujace V>400m/s (podejrzane): {bad}")
    print(f"  skok pionowy (alt): mediana={np.median(np.abs(da)):.1f}m  max={np.abs(da).max():.1f}m")

# --- ZAMROZENIA GPS (utrata fixa) ---
print("\n--- Zamrozenia GPS ---")
# najdluzszy ciag bez zmiany pozycji
no_change = np.abs(np.diff(e)) < 1e-6
runs = []
cur = 0
for nc in no_change:
    if nc: cur += 1
    else:
        if cur > 0: runs.append(cur)
        cur = 0
if runs:
    longest = max(runs) * tel.dt_mean
    print(f"  najdluzsze zamrozenie pozycji: {longest:.2f}s")
    print(f"  (typowe ~{1/np.median(np.diff(t[gps_change])):.0f}Hz = {np.median(np.diff(t[gps_change])):.2f}s odstep)")
