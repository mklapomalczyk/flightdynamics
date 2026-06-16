"""
diag_gps.py — diagnostyka struktury GPS i profilu napedu dla lotu.
Uzycie (z katalogu FlightSim lub field_test_data):
    python telemetry_analysis/diag_gps.py 16
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import parse_telemetry, resolve_data_file

fno = int(sys.argv[1])
tel = parse_telemetry(resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt"), verbose=False)

t_ign = tel.time[np.argmax(tel.press_cham > 2.0)] if tel.press_cham is not None and np.any(tel.press_cham>2) else 0
mask = (tel.time >= t_ign) & (tel.time <= tel.time[-1])
t = tel.time[mask]

print(f"=== Lot {fno} ===")
print(f"Zaplon: t={t_ign:.2f}s")

if tel.alt_onboard is not None:
    alt = tel.alt_onboard[mask]
    print(f"\nGPS wysokosc:")
    print(f"  start={alt[0]:.0f}m  max={np.nanmax(alt):.0f}m  apogeum w t={t[np.nanargmax(alt)]:.1f}s")
    uniq = np.unique(alt[~np.isnan(alt)])
    print(f"  unikalnych wartosci: {len(uniq)}")
    # czy GPS startuje pozno (zero przez pierwsze sekundy)?
    first_nonzero = np.argmax(np.abs(alt - alt[0]) > 5)
    print(f"  pierwsza zmiana >5m w t={t[first_nonzero]:.2f}s (GPS lag/brak fix w fazie napedu)")

if tel.vel_onboard is not None:
    v = tel.vel_onboard[mask]
    print(f"\nGPS predkosc onboard:")
    print(f"  max={np.nanmax(v):.0f}m/s")
    first_v = np.argmax(v > 5)
    print(f"  pierwsza predkosc >5m/s w t={t[first_v]:.2f}s")

acc = tel.acc_x[mask]
print(f"\nPrzyspieszenie osiowe acc_X:")
print(f"  max={np.nanmax(acc):.1f}g w t={t[np.nanargmax(acc)]:.2f}s")
above = t[acc > 3.0]
if len(above) > 0:
    print(f"  acc>3g od t={above[0]:.2f}s do t={above[-1]:.2f}s ({len(above)} probek)")
above2 = (acc > 2.0).astype(int)
transitions = np.where(np.diff(above2) != 0)[0]
print(f"  przejscia przez prog 2g: {len(transitions)} (>2 = wielofazowy naped)")

roll = tel.gyro_x[mask]
print(f"\nRoll gyro_X: srednia={np.nanmean(roll):.0f}st/s  scalkowany={np.nansum(roll)*tel.dt_mean:.0f}st")
sat = np.sum(np.abs(roll) > 1999)
print(f"  probek wysyconych (>1999): {sat} ({100*sat/len(roll):.1f}%)")
