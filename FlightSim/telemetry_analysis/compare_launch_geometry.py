"""
compare_launch_geometry.py
==========================
Porownuje kat startowy (azymut, elewacja) ZAPISANY w configs.txt z
wartoscia ODTWORZONA z telemetrii kazdego lotu:

  - azymut  <- slad naziemny GPS (heading zaplon->apogeum), bo azymutu
               NIE da sie policzyc z akcelerometru (grawitacja pionowa);
  - elewacja <- akcelerometr w spoczynku przed zaplonem: elewacja osi X
               nad poziomem = asin(acc_x / |a|), niezaleznie od rolla.

Motywacja: dla lotu 19 zapisany azymut (325 deg) byl ~19 deg przesuniety
wzgledem realnego sladu GPS (~344 deg), co tlumaczylo pozorny "crossrange"
1400 m jako artefakt rzutu na zly azymut, nie dryf wiatrowy. Ten skrypt
sprawdza czy blad jest systematyczny czy per-lot dla WSZYSTKICH lotow z
telemetria.

Uzywa istniejacych estymatorow z plot_flight_trajectory_6dof.py
(estimate_azimuth_from_telemetry / estimate_elevation_from_telemetry) —
zadnej nowej fizyki.

Tylko odczyt: NIE modyfikuje configs.txt ani modelu. Zapisuje tabele do
field_test_data/results/launch_geometry_comparison.csv.

Uzycie:
    python compare_launch_geometry.py
"""

import sys
import csv
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, resolve_data_file
from analyze_launch_weather import read_flights          # wszystkie loty + flaga analyze
from plot_flight_trajectory_6dof import (
    estimate_azimuth_from_telemetry,
    estimate_elevation_from_telemetry,
)

LOW_BASELINE_M   = 1000.0   # ponizej -> heading GPS mniej pewny
HIGH_OFFPLANE_DEG = 10.0    # powyzej -> IMU rolled/niewspolosiowy


def wrap180(deg):
    """Roznica katowa w [-180, 180] (najkrotszy luk)."""
    return ((deg + 180.0) % 360.0) - 180.0


def main():
    base = get_data_dir()
    flights = sorted(read_flights(base), key=lambda r: r["fno"])

    rows = []
    print(f"{'flt':>3} {'cfg_az':>6} {'GPS_az':>7} {'d_az':>6} | "
          f"{'cfg_el':>6} {'IMU_el':>7} {'d_el':>6} | "
          f"{'base_m':>7} {'offpl':>5} {'anal':>4}  flagi")
    print("-" * 82)

    for r in flights:
        fno = r["fno"]
        cfg_az = r["azimuth_deg"]
        cfg_el = r["elevation_deg"]
        analyze = "yes" if r.get("analyze") else ""

        if not resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt").exists():
            print(f"{fno:>3} {cfg_az:>6.0f} {'--':>7} {'--':>6} | "
                  f"{cfg_el:>6.0f} {'--':>7} {'--':>6} | "
                  f"{'--':>7} {'--':>5} {analyze:>4}  brak telemetrii")
            continue

        try:
            gps_az, baseline = estimate_azimuth_from_telemetry(base, fno)
            imu_el, off_plane = estimate_elevation_from_telemetry(base, fno)
        except Exception as e:
            print(f"{fno:>3} {cfg_az:>6.0f}  -- blad: {e}")
            continue

        d_az = wrap180(gps_az - cfg_az)
        d_el = imu_el - cfg_el

        flags = []
        if baseline < LOW_BASELINE_M:
            flags.append("krotka-baza")
        if off_plane > HIGH_OFFPLANE_DEG:
            flags.append("duzy-off-plane")
        flag_str = ",".join(flags)

        print(f"{fno:>3} {cfg_az:>6.0f} {gps_az:>7.1f} {d_az:>+6.1f} | "
              f"{cfg_el:>6.0f} {imu_el:>7.1f} {d_el:>+6.1f} | "
              f"{baseline:>7.0f} {off_plane:>5.1f} {analyze:>4}  {flag_str}")

        rows.append(dict(
            fno=fno, analyze=analyze,
            cfg_az=cfg_az, gps_az=round(gps_az, 2), d_az=round(d_az, 2),
            baseline_m=round(baseline, 1),
            cfg_el=cfg_el, imu_el=round(imu_el, 2), d_el=round(d_el, 2),
            off_plane_deg=round(off_plane, 2), flags=flag_str,
        ))

    if not rows:
        print("\nBrak lotow z telemetria — nic nie zapisano.")
        return

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "launch_geometry_comparison.csv"
    fields = ["fno", "analyze", "cfg_az", "gps_az", "d_az", "baseline_m",
              "cfg_el", "imu_el", "d_el", "off_plane_deg", "flags"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nZapisano: {csv_path}")
    print("(d_az = GPS - configs, znak dodatni = GPS zgodnie z ruchem "
          "wskazowek od zapisanego; d_el = IMU - configs)")


if __name__ == "__main__":
    main()
