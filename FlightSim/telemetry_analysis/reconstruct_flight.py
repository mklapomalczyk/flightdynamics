"""
reconstruct_flight.py
=====================
Glowny skrypt rekonstrukcji trajektorii lotu z IMU.
Laczy parser, kalibracje i strapdown INS, porownuje z telemetria pokladowa.

Uzycie:
    python reconstruct_flight.py 13
    python reconstruct_flight.py 13 --az 92 --el 45
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from telemetry_parser import parse_telemetry, resolve_data_file, get_data_dir
from imu_reconstruction import (calibrate, initial_orientation,
                                reconstruct, detect_ignition, G0)
from gps_fusion import fuse_imu_gps, latlon_to_enu


def load_config(flight_no, config_path="configs.txt"):
    """Wczytuje azymut i elewacje z configs.txt dla danego lotu."""
    cfg = {}
    p = resolve_data_file(config_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    header = lines[0].strip().split("\t")
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        try:
            fno = int(row.get("flight no", -1))
        except ValueError:
            continue
        if fno == flight_no:
            return {
                "cant":      float(row.get("cant angle", 0)),
                "head":      row.get("head configuration", "?"),
                "azimuth":   float(row.get("azimuth", 0)),
                "elevation": float(row.get("elevation", 45)),
            }
    return None


def detect_flight_window(tel):
    """Wykrywa okno lotu: zaplon (detect_ignition) do konca danych."""
    t_ignition = detect_ignition(tel)
    t_end = tel.time[-1]
    return t_ignition, t_end


def main():
    if len(sys.argv) < 2:
        print("Uzycie: python reconstruct_flight.py <flight_no>")
        sys.exit(1)

    flight_no = int(sys.argv[1])
    fname = resolve_data_file(f"ARTEMIDA_{flight_no}_LOT.txt")

    # Parsuj
    tel = parse_telemetry(fname)

    # Konfiguracja (azymut/elewacja)
    cfg = load_config(flight_no)
    if "--az" in sys.argv:
        az = float(sys.argv[sys.argv.index("--az")+1])
    else:
        az = cfg["azimuth"] if cfg else 92.0
    if "--el" in sys.argv:
        el = float(sys.argv[sys.argv.index("--el")+1])
    else:
        el = cfg["elevation"] if cfg else 45.0

    print(f"\nLot {flight_no}: azymut={az}° elewacja={el}°")
    if cfg:
        print(f"  cant={cfg['cant']}° glowica={cfg['head']}")

    # Kalibracja
    cal = calibrate(tel, t_rest_end=-1.0)
    print(f"\nKalibracja ({cal.n_rest} probek spoczynkowych):")
    print(f"  Bias zyroskopu [st/s]: "
          f"X={cal.gyro_bias[0]:+.3f} Y={cal.gyro_bias[1]:+.3f} Z={cal.gyro_bias[2]:+.3f}")
    print(f"  Skala akcelerometru: {cal.acc_scale:.4f} "
          f"(|a|_spoczynek = {1/cal.acc_scale:.4f}g)")

    # Orientacja poczatkowa
    # use_gravity=False daje lepszy wynik (RMS 59m vs 469m) — prosty obrot
    # z azymut+elewacja. Wariant z grawitacja wymaga dopracowania konwencji osi.
    q0 = initial_orientation(cal, az, el, use_gravity=False)
    print(f"  Kwaternion poczatkowy: "
          f"[{q0[0]:.3f}, {q0[1]:.3f}, {q0[2]:.3f}, {q0[3]:.3f}]")

    # Okno lotu
    t_ign, t_end = detect_flight_window(tel)
    print(f"\nOkno lotu: t={t_ign:.3f}s do {t_end:.3f}s")

    # Rekonstrukcja
    traj = reconstruct(tel, cal, q0, t_start=t_ign, t_end=t_end)
    print(f"\nRekonstrukcja IMU:")
    print(f"  Apogeum IMU: {traj.alt.max():.1f}m w t={traj.time[np.argmax(traj.alt)]:.2f}s")
    print(f"  Vmax IMU: {traj.speed.max():.1f}m/s")
    print(f"  Pozycja koncowa ENU: "
          f"E={traj.pos[-1,0]:.0f}m N={traj.pos[-1,1]:.0f}m U={traj.pos[-1,2]:.0f}m")

    # Fuzja IMU + GPS
    fused = fuse_imu_gps(traj, tel, t_ign,
                         alpha_pos=0.02, alpha_vel=0.01, acc_gate=3.0)
    print(f"\nFuzja IMU + GPS:")
    print(f"  Apogeum: {fused.alt.max():.1f}m")
    print(f"  Vmax: {fused.speed.max():.1f}m/s")
    print(f"  Probek z korekcja GPS: {int(np.sum(fused.gps_valid))}/{len(fused.time)}")
    print(f"  Odrzucone falszywe punkty GPS: {fused.n_gps_rejected}")

    # Telemetria pokladowa do porownania (w tym samym oknie)
    mask = (tel.time >= t_ign) & (tel.time <= t_end)
    t_ob   = tel.time[mask]
    alt_ob = tel.alt_onboard[mask] - tel.alt_onboard[mask][0]  # wzgledem startu
    vel_ob = tel.vel_onboard[mask]

    print(f"\nGPS pokladowy (odniesienie):")
    print(f"  Apogeum: {alt_ob.max():.1f}m")
    print(f"  Vmax: {vel_ob.max():.1f}m/s (ograniczony przez GPS)")

    # ---------------- Wykresy ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Rekonstrukcja IMU + fuzja GPS — Lot {flight_no} "
                 f"(az={az}° el={el}°)",
                 fontsize=13, fontweight="bold")

    # 1. Wysokosc vs czas — IMU, fused, GPS
    ax = axes[0, 0]
    ax.plot(traj.time, traj.alt, 'b-', lw=1.2, alpha=0.6, label="IMU (czysty)")
    ax.plot(fused.time, fused.alt, 'g-', lw=1.5, label="IMU+GPS")
    ax.plot(t_ob, alt_ob, 'r--', lw=1.2, label="GPS pokladowy")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Wysokosc [m]")
    ax.set_title("Wysokosc vs czas"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. Predkosc vs czas
    ax = axes[0, 1]
    ax.plot(traj.time, traj.speed, 'b-', lw=1.2, alpha=0.6, label="IMU |v|")
    ax.plot(fused.time, fused.speed, 'g-', lw=1.5, label="IMU+GPS")
    ax.plot(t_ob, vel_ob, 'r--', lw=1.2, label="GPS pokladowy")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Predkosc [m/s]")
    ax.set_title("Predkosc vs czas"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Trajektoria pozioma (E-N) z GPS
    ax = axes[0, 2]
    # GPS w ENU
    mask_g = (tel.time >= t_ign) & (tel.time <= t_end)
    e_g, n_g = latlon_to_enu(tel.lat[mask_g], tel.lon[mask_g],
                              tel.lat[mask_g][0], tel.lon[mask_g][0])
    ax.plot(traj.pos[:, 0], traj.pos[:, 1], 'b-', lw=1, alpha=0.5, label="IMU")
    ax.plot(fused.pos[:, 0], fused.pos[:, 1], 'g-', lw=1.5, label="IMU+GPS")
    ax.plot(e_g, n_g, 'r--', lw=1, label="GPS")
    ax.plot(0, 0, 'ko', ms=8, label="Start")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Trajektoria pozioma (ENU)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axis('equal')

    # 4. Surowe dane IMU — przyspieszenia
    ax = axes[1, 0]
    mask_raw = (tel.time >= t_ign - 1) & (tel.time <= t_end)
    ax.plot(tel.time[mask_raw], tel.acc_x[mask_raw], 'r-', lw=0.7, label="acc_X (os)")
    ax.plot(tel.time[mask_raw], tel.acc_y[mask_raw], 'g-', lw=0.7, label="acc_Y")
    ax.plot(tel.time[mask_raw], tel.acc_z[mask_raw], 'b-', lw=0.7, label="acc_Z")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Przyspieszenie [g]")
    ax.set_title("Surowe dane — akcelerometr")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. Surowe dane IMU — predkosci katowe
    ax = axes[1, 1]
    ax.plot(tel.time[mask_raw], tel.gyro_x[mask_raw], 'r-', lw=0.7, label="gyro_X (roll)")
    ax.plot(tel.time[mask_raw], tel.gyro_y[mask_raw], 'g-', lw=0.7, label="gyro_Y (pitch)")
    ax.plot(tel.time[mask_raw], tel.gyro_z[mask_raw], 'b-', lw=0.7, label="gyro_Z (yaw)")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Predkosc katowa [st/s]")
    ax.set_title("Surowe dane — zyroskop (po zamianie Y/Z)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 6. Katy Eulera
    ax = axes[1, 2]
    ax.plot(traj.time, traj.euler[:, 0], 'r-', lw=1, label="roll")
    ax.plot(traj.time, traj.euler[:, 1], 'g-', lw=1, label="pitch")
    ax.plot(traj.time, traj.euler[:, 2], 'b-', lw=1, label="yaw")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Kat [st]")
    ax.set_title("Orientacja (Euler)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = str(get_data_dir() / "results" / f"reconstruction_flight_{flight_no}.png")
    Path(out).parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()
