"""
test_axis_assignment.py
======================
Test roznych przypisan osi zyroskopu — porownanie z wysokoscia GPS.
Sprawdza ktore przypisanie osi pitch/yaw daje trajektorie blizsza GPS.

Warianty:
  A. oryginalne (gyro_x=roll, gyro_y, gyro_z)
  B. zamienione Y<->Z
  Dodatkowo testuje znaki osi.
"""

import numpy as np
import matplotlib.pyplot as plt
from telemetry_parser import parse_telemetry
from imu_reconstruction import (calibrate, initial_orientation,
                                reconstruct, Calibration, G0)
from dataclasses import replace
import copy


FLIGHT = 13
AZ, EL = 92.0, 45.0


def reconstruct_variant(tel, cal, q0, t_start, t_end,
                        swap_yz=False, sign_y=1, sign_z=1, sign_x=1):
    """Rekonstrukcja z modyfikacja przypisania osi zyroskopu."""
    tel2 = copy.deepcopy(tel)
    if swap_yz:
        tel2.gyro_y, tel2.gyro_z = tel.gyro_z.copy(), tel.gyro_y.copy()
    tel2.gyro_x = tel2.gyro_x * sign_x
    tel2.gyro_y = tel2.gyro_y * sign_y
    tel2.gyro_z = tel2.gyro_z * sign_z
    # Rekalibracja biasu dla zmienionych osi
    cal2 = calibrate(tel2, t_rest_end=-1.0)
    return reconstruct(tel2, cal2, q0, t_start=t_start, t_end=t_end)


def main():
    tel = parse_telemetry(f"ARTEMIDA_{FLIGHT}_LOT.txt", verbose=False)
    cal = calibrate(tel, t_rest_end=-1.0)
    q0  = initial_orientation(cal, AZ, EL)

    # Okno: zaplon do apogeum (faza gdzie GPS jest wiarygodny)
    t_ign = tel.time[np.argmax(tel.press_cham > 2.0)]
    # Apogeum z GPS
    mask_all = tel.time >= t_ign
    apo_idx  = np.argmax(tel.alt_onboard[mask_all])
    t_apo    = tel.time[mask_all][apo_idx]
    t_end    = t_apo + 2.0   # troche za apogeum

    print(f"Okno analizy: {t_ign:.2f}s do {t_end:.2f}s (apogeum GPS w {t_apo:.2f}s)")

    # GPS wysokosc (odniesienie)
    mask = (tel.time >= t_ign) & (tel.time <= t_end)
    t_gps   = tel.time[mask]
    alt_gps = tel.alt_onboard[mask] - tel.alt_onboard[mask][0]

    # Warianty
    variants = {
        "A: oryginalny":          dict(swap_yz=False),
        "B: swap Y<->Z":          dict(swap_yz=True),
        "C: swap + sign_y=-1":    dict(swap_yz=True, sign_y=-1),
        "D: swap + sign_z=-1":    dict(swap_yz=True, sign_z=-1),
    }

    results = {}
    for name, params in variants.items():
        traj = reconstruct_variant(tel, cal, q0, t_ign, t_end, **params)
        # Blad RMS wysokosci vs GPS (interpolacja na wspolna siatke)
        alt_imu = np.interp(t_gps, traj.time, traj.alt)
        rms = np.sqrt(np.mean((alt_imu - alt_gps)**2))
        apo_err = abs(traj.alt.max() - alt_gps.max())
        results[name] = {"traj": traj, "rms": rms, "apo_err": apo_err}
        print(f"  {name:25s}: RMS_alt={rms:6.1f}m  "
              f"apogeum_IMU={traj.alt.max():6.1f}m  "
              f"blad_apo={apo_err:6.1f}m")

    # Wykres
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Test przypisania osi — Lot {FLIGHT}", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(t_gps, alt_gps, 'k-', lw=2.5, label="GPS (odniesienie)", zorder=5)
    for name, r in results.items():
        ax.plot(r["traj"].time, r["traj"].alt, lw=1.2, label=name)
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Wysokosc [m]")
    ax.set_title("Wysokosc — warianty osi vs GPS")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    names = list(results.keys())
    rms_vals = [results[n]["rms"] for n in names]
    colors = ['#28a745' if r == min(rms_vals) else '#1f77b4' for r in rms_vals]
    ax.barh(names, rms_vals, color=colors)
    ax.set_xlabel("RMS bledu wysokosci [m]")
    ax.set_title("Blad RMS vs GPS (mniej = lepiej)")
    ax.grid(alpha=0.3, axis='x')
    for i, v in enumerate(rms_vals):
        ax.text(v, i, f" {v:.0f}m", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("test_axis_assignment.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("\nZapisano: test_axis_assignment.png")

    best = min(results, key=lambda n: results[n]["rms"])
    print(f"\nNajlepszy wariant: {best} (RMS={results[best]['rms']:.1f}m)")


if __name__ == "__main__":
    main()
