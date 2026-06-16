"""
batch_process.py
===============
Przetwarza wszystkie loty oznaczone 'yes' w configs.txt.

Dla kazdego lotu:
1. Waliduje strukture pliku
2. Parsuje telemetrie
3. Rekonstruuje trajektorie (IMU + fuzja GPS)
4. Zapisuje wykres do results/
5. Zbiera metryki do wspolnego CSV

Uruchomienie z katalogu field_test_data:
    python telemetry_analysis/batch_process.py
"""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # bez GUI
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import parse_telemetry, resolve_data_file, get_data_dir
from imu_reconstruction import (calibrate, initial_orientation, reconstruct,
                                detect_ignition)
from gps_fusion import fuse_imu_gps, latlon_to_enu
from validate_structure import validate_file, load_flights_to_analyze, read_header


RESULTS_DIR = get_data_dir() / "results"


def load_config_row(flight_no, config_path="configs.txt"):
    """Wczytuje pelny wiersz konfiguracji dla lotu."""
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
            if int(row.get("flight no", -1)) == flight_no:
                return row
        except ValueError:
            continue
    return None


def process_flight(flight_no, config_row, verbose=True):
    """
    Przetwarza pojedynczy lot. Zwraca slownik metryk lub None przy bledzie.
    """
    fname = resolve_data_file(f"ARTEMIDA_{flight_no}_LOT.txt")
    if not fname.exists():
        print(f"  [BRAK] {fname}")
        return None

    try:
        az = float(config_row.get("azimuth", 92)) if config_row else 92.0
        el = float(config_row.get("elevation", 45)) if config_row else 45.0
        cant = float(config_row.get("cant angle", 0)) if config_row else 0.0
        head = config_row.get("head configuration", "?") if config_row else "?"

        # Parsuj
        tel = parse_telemetry(fname, verbose=False)

        # Kalibracja
        cal = calibrate(tel, t_rest_end=-1.0)
        q0  = initial_orientation(cal, az, el, use_gravity=False)

        # Okno lotu
        t_ign = detect_ignition(tel)
        t_end = tel.time[-1]

        # Rekonstrukcja
        traj  = reconstruct(tel, cal, q0, t_start=t_ign, t_end=t_end)
        fused = fuse_imu_gps(traj, tel, t_ign)

        # GPS odniesienie
        mask = (tel.time >= t_ign) & (tel.time <= t_end)
        alt_gps = tel.alt_onboard[mask] - tel.alt_onboard[mask][0]
        vel_gps = tel.vel_onboard[mask]

        # Metryki
        metrics = {
            "flight_no":     flight_no,
            "cant":          cant,
            "head":          head,
            "azimuth":       az,
            "elevation":     el,
            "apogee_imu":    round(float(traj.alt.max()), 1),
            "apogee_fused":  round(float(fused.alt.max()), 1),
            "gps_rejected":  int(fused.n_gps_rejected),
            "apogee_gps":    round(float(alt_gps.max()), 1),
            "vmax_imu":      round(float(traj.speed.max()), 1),
            "vmax_fused":    round(float(fused.speed.max()), 1),
            "vmax_gps":      round(float(vel_gps.max()), 1),
            "roll_total_deg": round(float(np.degrees(
                np.sum(np.radians(tel.gyro_x[mask])) * tel.dt_mean)), 0),
            "t_ignition":    round(float(t_ign), 3),
            "duration_s":    round(float(t_end - t_ign), 1),
        }

        # Wykres
        _plot_flight(flight_no, traj, fused, tel, t_ign, t_end,
                     az, el, cant, head)

        if verbose:
            print(f"  [OK] Lot {flight_no}: apogeum IMU={metrics['apogee_imu']}m "
                  f"fused={metrics['apogee_fused']}m GPS={metrics['apogee_gps']}m  "
                  f"Vmax_imu={metrics['vmax_imu']}m/s")

        return metrics

    except Exception as e:
        print(f"  [BLAD] Lot {flight_no}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _plot_flight(flight_no, traj, fused, tel, t_ign, t_end, az, el, cant, head):
    """Generuje wykres rekonstrukcji dla lotu."""
    RESULTS_DIR.mkdir(exist_ok=True)

    mask = (tel.time >= t_ign) & (tel.time <= t_end)
    t_ob   = tel.time[mask]
    alt_ob = tel.alt_onboard[mask] - tel.alt_onboard[mask][0]
    vel_ob = tel.vel_onboard[mask]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Lot {flight_no} — cant={cant}° {head} "
                 f"(az={az}° el={el}°)",
                 fontsize=13, fontweight="bold")

    # 1. Wysokosc
    ax = axes[0, 0]
    ax.plot(traj.time, traj.alt, 'b-', lw=1, alpha=0.5, label="IMU")
    ax.plot(fused.time, fused.alt, 'g-', lw=1.5, label="IMU+GPS")
    ax.plot(t_ob, alt_ob, 'r--', lw=1.2, label="GPS")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Wysokosc [m]")
    ax.set_title("Wysokosc"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. Predkosc
    ax = axes[0, 1]
    ax.plot(traj.time, traj.speed, 'b-', lw=1, alpha=0.5, label="IMU")
    ax.plot(fused.time, fused.speed, 'g-', lw=1.5, label="IMU+GPS")
    ax.plot(t_ob, vel_ob, 'r--', lw=1.2, label="GPS")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("Predkosc [m/s]")
    ax.set_title("Predkosc"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Trajektoria pozioma
    ax = axes[0, 2]
    e_g, n_g = latlon_to_enu(tel.lat[mask], tel.lon[mask],
                              tel.lat[mask][0], tel.lon[mask][0])
    ax.plot(traj.pos[:, 0], traj.pos[:, 1], 'b-', lw=1, alpha=0.5, label="IMU")
    ax.plot(fused.pos[:, 0], fused.pos[:, 1], 'g-', lw=1.5, label="IMU+GPS")
    ax.plot(e_g, n_g, 'r--', lw=1, label="GPS")
    ax.plot(0, 0, 'ko', ms=8)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Trajektoria pozioma"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.axis('equal')

    # 4. Surowe akcelerometry
    ax = axes[1, 0]
    mr = (tel.time >= t_ign-1) & (tel.time <= t_end)
    ax.plot(tel.time[mr], tel.acc_x[mr], 'r-', lw=0.6, label="acc_X")
    ax.plot(tel.time[mr], tel.acc_y[mr], 'g-', lw=0.6, label="acc_Y")
    ax.plot(tel.time[mr], tel.acc_z[mr], 'b-', lw=0.6, label="acc_Z")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("[g]")
    ax.set_title("Akcelerometr (surowy)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. Surowe zyroskopy
    ax = axes[1, 1]
    ax.plot(tel.time[mr], tel.gyro_x[mr], 'r-', lw=0.6, label="gyro_X (roll)")
    ax.plot(tel.time[mr], tel.gyro_y[mr], 'g-', lw=0.6, label="gyro_Y (pitch)")
    ax.plot(tel.time[mr], tel.gyro_z[mr], 'b-', lw=0.6, label="gyro_Z (yaw)")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("[st/s]")
    ax.set_title("Zyroskop (po zamianie Y/Z)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 6. Euler
    ax = axes[1, 2]
    ax.plot(traj.time, traj.euler[:, 0], 'r-', lw=1, label="roll")
    ax.plot(traj.time, traj.euler[:, 1], 'g-', lw=1, label="pitch")
    ax.plot(traj.time, traj.euler[:, 2], 'b-', lw=1, label="yaw")
    ax.set_xlabel("Czas [s]"); ax.set_ylabel("[st]")
    ax.set_title("Orientacja Euler"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f"reconstruction_flight_{flight_no}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Loty do analizy
    flights = load_flights_to_analyze(str(resolve_data_file("configs.txt")))
    print(f"Loty do analizy (yes w configs.txt): {flights}\n")

    # Walidacja struktury
    print("="*60)
    print("KROK 1: Walidacja struktury plikow")
    print("="*60)
    ref_header = None
    for fno in flights:
        fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
        if ref_header is None and fname.exists():
            ref_header = read_header(fname)
        validate_file(fname, ref_header, verbose=True)

    # Przetwarzanie
    print("\n" + "="*60)
    print("KROK 2: Rekonstrukcja trajektorii")
    print("="*60)
    all_metrics = []
    for fno in flights:
        cfg_row = load_config_row(fno)
        m = process_flight(fno, cfg_row)
        if m:
            all_metrics.append(m)

    # Zapis zbiorczego CSV
    if all_metrics:
        csv_path = RESULTS_DIR / "validation_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n{'='*60}")
        print(f"Zapisano podsumowanie: {csv_path}")
        print(f"Przetworzono {len(all_metrics)}/{len(flights)} lotow")
        print(f"Wykresy w: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
