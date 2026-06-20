"""
estimate_thrust.py
===================
Estymacja krzywej ciagu T(t) z danych IMU + cisnienia w komorze spalania,
skalibrowana przez znana krzywa Cd(Mach) z fazy znizania (diag_drag/fit_cd_mach).

Idea:
  1. W krotkim oknie predkosci naddzwiekowej Ma=0.2-0.5 PODCZAS WZNOSZENIA
     (mid-burn, dobrze rozwiazane czasowo) znamy Cd(Ma) z dopasowania
     do fazy znizania -> D(t) = Cd(Ma)*q(t)*S jest znane.
  2. Akcelerometr mierzy SPECIFIC FORCE (sila wlasna, BEZ grawitacji):
         a_meas(t) = (T(t) - D(t)) / m(t)
     stad:
         T(t) = m(t) * a_meas(t) + D(t)
     w oknie kalibracyjnym (gdzie D(t) jest znane).
  3. Cisnienie w komorze Pc(t) jest miarka ciagu przez caly czas spalania
     (T = Cf * Pc * At, w przyblizeniu liniowo dla ustalonej geometrii
     dyszy). Dopasowujemy regresje liniowa T = k*Pc + c w oknie
     kalibracyjnym, nastepnie stosujemy te kalibracje na CALYM Pc(t)
     aby uzyskac ciag przez caly czas spalania (rowniez tam gdzie
     Cd jest nieznane/nieekstrapolowalne).
  4. Masa m(t) — model liniowy: m(t) = m_rocket - m_propellant*(t/t_burn),
     od pelnej masy startowej do masy po wypaleniu (m_coast).

Po przetworzeniu wszystkich lotow, krzywe T(t) sa przeskalowane do
wspolnej osi t/t_burn (znormalizowany czas spalania 0..1) i agregowane:
T_mean(frac), T_std(frac) — bo silniki maja naturalny rozrzut
miedzy egzemplarzami.

Wymaga: results/fit_cd_mach_curves.csv (wygenerowany przez run_all_flights.py)

Uzycie:
    python estimate_thrust.py                  # wszystkie loty z configs.txt
    python estimate_thrust.py 13 17 18          # wybrane loty
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_parser import parse_telemetry, get_data_dir
from imu_reconstruction import detect_ignition
from diag_drag import load_config, isa, detect_events, calib_acc_scale, S, G0
from run_all_flights import read_configs, check_data_exists


# --------------------------------------------------------------------------
def load_cd_curves(csv_path):
    """Wczytuje dopasowane krzywe Cd(Ma) per typ nosa z fit_cd_mach_curves.csv."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Brak {csv_path}. Najpierw uruchom: python run_all_flights.py")
    curves = {}  # nose -> (Ma_array, Cd_array)
    by_nose = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nose = row["nose"]
            by_nose.setdefault(nose, []).append(
                (float(row["Mach"]), float(row["Cd"])))
    for nose, pairs in by_nose.items():
        pairs.sort(key=lambda p: p[0])
        Ma = np.array([p[0] for p in pairs])
        Cd = np.array([p[1] for p in pairs])
        curves[nose] = (Ma, Cd)
    return curves


def cd_at(ma, nose, curves):
    Ma_nodes, Cd_nodes = curves[nose]
    return np.interp(ma, Ma_nodes, Cd_nodes, left=Cd_nodes[0], right=Cd_nodes[-1])


# --------------------------------------------------------------------------
def process_flight(fno, cfg, nose, curves, base):
    """
    Estymuje T(t) dla jednego lotu.
    Zwraca dict z polami: t, Pc, T_est, m, k, c, r2, n_calib
    lub None jesli nieudane.
    """
    fpath = Path(base) / f"ARTEMIDA_{fno}_LOT.txt"
    tel = parse_telemetry(fpath, verbose=False)

    t_ign_abs = detect_ignition(tel)
    i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign_abs)
    t = tel.time
    t_burn = t[i_bo] - t[i_ign]
    if t_burn < 0.3:
        print(f"[LOT {fno}] t_burn={t_burn:.2f}s — podejrzanie krotkie, pomijam")
        return None

    acc_scale, g_rest = calib_acc_scale(tel, t_ign_abs)

    win = np.arange(i_ign, i_bo + 1)
    tt  = t[win] - t[i_ign]                          # czas od zaplonu [s]
    ax_meas = tel.acc_x[win] * acc_scale * G0          # specific force [m/s^2]
    Pc      = tel.press_cham[win]                      # [bar]
    h0      = tel.alt_onboard[i_ign]
    elev    = np.radians(cfg["elevation"])

    # --- masa: model liniowy m_rocket -> m_coast w czasie t_burn ---
    m_rocket     = cfg["m_rocket"]
    m_propellant = cfg["m_propellant"]
    frac_burn    = np.clip(tt / t_burn, 0.0, 1.0)
    m_t = m_rocket - m_propellant * frac_burn

    # --- kinematyczna predkosc/wysokosc (do znalezienia Ma(t)) ---
    # dV/dt = a_meas - g*sin(elev)  (specific force NIE zawiera grawitacji)
    a_kin = ax_meas - G0 * np.sin(elev)
    V = np.concatenate([[0.0], np.cumsum(0.5 * (a_kin[1:] + a_kin[:-1]) * np.diff(tt))])
    Vz = V * np.sin(elev)
    h  = h0 + np.concatenate([[0.0], np.cumsum(0.5 * (Vz[1:] + Vz[:-1]) * np.diff(tt))])

    rho, a_snd, _, _ = isa(h)
    Ma = V / np.maximum(a_snd, 1.0)
    q  = 0.5 * rho * V**2

    # --- okno kalibracyjne: Ma w [0.2, 0.5] ---
    calib_mask = (Ma >= 0.2) & (Ma <= 0.5)
    n_calib = int(np.sum(calib_mask))
    if n_calib < 20:
        print(f"[LOT {fno}] tylko {n_calib} probek w oknie Ma 0.2-0.5 — pomijam")
        return None

    Cd_c = cd_at(Ma[calib_mask], nose, curves)
    D_c  = Cd_c * q[calib_mask] * S
    T_c  = m_t[calib_mask] * ax_meas[calib_mask] + D_c     # ciag w oknie kalibracyjnym [N]
    Pc_c = Pc[calib_mask]

    # --- regresja liniowa T = k*Pc + c w oknie kalibracyjnym ---
    A = np.vstack([Pc_c, np.ones_like(Pc_c)]).T
    (k, c), *_ = np.linalg.lstsq(A, T_c, rcond=None)
    T_fit_c = k * Pc_c + c
    ss_res = np.sum((T_c - T_fit_c)**2)
    ss_tot = np.sum((T_c - np.mean(T_c))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # --- zastosuj kalibracje na CALYM przebiegu spalania ---
    T_est = k * Pc + c
    T_est = np.maximum(T_est, 0.0)   # ciag nie moze byc ujemny

    print(f"[LOT {fno}] nos={nose}  t_burn={t_burn:.2f}s  "
          f"okno_kalib=[{tt[calib_mask][0]:.2f},{tt[calib_mask][-1]:.2f}]s "
          f"(n={n_calib})  k={k:.3f}N/bar  c={c:.2f}N  R2={r2:.3f}  "
          f"T_max={T_est.max():.1f}N")

    return dict(
        flight_no=fno, nose=nose, t=tt, Pc=Pc, T_est=T_est, m=m_t,
        t_burn=t_burn, k=k, c=c, r2=r2, n_calib=n_calib,
        calib_mask=calib_mask, T_calib=T_c, Pc_calib=Pc_c,
    )


# --------------------------------------------------------------------------
def save_per_flight_csv(result, out_dir):
    fno = result["flight_no"]
    csv_path = Path(out_dir) / f"thrust_flight_{fno}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "Pc_bar", "T_est_N", "m_kg"])
        for ti, pci, Ti, mi in zip(result["t"], result["Pc"], result["T_est"], result["m"]):
            writer.writerow([f"{ti:.4f}", f"{pci:.3f}", f"{Ti:.3f}", f"{mi:.4f}"])
    return csv_path


def save_calibration_csv(results, out_dir):
    csv_path = Path(out_dir) / "thrust_calibration.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flight_no", "nose", "t_burn_s", "k_N_per_bar",
                         "c_N", "R2", "n_calib_points"])
        for r in results:
            writer.writerow([r["flight_no"], r["nose"], f"{r['t_burn']:.3f}",
                             f"{r['k']:.4f}", f"{r['c']:.3f}", f"{r['r2']:.4f}",
                             r["n_calib"]])
    return csv_path


def burn_time_stats(results):
    """Srednia i odchylenie standardowe rzeczywistego czasu spalania [s]."""
    t_burns = np.array([r["t_burn"] for r in results])
    return dict(
        mean=float(np.mean(t_burns)), std=float(np.std(t_burns)),
        min=float(np.min(t_burns)), max=float(np.max(t_burns)),
        n=len(t_burns), per_flight={r["flight_no"]: r["t_burn"] for r in results},
    )


def save_burn_time_csv(bstats, out_dir):
    csv_path = Path(out_dir) / "burn_time_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flight_no", "t_burn_s"])
        for fno, tb in sorted(bstats["per_flight"].items()):
            writer.writerow([fno, f"{tb:.4f}"])
        writer.writerow([])
        writer.writerow(["mean_s", f"{bstats['mean']:.4f}"])
        writer.writerow(["std_s", f"{bstats['std']:.4f}"])
        writer.writerow(["min_s", f"{bstats['min']:.4f}"])
        writer.writerow(["max_s", f"{bstats['max']:.4f}"])
        writer.writerow(["n_flights", bstats["n"]])
    return csv_path


def save_aggregate_csv(results, out_dir, bstats, n_grid=101):
    """
    Resampluje T(t) kazdego lotu na wspolna os t/t_burn (0..1) i liczy
    srednia oraz odchylenie standardowe miedzy lotami w kazdym punkcie.

    Dodatkowo: t_abs_mean_s = frac * mean(t_burn) — przyblizona os
    czasu w sekundach (uzywajac SREDNIEGO czasu spalania), do orientacji
    rzeczywistej dlugosci spalania (nie tylko znormalizowanej).
    """
    frac_grid = np.linspace(0.0, 1.0, n_grid)
    T_matrix = []
    for r in results:
        frac = r["t"] / r["t_burn"]
        T_resampled = np.interp(frac_grid, frac, r["T_est"])
        T_matrix.append(T_resampled)
    T_matrix = np.array(T_matrix)   # (n_flights, n_grid)

    T_mean = T_matrix.mean(axis=0)
    T_std  = T_matrix.std(axis=0)
    t_abs_mean = frac_grid * bstats["mean"]

    csv_path = Path(out_dir) / "thrust_mean_std.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_frac_burn", "t_abs_mean_s", "T_mean_N", "T_std_N", "n_flights"])
        for frac, t_abs, mu, sd in zip(frac_grid, t_abs_mean, T_mean, T_std):
            writer.writerow([f"{frac:.4f}", f"{t_abs:.4f}", f"{mu:.3f}", f"{sd:.3f}", len(results)])

    return csv_path, frac_grid, T_mean, T_std, T_matrix


# --------------------------------------------------------------------------
def plot_results(results, frac_grid, T_mean, T_std, T_matrix, bstats, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax0 = axes[0]
    for r, T_row in zip(results, T_matrix):
        ax0.plot(r["t"], r["T_est"], lw=1, alpha=0.6, label=f"lot {r['flight_no']}")
        ax0.scatter(r["t"][r["calib_mask"]], r["T_calib"], s=8, alpha=0.5)
    ax0.set_xlabel("Czas od zaplonu [s]"); ax0.set_ylabel("Ciag T(t) [N]")
    ax0.set_title("Ciag per lot (punkty = wartosci z okna kalibracyjnego IMU+Cd)")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

    ax1 = axes[1]
    t_abs_mean = frac_grid * bstats["mean"]
    ax1.plot(frac_grid, T_mean, 'b-', lw=2, label="T_mean")
    ax1.fill_between(frac_grid, T_mean - T_std, T_mean + T_std,
                     alpha=0.25, color='b', label="±1 std (rozrzut miedzy lotami)")
    ax1.set_xlabel("Znormalizowany czas spalania t/t_burn"); ax1.set_ylabel("Ciag T [N]")
    ax1.set_title(f"Zagregowana krzywa ciagu ({len(results)} lotow)  "
                  f"t_burn={bstats['mean']:.2f}±{bstats['std']:.2f}s")

    # druga os X: przyblizone sekundy (skala przez sredni czas spalania)
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(frac_grid[::20])
    ax1_top.set_xticklabels([f"{v:.2f}" for v in t_abs_mean[::20]])
    ax1_top.set_xlabel("Przyblizony czas [s] (wg sredniego t_burn)")

    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    plt.tight_layout()
    out_png = Path(out_dir) / "thrust_estimate.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Estymacja ciagu z IMU + Pc, kalibrowana przez Cd")
    parser.add_argument("flights", nargs="*", type=int,
                        help="Numery lotow (domyslnie: wszystkie z configs.txt)")
    args = parser.parse_args()

    base    = get_data_dir()
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_cd_curves(out_dir / "fit_cd_mach_curves.csv")
    print(f"Wczytano krzywe Cd(Ma) dla: {list(curves.keys())}")

    rows = read_configs(base)
    if args.flights:
        rows = [r for r in rows if r["fno"] in args.flights]
    rows = [r for r in rows if check_data_exists(base, r["fno"])]

    results = []
    for r in rows:
        nose = r["nose"]
        if nose not in curves:
            print(f"[LOT {r['fno']}] brak krzywej Cd dla nos={nose} — pomijam")
            continue
        res = process_flight(r["fno"], r, nose, curves, base)
        if res is not None:
            results.append(res)
            save_per_flight_csv(res, out_dir)

    if not results:
        print("Brak wynikow do agregacji.")
        return

    save_calibration_csv(results, out_dir)

    bstats = burn_time_stats(results)
    save_burn_time_csv(bstats, out_dir)

    csv_path, frac_grid, T_mean, T_std, T_matrix = save_aggregate_csv(results, out_dir, bstats)
    print(f"Zapisano: {csv_path}")

    plot_results(results, frac_grid, T_mean, T_std, T_matrix, bstats, out_dir)

    print(f"\nPodsumowanie ({len(results)} lotow):")
    print(f"  Czas spalania t_burn = {bstats['mean']:.3f} ± {bstats['std']:.3f} s  "
          f"(min={bstats['min']:.3f}s  max={bstats['max']:.3f}s)")
    for fno, tb in sorted(bstats["per_flight"].items()):
        print(f"    lot {fno}: t_burn={tb:.3f}s")
    print(f"  T_mean peak = {T_mean.max():.1f} N przy t/t_burn={frac_grid[np.argmax(T_mean)]:.2f}  "
          f"(~{frac_grid[np.argmax(T_mean)]*bstats['mean']:.2f}s)")
    print(f"  T_std peak  = {T_std[np.argmax(T_mean)]:.1f} N "
          f"({100*T_std[np.argmax(T_mean)]/T_mean.max():.1f}% rozrzutu)")


if __name__ == "__main__":
    main()
