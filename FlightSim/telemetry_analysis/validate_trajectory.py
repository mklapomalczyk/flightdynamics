"""
validate_trajectory.py
=======================
Niezalezna walidacja calego modelu (Cd + ciag + masa) przez porownanie
przewidywanego apogeum z rzeczywistym apogeum GPS.

Idea:
  Zadny pojedynczy element kalibracji (Cd z fazy znizania, ciag z okna
  Ma=0.2-0.5 podczas wznoszenia) nie jest weryfikowany na danych, ktorych
  nie uzyto do jego wyznaczenia. Ten skrypt zamyka petle:
    1. Calkuje pelne rownania ruchu 2D (plaszczyzna pionowa) od zaplonu
       do apogeum, uzywajac:
         - rzeczywistego przebiegu cisnienia komory Pc(t) z telemetrii,
         - kalibracji ciagu T=k*Pc+c z thrust_calibration.csv,
         - dopasowanej krzywej Cd(Ma) z fit_cd_mach_curves.csv,
         - modelu masy liniowego m_rocket -> m_coast w czasie t_burn.
    2. Porownuje przewidywana wysokosc i czas apogeum z faktycznym
       apogeum GPS (niezalezne dane, nieuzyte w zadnej kalibracji).
  Rozbieznosc = miara calkowitej jakosci modelu (systematyczne bledy w Cd,
  ciagu lub masie sie kumuluja na drodze ~15-20s do apogeum).

Uzycie:
    python validate_trajectory.py                # wszystkie loty z kalibracji
    python validate_trajectory.py 13 14 17 18     # wybrane loty
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_parser import parse_telemetry, get_data_dir
from imu_reconstruction import detect_ignition
from diag_drag import load_config, isa, detect_events, baro_altitude, S, G0
from estimate_thrust import load_cd_curves, cd_at
from run_all_flights import check_data_exists


# --------------------------------------------------------------------------
def load_thrust_calibration(csv_path):
    """Wczytuje k,c,t_burn per lot z thrust_calibration.csv."""
    cal = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cal[int(row["flight_no"])] = dict(
                nose=row["nose"], t_burn=float(row["t_burn_s"]),
                k=float(row["k_N_per_bar"]), c=float(row["c_N"]),
            )
    return cal


# --------------------------------------------------------------------------
def simulate_ascent(tel, cfg, cal, curves, t_max_pad=10.0):
    """
    Calkuje rownania ruchu 2D od zaplonu do apogeum (Vz=0, terminal event).
    Zwraca dict z czasem/wysokoscia przewidywanego apogeum + przebiegiem.
    """
    nose = cal["nose"]
    k, c, t_burn = cal["k"], cal["c"], cal["t_burn"]
    m_rocket, m_propellant, m_coast = cfg["m_rocket"], cfg["m_propellant"], cfg["m_coast"]
    elev = np.radians(cfg["elevation"])

    t_ign_abs = detect_ignition(tel)
    i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign_abs)
    t = tel.time

    h_baro = baro_altitude(tel, i_ign)
    h0 = float(h_baro[i_ign])

    # Pc(t) jako funkcja czasu od zaplonu, interpolowalna na cala trajektorie
    t_rel_full = t - t[i_ign]
    Pc_full = tel.press_cham

    def Pc_at(tt):
        return np.interp(tt, t_rel_full, Pc_full, left=Pc_full[i_ign], right=0.0)

    def thrust_at(tt):
        if tt > t_burn:
            return 0.0
        return max(k * Pc_at(tt) + c, 0.0)

    def mass_at(tt):
        if tt >= t_burn:
            return m_coast
        return m_rocket - m_propellant * (tt / t_burn)

    def rhs(tt, state):
        h, Vh, Vz = state
        h = max(h, 0.0)
        rho, a_snd, _, _ = isa(h)
        V_total = np.sqrt(Vh**2 + Vz**2)
        Ma = V_total / max(a_snd, 1.0)
        Cd = cd_at(Ma, nose, curves)
        q = 0.5 * rho * V_total**2
        D = Cd * q * S

        T = thrust_at(tt)
        m = mass_at(tt)

        # Ciag dziala wzdluz osi rakiety (kat elewacji ze startu) -- w krotkim
        # czasie spalania (~1-2s) rakieta stabilizowana spinem nie "skreca"
        # w strone wektora predkosci. Opor zawsze przeciwny do V_total.
        dirT_h, dirT_z = np.cos(elev), np.sin(elev)
        if V_total > 1.0:
            dirD_h, dirD_z = Vh / V_total, Vz / V_total
        else:
            dirD_h, dirD_z = dirT_h, dirT_z

        ax = (T * dirT_h - D * dirD_h) / m
        az = (T * dirT_z - D * dirD_z) / m - G0
        return [Vz, ax, az]

    def apogee_event(tt, state):
        return state[2]   # Vz = 0
    apogee_event.terminal = True
    apogee_event.direction = -1

    t_apo_actual_guess = t[i_apo] - t[i_ign]
    t_span = (0.0, t_apo_actual_guess + t_max_pad)

    sol = solve_ivp(rhs, t_span, [h0, 0.0, 0.0], events=apogee_event,
                     max_step=0.02, rtol=1e-7, atol=1e-6, dense_output=True)

    if len(sol.t_events[0]) == 0:
        t_apo_pred = sol.t[-1]
        h_apo_pred = sol.y[0, -1]
    else:
        t_apo_pred = float(sol.t_events[0][0])
        h_apo_pred = float(sol.y_events[0][0][0])

    return dict(
        t_rel=sol.t, h=sol.y[0], Vh=sol.y[1], Vz=sol.y[2],
        t_apo_pred=t_apo_pred, h_apo_pred=h_apo_pred,
        t_apo_actual=float(t[i_apo] - t[i_ign]),
        h_apo_actual=float(h_baro[i_apo]),
        h0=h0,
    )


# --------------------------------------------------------------------------
def process_flight(fno, cfg, cal, curves, base):
    fpath = Path(base) / f"ARTEMIDA_{fno}_LOT.txt"
    tel = parse_telemetry(fpath, verbose=False)
    res = simulate_ascent(tel, cfg, cal, curves)

    dh = res["h_apo_pred"] - res["h_apo_actual"]
    dt = res["t_apo_pred"] - res["t_apo_actual"]
    pct = 100.0 * dh / res["h_apo_actual"]

    print(f"[LOT {fno}] apogeum: pred={res['h_apo_pred']:.1f}m  "
          f"actual={res['h_apo_actual']:.1f}m  dh={dh:+.1f}m ({pct:+.1f}%)  "
          f"t_pred={res['t_apo_pred']:.2f}s  t_actual={res['t_apo_actual']:.2f}s  "
          f"dt={dt:+.2f}s")

    return dict(flight_no=fno, nose=cal["nose"],
                h_apo_pred=res["h_apo_pred"], h_apo_actual=res["h_apo_actual"],
                dh_m=dh, dh_pct=pct,
                t_apo_pred=res["t_apo_pred"], t_apo_actual=res["t_apo_actual"],
                dt_s=dt, traj=res)


# --------------------------------------------------------------------------
def save_csv(results, out_dir):
    csv_path = Path(out_dir) / "trajectory_closure_summary.csv"
    fieldnames = ["flight_no", "nose", "h_apo_pred_m", "h_apo_actual_m",
                  "dh_m", "dh_pct", "t_apo_pred_s", "t_apo_actual_s", "dt_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda r: r["flight_no"]):
            writer.writerow(dict(
                flight_no=r["flight_no"], nose=r["nose"],
                h_apo_pred_m=round(r["h_apo_pred"], 1),
                h_apo_actual_m=round(r["h_apo_actual"], 1),
                dh_m=round(r["dh_m"], 1), dh_pct=round(r["dh_pct"], 2),
                t_apo_pred_s=round(r["t_apo_pred"], 3),
                t_apo_actual_s=round(r["t_apo_actual"], 3),
                dt_s=round(r["dt_s"], 3),
            ))
    print(f"Zapisano: {csv_path}")
    return csv_path


def plot_results(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax0 = axes[0]
    for r in results:
        ax0.plot(r["traj"]["t_rel"], r["traj"]["h"], lw=1.3,
                  label=f"lot {r['flight_no']} (model)")
        ax0.scatter([r["t_apo_actual"]], [r["h_apo_actual"]], marker='x', s=60,
                     c='k')
    ax0.set_xlabel("Czas od zaplonu [s]"); ax0.set_ylabel("h [m]")
    ax0.set_title("Trajektoria modelowa (krzyzyk = apogeum GPS rzeczywiste)")
    ax0.legend(fontsize=7); ax0.grid(alpha=0.3)

    ax1 = axes[1]
    fnos = [r["flight_no"] for r in results]
    dh_pct = [r["dh_pct"] for r in results]
    ax1.bar(range(len(fnos)), dh_pct, color='tab:blue', alpha=0.7)
    ax1.set_xticks(range(len(fnos)))
    ax1.set_xticklabels([str(f) for f in fnos])
    ax1.axhline(0, color='k', lw=0.8)
    ax1.set_xlabel("Lot"); ax1.set_ylabel("Blad wysokosci apogeum [%]")
    ax1.set_title("Blad zamkniecia trajektorii (model vs GPS)")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    out_png = Path(out_dir) / "trajectory_closure.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end apogee closure validation")
    parser.add_argument("flights", nargs="*", type=int,
                        help="numery lotow (domyslnie wszystkie z thrust_calibration.csv)")
    args = parser.parse_args()

    base = get_data_dir()
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_path = out_dir / "thrust_calibration.csv"
    cd_path = out_dir / "fit_cd_mach_curves.csv"
    if not cal_path.exists() or not cd_path.exists():
        raise FileNotFoundError(
            "Brak thrust_calibration.csv lub fit_cd_mach_curves.csv. "
            "Najpierw uruchom run_all_flights.py i estimate_thrust.py")

    cal = load_thrust_calibration(cal_path)
    curves = load_cd_curves(cd_path)

    flights = args.flights if args.flights else sorted(cal.keys())

    results = []
    for fno in flights:
        if fno not in cal:
            print(f"[LOT {fno}] brak kalibracji ciagu — pomijam")
            continue
        if not check_data_exists(base, fno):
            print(f"[LOT {fno}] brak danych — pomijam")
            continue
        cfg = load_config(fno, base)
        if cfg is None:
            continue
        try:
            res = process_flight(fno, cfg, cal[fno], curves, base)
            results.append(res)
        except Exception as e:
            print(f"[LOT {fno}] BLAD: {e}")

    if not results:
        print("Brak wynikow.")
        return

    dh_pcts = np.array([r["dh_pct"] for r in results])
    print(f"\nBlad wysokosci apogeum: mean={np.mean(dh_pcts):+.2f}%  "
          f"std={np.std(dh_pcts):.2f}%  |max|={np.max(np.abs(dh_pcts)):.2f}%")

    save_csv(results, out_dir)
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
