"""
monte_carlo_thrust.py
======================
Monte Carlo: jak duzy rozrzut apogeum wynika z naturalnej zmiennosci
silnika (amplituda ciagu + czas spalania) miedzy egzemplarzami, w
porownaniu do rozrzutu apogeum widzianego w testach polowych.

Nie modyfikuje MAIN.py ani configurations/*.yaml — jest to niezalezny,
samodzielny model 2D wznoszenia (taki jak validate_trajectory.py), ktory
korzysta wylacznie z danych z field_test_data/results/:
  - thrust_mean_std.csv   (T_mean(t/t_burn), T_std(t/t_burn))
  - burn_time_stats.csv   (mean/std t_burn)
  - aero_table_missile.pkl (DATCOM CA(alpha=0), zwalidowany Cd(Ma))
  - configs.txt           (masa/elewacja per lot, do wartosci nominalnych)

Model zmiennosci silnika:
  - jeden losowany wspolczynnik skali amplitudy ciagu na przebieg
    (silniki sa "slabsze"/"silniejsze" jako cala partia, nie szum
    punkt-po-punkcie) — rozrzut wzgledny estymowany z mean(T_std/T_mean),
  - niezaleznie losowany czas spalania t_burn ~ N(mean, std) z
    burn_time_stats.csv (przyciety do [min, max] zaobserwowanego).

Uzycie:
    python monte_carlo_thrust.py                 # 500 przebiegow, nos 'ostra'
    python monte_carlo_thrust.py --n 1000 --nose tepa
"""

import sys
import csv
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir
from diag_drag import load_config, isa, G0, D_CAL
from run_all_flights import read_configs, check_data_exists

S = np.pi * D_CAL**2 / 4.0


# --------------------------------------------------------------------------
def load_thrust_mean_std(csv_path):
    frac, t_abs, T_mean, T_std = [], [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frac.append(float(row["t_frac_burn"]))
            t_abs.append(float(row["t_abs_mean_s"]))
            T_mean.append(float(row["T_mean_N"]))
            T_std.append(float(row["T_std_N"]))
    return np.array(frac), np.array(T_mean), np.array(T_std)


def load_burn_time_stats(csv_path):
    """Plik zawiera tabele per-lot, puste pole, a potem 'etykieta,wartosc'
    podsumowanie (mean_s/std_s/min_s/max_s/n_flights) — bierzemy tylko to."""
    stats = {}
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] in ("mean_s", "std_s", "min_s", "max_s", "n_flights"):
                stats[parts[0]] = parts[1]
    return dict(mean_s=float(stats["mean_s"]), std_s=float(stats["std_s"]),
                min_s=float(stats["min_s"]), max_s=float(stats["max_s"]))


def load_datcom_ca0(pkl_path):
    with open(pkl_path, "rb") as f:
        a = pickle.load(f)
    alpha = np.array(a.alpha_table)
    mach = np.array(a.mach_table)
    CA = np.array(a.CA_table)
    ia0 = int(np.argmin(np.abs(alpha)))
    return mach, CA[ia0]


def nominal_config(base, nose):
    """Srednie m_rocket/m_propellant/m_coast/elevation z lotow danego nosa
    (configs.txt) — wartosci nominalne, NIE z configurations/*.yaml."""
    rows = read_configs(base)
    sel = [r for r in rows if r["nose"] == nose and check_data_exists(base, r["fno"])]
    if not sel:
        raise RuntimeError(f"Brak lotow dla nosa '{nose}' w configs.txt.")
    cfgs = []
    for r in sel:
        cfg = load_config(r["fno"], base)
        if cfg is not None:
            cfgs.append(cfg)
    m_rocket = np.mean([c["m_rocket"] for c in cfgs])
    m_propellant = np.mean([c["m_propellant"] for c in cfgs])
    m_coast = np.mean([c["m_coast"] for c in cfgs])
    elevation = np.mean([c["elevation"] for c in cfgs])
    flights = [r["fno"] for r in sel]
    return dict(m_rocket=m_rocket, m_propellant=m_propellant,
                m_coast=m_coast, elevation=elevation), flights


def actual_apogees(base, flights):
    """Apogea GPS dla podanych lotow, czytane z trajectory_closure_summary.csv
    (jesli istnieje) — niezalezne dane referencyjne."""
    csv_path = Path(base) / "results" / "trajectory_closure_summary.csv"
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fno = int(row["flight_no"])
            if fno in flights:
                out[fno] = float(row["h_apo_actual_m"])
    return out


def actual_v_max(base, flights):
    """V_max (hybryda akcel./GPS+baro, validate_trajectory.py) dla podanych
    lotow, czytane z trajectory_closure_summary.csv — niezalezne dane
    referencyjne."""
    csv_path = Path(base) / "results" / "trajectory_closure_summary.csv"
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fno = int(row["flight_no"])
            if fno in flights and row.get("V_max_actual_mps"):
                out[fno] = float(row["V_max_actual_mps"])
    return out


# --------------------------------------------------------------------------
def simulate_one(cfg, mach_ca, ca0, thrust_frac, thrust_mean, scale, t_burn,
                  ratio_powered=None, post_burn_s=0.0, h0=100.0, t_max_pad=10.0):
    """Calkuje wznoszenie 2D do apogeum (Vz=0) dla jednej wylosowanej
    realizacji silnika. Ciag wzdluz elewacji startu (rakieta stabilizowana
    spinem, krotki czas spalania), opor wzdluz wektora predkosci.

    ratio_powered : tablica (na siatce mach_ca) lub None
        Wzgledny spadek Cd przy "zasilanej" denku (plomien zaslania base
        drag), z base_drag_phase_factor.py. Stosowany dla tt < t_burn +
        post_burn_s (spalanie + okno resztkowego cisnienia w komorze po
        wypaleniu); poza tym oknem uzywany czysty Cd z DATCOM."""
    elev = np.radians(cfg["elevation"])
    m_rocket, m_propellant, m_coast = cfg["m_rocket"], cfg["m_propellant"], cfg["m_coast"]
    t_powered_end = t_burn + post_burn_s

    def thrust_at(tt):
        if tt > t_burn or tt < 0:
            return 0.0
        return max(scale * np.interp(tt / t_burn, thrust_frac, thrust_mean), 0.0)

    def mass_at(tt):
        if tt >= t_burn:
            return m_coast
        return m_rocket - m_propellant * (tt / t_burn)

    def cd_at(ma, tt):
        cd0 = np.interp(ma, mach_ca, ca0, left=ca0[0], right=ca0[-1])
        if ratio_powered is not None and tt < t_powered_end:
            r = np.interp(ma, mach_ca, ratio_powered, left=ratio_powered[0], right=ratio_powered[-1])
            return cd0 * r
        return cd0

    def rhs(tt, state):
        h, Vh, Vz = state
        h = max(h, 0.0)
        rho, a_snd, _, _ = isa(h)
        V_total = np.sqrt(Vh**2 + Vz**2)
        Ma = V_total / max(a_snd, 1.0)
        Cd = cd_at(Ma, tt)
        q = 0.5 * rho * V_total**2
        D = Cd * q * S

        T = thrust_at(tt)
        m = mass_at(tt)

        dirT_h, dirT_z = np.cos(elev), np.sin(elev)
        if V_total > 1.0:
            dirD_h, dirD_z = Vh / V_total, Vz / V_total
        else:
            dirD_h, dirD_z = dirT_h, dirT_z

        ax = (T * dirT_h - D * dirD_h) / m
        az = (T * dirT_z - D * dirD_z) / m - G0
        return [Vz, ax, az]

    def apogee_event(tt, state):
        return state[2]
    apogee_event.terminal = True
    apogee_event.direction = -1

    t_span = (0.0, t_burn + 30.0 + t_max_pad)
    sol = solve_ivp(rhs, t_span, [h0, 0.0, 0.0], events=apogee_event,
                     max_step=0.02, rtol=1e-7, atol=1e-6)

    V_total = np.sqrt(sol.y[1]**2 + sol.y[2]**2)
    V_max = float(np.max(V_total))

    if len(sol.t_events[0]) == 0:
        return float(sol.t[-1]), float(sol.y[0, -1]), V_max
    return float(sol.t_events[0][0]), float(sol.y_events[0][0][0]), V_max


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo apogee spread from motor (thrust/burn-time) variability")
    parser.add_argument("--n", type=int, default=500, help="liczba przebiegow Monte Carlo")
    parser.add_argument("--nose", choices=["ostra", "tepa"], default="ostra")
    parser.add_argument("--case", default="rocket_70mm_baseline",
                        help="katalog DATCOM (datcom_runs/<case>/aero_table_missile.pkl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_drag_fix", action="store_true",
                        help="zastosuj wzgledny spadek Cd (Fleeman, base_drag_phase_factor.py) "
                             "podczas spalania + okna resztkowego po wypaleniu")
    parser.add_argument("--post_burn_s", type=float, default=1.5,
                        help="okno [s] po wypaleniu, w ktorym wciaz zaklada sie zaslonione "
                             "denko (resztkowe cisnienie w komorze) — uzywane tylko z --base_drag_fix")
    args = parser.parse_args()

    base = get_data_dir()
    out_dir = Path(base) / "results"
    root = Path(base).parent
    pkl_path = root / "datcom_runs" / args.case / "aero_table_missile.pkl"
    yaml_path = root / "configurations" / f"{args.case}.yaml"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Brak {pkl_path}. Uruchom MAIN.py najpierw.")

    thrust_frac, thrust_mean, thrust_std = load_thrust_mean_std(out_dir / "thrust_mean_std.csv")
    burn = load_burn_time_stats(out_dir / "burn_time_stats.csv")
    mach_ca, ca0 = load_datcom_ca0(pkl_path)

    ratio_powered = None
    if args.base_drag_fix:
        from base_drag_phase_factor import base_drag_ratio
        ratio_powered = base_drag_ratio(yaml_path, mach_ca)
        print(f"Korekta base drag (Fleeman, plomien zaslania denko): "
              f"ratio_Cd w [{ratio_powered.min():.3f}, {ratio_powered.max():.3f}], "
              f"okno po wypaleniu={args.post_burn_s:.2f}s")

    cfg, flights = nominal_config(base, args.nose)
    actual = actual_apogees(base, flights)
    actual_v = actual_v_max(base, flights)

    # rozrzut wzgledny amplitudy ciagu (pomijajac pierwsze/ostatnie probki, gdzie
    # T_mean->0 i T_std/T_mean rozdmuchuje sie sztucznie)
    mid = (thrust_frac > 0.05) & (thrust_frac < 0.95)
    rel_std = float(np.mean(thrust_std[mid] / thrust_mean[mid]))

    print(f"Konfiguracja nominalna (nos '{args.nose}', loty {flights}):")
    print(f"  m_rocket={cfg['m_rocket']:.3f} kg  m_propellant={cfg['m_propellant']:.3f} kg  "
          f"m_coast={cfg['m_coast']:.3f} kg  elevation={cfg['elevation']:.1f} deg")
    print(f"  t_burn: mean={burn['mean_s']:.3f}s std={burn['std_s']:.3f}s "
          f"[{burn['min_s']:.3f}, {burn['max_s']:.3f}]")
    print(f"  rozrzut wzgledny amplitudy ciagu (mean T_std/T_mean, 5-95% spalania): {rel_std:.3f}")

    rng = np.random.default_rng(args.seed)
    h_apo = np.empty(args.n)
    t_apo = np.empty(args.n)
    v_max = np.empty(args.n)
    for i in range(args.n):
        scale = max(rng.normal(1.0, rel_std), 0.3)
        t_burn = float(np.clip(rng.normal(burn["mean_s"], burn["std_s"]),
                                burn["min_s"], burn["max_s"]))
        t_apo[i], h_apo[i], v_max[i] = simulate_one(cfg, mach_ca, ca0, thrust_frac,
                                                      thrust_mean, scale, t_burn,
                                                      ratio_powered=ratio_powered,
                                                      post_burn_s=args.post_burn_s)

    print(f"\nMonte Carlo ({args.n} przebiegow), apogeum:")
    print(f"  predykcja: mean={np.mean(h_apo):.1f} m  std={np.std(h_apo):.1f} m  "
          f"[{np.min(h_apo):.1f}, {np.max(h_apo):.1f}]")
    if actual:
        act_vals = np.array(list(actual.values()))
        print(f"  rzeczywiste apogea GPS (loty {list(actual.keys())}): "
              f"mean={np.mean(act_vals):.1f} m  std={np.std(act_vals):.1f} m  "
              f"[{np.min(act_vals):.1f}, {np.max(act_vals):.1f}]")
        bias_pct = 100.0 * (np.mean(h_apo) - np.mean(act_vals)) / np.mean(act_vals)
        print(f"  systematyczne odchylenie modelu (mean_pred vs mean_actual): {bias_pct:+.1f}%")

    print(f"\nMonte Carlo ({args.n} przebiegow), V_max:")
    print(f"  predykcja: mean={np.mean(v_max):.1f} m/s  std={np.std(v_max):.1f} m/s  "
          f"[{np.min(v_max):.1f}, {np.max(v_max):.1f}]")
    if actual_v:
        actv_vals = np.array(list(actual_v.values()))
        print(f"  rzeczywiste V_max (akcel.+GPS/baro, loty {list(actual_v.keys())}): "
              f"mean={np.mean(actv_vals):.1f} m/s  std={np.std(actv_vals):.1f} m/s  "
              f"[{np.min(actv_vals):.1f}, {np.max(actv_vals):.1f}]")
        bias_v_pct = 100.0 * (np.mean(v_max) - np.mean(actv_vals)) / np.mean(actv_vals)
        print(f"  systematyczne odchylenie modelu (mean_pred vs mean_actual): {bias_v_pct:+.1f}%")

    csv_path = out_dir / f"monte_carlo_apogee_{args.nose}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "h_apo_pred_m", "t_apo_pred_s", "v_max_pred_mps"])
        for i in range(args.n):
            writer.writerow([i, round(h_apo[i], 2), round(t_apo[i], 3), round(v_max[i], 2)])
    print(f"\nZapisano: {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(h_apo, bins=30, color='tab:blue', alpha=0.7,
             label=f"Monte Carlo predykcja (n={args.n}, rozrzut silnika)")
    ax1.axvline(np.mean(h_apo), color='tab:blue', ls='--', lw=1.5,
                label=f"pred. mean={np.mean(h_apo):.0f} m")
    for fno, h in actual.items():
        ax1.axvline(h, color='tab:red', lw=1.5, alpha=0.8)
    if actual:
        ax1.axvline(list(actual.values())[0], color='tab:red', lw=1.5, alpha=0.8,
                    label="apogea GPS rzeczywiste (loty)")
    ax1.set_xlabel("Apogeum [m]")
    ax1.set_ylabel("Liczba przebiegow")
    ax1.set_title(f"Apogeum — nos '{args.nose}'")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(v_max, bins=30, color='tab:green', alpha=0.7,
             label=f"Monte Carlo predykcja (n={args.n}, rozrzut silnika)")
    ax2.axvline(np.mean(v_max), color='tab:green', ls='--', lw=1.5,
                label=f"pred. mean={np.mean(v_max):.0f} m/s")
    for fno, v in actual_v.items():
        ax2.axvline(v, color='tab:red', lw=1.5, alpha=0.8)
    if actual_v:
        ax2.axvline(list(actual_v.values())[0], color='tab:red', lw=1.5, alpha=0.8,
                    label="V_max rzeczywiste (akcel.+GPS/baro, loty)")
    ax2.set_xlabel("V_max [m/s]")
    ax2.set_ylabel("Liczba przebiegow")
    ax2.set_title(f"Predkosc maksymalna — nos '{args.nose}'")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Monte Carlo apogeum i V_max vs rozrzut polowy — nos '{args.nose}'")
    plt.tight_layout()
    out_png = out_dir / f"monte_carlo_apogee_{args.nose}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
