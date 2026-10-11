"""
fit_wind_altitude_profile.py
==============================
Testuje hipoteze, ze rzeczywisty wiatr na wysokosci lotu jest SILNIEJSZY
niz zaklada domyslny profil potegowy (alpha_exp=0.16, typowy dla otwartego
terenu przy powierzchni) uzywany w build_wind_steady(). Trzeci (po rollu
i sztywnosci pitch-yaw) kandydat na wyjasnienie niedomodelowanego
crossrange -- wciaz na poziomie TRAJEKTORII (posrednie, nie zwalidowane
niezaleznym czujnikiem).

V(h) = V_ref(10m) * (h/10)^alpha_exp -- wiekszy alpha_exp = szybszy
wzrost predkosci wiatru z wysokoscia. Dla lotu 19 (apogeum ~1500m):
  alpha_exp=0.16 (domyslny, otwarty teren)      -> V(1500m) =~ 35 m/s
  alpha_exp=0.30 (umiarkowana zabudowa/szorstkosc) -> V(1500m) =~ 70 m/s
  alpha_exp=0.40 (agresywne, na granicy realizmu)  -> V(1500m) =~ 107 m/s

Model uzywa juz najlepszych dotad znalezionych korekt: roll (CLL x15.5,
Clp x3.0, zwalidowane niezaleznym czujnikiem AX) i pitch/yaw sztywnosc/
tlumienie (Cm x3.0, Cmq x0.3, dopasowanie POSREDNIE, bez niezaleznego
czujnika -- patrz fit_pitch_yaw_stiffness.py).

Nie modyfikuje configurations/*.yaml ani MAIN.py.

Uzycie:
    python fit_wind_altitude_profile.py --no-rerun-datcom 19
"""

import sys
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir
from run_6dof_cant_montecarlo import get_aero_for_cant
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_geometry
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from core.state6 import State6DOF

from analyze_per_flight_6dof import read_flights, build_flight_thrust, build_scaled_mass
from plot_flight_trajectory_6dof import model_time_series, actual_time_series, build_wind_steady

ROLL_CLL_SCALE = 15.5
ROLL_CLP_SCALE = 3.0
PITCH_YAW_CM_SCALE  = 3.0
PITCH_YAW_CMQ_SCALE = 0.3

ALPHA_EXPS = [0.16, 0.20, 0.25, 0.30, 0.40, 0.50]


def rmse(t_ref, y_ref, t_m, y_m, t_max):
    mask = t_ref <= min(t_max, t_ref[-1])
    if mask.sum() < 5:
        return float("nan")
    y_interp = np.interp(t_ref[mask], t_m, y_m)
    return float(np.sqrt(np.mean((y_interp - y_ref[mask]) ** 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Testuje silniejszy profil wiatru z wysokoscia (alpha_exp)")
    parser.add_argument("fno", type=int)
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--t-fit-max", type=float, default=36.0)
    args = parser.parse_args()
    fno = args.fno

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    flights_by_fno = {r["fno"]: r for r in read_flights(base)}
    if fno not in flights_by_fno:
        print(f"Lot {fno}: brak w configs.txt.")
        return
    r = flights_by_fno[fno]
    case = args.case_ostra if r["nose"] == "ostra" else args.case_tepa
    cfg_base = load_config(str(root / "configurations" / f"{case}.yaml"))
    t_ignition = cfg_base.propulsion.t_ignition
    t_burn = cfg_base.propulsion.t_burn

    aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
    if aero.CLL_table is not None:
        aero.CLL_table = aero.CLL_table * ROLL_CLL_SCALE
    if aero.Clp_table is not None:
        aero.Clp_table = aero.Clp_table * ROLL_CLP_SCALE
    if aero.Cm_table is not None:
        aero.Cm_table = aero.Cm_table * PITCH_YAW_CM_SCALE
    if aero.Cmq_table is not None:
        aero.Cmq_table = aero.Cmq_table * PITCH_YAW_CMQ_SCALE
    print(f"Lot {fno}: roll (CLL x{ROLL_CLL_SCALE}, Clp x{ROLL_CLP_SCALE}) i pitch/yaw "
          f"(Cm x{PITCH_YAW_CM_SCALE}, Cmq x{PITCH_YAW_CMQ_SCALE}) juz skorygowane")

    geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
    geom.cant_angle_rad = math.radians(r["cant"])
    mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
    atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)
    prop_flight = build_flight_thrust(base, fno, t_ignition)
    if prop_flight is None:
        print(f"Lot {fno}: brak profilu ciagu.")
        return

    elev_deg = r["elevation"]
    azimuth_deg = r["azimuth"]
    actual = actual_time_series(base, fno, azimuth_deg, t_burn, elev_deg)
    initial_state = State6DOF.initial(elevation_deg=elev_deg, azimuth_deg=azimuth_deg)

    print(f"Lot {fno}: {len(ALPHA_EXPS)} wartosci alpha_exp...")
    results = []
    model_runs = {}
    for a_exp in ALPHA_EXPS:
        wind = build_wind_steady(base, fno, azimuth_deg, alpha_exp=a_exp)
        if wind is None:
            print(f"Lot {fno}: brak zmierzonego wiatru.")
            return
        model = model_time_series(aero, geom, atm, gravity, launcher, mass, prop_flight,
                                   initial_state, wind_model=wind)
        rc = rmse(actual["t"], actual["crossrange"], model["t"], model["crossrange"], args.t_fit_max)
        rd = rmse(actual["t"], actual["downrange"], model["t"], model["downrange"], args.t_fit_max)
        h_apo = float(np.max(model["h"]))
        v_1500 = wind.speed_at(1500.0)
        print(f"  alpha_exp={a_exp:.2f}  V(1500m)={v_1500:5.1f}m/s  status={model['status']:>8}  "
              f"RMSE_crossrange={rc:7.0f}  RMSE_downrange={rd:7.0f}  h_apo={h_apo:6.0f}m  "
              f"crossrange_end={model['crossrange'][-1]:7.0f}m")
        results.append(dict(alpha_exp=a_exp, v_at_1500m=v_1500, rmse_crossrange=rc,
                             rmse_downrange=rd, h_apo=h_apo,
                             crossrange_end=float(model["crossrange"][-1]), status=model["status"]))
        model_runs[a_exp] = model

    csv_path = out_dir / f"wind_altitude_profile_fit_flight_{fno}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Zapisano: {csv_path}")

    # --- wykres: wszystkie alpha_exp na crossrange(t), plus wysokosc ---- #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ALPHA_EXPS)))
    ax = axes[0]
    for a_exp, c in zip(ALPHA_EXPS, colors):
        m = model_runs[a_exp]
        ax.plot(m["t"], m["crossrange"], color=c, lw=1.5, label=f"alpha_exp={a_exp:.2f}")
    ax.plot(actual["t"], actual["crossrange"], color="tab:orange", lw=2.0, ls="--", label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("crossrange [m]"); ax.set_title("Crossrange(t) vs profil wiatru")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for a_exp, c in zip(ALPHA_EXPS, colors):
        m = model_runs[a_exp]
        ax.plot(m["t"], m["h"], color=c, lw=1.5, label=f"alpha_exp={a_exp:.2f}")
    ax.plot(actual["t"], actual["h"], color="tab:orange", lw=2.0, ls="--", label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("wysokosc AGL [m]"); ax.set_title("Trajektoria wertykalna")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Lot {fno}: wplyw profilu wiatru z wysokoscia (roll+pitch/yaw juz skorygowane)")
    fig.tight_layout()
    out_png = out_dir / f"wind_altitude_profile_fit_flight_{fno}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
