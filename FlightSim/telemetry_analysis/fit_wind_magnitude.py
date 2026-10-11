"""
fit_wind_magnitude.py
========================
Testuje, czy jakakolwiek WIELKOSC (predkosc) wiatru -- zamiast kierunku
(patrz sweep offsetu w tej samej sesji, wykluczyl kierunek jako
wyjasnienie) -- zblizyloby model do rzeczywistego crossrange lotu 19
(~1410m). Testuje trzy modele wiatru (steady/gust/pulse) x mnoznik
predkosci (speed_scale) x opcjonalnie inny kierunek (dir-offset).

Model uzywa juz najlepszych dotad znalezionych korekt: roll (CLL x15.5,
Clp x3.0, zwalidowane niezaleznym czujnikiem AX) i pitch/yaw sztywnosc/
tlumienie (Cm x3.0, Cmq x0.3, dopasowanie POSREDNIE).

Nie modyfikuje configurations/*.yaml ani MAIN.py.

Uzycie:
    python fit_wind_magnitude.py --no-rerun-datcom --wind-model steady 19
    python fit_wind_magnitude.py --no-rerun-datcom --wind-model gust 19
    python fit_wind_magnitude.py --no-rerun-datcom --wind-model pulse --pulse-t-center 1.5 19
    python fit_wind_magnitude.py --no-rerun-datcom --wind-model steady --dir-offset 300 19
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
from models.wind import PowerLawWind, PowerLawGustWind, PowerLawGustPulseWind
from core.state6 import State6DOF

from analyze_per_flight_6dof import read_flights, build_flight_thrust, build_scaled_mass
from analyze_wind_sensitivity import read_measured_wind
from plot_flight_trajectory_6dof import model_time_series, actual_time_series, GUST_PERIOD_S, GUST_PHASE_RAD

ROLL_CLL_SCALE = 15.5
ROLL_CLP_SCALE = 3.0
PITCH_YAW_CM_SCALE  = 3.0
PITCH_YAW_CMQ_SCALE = 0.3

SPEED_SCALES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


def build_wind(model_name, mw, dir_from_deg, azimuth_deg, speed_scale, pulse_t_center):
    speed_ref = mw["mean_speed_mps"] * speed_scale
    if model_name == "steady":
        return PowerLawWind(speed_ref_mps=speed_ref, dir_from_deg=dir_from_deg,
                             azimuth_deg=azimuth_deg, h_ref_m=10.0, alpha_exp=0.16)
    elif model_name == "gust":
        gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
        return PowerLawGustWind(speed_ref_mps=speed_ref, dir_from_deg=dir_from_deg,
                                 azimuth_deg=azimuth_deg, h_ref_m=10.0, alpha_exp=0.16,
                                 gust_amp=gust_amp, gust_period_s=GUST_PERIOD_S,
                                 gust_phase_rad=GUST_PHASE_RAD)
    elif model_name == "pulse":
        gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
        return PowerLawGustPulseWind(speed_ref_mps=speed_ref, dir_from_deg=dir_from_deg,
                                      azimuth_deg=azimuth_deg, h_ref_m=10.0, alpha_exp=0.16,
                                      gust_amp=gust_amp, t_center_s=pulse_t_center, sigma_s=2.0)
    else:
        raise ValueError(model_name)


def main():
    parser = argparse.ArgumentParser(description="Sweep predkosci wiatru (roznych modeli) dla lotu 19")
    parser.add_argument("fno", type=int)
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--wind-model", choices=["steady", "gust", "pulse"], default="steady")
    parser.add_argument("--pulse-t-center", type=float, default=1.5)
    parser.add_argument("--dir-offset", type=float, default=0.0,
                         help="offset [deg] na kierunek wiatru wzgledem obliczonego z Open-Meteo")
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
    initial_state = State6DOF.initial(elevation_deg=elev_deg, azimuth_deg=azimuth_deg)
    actual = actual_time_series(base, fno, azimuth_deg, t_burn, elev_deg)

    mw = read_measured_wind(base, fno)
    if mw is None:
        print(f"Lot {fno}: brak zmierzonego wiatru.")
        return
    dir_from_base = (azimuth_deg + mw["rel_az_deg"] + args.dir_offset) % 360.0

    print(f"Lot {fno}: model wiatru={args.wind_model}, dir_from={dir_from_base:.1f} deg "
          f"(offset={args.dir_offset:.0f}), roll+pitch/yaw juz skorygowane")
    print(f"Telemetry: crossrange_end={actual['crossrange'][-1]:.0f}  "
          f"downrange_end={actual['downrange'][-1]:.0f}  h_apo={actual['h_apo']:.0f}")

    results = []
    model_runs = {}
    for scale in SPEED_SCALES:
        wind = build_wind(args.wind_model, mw, dir_from_base, azimuth_deg, scale, args.pulse_t_center)
        model = model_time_series(aero, geom, atm, gravity, launcher, mass, prop_flight,
                                   initial_state, wind_model=wind)
        mask = actual["t"] <= min(args.t_fit_max, actual["t"][-1])
        rc = np.sqrt(np.mean((np.interp(actual["t"][mask], model["t"], model["crossrange"])
                               - actual["crossrange"][mask]) ** 2))
        rd = np.sqrt(np.mean((np.interp(actual["t"][mask], model["t"], model["downrange"])
                               - actual["downrange"][mask]) ** 2))
        h_apo = float(np.max(model["h"]))
        cross_end = float(model["crossrange"][-1])
        print(f"  speed_scale={scale:>4.1f}  V_ref={mw['mean_speed_mps']*scale:6.1f}m/s  "
              f"status={model['status']:>8}  crossrange_end={cross_end:8.0f}  "
              f"downrange_end={model['downrange'][-1]:8.0f}  h_apo={h_apo:6.0f}  "
              f"RMSE_cross={rc:7.0f}  RMSE_down={rd:7.0f}")
        results.append(dict(speed_scale=scale, v_ref_mps=mw["mean_speed_mps"] * scale,
                             status=model["status"], crossrange_end=cross_end,
                             downrange_end=float(model["downrange"][-1]), h_apo=h_apo,
                             rmse_crossrange=rc, rmse_downrange=rd))
        model_runs[scale] = model

    csv_path = out_dir / f"wind_magnitude_fit_flight_{fno}_{args.wind_model}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Zapisano: {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(SPEED_SCALES)))
    ax = axes[0]
    for scale, c in zip(SPEED_SCALES, colors):
        m = model_runs[scale]
        ax.plot(m["t"], m["crossrange"], color=c, lw=1.5, label=f"x{scale:.1f}")
    ax.plot(actual["t"], actual["crossrange"], color="tab:orange", lw=2.2, ls="--", label="dane polowe")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("czas [s]"); ax.set_ylabel("crossrange [m]")
    ax.set_title(f"Crossrange(t) vs skala predkosci wiatru ({args.wind_model})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for scale, c in zip(SPEED_SCALES, colors):
        m = model_runs[scale]
        ax.plot(m["t"], m["h"], color=c, lw=1.5, label=f"x{scale:.1f}")
    ax.plot(actual["t"], actual["h"], color="tab:orange", lw=2.2, ls="--", label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("wysokosc AGL [m]")
    ax.set_title("Trajektoria wertykalna")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Lot {fno}: wplyw skali predkosci wiatru, model={args.wind_model}, "
                f"dir_offset={args.dir_offset:.0f}deg (roll+pitch/yaw skorygowane)")
    fig.tight_layout()
    out_png = out_dir / f"wind_magnitude_fit_flight_{fno}_{args.wind_model}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
