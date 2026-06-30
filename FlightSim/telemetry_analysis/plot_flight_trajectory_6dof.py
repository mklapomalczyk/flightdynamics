"""
plot_flight_trajectory_6dof.py
================================
Wykres trajektorii i predkosci PELNEGO przebiegu czasowego (nie tylko
apogeum/impact), model 6DOF (wariant 'adjusted' -- rzeczywisty profil
ciagu tego lotu, masa/elewacja/azymut/atmosfera tego lotu, jak w
analyze_per_flight_6dof.py) vs dane polowe, dla wybranych lotow.

4 panele per lot: wysokosc AGL(t), downrange(t), crossrange(t),
predkosc(t). Predkosc rzeczywista to ta sama hybryda co w
validate_trajectory.py (spalanie: calkowanie akcelerometru; coast:
GPS Vh + rozniczka baro dla Vz) -- NIE predkosc onboard wprost.

Uzycie (lokalnie, z prawdziwym DATCOM):
    python plot_flight_trajectory_6dof.py 16 17

Sanity-check w kontenerze (bez DATCOM):
    python plot_flight_trajectory_6dof.py --no-rerun-datcom 16 17
"""

import sys
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, parse_telemetry, resolve_data_file
from run_6dof_cant_montecarlo import get_aero_for_cant
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_geometry
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from imu_reconstruction import detect_ignition
from diag_drag import detect_events, baro_altitude, calib_acc_scale, smooth_gps_derivative, G0
from gps_fusion import latlon_to_enu
from models.wind import PowerLawGustWind
from analyze_wind_sensitivity import read_measured_wind

from analyze_per_flight_6dof import (
    read_flights, build_flight_thrust, build_scaled_mass, downrange_crossrange,
    GUST_PERIOD_S, GUST_PHASE_RAD,
)


def build_wind(base, fno, azimuth_deg):
    """PowerLawGustWind z faktycznie zmierzonego wiatru dla tego lotu
    (Open-Meteo, field_test_data/results/launch_weather_openmeteo.csv) --
    gust_amp = gust/mean - 1. Zwraca None jesli brak pliku/wiersza dla
    tego lotu."""
    mw = read_measured_wind(base, fno)
    if mw is None or mw["mean_speed_mps"] <= 0:
        return None
    dir_from_deg = (azimuth_deg + mw["rel_az_deg"]) % 360.0
    gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
    return PowerLawGustWind(
        speed_ref_mps=mw["mean_speed_mps"], dir_from_deg=dir_from_deg, azimuth_deg=azimuth_deg,
        h_ref_m=10.0, alpha_exp=0.16,
        gust_amp=gust_amp, gust_period_s=GUST_PERIOD_S, gust_phase_rad=GUST_PHASE_RAD,
    )


def actual_time_series(base, fno, azimuth_deg, t_burn, elev_deg):
    """Szeregi czasowe rzeczywiste (od zaplonu, AGL): t, h, downrange,
    crossrange, V -- ta sama hybryda predkosci co simulate_ascent() w
    validate_trajectory.py (spalanie: akcelerometr; coast: GPS Vh +
    rozniczka baro Vz), na pelnym zakresie i_ign..i_end."""
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    tel = parse_telemetry(fpath, verbose=False)
    t_ign_abs = detect_ignition(tel)
    i_ign, _, i_apo, i_end = detect_events(tel, t_ign_abs)

    h_baro = baro_altitude(tel, i_ign)
    h0 = h_baro[i_ign]
    t = tel.time
    seg = slice(i_ign, i_end + 1)
    t_rel = t[seg] - t[i_ign]
    h = h_baro[seg] - h0

    lat0, lon0 = tel.lat[i_ign], tel.lon[i_ign]
    e, n = latlon_to_enu(tel.lat[seg], tel.lon[seg], lat0, lon0)
    downrange, crossrange = downrange_crossrange(e, n, azimuth_deg)

    elev = math.radians(elev_deg)
    acc_scale, _ = calib_acc_scale(tel, t_ign_abs)
    a_meas = tel.acc_x[seg] * acc_scale * G0
    a_kin = a_meas - G0 * np.sin(elev)
    V_acc = np.concatenate([[0.0],
        np.cumsum(0.5 * (a_kin[1:] + a_kin[:-1]) * np.diff(t_rel))])
    V_acc = np.abs(V_acc)

    Vz_baro = smooth_gps_derivative(h_baro, t, window_s=0.15)[seg]
    Vh_gps = tel.vel_onboard[seg]
    V_gps = np.sqrt(Vh_gps ** 2 + Vz_baro ** 2)
    dh_step = np.abs(np.diff(h_baro[seg], prepend=h_baro[seg][0]))
    half_win = int(0.15 / 0.004)
    bad = np.convolve(dh_step > 5.0, np.ones(2 * half_win + 1), mode='same') > 0
    V_gps = np.where(bad, np.nan, V_gps)

    i_switch = int(np.argmin(np.abs(t_rel - (t_burn + 0.3))))
    V = np.concatenate([V_acc[:i_switch], V_gps[i_switch:]])

    return dict(t=t_rel, h=h, downrange=downrange, crossrange=crossrange, V=V,
                t_apo=t[i_apo] - t[i_ign], h_apo=h_baro[i_apo] - h0)


def model_time_series(aero, geom, atm, gravity, launcher, mass, prop, initial_state, wind_model=None):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
        wind_model=wind_model,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-6, atol=1e-8, max_step=0.05)
    return dict(t=result.t, h=-result.z, downrange=result.x, crossrange=result.y,
                V=result.speed, status=result.status)


def plot_flight(fno, model, actual, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(model["t"], model["h"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["h"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.scatter([actual["t_apo"]], [actual["h_apo"]], marker='x', s=60, c='k', zorder=5,
               label="apogeum GPS")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("wysokosc AGL [m]")
    ax.set_title("Trajektoria wertykalna"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(model["t"], model["downrange"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["downrange"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("downrange [m]")
    ax.set_title("Downrange(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(model["t"], model["crossrange"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["crossrange"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("crossrange [m]")
    ax.set_title("Crossrange(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(model["t"], model["V"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["V"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("predkosc [m/s]")
    ax.set_title("Predkosc(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Lot {fno}: model 6DOF (adjusted+wiatr, profil ciagu tego lotu + zmierzony "
                 f"wiatr Open-Meteo) vs dane polowe (status modelu: {model['status']})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Trajektoria/predkosc(t) model 6DOF vs dane polowe")
    parser.add_argument("flights", type=int, nargs="+", help="numery lotow, np. 16 17")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    cfg_base_by_nose = {
        nose: load_config(str(root / "configurations" / f"{case}.yaml"))
        for nose, case in case_by_nose.items()
    }
    t_ignition_by_nose = {nose: cfg.propulsion.t_ignition for nose, cfg in cfg_base_by_nose.items()}

    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)

    flights_by_fno = {r["fno"]: r for r in read_flights(base)}

    for fno in args.flights:
        if fno not in flights_by_fno:
            print(f"Lot {fno}: brak w configs.txt (to analyze? != yes) -- pomijam.")
            continue
        r = flights_by_fno[fno]
        case = case_by_nose[r["nose"]]
        cfg_base = cfg_base_by_nose[r["nose"]]
        t_ignition = t_ignition_by_nose[r["nose"]]
        t_burn = cfg_base.propulsion.t_burn

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
        geom.cant_angle_rad = math.radians(r["cant"])

        mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
        initial_state = State6DOF.initial(elevation_deg=r["elevation"], azimuth_deg=r["azimuth"])

        prop_flight = build_flight_thrust(base, fno, t_ignition)
        if prop_flight is None:
            print(f"Lot {fno}: brak thrust_flight_{fno}.csv (kalibracja ciagu) -- pomijam.")
            continue

        wind = build_wind(base, fno, r["azimuth"])
        if wind is None:
            print(f"Lot {fno}: brak zmierzonego wiatru (Open-Meteo) -- pomijam.")
            continue

        model = model_time_series(aero, geom, atm, gravity, launcher, mass, prop_flight, initial_state,
                                   wind_model=wind)
        actual = actual_time_series(base, fno, r["azimuth"], t_burn, r["elevation"])

        plot_flight(fno, model, actual, out_dir / f"trajectory_6dof_flight_{fno}.png")


if __name__ == "__main__":
    main()
