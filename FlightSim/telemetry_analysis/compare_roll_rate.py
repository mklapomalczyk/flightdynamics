"""
compare_roll_rate.py
=====================
Porownuje predkosc obrotowa (roll rate) modelu 6DOF z telemetria dla
danego lotu, uzywajac kolumny 'Predkosc obrotowa AX' -- dedykowanego
czujnika o szerszym zakresie niz standardowy zyroskop gyro_X (ktory
wysyca sie na +/-2000 st/s, podczas gdy realny roll canted-fin rakiety
siega kilku tysiecy st/s). Rekonstrukcja AX -> podpisana predkosc
(reconstruct_roll_from_ax) juz istnieje w imu_reconstruction.py.

Motywacja: podczas walidacji lotu 19 model 6DOF przewidywal szczytowa
predkosc obrotowa ~850-900 st/s (napedzana przez CLL z realnego DATCOM +
ANALITYCZNE przyblizenie Clp -- prawdziwy $RLLO z Missile DATCOM nie
dziala z ta wersja, patrz aero.py). Telemetria (kolumna AX,
zrekonstruowana) pokazuje realna predkosc obrotowa 2-3x wieksza i
znacznie wolniej zanikajaca w fazie balistycznej -- silna przeslanka, ze
analityczny Clp (tlumienie toczenia) w modelu jest za DUZY co do modulu
(za duzo tlumienia), a nie ze brakuje jakiegos zrodla napedu.

Nie modyfikuje configurations/*.yaml ani MAIN.py.

Uzycie:
    python compare_roll_rate.py 19
    python compare_roll_rate.py --no-rerun-datcom --clp-scale 0.35 19
"""

import sys
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from imu_reconstruction import detect_ignition, reconstruct_roll_from_ax_const_sign
from diag_drag import detect_events

from analyze_per_flight_6dof import read_flights, build_flight_thrust, build_scaled_mass
from plot_flight_trajectory_6dof import (
    build_wind_steady, estimate_elevation_from_telemetry, estimate_azimuth_from_telemetry,
)


def telemetry_roll_rate(base, fno, t_unreliable_s=0.7):
    """Zrekonstruowana predkosc obrotowa z kolumny AX -- reconstruct_roll_from_ax_const_sign
    (STALY znak, ustalony z gyro_X w oknie [5,8]s od zaplonu, zamiast
    kruchej logiki 'ostatni znany znak' per-probka -- ta ostatnia dawala
    pozorna, erratyczna oscylacje +/- ktora byla WYLACZNIE artefaktem
    zawodnosci odzyskiwania znaku przy czestym wysyceniu gyro_X, NIE
    odzwierciedleniem prawdziwego sygnalu: |AX_scaled| samo w sobie jest
    gladkie i ciagle od t~1s po zaplonie, ze szczytem w okolicy burnout).
    Tylko pierwsze t_unreliable_s sekund (rakieta ~w spoczynku na
    szynie/tuz po starcie, zanim cisnienie dynamiczne zbuduje sygnal
    powyzej szumu) sa oznaczone jako niepewne."""
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    tel = parse_telemetry(fpath, verbose=False)
    t_ign = detect_ignition(tel)
    i_ign, _, i_apo, i_end = detect_events(tel, t_ign)
    roll_rate = reconstruct_roll_from_ax_const_sign(tel, t_ign)
    if roll_rate is None:
        return None
    t_rel = tel.time - t_ign
    seg = slice(i_ign, i_end + 1)
    return dict(t=t_rel[seg], roll_rate=roll_rate[seg],
                reliable=t_rel[seg] >= t_unreliable_s)


def main():
    parser = argparse.ArgumentParser(
        description="Model vs telemetria (kolumna AX): predkosc obrotowa (roll)")
    parser.add_argument("flights", type=int, nargs="+")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--clp-scale", type=float, default=1.0,
                         help="mnoznik na analityczny Clp_table (domyslnie 1.0 -- "
                              "bez zmian). Np. 0.35 testuje hipotoze, ze analityczne "
                              "tlumienie toczenia jest ~3x za duze wzgledem "
                              "zrekonstruowanej z AX realnej predkosci obrotowej.")
    parser.add_argument("--no-wind", action="store_true")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    cfg_base_by_nose = {
        nose: load_config(str(root / "configurations" / f"{case}.yaml"))
        for nose, case in case_by_nose.items()
    }
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
        t_ignition = cfg_base.propulsion.t_ignition

        tel_roll = telemetry_roll_rate(base, fno)
        if tel_roll is None:
            print(f"Lot {fno}: brak kolumny AX w telemetrii -- pomijam.")
            continue

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        if args.clp_scale != 1.0:
            if aero.Clp_table is not None:
                aero.Clp_table = aero.Clp_table * args.clp_scale
                print(f"Lot {fno}: Clp_table przeskalowany x{args.clp_scale:.2f}")
            else:
                print(f"Lot {fno}: brak Clp_table -- --clp-scale bez efektu.")

        geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
        geom.cant_angle_rad = math.radians(r["cant"])
        mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])

        elev_deg, _ = estimate_elevation_from_telemetry(base, fno)
        azimuth_deg = r["azimuth"]   # NIE 'auto' -- GPS-owy azymut juz zawiera znoszenie wiatrem

        initial_state = State6DOF.initial(elevation_deg=elev_deg, azimuth_deg=azimuth_deg)
        prop_flight = build_flight_thrust(base, fno, t_ignition)
        if prop_flight is None:
            print(f"Lot {fno}: brak thrust_flight_{fno}.csv -- pomijam.")
            continue

        wind = None if args.no_wind else build_wind_steady(base, fno, azimuth_deg)

        force_model = ForceModel6DOF(
            atmosphere=atm, mass_model=mass, aero_model=aero, gravity=gravity,
            geometry=geom, propulsion=prop_flight, launcher=launcher, wind_model=wind,
        )
        result = run_simulation_6dof(force_model, initial_state, t_max=40, dt_output=0.02,
                                      rtol=1e-6, atol=1e-8, max_step=0.05)
        p_deg_s = np.degrees(result.p)

        print(f"\nLot {fno} (status={result.status}, elev={elev_deg:.1f}, "
              f"az={azimuth_deg:.1f}, Clp_scale={args.clp_scale:.2f}):")
        print(f"  model:  max|p|={np.max(np.abs(p_deg_s)):.0f} deg/s "
              f"at t={result.t[np.argmax(np.abs(p_deg_s))]:.2f}s")
        rel = tel_roll["reliable"]
        peak_i = np.nanargmax(np.abs(tel_roll["roll_rate"][rel]))
        print(f"  telemetria (AX, znak staly): "
              f"szczyt={tel_roll['roll_rate'][rel][peak_i]:.0f} deg/s "
              f"at t={tel_roll['t'][rel][peak_i]:.2f}s")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(result.t, p_deg_s, color="tab:blue", lw=1.5,
                label=f"model 6DOF (Clp x{args.clp_scale:.2f})")
        t_tel, rr = tel_roll["t"], tel_roll["roll_rate"]
        ax.plot(t_tel[~rel], rr[~rel], color="tab:orange", lw=0.8, alpha=0.35,
                label="telemetria AX (t<0.7s -- sygnal ponizej szumu)")
        ax.plot(t_tel[rel], rr[rel], color="tab:orange", lw=1.5,
                label="telemetria AX (znak staly, ustalony z gyro_X @5-8s)")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("czas od zaplonu [s]")
        ax.set_ylabel("predkosc obrotowa p [deg/s]")
        ax.set_title(f"Lot {fno}: model vs telemetria (czujnik AX, szeroki zakres)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        suffix = f"_clp{args.clp_scale:.2f}" if args.clp_scale != 1.0 else ""
        out_png = out_dir / f"roll_rate_vs_telemetry_flight_{fno}{suffix}.png"
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Zapisano: {out_png}")


if __name__ == "__main__":
    main()
