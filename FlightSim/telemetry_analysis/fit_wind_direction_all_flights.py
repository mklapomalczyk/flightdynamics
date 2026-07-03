"""
fit_wind_direction_all_flights.py
====================================
Dla kazdego lotu z telemetria+wiatrem (Open-Meteo), przy REALNEJ zmierzonej
predkosci wiatru (bez skalowania, speed_scale=1.0), przeszukuje kierunek
wiatru (offset od wartosci wyliczonej z Open-Meteo) i znajduje ten, ktory
najlepiej pasuje JEDNOCZESNIE do apogeum, downrange i crossrange (nie
tylko crossrange, jak w sweep'u dla lotu 19 wczesniej w tej sesji).

Model uzywa juz najlepszych dotad znalezionych korekt: roll (CLL x15.5,
Clp x3.0) i pitch/yaw sztywnosc/tlumienie (Cm x3.0, Cmq x0.3) -- oba
dopasowane dla lotu 19, tu zastosowane uniwersalnie (nie ponownie
dopasowywane per lot).

Motywacja: sprawdzic czy najlepszy kierunek wiatru jest SPOJNY miedzy
lotami (co wskazywaloby na systematyczny blad np. w konwersji rel_az/
azymutu) czy calkowicie rozny per lot (co wskazywaloby na po prostu
niedokladnosc godzinowych danych Open-Meteo wzgledem realnych warunkow
lokalnych, rozna kazdego dnia/godziny).

Uzywa PowerLawWind (stale, bez sztucznego podmuchu) -- "predkosc wiatru
z danych Open-Meteo" dosl. jak podaje uzytkownik, bez domieszki
niezwalidowanego ksztaltu podmuchu w czasie.

Nie modyfikuje configurations/*.yaml ani MAIN.py.

Uzycie:
    python fit_wind_direction_all_flights.py --no-rerun-datcom
    python fit_wind_direction_all_flights.py --no-rerun-datcom 13 17 18 19
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
from models.wind import PowerLawWind
from core.state6 import State6DOF
from datcom_io.config_reader import load_config

from analyze_per_flight_6dof import read_flights
from analyze_wind_sensitivity import read_measured_wind
from plot_flight_trajectory_6dof import model_time_series, actual_time_series
from check_roll_factors_all_flights import build_flight_model_inputs

ROLL_CLL_SCALE = 15.5
ROLL_CLP_SCALE = 3.0
PITCH_YAW_CM_SCALE  = 3.0
PITCH_YAW_CMQ_SCALE = 0.3

OFFSETS = list(range(0, 360, 60))   # co 60 stopni, 6 punktow (grubsze -- wiele lotow)


def combined_error(model, actual, t_max):
    mask = actual["t"] <= min(t_max, actual["t"][-1])
    rc = np.sqrt(np.mean((np.interp(actual["t"][mask], model["t"], model["crossrange"])
                           - actual["crossrange"][mask]) ** 2))
    rd = np.sqrt(np.mean((np.interp(actual["t"][mask], model["t"], model["downrange"])
                           - actual["downrange"][mask]) ** 2))
    h_apo_err = abs(float(np.max(model["h"])) - actual["h_apo"])
    return rc, rd, h_apo_err


def main():
    parser = argparse.ArgumentParser(
        description="Przeszukuje kierunek wiatru per lot, realna predkosc z Open-Meteo")
    parser.add_argument("flights", type=int, nargs="*")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--t-fit-max", type=float, default=36.0)
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_flights = read_flights(base)
    flights = [r for r in all_flights if (not args.flights or r["fno"] in args.flights)]

    summary = []
    for r in flights:
        fno = r["fno"]
        mw = read_measured_wind(base, fno)
        if mw is None or mw["mean_speed_mps"] <= 0:
            print(f"Lot {fno}: brak zmierzonego wiatru -- pomijam.")
            continue

        inputs = build_flight_model_inputs(base, root, r, args.case_ostra, args.case_tepa,
                                            force_rerun=not args.no_rerun_datcom)
        if inputs is None:
            print(f"Lot {fno}: brak profilu ciagu -- pomijam.")
            continue
        if not inputs["cant_matched"]:
            print(f"Lot {fno}: aero NIE dopasowane do cant tego lotu -- pomijam "
                  f"(brak lokalnego cache dla tego cant_angle).")
            continue

        aero = inputs["aero"]
        if aero.CLL_table is not None:
            aero.CLL_table = aero.CLL_table * ROLL_CLL_SCALE
        if aero.Clp_table is not None:
            aero.Clp_table = aero.Clp_table * ROLL_CLP_SCALE
        if aero.Cm_table is not None:
            aero.Cm_table = aero.Cm_table * PITCH_YAW_CM_SCALE
        if aero.Cmq_table is not None:
            aero.Cmq_table = aero.Cmq_table * PITCH_YAW_CMQ_SCALE

        case = args.case_ostra if r["nose"] == "ostra" else args.case_tepa
        cfg_base = load_config(str(root / "configurations" / f"{case}.yaml"))
        actual = actual_time_series(base, fno, r["azimuth"], cfg_base.propulsion.t_burn, r["elevation"])
        initial_state = State6DOF.initial(elevation_deg=r["elevation"], azimuth_deg=r["azimuth"])

        dir_from_base = (r["azimuth"] + mw["rel_az_deg"]) % 360.0
        print(f"\nLot {fno} (cant={r['cant']:.2f}, wiatr={mw['mean_speed_mps']:.1f}m/s, "
              f"dir_from_base={dir_from_base:.1f}):")
        print(f"  Telemetria: crossrange_end={actual['crossrange'][-1]:.0f}  "
              f"downrange_end={actual['downrange'][-1]:.0f}  h_apo={actual['h_apo']:.0f}")

        best = None
        for offset in OFFSETS:
            dfd = (dir_from_base + offset) % 360.0
            wind = PowerLawWind(speed_ref_mps=mw["mean_speed_mps"], dir_from_deg=dfd,
                                 azimuth_deg=r["azimuth"], h_ref_m=10.0, alpha_exp=0.16)
            model = model_time_series(aero, geom=inputs["geom"], atm=inputs["atm"],
                                       gravity=inputs["gravity"], launcher=inputs["launcher"],
                                       mass=inputs["mass"], prop=inputs["prop"],
                                       initial_state=initial_state, wind_model=wind)
            if model["status"] != "ok":
                print(f"  offset={offset:>3}  dir_from={dfd:>6.1f}  status={model['status']} -- pomijam")
                continue
            rc, rd, h_err = combined_error(model, actual, args.t_fit_max)
            # znormalizowany laczny blad -- kazda skladowa wzgledem typowej skali bledu w tej sesji
            combined = rc/500.0 + rd/500.0 + h_err/300.0
            print(f"  offset={offset:>3}  dir_from={dfd:>6.1f}  "
                  f"crossrange_end={model['crossrange'][-1]:>7.0f}  downrange_end={model['downrange'][-1]:>7.0f}  "
                  f"h_apo={np.max(model['h']):>6.0f}  RMSE_cross={rc:>6.0f}  RMSE_down={rd:>6.0f}  "
                  f"h_apo_err={h_err:>5.0f}  combined={combined:.2f}")
            if best is None or combined < best[0]:
                best = (combined, offset, dfd, rc, rd, h_err,
                        model["crossrange"][-1], model["downrange"][-1], float(np.max(model["h"])))

        if best is None:
            print(f"  Lot {fno}: brak udanych symulacji.")
            continue
        combined, offset, dfd, rc, rd, h_err, cross_end, down_end, h_apo = best
        print(f"  NAJLEPSZY: offset={offset} dir_from={dfd:.1f}  combined_err={combined:.2f}  "
              f"(crossrange={cross_end:.0f}, downrange={down_end:.0f}, h_apo={h_apo:.0f})")
        summary.append(dict(fno=fno, cant_deg=r["cant"], azimuth_deg=r["azimuth"],
                             dir_from_base=dir_from_base, best_offset=offset, best_dir_from=dfd,
                             rmse_crossrange=rc, rmse_downrange=rd, h_apo_err=h_err,
                             combined_err=combined, crossrange_end=cross_end,
                             downrange_end=down_end, h_apo=h_apo,
                             tel_crossrange_end=actual["crossrange"][-1],
                             tel_downrange_end=actual["downrange"][-1], tel_h_apo=actual["h_apo"]))

    if not summary:
        print("\nBrak lotow do podsumowania.")
        return

    csv_path = out_dir / "wind_direction_best_fit_all_flights.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nZapisano: {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for s in summary:
        ax.scatter(s["fno"], s["best_offset"], s=100, color="tab:blue")
        ax.annotate(f"cant={s['cant_deg']:.1f}", (s["fno"], s["best_offset"]),
                    fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("nr lotu")
    ax.set_ylabel("najlepszy offset kierunku wiatru [deg]")
    ax.set_title("Najlepszy offset kierunku wiatru (wzgledem Open-Meteo) per lot\n"
                "(realna predkosc wiatru, laczny blad apogeum+downrange+crossrange)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / "wind_direction_best_fit_all_flights.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
