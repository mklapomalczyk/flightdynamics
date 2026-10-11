"""
fit_pitch_yaw_stiffness.py
============================
Testuje, czy sztywnosc/tlumienie pitch-yaw (Cm_table -- statyczny moment
przywracajacy z realnego DATCOM, i Cmq_table -- tlumienie, ANALITYCZNE
przyblizenie z CNA_fins, patrz aero.py::_compute_cmq_table) sa
niedoszacowane, PODOBNIE jak dla rolla (gdzie CLL/Clp okazaly sie
niedoszacowane, zwalidowane przez niezalezny czujnik AX).

WAZNA ROZNICA wzgledem walidacji rolla: NIE MA tu niezaleznego, wiarygodnego
czujnika predkosci pitch/yaw. Surowe gyro_y/gyro_z podczas t<10s pokazuja
setki-do-1000+ st/s, ale sprawdzone widmowo (FFT) -- pokazuja skladowa w
paśmie ~14-21 Hz odpowiadajacej CZESTOTLIWOSCI OBROTU rolla w tym oknie
(zwalidowanej niezaleznie przez czujnik AX), co wskazuje na przeciek
sygnalu rolla (cross-axis) do kanalow Y/Z, nie prawdziwy ruch pitch/yaw --
a fizyczny dowod (rakieta odzyskana calo, bez oznak koziolkowania) wprost
przeczy realnym predkosciom pitch/yaw rzedu setek-tysiecy st/s. Dlatego
NIE dopasowujemy do gyro_y/z (byloby to dopasowywanie do szumu czujnika).

Zamiast tego: dopasowanie POSREDNIE, na poziomie TRAJEKTORII (downrange/
crossrange(t) vs dane polowe) -- SLABSZA walidacja niz dla rolla (ryzyko
kompensowania wielu roznych bledow jednoczesnie jednym parametrem), ale
jedyna dostepna bez wiarygodnego czujnika predkosci katowej pitch/yaw.
Model uzywa juz zwalidowanej korekty rolla (CLL x15.5, Clp x3.0) i
realnego, stalego (nie sinusoidalnego) wiatru (patrz build_wind_steady),
bo to najlepiej zachowujacy sie z przetestowanych modeli wiatru.

Cm_scale: mnoznik na Cm_table (statyczny moment przywracajacy, REALNY
DATCOM z glownego przebiegu -- mniej podejrzany, mniejszy zakres siatki).
Cmq_scale: mnoznik na Cmq_table (tlumienie, ANALITYCZNA formula z CNA_fins,
analogiczna do Clp rolla ktore bylo x3 za male -- bardziej podejrzany,
szerszy zakres siatki).

Nie modyfikuje configurations/*.yaml ani MAIN.py.

Uzycie:
    python fit_pitch_yaw_stiffness.py --no-rerun-datcom 19
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

ROLL_CLL_SCALE = 15.5   # znalezione dopasowanie predkosci obrotowej (lot 19)
ROLL_CLP_SCALE = 3.0

CM_SCALES  = [0.5, 1.0, 1.5, 2.0, 3.0]
CMQ_SCALES = [0.3, 1.0, 3.0, 5.0]


def run_with_stiffness(aero, geom, mass, atm, gravity, launcher, prop, wind,
                        elev_deg, azimuth_deg, cm_scale, cmq_scale):
    orig_cm = aero.Cm_table
    orig_cmq = aero.Cmq_table
    try:
        aero.Cm_table = orig_cm * cm_scale
        if aero.Cmq_table is not None:
            aero.Cmq_table = orig_cmq * cmq_scale
        initial_state = State6DOF.initial(elevation_deg=elev_deg, azimuth_deg=azimuth_deg)
        return model_time_series(aero, geom, atm, gravity, launcher, mass, prop,
                                  initial_state, wind_model=wind)
    finally:
        aero.Cm_table = orig_cm
        aero.Cmq_table = orig_cmq


def rmse(t_ref, y_ref, t_m, y_m, t_max):
    mask = t_ref <= min(t_max, t_ref[-1])
    if mask.sum() < 5:
        return float("nan")
    y_interp = np.interp(t_ref[mask], t_m, y_m)
    return float(np.sqrt(np.mean((y_interp - y_ref[mask]) ** 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Posrednie (poziom trajektorii) dopasowanie sztywnosci/tlumienia pitch-yaw")
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
    if aero.Cm_table is None:
        print(f"Lot {fno}: brak Cm_table -- nie mozna dopasowac.")
        return
    aero.CLL_table = aero.CLL_table * ROLL_CLL_SCALE if aero.CLL_table is not None else None
    aero.Clp_table = aero.Clp_table * ROLL_CLP_SCALE if aero.Clp_table is not None else None
    print(f"Lot {fno}: roll juz skorygowany (CLL x{ROLL_CLL_SCALE}, Clp x{ROLL_CLP_SCALE})")

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
    wind = build_wind_steady(base, fno, azimuth_deg)
    if wind is None:
        print(f"Lot {fno}: brak zmierzonego wiatru -- nie mozna dopasowac (potrzebny do "
              f"wzbudzenia obserwowalnego crossrange).")
        return

    actual = actual_time_series(base, fno, azimuth_deg, t_burn, elev_deg)

    print(f"Lot {fno}: siatka {len(CM_SCALES)}x{len(CMQ_SCALES)} = "
          f"{len(CM_SCALES)*len(CMQ_SCALES)} symulacji "
          f"(dopasowanie posrednie na downrange/crossrange(t), t<={args.t_fit_max:.0f}s)...")

    results = {}
    best = None
    for cm_s in CM_SCALES:
        for cmq_s in CMQ_SCALES:
            model = run_with_stiffness(aero, geom, mass, atm, gravity, launcher, prop_flight,
                                        wind, elev_deg, azimuth_deg, cm_s, cmq_s)
            rmse_cross = rmse(actual["t"], actual["crossrange"], model["t"], model["crossrange"],
                               args.t_fit_max)
            rmse_down = rmse(actual["t"], actual["downrange"], model["t"], model["downrange"],
                              args.t_fit_max)
            h_apo_m = float(np.max(model["h"]))
            results[(cm_s, cmq_s)] = dict(rmse_cross=rmse_cross, rmse_down=rmse_down,
                                           h_apo=h_apo_m, status=model["status"])
            print(f"  Cm x{cm_s:.1f}  Cmq x{cmq_s:.1f}  status={model['status']:>8}  "
                  f"RMSE_crossrange={rmse_cross:7.0f}  RMSE_downrange={rmse_down:7.0f}  "
                  f"h_apo={h_apo_m:6.0f}m")
            if model["status"] == "ok" and (best is None or rmse_cross < best[0]):
                best = (rmse_cross, cm_s, cmq_s)

    if best is None:
        print("\nBrak udanych (status=ok) symulacji -- nie mozna wybrac najlepszej.")
        return
    rmse_best, cm_best, cmq_best = best
    print(f"\nNajlepsze (wg RMSE crossrange, status=ok): Cm x{cm_best:.1f}, Cmq x{cmq_best:.1f} "
          f"(RMSE_crossrange={rmse_best:.0f}m)")
    baseline = results[(1.0, 1.0)]
    print(f"Bazowy (Cm x1.0, Cmq x1.0): RMSE_crossrange={baseline['rmse_cross']:.0f}m")

    # --- CSV -------------------------------------------------------------- #
    rows = []
    for (cm_s, cmq_s), d in results.items():
        rows.append(dict(cm_scale=cm_s, cmq_scale=cmq_s, **d))
    csv_path = out_dir / f"pitch_yaw_stiffness_fit_flight_{fno}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Zapisano: {csv_path}")

    # --- wykres: najlepszy vs bazowy vs telemetria ------------------------ #
    model_best = run_with_stiffness(aero, geom, mass, atm, gravity, launcher, prop_flight,
                                     wind, elev_deg, azimuth_deg, cm_best, cmq_best)
    model_base = run_with_stiffness(aero, geom, mass, atm, gravity, launcher, prop_flight,
                                     wind, elev_deg, azimuth_deg, 1.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(model_base["t"], model_base["h"], color="gray", lw=1.2, ls="--", label="model (Cm x1, Cmq x1)")
    ax.plot(model_best["t"], model_best["h"], color="tab:blue", lw=1.6,
            label=f"model (Cm x{cm_best:.1f}, Cmq x{cmq_best:.1f})")
    ax.plot(actual["t"], actual["h"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("wysokosc AGL [m]"); ax.set_title("Trajektoria wertykalna")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(model_base["t"], model_base["downrange"], color="gray", lw=1.2, ls="--", label="model bazowy")
    ax.plot(model_best["t"], model_best["downrange"], color="tab:blue", lw=1.6, label="model najlepszy")
    ax.plot(actual["t"], actual["downrange"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("downrange [m]"); ax.set_title("Downrange(t)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(model_base["t"], model_base["crossrange"], color="gray", lw=1.2, ls="--", label="model bazowy")
    ax.plot(model_best["t"], model_best["crossrange"], color="tab:blue", lw=1.6, label="model najlepszy")
    ax.plot(actual["t"], actual["crossrange"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("crossrange [m]"); ax.set_title("Crossrange(t)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(model_base["t"], model_base["V"], color="gray", lw=1.2, ls="--", label="model bazowy")
    ax.plot(model_best["t"], model_best["V"], color="tab:blue", lw=1.6, label="model najlepszy")
    ax.plot(actual["t"], actual["V"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas [s]"); ax.set_ylabel("predkosc [m/s]"); ax.set_title("Predkosc(t)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Lot {fno}: dopasowanie POSREDNIE sztywnosci/tlumienia pitch-yaw "
                f"(Cm x{cm_best:.1f}, Cmq x{cmq_best:.1f}) -- SLABSZA walidacja niz rolla "
                f"(brak wiarygodnego czujnika predkosci pitch/yaw)")
    fig.tight_layout()
    out_png = out_dir / f"pitch_yaw_stiffness_fit_flight_{fno}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
