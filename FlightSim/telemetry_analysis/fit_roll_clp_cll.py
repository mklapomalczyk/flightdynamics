"""
fit_roll_clp_cll.py
====================
Szuka najlepszego dopasowania modelu 6DOF do zrekonstruowanej z telemetrii
(kolumna AX, stały znak — patrz reconstruct_roll_from_ax_const_sign) predkosci
obrotowej p(t), skalujac DWA wspolczynniki toczenia niezaleznie:

  - CLL_scale  -- mnoznik momentu napedowego (od zaklinowania pletw,
                  realny DATCOM)
  - Clp_scale  -- mnoznik tlumienia toczenia (analityczne przyblizenie,
                  patrz aero.py -- prawdziwy $RLLO nie dziala z ta
                  wersja Missile DATCOM)

Motywacja: samo skalowanie Clp (tlumienie) NIE moze jednoczesnie
wyjasnic obserwowanej amplitudy I czasu narastania -- mniejsze tlumienie
podnosi rownowagowa predkosc obrotowa, ale WYDLUZA stala czasowa (szczyt
przesuwa sie w czasie pozniej), podczas gdy telemetria pokazuje bardzo
SZYBKIE narastanie do szczytu ~-5500 st/s juz w okolicy burnout
(t_burn=1.712s). To wskazuje, ze potrzebne jest WIEKSZE CLL (szybsze
narastanie) LUB MNIEJSZE Clp (wyzszy sufit) LUB oba jednoczesnie --
siatka przeszukuje obie osie.

Metoda: grid search (nie optymalizator gradientowy -- funkcja kosztu
oparta o symulacje ODE, tania siatka jest bardziej przejrzysta i
odporna). Koszt = RMSE miedzy modelem p(t) (interpolowanym do znacznikow
czasu telemetrii) a telemetria roll_rate, na segmencie wiarygodnym
(t>=0.7s, patrz reconstruct_roll_from_ax_const_sign).

Nie modyfikuje configurations/*.yaml ani MAIN.py -- skaluje tabele aero
w pamieci, na kopii wczytanego obiektu.

Uzycie:
    python fit_roll_clp_cll.py 19
    python fit_roll_clp_cll.py --no-rerun-datcom 19
"""

import sys
import argparse
import math
import copy
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
from plot_flight_trajectory_6dof import build_wind_steady, estimate_elevation_from_telemetry

CLL_SCALES = [1.0, 2.0, 3.0, 5.0]
CLP_SCALES = [0.1, 0.2, 0.5, 1.0]


def run_model_roll(aero, geom, mass, atm, gravity, launcher, prop, wind,
                    elev_deg, azimuth_deg, cll_scale, clp_scale, t_max=30.0):
    aero_local = copy.copy(aero)
    if aero.CLL_table is not None:
        aero_local.CLL_table = aero.CLL_table * cll_scale
    if aero.Clp_table is not None:
        aero_local.Clp_table = aero.Clp_table * clp_scale

    st = State6DOF.initial(elevation_deg=elev_deg, azimuth_deg=azimuth_deg)
    fm = ForceModel6DOF(atmosphere=atm, mass_model=mass, aero_model=aero_local,
                         gravity=gravity, geometry=geom, propulsion=prop,
                         launcher=launcher, wind_model=wind)
    result = run_simulation_6dof(fm, st, t_max=t_max, dt_output=0.05,
                                  rtol=1e-6, atol=1e-8, max_step=0.05)
    return result.t, np.degrees(result.p)


def main():
    parser = argparse.ArgumentParser(description="Dopasowanie CLL/Clp do predkosci obrotowej z telemetrii (AX)")
    parser.add_argument("fno", type=int)
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--t-fit-max", type=float, default=30.0,
                         help="koniec okna dopasowania [s] od zaplonu (domyslnie 30s)")
    args = parser.parse_args()
    fno = args.fno

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    flights_by_fno = {r["fno"]: r for r in read_flights(base)}
    if fno not in flights_by_fno:
        print(f"Lot {fno}: brak w configs.txt (to analyze? != yes).")
        return
    r = flights_by_fno[fno]
    case = args.case_ostra if r["nose"] == "ostra" else args.case_tepa
    cfg_base = load_config(str(root / "configurations" / f"{case}.yaml"))
    t_ignition = cfg_base.propulsion.t_ignition

    # --- telemetria (referencja) ------------------------------------- #
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    tel = parse_telemetry(fpath, verbose=False)
    t_ign = detect_ignition(tel)
    i_ign, _, i_apo, i_end = detect_events(tel, t_ign)
    roll_rate = reconstruct_roll_from_ax_const_sign(tel, t_ign)
    if roll_rate is None:
        print(f"Lot {fno}: brak kolumny AX -- nie mozna dopasowac.")
        return
    t_tel = (tel.time - t_ign)[i_ign:i_end + 1]
    rr_tel = roll_rate[i_ign:i_end + 1]
    fit_mask = (t_tel >= 0.7) & (t_tel <= args.t_fit_max)
    t_fit, rr_fit = t_tel[fit_mask], rr_tel[fit_mask]

    # --- model setup --------------------------------------------------- #
    aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
    geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
    geom.cant_angle_rad = math.radians(r["cant"])
    mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
    atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)
    prop_flight = build_flight_thrust(base, fno, t_ignition)
    if prop_flight is None:
        print(f"Lot {fno}: brak thrust_flight_{fno}.csv.")
        return

    elev_deg, _ = estimate_elevation_from_telemetry(base, fno)
    azimuth_deg = r["azimuth"]
    wind = build_wind_steady(base, fno, azimuth_deg)

    if aero.CLL_table is None or aero.Clp_table is None:
        print(f"Lot {fno}: brak CLL_table lub Clp_table w aero -- nie mozna dopasowac.")
        return

    # --- siatka ---------------------------------------------------------- #
    print(f"Lot {fno}: siatka {len(CLL_SCALES)}x{len(CLP_SCALES)} = "
          f"{len(CLL_SCALES)*len(CLP_SCALES)} symulacji...")
    results = {}
    best = None
    for cll_s in CLL_SCALES:
        for clp_s in CLP_SCALES:
            t_m, p_m = run_model_roll(aero, geom, mass, atm, gravity, launcher,
                                       prop_flight, wind, elev_deg, azimuth_deg,
                                       cll_s, clp_s, t_max=args.t_fit_max + 2.0)
            p_interp = np.interp(t_fit, t_m, p_m)
            rmse = float(np.sqrt(np.mean((p_interp - rr_fit) ** 2)))
            peak_m = p_m[np.argmax(np.abs(p_m))]
            t_peak_m = t_m[np.argmax(np.abs(p_m))]
            results[(cll_s, clp_s)] = dict(rmse=rmse, peak=peak_m, t_peak=t_peak_m)
            print(f"  CLL x{cll_s:>4.1f}  Clp x{clp_s:>4.2f}  ->  RMSE={rmse:7.0f}  "
                  f"peak={peak_m:7.0f} @t={t_peak_m:.2f}s")
            if best is None or rmse < best[0]:
                best = (rmse, cll_s, clp_s)

    rmse_best, cll_best, clp_best = best
    print(f"\nNajlepsze dopasowanie: CLL x{cll_best:.2f}, Clp x{clp_best:.2f}  "
          f"(RMSE={rmse_best:.0f} deg/s)")
    print(f"Telemetria: szczyt={rr_fit[np.argmax(np.abs(rr_fit))]:.0f} deg/s "
          f"@t={t_fit[np.argmax(np.abs(rr_fit))]:.2f}s")

    # --- wykres najlepszego dopasowania ----------------------------------- #
    t_best, p_best = run_model_roll(aero, geom, mass, atm, gravity, launcher,
                                     prop_flight, wind, elev_deg, azimuth_deg,
                                     cll_best, clp_best, t_max=40.0)
    t_base, p_base = run_model_roll(aero, geom, mass, atm, gravity, launcher,
                                     prop_flight, wind, elev_deg, azimuth_deg,
                                     1.0, 1.0, t_max=40.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_tel[t_tel < 0.7], rr_tel[t_tel < 0.7], color="tab:orange", lw=0.8, alpha=0.35,
            label="telemetria AX (t<0.7s -- sygnal ponizej szumu)")
    ax.plot(t_tel[t_tel >= 0.7], rr_tel[t_tel >= 0.7], color="tab:orange", lw=1.5,
            label="telemetria AX (znak staly)")
    ax.plot(t_base, p_base, color="gray", lw=1.2, ls="--",
            label="model 6DOF (CLL x1.0, Clp x1.0 -- bazowy)")
    ax.plot(t_best, p_best, color="tab:blue", lw=1.8,
            label=f"model 6DOF (CLL x{cll_best:.2f}, Clp x{clp_best:.2f} -- najlepsze)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("czas od zaplonu [s]")
    ax.set_ylabel("predkosc obrotowa p [deg/s]")
    ax.set_title(f"Lot {fno}: dopasowanie CLL/Clp do predkosci obrotowej (czujnik AX)\n"
                f"najlepsze: CLL x{cll_best:.2f}, Clp x{clp_best:.2f}, RMSE={rmse_best:.0f} deg/s "
                f"(dopasowanie na t=0.7-{args.t_fit_max:.0f}s)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / f"roll_rate_fit_flight_{fno}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_png}")

    # --- heatmapa RMSE (siatka NIErownomierna -- kategorialne osie,
    # nie pcolormesh/imshow z extent, zeby nie zniekształcić pozycji) ---- #
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rmse_grid = np.array([[results[(c, p)]["rmse"] for c in CLL_SCALES] for p in CLP_SCALES])
    im = ax2.imshow(rmse_grid, aspect="auto", origin="lower", cmap="viridis_r")
    ax2.set_xticks(range(len(CLL_SCALES)))
    ax2.set_xticklabels([f"{c:.1f}" for c in CLL_SCALES])
    ax2.set_yticks(range(len(CLP_SCALES)))
    ax2.set_yticklabels([f"{p:.2f}" for p in CLP_SCALES])
    ax2.set_xlabel("CLL scale")
    ax2.set_ylabel("Clp scale")
    ax2.set_title(f"Lot {fno}: RMSE(CLL_scale, Clp_scale) [deg/s]")
    ax2.scatter([CLL_SCALES.index(cll_best)], [CLP_SCALES.index(clp_best)],
                color="red", marker="*", s=200, label="najlepsze")
    ax2.legend()
    fig2.colorbar(im, ax=ax2, label="RMSE [deg/s]")
    out_png2 = out_dir / f"roll_rate_fit_heatmap_flight_{fno}.png"
    fig2.savefig(out_png2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"Zapisano: {out_png2}")


if __name__ == "__main__":
    main()
