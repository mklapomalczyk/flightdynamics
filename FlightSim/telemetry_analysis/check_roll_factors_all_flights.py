"""
check_roll_factors_all_flights.py
==================================
Sprawdza, czy skalowanie CLL/Clp znalezione dla lotu 19 (CLL x15.5,
Clp x3.0 -- patrz fit_roll_clp_cll.py) generalizuje sie na inne loty, oraz
(opcjonalnie, --fit) szuka NIEZALEZNEGO najlepszego dopasowania per lot.

Dwa tryby:
  1. Domyslny: dla kazdego lotu z telemetria+AX+ciagiem, rysuje p(t) modelu
     PRZY STALYCH wspolczynnikach (domyslnie z lotu 19) na tle
     zrekonstruowanej z AX predkosci obrotowej (znak staly, patrz
     reconstruct_roll_from_ax_const_sign w imu_reconstruction.py). Szybkie
     (1 symulacja/lot) -- pierwszy, wzrokowy test czy stale wspolczynniki
     dzialaja wszedzie.
  2. --fit: dodatkowo male przeszukiwanie siatki per lot (podobne do
     fit_roll_clp_cll.py) niezaleznie dla kazdego lotu, zeby sprawdzic czy
     KAZDY lot potrzebuje podobnego wspolczynnika, czy rozni sie znaczaco
     (np. wraz z zaklinowaniem pletw, predkoscia, machem...).

UWAGA fizyczna: loty z cant_angle=0 (patrz configs.txt) nie generuja
momentu naped-toczenia od zaklinowania pletw (CLL_table ~ 0 niezaleznie
od cant_scale) -- to NATURALNY test kontrolny: jesli telemetria takze
pokazuje bliski zera roll dla tych lotow, wzmacnia to hipoteze, ze caly
brakujacy roll pochodzi z niedoszacowanego cant-driven CLL (a nie z
innego, cant-niezaleznego zrodla momentu, np. asymetrii silnika/pletw).
Jesli telemetria POKAZUJE spory roll mimo cant=0, to wskazuje na
dodatkowe zrodlo momentu nieuwzglednione w modelu.

Do testow offline (bez lokalnego MissileDATCOM.exe): --no-rerun-datcom
probuje najpierw wczytac JUZ ISTNIEJACY cache dopasowany do cant_angle
danego lotu (datcom_runs/<case>_cant<X>/aero_table_missile.pkl, ta sama
konwencja co run_6dof_cant_montecarlo.make_cant_case_name) -- fizycznie
poprawne TYLKO gdy taki cache faktycznie istnieje dla tego cant. Lokalnie
(z realnym DATCOM.exe) pomin --no-rerun-datcom, aby kazdy lot dostal
swiezo wygenerowana, prawdziwie dopasowana do wlasnego cant_angle tabele
aero.

Wyjscie (field_test_data/results/):
  - roll_rate_check_flight_<fno>.png        -- per lot (tryb 1)
  - roll_rate_factors_summary.csv           -- podsumowanie wszystkich lotow
  - roll_rate_factors_compare.png           -- (--fit) porownanie
                                                najlepszych CLL/Clp per lot

Uzycie:
    python check_roll_factors_all_flights.py
    python check_roll_factors_all_flights.py --no-rerun-datcom
    python check_roll_factors_all_flights.py --fit --no-rerun-datcom
    python check_roll_factors_all_flights.py --cll-scale 15.5 --clp-scale 3.0 19 20 21
"""

import sys
import csv
import copy
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, parse_telemetry, resolve_data_file
from run_6dof_cant_montecarlo import get_aero_for_cant, make_cant_case_name
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_geometry
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from core.state6 import State6DOF
from imu_reconstruction import detect_ignition, reconstruct_roll_from_ax_const_sign
from diag_drag import detect_events

from analyze_per_flight_6dof import read_flights, build_flight_thrust, build_scaled_mass
from plot_flight_trajectory_6dof import build_wind_steady, estimate_elevation_from_telemetry
from fit_roll_clp_cll import run_model_roll

DEFAULT_CLL_SCALE = 15.5
DEFAULT_CLP_SCALE = 3.0

# Siatka dla --fit -- szersza niz fit_roll_clp_cll.py bo nie wiadomo z
# gory czy inne loty potrzebuja podobnego rzedu wielkosci co lot 19.
FIT_CLL_SCALES = [1.0, 5.0, 10.0, 15.5, 25.0, 40.0]
FIT_CLP_SCALES = [1.0, 2.0, 3.0, 5.0]


def load_aero_offline_first(base, root, case_name, cant_deg, force_rerun):
    """Dla --no-rerun-datcom: probuje najpierw wczytac cache DOPASOWANY do
    cant_deg tego lotu (jesli istnieje z wczesniejszego prawdziwego
    przebiegu DATCOM), zamiast bazowego case'u (ktory get_aero_for_cant
    zwraca przy force_rerun=False niezaleznie od cant_deg -- patrz
    docstring get_aero_for_cant). Zwraca (aero, is_cant_matched)."""
    if force_rerun:
        aero, _ = get_aero_for_cant(case_name, cant_deg, force_rerun=True)
        return aero, True

    import pickle
    cant_case = make_cant_case_name(case_name, cant_deg)
    cant_pkl = Path(base).parent / "datcom_runs" / cant_case / "aero_table_missile.pkl"
    if cant_pkl.exists():
        with open(cant_pkl, "rb") as f:
            return pickle.load(f), True

    aero, _ = get_aero_for_cant(case_name, cant_deg, force_rerun=False)
    return aero, False


def build_flight_model_inputs(base, root, r, case_ostra, case_tepa, force_rerun):
    case = case_ostra if r["nose"] == "ostra" else case_tepa
    cfg_base = load_config(str(root / "configurations" / f"{case}.yaml"))
    t_ignition = cfg_base.propulsion.t_ignition

    aero, cant_matched = load_aero_offline_first(base, root, case, r["cant"], force_rerun)
    geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
    geom.cant_angle_rad = math.radians(r["cant"])
    mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
    atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)
    prop_flight = build_flight_thrust(base, r["fno"], t_ignition)
    if prop_flight is None:
        return None

    elev_deg, _ = estimate_elevation_from_telemetry(base, r["fno"])
    azimuth_deg = r["azimuth"]
    wind = build_wind_steady(base, r["fno"], azimuth_deg)

    return dict(aero=aero, cant_matched=cant_matched, geom=geom, mass=mass, atm=atm,
                gravity=gravity, launcher=launcher, prop=prop_flight, wind=wind,
                elev_deg=elev_deg, azimuth_deg=azimuth_deg)


def telemetry_roll(base, fno):
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    if not fpath.exists():
        return None
    tel = parse_telemetry(fpath, verbose=False)
    t_ign = detect_ignition(tel)
    i_ign, _, i_apo, i_end = detect_events(tel, t_ign)
    roll_rate = reconstruct_roll_from_ax_const_sign(tel, t_ign)
    if roll_rate is None:
        return None
    t_rel = tel.time - t_ign
    seg = slice(i_ign, i_end + 1)
    return dict(t=t_rel[seg], roll_rate=roll_rate[seg])


def rmse_vs_model(t_tel, rr_tel, t_model, p_model, t_min=0.7, t_max=30.0):
    mask = (t_tel >= t_min) & (t_tel <= min(t_max, t_tel[-1]))
    if mask.sum() < 5:
        return float("nan")
    p_interp = np.interp(t_tel[mask], t_model, p_model)
    return float(np.sqrt(np.mean((p_interp - rr_tel[mask]) ** 2)))


def fit_one_flight(inputs, t_fit, rr_fit, t_fit_max):
    best = None
    for cll_s in FIT_CLL_SCALES:
        for clp_s in FIT_CLP_SCALES:
            t_m, p_m = run_model_roll(
                inputs["aero"], inputs["geom"], inputs["mass"], inputs["atm"],
                inputs["gravity"], inputs["launcher"], inputs["prop"], inputs["wind"],
                inputs["elev_deg"], inputs["azimuth_deg"], cll_s, clp_s,
                t_max=t_fit_max + 2.0)
            p_interp = np.interp(t_fit, t_m, p_m)
            rmse = float(np.sqrt(np.mean((p_interp - rr_fit) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, cll_s, clp_s)
    return best  # (rmse, cll_scale, clp_scale)


def main():
    parser = argparse.ArgumentParser(
        description="Sprawdza czy CLL/Clp z lotu 19 generalizuja sie na inne loty")
    parser.add_argument("flights", type=int, nargs="*",
                         help="numery lotow (domyslnie: wszystkie z configs.txt, to analyze?=yes)")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--cll-scale", type=float, default=DEFAULT_CLL_SCALE)
    parser.add_argument("--clp-scale", type=float, default=DEFAULT_CLP_SCALE)
    parser.add_argument("--fit", action="store_true",
                         help="dodatkowo szuka niezaleznego najlepszego CLL/Clp per lot "
                             f"(siatka {len(FIT_CLL_SCALES)}x{len(FIT_CLP_SCALES)} -- kosztowne)")
    parser.add_argument("--t-fit-max", type=float, default=30.0)
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_flights = read_flights(base)
    if args.flights:
        flights = [r for r in all_flights if r["fno"] in args.flights]
    else:
        flights = all_flights

    summary = []
    print(f"{'flt':>3} {'cant':>5} {'nose':>5} {'cant_ok':>7} | "
          f"{'tel_peak':>9} {'tel_t':>6} | {'mod_peak':>9} {'mod_t':>6} {'RMSE_fix':>9}")
    print("-" * 78)

    for r in flights:
        fno = r["fno"]
        tel_roll = telemetry_roll(base, fno)
        if tel_roll is None:
            print(f"{fno:>3}  -- brak telemetrii lub kolumny AX, pomijam")
            continue

        inputs = build_flight_model_inputs(base, root, r, args.case_ostra, args.case_tepa,
                                            force_rerun=not args.no_rerun_datcom)
        if inputs is None:
            print(f"{fno:>3}  -- brak profilu ciagu, pomijam")
            continue

        t_tel, rr_tel = tel_roll["t"], tel_roll["roll_rate"]
        i_tel_peak = np.nanargmax(np.abs(rr_tel[t_tel >= 0.7])) if np.any(t_tel >= 0.7) else 0
        tel_peak = rr_tel[t_tel >= 0.7][i_tel_peak] if np.any(t_tel >= 0.7) else float("nan")
        tel_t_peak = t_tel[t_tel >= 0.7][i_tel_peak] if np.any(t_tel >= 0.7) else float("nan")

        t_fix, p_fix = run_model_roll(
            inputs["aero"], inputs["geom"], inputs["mass"], inputs["atm"],
            inputs["gravity"], inputs["launcher"], inputs["prop"], inputs["wind"],
            inputs["elev_deg"], inputs["azimuth_deg"], args.cll_scale, args.clp_scale,
            t_max=min(40.0, t_tel[-1] + 5.0))
        mod_i_peak = np.argmax(np.abs(p_fix))
        rmse_fix = rmse_vs_model(t_tel, rr_tel, t_fix, p_fix, t_max=args.t_fit_max)

        cant_ok = "yes" if inputs["cant_matched"] else "NIE-dopasowany"
        print(f"{fno:>3} {r['cant']:>5.2f} {r['nose']:>5} {cant_ok:>7} | "
              f"{tel_peak:>9.0f} {tel_t_peak:>6.2f} | "
              f"{p_fix[mod_i_peak]:>9.0f} {t_fix[mod_i_peak]:>6.2f} {rmse_fix:>9.0f}")

        row = dict(fno=fno, cant_deg=r["cant"], nose=r["nose"],
                   cant_matched=inputs["cant_matched"],
                   tel_peak=tel_peak, tel_t_peak=tel_t_peak,
                   model_peak_fixed=p_fix[mod_i_peak], model_t_peak_fixed=t_fix[mod_i_peak],
                   rmse_fixed=rmse_fix,
                   cll_scale_fixed=args.cll_scale, clp_scale_fixed=args.clp_scale)

        t_base, p_base = run_model_roll(
            inputs["aero"], inputs["geom"], inputs["mass"], inputs["atm"],
            inputs["gravity"], inputs["launcher"], inputs["prop"], inputs["wind"],
            inputs["elev_deg"], inputs["azimuth_deg"], 1.0, 1.0,
            t_max=min(40.0, t_tel[-1] + 5.0))

        best_cll = best_clp = best_rmse = float("nan")
        t_best = p_best = None
        if args.fit:
            fit_mask = (t_tel >= 0.7) & (t_tel <= args.t_fit_max)
            if r["cant"] == 0.0:
                print(f"     lot {fno}: cant=0 -- CLL_table ~0, pomijam dopasowanie CLL "
                      f"(oczekiwany model roll ~0 niezaleznie od CLL_scale)")
            elif fit_mask.sum() >= 5:
                best_rmse, best_cll, best_clp = fit_one_flight(
                    inputs, t_tel[fit_mask], rr_tel[fit_mask], args.t_fit_max)
                t_best, p_best = run_model_roll(
                    inputs["aero"], inputs["geom"], inputs["mass"], inputs["atm"],
                    inputs["gravity"], inputs["launcher"], inputs["prop"], inputs["wind"],
                    inputs["elev_deg"], inputs["azimuth_deg"], best_cll, best_clp,
                    t_max=min(40.0, t_tel[-1] + 5.0))
                print(f"     lot {fno}: najlepsze CLL x{best_cll:.1f}, Clp x{best_clp:.1f} "
                      f"(RMSE={best_rmse:.0f} deg/s)")
        row.update(cll_scale_best=best_cll, clp_scale_best=best_clp, rmse_best=best_rmse)
        summary.append(row)

        # --- wykres per lot ------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_tel[t_tel < 0.7], rr_tel[t_tel < 0.7], color="tab:orange", lw=0.8, alpha=0.35,
                label="telemetria AX (t<0.7s -- ponizej szumu)")
        ax.plot(t_tel[t_tel >= 0.7], rr_tel[t_tel >= 0.7], color="tab:orange", lw=1.5,
                label="telemetria AX (znak staly)")
        ax.plot(t_base, p_base, color="gray", lw=1.0, ls="--", label="model (CLL x1, Clp x1)")
        ax.plot(t_fix, p_fix, color="tab:blue", lw=1.8,
                label=f"model (CLL x{args.cll_scale:.1f}, Clp x{args.clp_scale:.1f} -- z lotu 19)")
        if t_best is not None:
            ax.plot(t_best, p_best, color="tab:green", lw=1.8, ls=":",
                    label=f"model (CLL x{best_cll:.1f}, Clp x{best_clp:.1f} -- najlepsze dla tego lotu)")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("czas od zaplonu [s]")
        ax.set_ylabel("predkosc obrotowa p [deg/s]")
        cant_note = "" if inputs["cant_matched"] else "  [UWAGA: aero NIE dopasowane do cant tego lotu]"
        ax.set_title(f"Lot {fno} (cant={r['cant']:.2f}°, {r['nose']}){cant_note}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = out_dir / f"roll_rate_check_flight_{fno}.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)

    if not summary:
        print("\nBrak lotow do podsumowania.")
        return

    csv_path = out_dir / "roll_rate_factors_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nZapisano: {csv_path}")

    if args.fit:
        fittable = [s for s in summary if s["cant_deg"] != 0.0 and not math.isnan(s["cll_scale_best"])]
        if fittable:
            fig, ax = plt.subplots(figsize=(8, 6))
            xs = [s["cll_scale_best"] for s in fittable]
            ys = [s["clp_scale_best"] for s in fittable]
            for s in fittable:
                ax.scatter(s["cll_scale_best"], s["clp_scale_best"], s=80)
                ax.annotate(f"lot {s['fno']}\ncant={s['cant_deg']:.2f}°",
                            (s["cll_scale_best"], s["clp_scale_best"]),
                            fontsize=8, xytext=(6, 6), textcoords="offset points")
            ax.scatter([args.cll_scale], [args.clp_scale], color="red", marker="*", s=250,
                       label=f"lot 19 (referencja: x{args.cll_scale:.1f}, x{args.clp_scale:.1f})")
            ax.set_xlabel("najlepszy CLL_scale")
            ax.set_ylabel("najlepszy Clp_scale")
            ax.set_title("Porownanie najlepszych wspolczynnikow CLL/Clp miedzy lotami\n"
                         "(blisko siebie = spojny, uogolnialny blad DATCOM; rozrzut = zalezny od lotu)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            out_png = out_dir / "roll_rate_factors_compare.png"
            fig.savefig(out_png, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"Zapisano: {out_png}")
        else:
            print("Brak lotow z cant!=0 i udanym dopasowaniem -- pomijam wykres porownawczy.")


if __name__ == "__main__":
    main()
