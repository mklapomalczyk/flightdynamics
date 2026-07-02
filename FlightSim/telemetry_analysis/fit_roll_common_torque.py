"""
fit_roll_common_torque.py
==========================
Testuje reframed hipoteze z check_roll_factors_all_flights.py: ze
dominujacym realnym zrodlem toczenia NIE jest moment od zaklinowania
pletw (cant_angle, CLL z DATCOM), tylko wspolne dla calej floty zrodlo
NIEZALEZNE od cant (np. asymetria/wir spalin silnika, tolerancja
produkcyjna pletw/korpusu) -- poszlaka: loty z cant=0 (13,17,18) pokazuja
realny roll POROWNYWALNY do lotow z cant!=0 (np. lot 19), co czysty
model cant-driven (CLL*cant_angle) nie moze wyjasnic, i skala nie rosnie
proporcjonalnie do cant_angle miedzy lotami (15/16 przy 1.6 st. NIE
pokazuja proporcjonalnie wiecej rolla niz 19 przy 0.6 st.).

Metoda: dopasowuje DWA WSPOLNE (dla wszystkich lotow naraz) parametry:

  - Cl0_manufacturing  -- STALY (niezalezny od cant_angle) wspolczynnik
                          momentu tocznego [-], patrz nowy hak w
                          force_model6.py (MA_roll_const =
                          Cl0_manufacturing * q_dyn * S_ref * d_ref).
                          Testowany zamiast CLL_scale (ktory pozostaje
                          x1.0 -- ufamy realnej z DATCOM, cant-driven
                          wartosci CLL_table takiej jaka jest, per lot).
  - Clp_scale          -- mnoznik tlumienia toczenia (jak wczesniej).

Ocena: RMSE per lot dla WSPOLNEGO (Cl0, Clp_scale) porownane z (a) modelem
bazowym (Cl0=0, Clp_scale=1 -- obecny stan), (b) niezaleznym
najlepszym dopasowaniem per lot z check_roll_factors_all_flights.py
(--fit), zeby ocenic ile zmiennosci miedzy lotami wyjasnia JEDEN wspolny
parametr.

UWAGA (znak): configs.txt zapisuje tylko WIELKOSC cant_angle, nie kierunek
fizycznego zaklinowania -- loty 15/16 pokazuja roll o PRZECIWNYM znaku niz
19/20/21 mimo tego samego typu dziobu ("ostra"), co sugeruje odwrocony
kierunek zaklinowania/montazu miedzy partiami rakiet. Pojedynczy wspolny
Cl0 (jeden znak) NIE moze dopasowac obu grup jednoczesnie -- raportowane
osobno per lot, nie ukrywane.

Nie modyfikuje configurations/*.yaml ani MAIN.py.

Uzycie:
    python fit_roll_common_torque.py --no-rerun-datcom
    python fit_roll_common_torque.py --no-rerun-datcom 13 17 18   # tylko cant=0
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
from core.state6 import State6DOF
from core.solver6 import run_simulation_6dof
from forces.force_model6 import ForceModel6DOF
from analyze_per_flight_6dof import read_flights

from check_roll_factors_all_flights import (
    build_flight_model_inputs, telemetry_roll, rmse_vs_model,
)

CL0_VALUES = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
CLP_SCALES = [1.0, 2.0, 3.0]


def run_with_cl0(inputs, cl0, clp_scale, t_max):
    aero = inputs["aero"]
    orig_clp = aero.Clp_table
    orig_cl0 = getattr(aero, "Cl0_manufacturing", 0.0)
    try:
        if aero.Clp_table is not None:
            aero.Clp_table = orig_clp * clp_scale
        aero.Cl0_manufacturing = cl0
        st = State6DOF.initial(elevation_deg=inputs["elev_deg"], azimuth_deg=inputs["azimuth_deg"])
        fm = ForceModel6DOF(atmosphere=inputs["atm"], mass_model=inputs["mass"], aero_model=aero,
                             gravity=inputs["gravity"], geometry=inputs["geom"], propulsion=inputs["prop"],
                             launcher=inputs["launcher"], wind_model=inputs["wind"])
        result = run_simulation_6dof(fm, st, t_max=t_max, dt_output=0.05,
                                      rtol=1e-6, atol=1e-8, max_step=0.05)
        return result.t, np.degrees(result.p)
    finally:
        aero.Clp_table = orig_clp
        aero.Cl0_manufacturing = orig_cl0


def main():
    parser = argparse.ArgumentParser(
        description="Dopasowanie WSPOLNEGO (cant-niezaleznego) momentu tocznego do wszystkich lotow")
    parser.add_argument("flights", type=int, nargs="*")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--t-fit-max", type=float, default=25.0)
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_flights = read_flights(base)
    flights = [r for r in all_flights if (not args.flights or r["fno"] in args.flights)]

    flight_data = []
    for r in flights:
        tel_roll = telemetry_roll(base, r["fno"])
        if tel_roll is None:
            print(f"Lot {r['fno']}: brak telemetrii/AX -- pomijam.")
            continue
        inputs = build_flight_model_inputs(base, root, r, args.case_ostra, args.case_tepa,
                                            force_rerun=not args.no_rerun_datcom)
        if inputs is None:
            print(f"Lot {r['fno']}: brak profilu ciagu -- pomijam.")
            continue
        t_tel, rr_tel = tel_roll["t"], tel_roll["roll_rate"]
        fit_mask = (t_tel >= 0.7) & (t_tel <= args.t_fit_max)
        if fit_mask.sum() < 5:
            print(f"Lot {r['fno']}: za malo probek w oknie dopasowania -- pomijam.")
            continue
        flight_data.append(dict(fno=r["fno"], cant=r["cant"], nose=r["nose"],
                                 inputs=inputs, t_fit=t_tel[fit_mask], rr_fit=rr_tel[fit_mask]))

    if not flight_data:
        print("Brak lotow do dopasowania.")
        return

    print(f"Przeszukiwanie WSPOLNEGO (Cl0, Clp_scale) dla {len(flight_data)} lotow: "
          f"{len(CL0_VALUES)}x{len(CLP_SCALES)} kombinacji x {len(flight_data)} lotow = "
          f"{len(CL0_VALUES)*len(CLP_SCALES)*len(flight_data)} symulacji...")

    best = None
    all_results = {}
    for cl0 in CL0_VALUES:
        for clp_s in CLP_SCALES:
            per_flight_rmse = []
            for fd in flight_data:
                t_m, p_m = run_with_cl0(fd["inputs"], cl0, clp_s, t_max=args.t_fit_max + 2.0)
                rmse = rmse_vs_model(fd["t_fit"], fd["rr_fit"], t_m, p_m,
                                      t_min=fd["t_fit"][0], t_max=args.t_fit_max)
                per_flight_rmse.append(rmse)
            pooled = float(np.mean(per_flight_rmse))
            all_results[(cl0, clp_s)] = per_flight_rmse
            print(f"  Cl0={cl0:+.3f}  Clp x{clp_s:.1f}  ->  RMSE pooled(mean)={pooled:7.0f}  "
                  f"per-lot={['%.0f'%v for v in per_flight_rmse]}")
            if best is None or pooled < best[0]:
                best = (pooled, cl0, clp_s)

    pooled_best, cl0_best, clp_best = best
    print(f"\nNajlepszy WSPOLNY: Cl0={cl0_best:+.3f}, Clp x{clp_best:.1f} "
          f"(RMSE pooled mean={pooled_best:.0f} deg/s)")

    # --- podsumowanie per lot: wspolny vs bazowy ------------------------- #
    summary = []
    print(f"\n{'flt':>3} {'cant':>5} | {'RMSE bazowy':>12} {'RMSE wspolny':>13}")
    for fd in flight_data:
        t_base, p_base = run_with_cl0(fd["inputs"], 0.0, 1.0, t_max=args.t_fit_max + 2.0)
        rmse_base = rmse_vs_model(fd["t_fit"], fd["rr_fit"], t_base, p_base,
                                   t_min=fd["t_fit"][0], t_max=args.t_fit_max)
        rmse_common = all_results[(cl0_best, clp_best)][flight_data.index(fd)]
        print(f"{fd['fno']:>3} {fd['cant']:>5.2f} | {rmse_base:>12.0f} {rmse_common:>13.0f}")
        summary.append(dict(fno=fd["fno"], cant_deg=fd["cant"], nose=fd["nose"],
                             rmse_baseline=rmse_base, rmse_common_cl0=rmse_common,
                             cl0_common=cl0_best, clp_scale_common=clp_best))

    csv_path = out_dir / "roll_common_torque_fit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nZapisano: {csv_path}")

    # --- wykresy per lot: telemetria vs wspolny model -------------------- #
    for fd in flight_data:
        t_m, p_m = run_with_cl0(fd["inputs"], cl0_best, clp_best, t_max=40.0)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(fd["t_fit"], fd["rr_fit"], color="tab:orange", lw=1.3, label="telemetria AX")
        ax.plot(t_m, p_m, color="tab:blue", lw=1.6,
                label=f"model (Cl0={cl0_best:+.3f}, Clp x{clp_best:.1f} -- wspolny)")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("czas od zaplonu [s]")
        ax.set_ylabel("predkosc obrotowa p [deg/s]")
        ax.set_title(f"Lot {fd['fno']} (cant={fd['cant']:.2f}°) -- wspolny model wszystkich lotow")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = out_dir / f"roll_common_torque_flight_{fd['fno']}.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"Zapisano wykresy per lot: roll_common_torque_flight_<fno>.png")


if __name__ == "__main__":
    main()
