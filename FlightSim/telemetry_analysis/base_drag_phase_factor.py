"""
base_drag_phase_factor.py
==========================
Hipoteza: Cd jest wyzszy w fazie bezsilowej (coast) niz w fazie spalania,
bo plomien spalin wypelnia denko podczas pracy silnika i eliminuje opor
denny (base drag); po wypaleniu (i jeszcze chwile po, przy resztkowym
ciagu/cisnieniu w komorze) denko jest "otwarte" na podciśnienie sladu
i base drag wraca w pelni.

Metoda — NIEZALEZNA od pomiarow ciagu/Cd z telemetrii (zero krazenia
danych, w przeciwienstwie do liczenia Cd ze znanego ciagu, ktory sam
pochodzi z Cd):
  1. Wzor Fleemana na opor denny (datcom_io/barrowman.py _CA_base) zalezy
     od geometrii dyszy: xi_s = 1 - (d_e/d)^2 / xi_k.
        - "bezsilowa" (unpowered): d_e = rzeczywista srednica dyszy z YAML
          (denko otwarte poza otworem dyszy) — to jest dokladnie
          standardowy wzor Fleemana, uzywany domyslnie.
        - "zasilana" (powered): plomien wypelnia CALE denko -> brak
          podciśnienia na denku -> CA_base = 0 (d_e = d_k, xi_s = 0).
  2. Reszta skladowych CA (tarcie kadluba/statecznikow, opor falowy)
     NIE zalezy od stanu silnika — zmienia sie tylko CA_base.
  3. ratio(M) = CA_total_powered(M) / CA_total_unpowered(M)  (Fleeman, calosc)
  4. CA_datcom_unpowered(M) = CA_datcom(M)              (tabela DATCOM, jak dotychczas)
     CA_datcom_powered(M)   = CA_datcom(M) * ratio(M)   (skalowana o ten sam
                                                          wzgledny spadek, ktory
                                                          przewiduje Fleeman)
  5. Porownanie: obie krzywe DATCOM (powered/unpowered) na tle zbinowanego
     Cd(Ma) z fazy zniżania (testy polowe) — "unpowered" powinno pasowac
     tak jak dotychczas (compare_cd_datcom.py); "powered" jest krzywa,
     ktora proponujemy uzywac w symulatorze podczas spalania (+ ewentualne
     okno resztkowe po wypaleniu).

Nie modyfikuje configurations/*.yaml ani MAIN.py — czyta tylko YAML i
istniejacy cache DATCOM (aero_table_missile.pkl).

Uzycie:
    python base_drag_phase_factor.py
    python base_drag_phase_factor.py --case rocket_70mm_baseline --nose ostra
"""

import sys
import csv
import pickle
import argparse
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datcom_io.config_reader import load_config as load_yaml_config
from datcom_io.barrowman import FleemanCalculator
from compare_cd_datcom import flights_for_nose, collect_field_cd, bin_field_cd


# --------------------------------------------------------------------------
def fleeman_ca_total(calc, mach, d_e):
    """CA total (alpha=0) z Fleemana dla podanej srednicy dyszy d_e —
    tylko CA_base zalezy od d_e, reszta skladowych jest wspolna."""
    calc.d_e = d_e
    CA_fb, CA_ff, CA_wb, CA_wf, CA_base = calc._CA(mach)
    return CA_fb + CA_ff + CA_wb + CA_wf + CA_base


def base_drag_ratio(yaml_path, mach_grid):
    """ratio(M) = CA_total('plomien wypelnia denko') / CA_total('denko z otworem dyszy')."""
    cfg  = load_yaml_config(yaml_path)
    calc = FleemanCalculator(cfg)

    d_e_unpowered = cfg.propulsion.nozzle_diameter if cfg.propulsion else 0.0
    d_e_powered   = calc.d_k   # denko w calosci "zaslepione" plomieniem -> xi_s=0 -> CA_base=0

    ratio = np.array([
        fleeman_ca_total(calc, m, d_e_powered) / fleeman_ca_total(calc, m, d_e_unpowered)
        for m in mach_grid
    ])
    return ratio


def load_datcom_ca0(pkl_path):
    with open(pkl_path, "rb") as f:
        a = pickle.load(f)
    alpha = np.array(a.alpha_table)
    mach  = np.array(a.mach_table)
    CA    = np.array(a.CA_table)
    ia0   = int(np.argmin(np.abs(alpha)))
    return mach, CA[ia0]


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fleeman base-drag powered/unpowered ratio applied to DATCOM Cd(Ma)")
    parser.add_argument("--case", default="rocket_70mm_baseline")
    parser.add_argument("--nose", choices=["ostra", "tepa"], default="ostra")
    parser.add_argument("--ma_lo", type=float, default=0.20)
    parser.add_argument("--ma_hi", type=float, default=0.45)
    parser.add_argument("--ma_step", type=float, default=0.025)
    parser.add_argument("--min_n", type=int, default=5)
    args = parser.parse_args()

    from telemetry_parser import get_data_dir
    base      = get_data_dir()
    root      = Path(base).parent
    yaml_path = root / "configurations" / f"{args.case}.yaml"
    pkl_path  = root / "datcom_runs" / args.case / "aero_table_missile.pkl"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Brak {yaml_path}.")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Brak {pkl_path}. Uruchom MAIN.py najpierw.")

    mach_datcom, ca0_datcom = load_datcom_ca0(pkl_path)
    ratio = base_drag_ratio(yaml_path, mach_datcom)

    ca0_unpowered = ca0_datcom
    ca0_powered   = ca0_datcom * ratio

    print(f"Konfiguracja: {args.case}")
    print(f"{'Mach':>6} {'CA_unpowered':>13} {'ratio_Fleeman':>14} {'CA_powered':>11}")
    for m, u, r, p in zip(mach_datcom, ca0_unpowered, ratio, ca0_powered):
        print(f"{m:6.3f} {u:13.4f} {r:14.4f} {p:11.4f}")

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"base_drag_phase_factor_{args.case}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Mach", "CA_datcom_unpowered", "ratio_fleeman_powered_over_unpowered", "CA_datcom_powered"])
        for m, u, r, p in zip(mach_datcom, ca0_unpowered, ratio, ca0_powered):
            writer.writerow([round(float(m), 4), round(float(u), 4), round(float(r), 4), round(float(p), 4)])
    print(f"\nZapisano: {csv_path}")

    # Punkty Cd z testow polowych (faza zniżania -> reprezentuje unpowered)
    flights = flights_for_nose(base, args.nose)
    print(f"\nZbieranie punktow Cd(Ma) z testow polowych (faza zniżania, nos '{args.nose}'):")
    Ma, Cd = collect_field_cd(base, flights)
    centers, meds, stds, ns = bin_field_cd(Ma, Cd, args.ma_lo, args.ma_hi, args.ma_step, args.min_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mach_datcom, ca0_unpowered, 's--', ms=6, lw=1.5, color='tab:red',
            label="DATCOM — bezsilowa (denko z otworem dyszy)")
    ax.plot(mach_datcom, ca0_powered, 'o-', ms=6, lw=1.5, color='tab:orange',
            label="DATCOM × ratio Fleeman — zasilana (plomien zaslania denko)")
    if len(centers) > 0:
        ax.errorbar(centers, meds, yerr=stds, fmt='o', ms=7, capsize=4, lw=1.5,
                     color='tab:blue', label=f"testy polowe (zniżanie, bezsilowa) — '{args.nose}'")
    ax.set_xlabel("Mach")
    ax.set_ylabel("Cd")
    ax.set_title(f"Base drag: zasilana vs bezsilowa (Fleeman) — {args.case}, nos '{args.nose}'")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / f"base_drag_phase_factor_{args.case}_{args.nose}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
