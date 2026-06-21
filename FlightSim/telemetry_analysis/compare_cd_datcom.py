"""
compare_cd_datcom.py
=====================
Porownanie Cd(Ma) z modelu DATCOM (aero_table_missile.pkl, CA przy
alpha=0) z Cd(Ma) wyznaczonym z testow polowych, dla wybranej
konfiguracji nosa (tepa/ostra) i zakresu Mach.

Porownanie bin-do-bin (NIE mean-vs-mean):
  - Dla kazdego lotu w wybranej grupie licza sie punkty Cd(Ma) z fazy
    znizania (metoda GPS+baro, diag_drag.cd_on), scalone ze wszystkich
    lotow w grupie.
  - Te punkty sa binowane na siatce Ma (krok --ma_step), dajac
    median+std per bin (rozrzut miedzy-lotowy i wewnatrz-lotowy).
  - DATCOM CA(alpha=0) interpolowane na te sama siatke Ma.
  - Wynik: czy DATCOM lezy w paśmie błędu danych polowych, czy poza nim.

Uzycie:
    python compare_cd_datcom.py
    python compare_cd_datcom.py --nose tepa --ma_lo 0.2 --ma_hi 0.5
    python compare_cd_datcom.py --case rocket_70mm_baseline --flights 14 15 16
"""

import sys
import csv
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import parse_telemetry, get_data_dir
from diag_drag import load_config, detect_events, baro_altitude, cd_on, D_CAL
from imu_reconstruction import detect_ignition
from run_all_flights import read_configs, check_data_exists

S = np.pi * D_CAL**2 / 4.0


# --------------------------------------------------------------------------
def flights_for_nose(base, nose):
    """Numery lotow 'to analyze?==yes' o danym typie nosa (z configs.txt)."""
    rows = read_configs(base)
    return [r["fno"] for r in rows
            if r["nose"] == nose and check_data_exists(base, r["fno"])]


def collect_field_cd(base, flights):
    """Zbiera punkty Cd(Ma) z fazy znizania dla wszystkich podanych lotow."""
    Ma_all, Cd_all = [], []
    for fno in flights:
        fpath = Path(base) / f"ARTEMIDA_{fno}_LOT.txt"
        if not fpath.exists():
            continue
        cfg = load_config(fno, base)
        if cfg is None:
            continue
        tel = parse_telemetry(fpath, verbose=False)
        t_ign = detect_ignition(tel)
        i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign)
        t = tel.time
        des = (np.arange(len(t)) >= i_apo) & (np.arange(len(t)) <= i_end)
        if np.sum(des) < 20:
            continue
        h_baro = baro_altitude(tel, i_ign)
        D = cd_on(tel, des, cfg["m_coast"], S, h_full=h_baro)

        t_apo = t[i_apo]
        valid = (D['t'] > t_apo + 2.0) & np.isfinite(D['Cd']) & (D['q'] > 100)
        chute_mask = np.abs(D['dVh_dt']) > 10.0
        if np.any(chute_mask[valid]):
            i_chute = np.where(valid & chute_mask)[0][0]
            valid = valid & (D['t'] < D['t'][i_chute])

        if np.sum(valid) == 0:
            print(f"  lot {fno}: 0 pkt — pomijam")
            continue
        Ma_all.append(D['M'][valid])
        Cd_all.append(D['Cd'][valid])
        print(f"  lot {fno}: {np.sum(valid)} pkt (Ma {D['M'][valid].min():.2f}-{D['M'][valid].max():.2f})")

    if not Ma_all:
        raise RuntimeError("Brak punktow Cd z danych polowych dla podanych lotow.")
    return np.concatenate(Ma_all), np.concatenate(Cd_all)


def bin_field_cd(Ma, Cd, lo, hi, step, min_n=5):
    edges = np.arange(lo, hi + step, step)
    centers, meds, stds, ns = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (Ma >= a) & (Ma < b) & np.isfinite(Cd)
        if np.sum(sel) >= min_n:
            centers.append(0.5 * (a + b))
            meds.append(np.median(Cd[sel]))
            stds.append(np.std(Cd[sel]))
            ns.append(int(np.sum(sel)))
    return np.array(centers), np.array(meds), np.array(stds), np.array(ns)


# --------------------------------------------------------------------------
def load_datcom_ca0(pkl_path):
    with open(pkl_path, "rb") as f:
        a = pickle.load(f)
    alpha = np.array(a.alpha_table)
    mach = np.array(a.mach_table)
    CA = np.array(a.CA_table)   # shape (n_alpha, n_mach)
    ia0 = int(np.argmin(np.abs(alpha)))
    return mach, CA[ia0]


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DATCOM vs field-test Cd(Ma) comparison")
    parser.add_argument("--case", default="rocket_70mm_baseline",
                        help="nazwa konfiguracji DATCOM (katalog w datcom_runs/)")
    parser.add_argument("--nose", choices=["ostra", "tepa"], default="ostra",
                        help="typ nosa (grupa lotow z configs.txt) do porownania")
    parser.add_argument("--flights", nargs="*", type=int, default=None,
                        help="recznie wybrane numery lotow (domyslnie wszystkie z --nose)")
    parser.add_argument("--ma_lo", type=float, default=0.20)
    parser.add_argument("--ma_hi", type=float, default=0.45)
    parser.add_argument("--ma_step", type=float, default=0.025)
    parser.add_argument("--min_n", type=int, default=5,
                        help="minimalna liczba punktow w binie Mach, by go uwzglednic")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent   # field_test_data -> FlightSim
    pkl_path = root / "datcom_runs" / args.case / "aero_table_missile.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Brak {pkl_path}. Uruchom MAIN.py (z odpowiednim CASE_NAME) "
            f"najpierw, aby wygenerowac cache DATCOM.")

    flights = args.flights if args.flights else flights_for_nose(base, args.nose)
    if not flights:
        raise RuntimeError(f"Brak lotow dla nosa '{args.nose}' w configs.txt.")

    print(f"Konfiguracja DATCOM: {args.case}   Nos: {args.nose}   Loty: {flights}")
    print("Zbieranie punktow Cd(Ma) z testow polowych:")
    Ma, Cd = collect_field_cd(base, flights)
    centers, meds, stds, ns = bin_field_cd(Ma, Cd, args.ma_lo, args.ma_hi, args.ma_step, args.min_n)
    if len(centers) == 0:
        raise RuntimeError("Brak binow Mach z wystarczajaca liczba punktow — zmniejsz --min_n lub poszerz zakres.")

    mach_datcom, ca0_datcom = load_datcom_ca0(pkl_path)
    ca0_at_centers = np.interp(centers, mach_datcom, ca0_datcom)

    print(f"\n{'Mach':>6} {'Cd_pole':>8} {'std':>6} {'n':>5} {'Cd_DATCOM':>10} {'diff':>7} {'wewn. band?':>12}")
    rows = []
    for c, m, s, n, d in zip(centers, meds, stds, ns, ca0_at_centers):
        diff = d - m
        inside = "TAK" if abs(diff) <= s else "NIE"
        print(f"{c:6.3f} {m:8.4f} {s:6.4f} {n:5d} {d:10.4f} {diff:+7.4f} {inside:>12}")
        rows.append(dict(Mach=round(float(c), 4), Cd_field_med=round(float(m), 4),
                          Cd_field_std=round(float(s), 4), n=int(n),
                          Cd_datcom=round(float(d), 4), diff=round(float(diff), 4),
                          inside_band=inside))

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.nose}"
    csv_path = out_dir / f"cd_datcom_vs_field_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nZapisano: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(centers, meds, yerr=stds, fmt='o-', ms=7, capsize=4, lw=1.5,
                 color='tab:blue', label=f"testy polowe — '{args.nose}' (mediana ± sd, bin-do-bin)")
    ax.plot(mach_datcom, ca0_datcom, 's--', ms=6, lw=1.5, color='tab:red',
             label=f"DATCOM CA(alpha=0) — {args.case}")
    ax.set_xlim(args.ma_lo - 0.02, args.ma_hi + 0.1)
    ax.set_xlabel("Mach")
    ax.set_ylabel("Cd")
    ax.set_title(f"Cd(Ma) — DATCOM vs testy polowe ('{args.nose}'), zakres Ma {args.ma_lo}-{args.ma_hi}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / f"cd_datcom_vs_field_{tag}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")

    n_inside = sum(1 for r in rows if r["inside_band"] == "TAK")
    print(f"\n{n_inside}/{len(rows)} binow Mach: DATCOM lezy w pasmie ±1 std danych polowych.")


if __name__ == "__main__":
    main()
