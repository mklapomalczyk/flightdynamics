"""
compare_cd_datcom.py
=====================
Porownanie Cd(Ma) z DATCOM (aero_table_missile.pkl, CA przy alpha=0)
z Cd(Ma) wyznaczonym z testow polowych dla konfiguracji "ostra",
w zakresie Ma 0.2-0.45 (gdzie mamy najwiecej wiarygodnych danych
polowych, patrz fit_cd_mach_curves.csv / diag_drag).

Porownanie bin-do-bin (NIE mean-vs-mean):
  - Dla kazdego lotu "ostra" licza sie punkty Cd(Ma) z fazy znizania
    (metoda GPS+baro, diag_drag.cd_on), scalone ze wszystkich lotow.
  - Te punkty sa binowane na siatce Ma (krok 0.025, jak w fit_cd_mach),
    dajac median+std per bin (rozrzut miedzy-lotowy I wewnatrz-lotowy).
  - DATCOM CA(alpha=0) interpolowane na te sama siatke Ma.
  - Wynik: czy DATCOM lezy w paśmie błędu danych polowych, czy poza nim.

Uzycie:
    python compare_cd_datcom.py
"""

import sys
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import parse_telemetry, get_data_dir
from diag_drag import load_config, detect_events, baro_altitude, cd_on, bin_stats
from imu_reconstruction import detect_ignition

MA_LO, MA_HI, MA_STEP = 0.20, 0.45, 0.025
OSTRA_FLIGHTS = [14, 15, 16, 18, 19, 20, 21]
DATCOM_PKL = "datcom_runs/rocket_70mm_baseline/aero_table_missile.pkl"


# --------------------------------------------------------------------------
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
        D = cd_on(tel, des, cfg["m_coast"], np.pi * 0.070**2 / 4.0, h_full=h_baro)

        t_apo = t[i_apo]
        valid = (D['t'] > t_apo + 2.0) & np.isfinite(D['Cd']) & (D['q'] > 100)
        chute_mask = np.abs(D['dVh_dt']) > 10.0
        if np.any(chute_mask[valid]):
            i_chute = np.where(valid & chute_mask)[0][0]
            valid = valid & (D['t'] < D['t'][i_chute])

        Ma_all.append(D['M'][valid])
        Cd_all.append(D['Cd'][valid])
        print(f"  lot {fno}: {np.sum(valid)} pkt (Ma {D['M'][valid].min():.2f}-{D['M'][valid].max():.2f})")

    return np.concatenate(Ma_all), np.concatenate(Cd_all)


def bin_field_cd(Ma, Cd, lo=MA_LO, hi=MA_HI, step=MA_STEP):
    edges = np.arange(lo, hi + step, step)
    centers, meds, stds, ns = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (Ma >= a) & (Ma < b) & np.isfinite(Cd)
        if np.sum(sel) >= 5:
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
    base = get_data_dir()
    root = Path(base).parent   # field_test_data -> FlightSim
    pkl_path = root / DATCOM_PKL
    if not pkl_path.exists():
        raise FileNotFoundError(f"Brak {pkl_path}. Uruchom MAIN.py najpierw (generuje cache DATCOM).")

    print("Zbieranie punktow Cd(Ma) z testow polowych — konfiguracja 'ostra':")
    Ma, Cd = collect_field_cd(base, OSTRA_FLIGHTS)
    centers, meds, stds, ns = bin_field_cd(Ma, Cd)

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
    csv_path = out_dir / "cd_datcom_vs_field_ostra.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nZapisano: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(centers, meds, yerr=stds, fmt='o-', ms=7, capsize=4, lw=1.5,
                 color='tab:blue', label="testy polowe — 'ostra' (mediana ± sd, bin-do-bin)")
    ax.plot(mach_datcom, ca0_datcom, 's--', ms=6, lw=1.5, color='tab:red',
             label="DATCOM CA(alpha=0)")
    ax.set_xlim(MA_LO - 0.02, MA_HI + 0.1)
    ax.set_xlabel("Mach")
    ax.set_ylabel("Cd")
    ax.set_title("Cd(Ma) — DATCOM vs testy polowe ('ostra'), zakres Ma 0.2-0.45")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / "cd_datcom_vs_field_ostra.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")

    n_inside = sum(1 for r in rows if r["inside_band"] == "TAK")
    print(f"\n{n_inside}/{len(rows)} binow Mach: DATCOM lezy w pasmie ±1 std danych polowych.")


if __name__ == "__main__":
    main()
