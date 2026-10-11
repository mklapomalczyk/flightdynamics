"""
roll_dynamics.py
================
Ekstrakcja zaleznosci predkosci obrotowej (roll) od kata pochylenia
plaszczyzn statecznikow (cant angle), z danych zyroskopu gyro_x.

Teoria (klasyczna rownowaga spinu rakiety ze statecznikami pochylonymi
o kat cant):
    przy ustalonym spinie (rownowaga momentu napedzajacego od statecznikow
    i momentu tlumiacego aerodynamicznego) bezwymiarowa predkosc obrotowa
        p* = p * d / (2V)
    jest w przyblizeniu stala podczas wznoszenia/coastu przy danym kacie
    cant i zalezy w przyblizeniu LINIOWO od kata cant (dla malych katow):
        p* = k_roll * cant_deg
    gdzie k_roll zawiera w sobie Cl_delta/Clp (sprzezenie momentu od
    pochylenia statecznikow z tlumieniem aerodynamicznym rakiety).

Procedura:
  1. Dla kazdego lotu z configs.txt (to analyze? = yes) wyznacz p* w fazie
     coastu (po burnoucie, przed apogeum) — gdzie ciag = 0 i predkosc
     calkowita V jest znana z GPS (Vh) + baro (Vz), rownowaga spinu
     najbardziej stabilna.
  2. Zagreguj p*_med per lot (mediana w oknie coast), z odchyleniem std.
  3. Dopasuj liniowa regresje p*_med = k_roll * cant_deg (przez 0,0 —
     brak cant => brak wymuszonego spinu) metoda najmniejszych kwadratow.
  4. Zapisz CSV per-lot i wykres p* vs cant z dopasowana prosta.

Uzycie:
    python roll_dynamics.py                 # wszystkie loty z configs.txt
    python roll_dynamics.py 13 14 17 18     # wybrane loty
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_parser import parse_telemetry, get_data_dir
from imu_reconstruction import detect_ignition
from diag_drag import (load_config, isa, detect_events, baro_altitude,
                        smooth_gps_derivative, D_CAL)
from run_all_flights import read_configs, check_data_exists

D_CAL_M = D_CAL   # srednica rakiety [m], do normalizacji p*


# --------------------------------------------------------------------------
def process_flight(fno, cfg, base):
    """
    Wyznacza p* = p*d/(2V) w fazie coastu (po burnoucie, przed apogeum).
    Zwraca dict z polami: flight_no, cant, t, p_star, p_star_med, p_star_std
    lub None jesli nieudane.
    """
    fpath = Path(base) / f"ARTEMIDA_{fno}_LOT.txt"
    tel = parse_telemetry(fpath, verbose=False)

    t_ign = detect_ignition(tel)
    i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign)
    t = tel.time

    # Faza coastu: od burnoutu (+0.3s zapas na przejscie) do apogeum
    t_start = t[i_bo] + 0.3
    i_start = int(np.argmin(np.abs(t - t_start)))
    if i_start >= i_apo - 10:
        print(f"[LOT {fno}] zbyt krotki coast — pomijam")
        return None

    coast = np.arange(i_start, i_apo)

    # Predkosc calkowita w coastcie: Vh z GPS, Vz z baro (lepsza rozdzielczosc
    # niz GPS przy stosunkowo wolnym wznoszeniu po burnoucie)
    h_baro = baro_altitude(tel, i_ign)
    Vz_full = smooth_gps_derivative(h_baro, t, window_s=0.3)
    Vh = tel.vel_onboard[coast]
    Vz = Vz_full[coast]
    V_total = np.sqrt(Vh**2 + Vz**2)

    p_deg_s = tel.gyro_x[coast]               # predkosc obrotowa [st/s]
    p_rad_s = np.radians(p_deg_s)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_star = p_rad_s * D_CAL_M / (2.0 * V_total)

    valid = np.isfinite(p_star) & (V_total > 10.0)
    if np.sum(valid) < 10:
        print(f"[LOT {fno}] za malo waznych probek p* — pomijam")
        return None

    p_star_v = p_star[valid]
    p_star_med = float(np.median(p_star_v))
    p_star_std = float(np.std(p_star_v))

    print(f"[LOT {fno}] cant={cfg['cant']:.1f}deg  "
          f"p*_med={p_star_med:+.4f}  p*_std={p_star_std:.4f}  "
          f"n={np.sum(valid)}  V_med={np.median(V_total[valid]):.1f}m/s")

    return dict(
        flight_no=fno, cant=cfg["cant"], nose=cfg.get("head", "?"),
        t=t[coast][valid], p_star=p_star_v,
        p_star_med=p_star_med, p_star_std=p_star_std,
        n=int(np.sum(valid)),
    )


# --------------------------------------------------------------------------
def fit_roll_slope(results):
    """
    Dopasowuje p*_med = k_roll * cant_deg (przez 0,0) metoda najmniejszych
    kwadratow, wazone odwrotnoscia wariancji (1/std^2) gdy dostepne.

    Uzywamy wartosci absolutnej cant i znaku p* zgodnego z kierunkiem
    pochylenia statecznikow (zaplikowane z configs.txt jako wartosc cant,
    zakladamy ten sam kierunek montazu dla wszystkich lotow w grupie).
    """
    cants = np.array([r["cant"] for r in results])
    pstars = np.array([r["p_star_med"] for r in results])
    stds = np.array([max(r["p_star_std"], 1e-4) for r in results])
    w = 1.0 / stds**2

    # regresja przez 0: k = sum(w*cant*pstar)/sum(w*cant^2)
    num = np.sum(w * cants * pstars)
    den = np.sum(w * cants**2)
    if den < 1e-9:
        return None, cants, pstars
    k_roll = num / den

    resid = pstars - k_roll * cants
    ss_res = np.sum(w * resid**2)
    ss_tot = np.sum(w * (pstars - np.average(pstars, weights=w))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    return k_roll, cants, pstars, r2


# --------------------------------------------------------------------------
def save_csv(results, out_dir):
    csv_path = Path(out_dir) / "roll_dynamics_summary.csv"
    fieldnames = ["flight_no", "nose", "cant_deg", "p_star_med", "p_star_std", "n"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda r: r["flight_no"]):
            writer.writerow(dict(
                flight_no=r["flight_no"], nose=r["nose"], cant_deg=r["cant"],
                p_star_med=round(r["p_star_med"], 5),
                p_star_std=round(r["p_star_std"], 5), n=r["n"],
            ))
    print(f"Zapisano: {csv_path}")
    return csv_path


def plot_results(results, k_roll, cants, pstars, r2, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        ax.errorbar(r["cant"], r["p_star_med"], yerr=r["p_star_std"],
                     fmt='o', ms=8, capsize=4, label=f"lot {r['flight_no']}")
    if k_roll is not None:
        cant_grid = np.linspace(0, max(cants.max(), 0.1) * 1.1, 50)
        ax.plot(cant_grid, k_roll * cant_grid, 'k--', lw=1.5,
                 label=f"dopasowanie: p*={k_roll:.4f}*cant  (R²={r2:.3f})")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("Kat pochylenia statecznikow (cant) [deg]")
    ax.set_ylabel("Bezwymiarowa predkosc obrotowa  p* = p·d/(2V)")
    ax.set_title("Roll dynamics — p* vs cant angle (faza coastu)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_png = Path(out_dir) / "roll_dynamics.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Roll-rate vs cant-angle analysis")
    parser.add_argument("flights", nargs="*", type=int,
                        help="numery lotow (domyslnie wszystkie z configs.txt)")
    args = parser.parse_args()

    base = get_data_dir()
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_configs(base)
    if args.flights:
        rows = [r for r in rows if r["fno"] in args.flights]
    rows = [r for r in rows if check_data_exists(base, r["fno"])]

    results = []
    for r in rows:
        cfg = dict(
            cant=r["cant"], azimuth=r["azimuth"], elevation=r["elevation"],
            head=r["head_raw"], m_rocket=r["m_rocket"],
            m_propellant=r["m_propellant"], m_coast=r["m_coast"],
        )
        res = process_flight(r["fno"], cfg, base)
        if res is not None:
            results.append(res)

    if not results:
        print("Brak wynikow.")
        return

    out = fit_roll_slope(results)
    if out[0] is None:
        k_roll, cants, pstars = out[1], out[2], None
        r2 = float("nan")
    else:
        k_roll, cants, pstars, r2 = out

    print(f"\nDopasowanie: p* = {k_roll:.4f} * cant_deg   (R²={r2:.3f}, n_lotow={len(results)})")

    save_csv(results, out_dir)
    plot_results(results, k_roll, cants, pstars, r2, out_dir)


if __name__ == "__main__":
    main()
