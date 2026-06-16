"""
run_all_flights.py
==================
Batch drag validation across all flights listed in configs.txt.

Steps:
  1. Read configs.txt, filter flights with 'to analyze? == yes'
     and existing data files.
  2. Run diag_drag.analyze() for every qualifying flight individually
     -> one diagnostic PNG per flight.
  3. Group flights by nose configuration (head configuration column):
       tepа  / tępa  -> blunt nose
       ostra          -> sharp nose
  4. Run fit_cd_mach.run() separately for each nose group
     -> one fitted Cd(Mach) curve per group + trajectory comparison plots.
  5. Print a summary table at the end.

Usage:
    python run_all_flights.py
    python run_all_flights.py --n_starts 20 --lam 0.05
    python run_all_flights.py --skip_diag     # skip per-flight diagnostic plots
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# ---- make sure this module's directory is on the path ----
import os
sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_parser import parse_telemetry, get_data_dir
from diag_drag import load_config, isa, detect_events, analyze as diag_analyze, S
from imu_reconstruction import detect_ignition
import fit_cd_mach as fcm


# --------------------------------------------------------------------------
def read_configs(base):
    """
    Parses configs.txt.
    Returns list of dicts for flights marked 'to analyze? == yes'.
    """
    cfg_path = Path(base) / "configs.txt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"configs.txt not found in {base}")

    lines = open(cfg_path, encoding="utf-8", errors="replace").readlines()
    header = [h.strip() for h in lines[0].strip().split("\t")]

    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split("\t")]
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        try:
            fno = int(row["flight no"])
        except (ValueError, KeyError):
            continue

        analyze_flag = row.get("to analyze?", "no").lower().strip()
        if analyze_flag != "yes":
            continue

        def f(k):
            try: return float(row.get(k, 0))
            except ValueError: return 0.0

        m_empty      = f("m_empty")
        m_full       = f("m_full")
        m_rocket     = f("m_rocket")
        m_propellant = m_full - m_empty
        m_coast      = m_rocket - m_propellant

        head = row.get("head configuration", "").lower().strip()
        # normalise Polish characters
        if "t" in head and ("pa" in head or "ępa" in head or "epa" in head):
            nose = "tepa"    # tępa = blunt
        else:
            nose = "ostra"   # ostra = sharp

        rows.append(dict(
            fno=fno, nose=nose, head_raw=row.get("head configuration", ""),
            cant=f("cant angle"), azimuth=f("azimuth"), elevation=f("elevation"),
            m_empty=m_empty, m_full=m_full, m_rocket=m_rocket,
            m_propellant=m_propellant, m_coast=m_coast,
        ))

    rows.sort(key=lambda r: r["fno"])
    return rows


# --------------------------------------------------------------------------
def check_data_exists(base, fno):
    return (Path(base) / f"ARTEMIDA_{fno}_LOT.txt").exists()


# --------------------------------------------------------------------------
def run_diagnostics(base, out_dir, rows, verbose=True):
    """Run diag_drag for every flight, return per-flight Cd summary."""
    results = {}
    for r in rows:
        fno = r["fno"]
        fpath = Path(base) / f"ARTEMIDA_{fno}_LOT.txt"
        if not fpath.exists():
            print(f"\n[LOT {fno}] brak pliku — pomijam diagnostyke")
            continue

        cfg = dict(
            m_coast=r["m_coast"], m_rocket=r["m_rocket"],
            m_propellant=r["m_propellant"],
            cant=r["cant"], azimuth=r["azimuth"], elevation=r["elevation"],
            head=r["head_raw"],
        )

        tel = parse_telemetry(fpath, verbose=False)
        out_png = str(Path(out_dir) / f"drag_flight_{fno}.png")
        D = diag_analyze(tel, cfg, S, fno, out_png)
        if D is not None:
            valid = np.isfinite(D["Cd"]) & (D["q"] > 100)
            results[fno] = dict(
                nose=r["nose"],
                Cd_med=float(np.nanmedian(D["Cd"][valid])) if valid.any() else np.nan,
                Ma_min=float(D["M"][valid].min()) if valid.any() else np.nan,
                Ma_max=float(D["M"][valid].max()) if valid.any() else np.nan,
            )
    return results


# --------------------------------------------------------------------------
def run_group_fit(base, out_dir, rows, nose_type, n_starts, lam):
    """Fit Cd(Mach) for all flights of a given nose type."""
    group = [r["fno"] for r in rows
             if r["nose"] == nose_type and check_data_exists(base, r["fno"])]
    if not group:
        print(f"\n[{nose_type.upper()}] brak lotow — pomijam dopasowanie")
        return None, group

    label = "tepa (blunt)" if nose_type == "tepa" else "ostra (sharp)"
    print(f"\n{'='*60}")
    print(f"Dopasowanie Cd(Ma) — {label}  loty: {group}")
    print(f"{'='*60}")

    best_cd = fcm.run(
        flight_nos=group,
        base=base,
        n_starts=n_starts,
        lam=lam,
        out_dir=out_dir,
    )
    return best_cd, group


# --------------------------------------------------------------------------
def print_summary(diag_results, cd_tepa, group_tepa, cd_ostra, group_ostra):
    print("\n" + "="*60)
    print("PODSUMOWANIE — Cd mediana z diagnostyki GPS")
    print("="*60)
    print(f"  {'Lot':>4}  {'Nos':>6}  {'Cd_med':>7}  {'Ma_min':>7}  {'Ma_max':>7}")
    for fno, r in sorted(diag_results.items()):
        print(f"  {fno:4d}  {r['nose']:>6}  {r['Cd_med']:7.3f}  "
              f"{r['Ma_min']:7.3f}  {r['Ma_max']:7.3f}")

    print()
    if cd_tepa is not None:
        print(f"Krzywa Cd(Ma) — TEPA (blunt), loty {group_tepa}:")
        print(f"  {'Mach':>6}  {'Cd':>6}")
        for ma, cd in zip(fcm.MA_NODES, cd_tepa):
            print(f"  {ma:6.3f}  {cd:6.4f}")

    if cd_ostra is not None:
        print(f"\nKrzywa Cd(Ma) — OSTRA (sharp), loty {group_ostra}:")
        print(f"  {'Mach':>6}  {'Cd':>6}")
        for ma, cd in zip(fcm.MA_NODES, cd_ostra):
            print(f"  {ma:6.3f}  {cd:6.4f}")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch drag validation")
    parser.add_argument("--n_starts", type=int, default=15,
                        help="Liczba startow optymalizatora (domyslnie 15)")
    parser.add_argument("--lam", type=float, default=0.1,
                        help="Wspolczynnik regularyzacji lambda (domyslnie 0.1)")
    parser.add_argument("--skip_diag", action="store_true",
                        help="Pomijaj wykresy diagnostyczne per-flight")
    args = parser.parse_args()

    base    = get_data_dir()
    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wczytaj konfiguracje
    rows = read_configs(base)
    available = [r for r in rows if check_data_exists(base, r["fno"])]

    print(f"Znaleziono {len(rows)} lotow do analizy w configs.txt, "
          f"{len(available)} ma pliki danych.")
    print(f"{'Lot':>4}  {'Nos':>6}  {'Cant':>5}  {'m_coast':>8}  {'Plik':>5}")
    for r in rows:
        exists = "TAK" if check_data_exists(base, r["fno"]) else "BRAK"
        print(f"{r['fno']:4d}  {r['nose']:>6}  {r['cant']:5.1f}  "
              f"{r['m_coast']:8.3f}  {exists}")

    # 2. Diagnostyka per-flight
    if not args.skip_diag:
        print(f"\n{'='*60}")
        print("Diagnostyka Cd (metoda GPS) — kazdy lot osobno")
        print(f"{'='*60}")
        diag_results = run_diagnostics(base, out_dir, available)
    else:
        diag_results = {}

    # 3. Dopasowanie krzywej Cd(Ma) osobno dla kazdej konfiguracji nosa
    cd_tepa,  group_tepa  = run_group_fit(base, out_dir, available,
                                           "tepa",  args.n_starts, args.lam)
    cd_ostra, group_ostra = run_group_fit(base, out_dir, available,
                                           "ostra", args.n_starts, args.lam)

    # 4. Podsumowanie
    if diag_results:
        print_summary(diag_results, cd_tepa, group_tepa, cd_ostra, group_ostra)

    print(f"\nWyniki zapisane w: {out_dir}")


if __name__ == "__main__":
    main()
