"""
compare_validation.py
=====================
Porownanie wynikow walidacji PRZED i PO zmianie w modelu.

Walidacja per lot zyje w telemetry_analysis/analyze_per_flight_6dof.py i
zapisuje field_test_data/results/per_flight_6dof_per_nose.csv. Ten skrypt
pozwala zrobic z tego pliku migawke, a potem porownac z nia nowy wynik —
zeby bylo widac, co dokladnie zmiana w modelu poprawila, a co pogorszyla.

Typowy przebieg:
    # 1. przed zmiana (albo tuz po pobraniu starych wynikow)
    python telemetry_analysis/analyze_per_flight_6dof.py
    python compare_validation.py --save-baseline przed_LREF

    # 2. po zmianie w modelu i przeliczeniu DATCOM
    python telemetry_analysis/analyze_per_flight_6dof.py
    python compare_validation.py --vs przed_LREF

Migawki leza w field_test_data/results/validation_snapshots/.

Uzycie:
    python compare_validation.py --list
    python compare_validation.py --save-baseline <nazwa>
    python compare_validation.py --vs <nazwa>
    python compare_validation.py --vs <nazwa> --plot
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "field_test_data" / "results"
CSV_NAME = "per_flight_6dof_per_nose.csv"
SNAP_DIR = RESULTS / "validation_snapshots"

# (kolumna_modelu, kolumna_pomiaru, etykieta, jednostka, "mniej znaczy lepiej")
METRICS = [
    ("h_apo_pred_adjusted",              "h_apo_actual",            "apogeum",    "m"),
    ("v_max_pred_adjusted",              "v_max_actual",            "Vmax",       "m/s"),
    ("downrange_apo_pred_adjusted_m",    "downrange_apo_actual_m",  "downrange",  "m"),
    ("crossrange_apo_pred_adjusted_m",   "crossrange_apo_actual_m", "crossrange", "m"),
]


def read_csv(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {int(r["fno"]): r for r in csv.DictReader(f)}


def fnum(row, key):
    try:
        v = float(row.get(key, ""))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def err_stats(data, pred_key, act_key):
    """Bledy bezwzgledne model-pomiar dla wszystkich lotow z danymi."""
    out = {}
    for fno, row in data.items():
        p, a = fnum(row, pred_key), fnum(row, act_key)
        if np.isfinite(p) and np.isfinite(a):
            out[fno] = (p, a, p - a)
    return out


def cmd_list():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    snaps = sorted(SNAP_DIR.glob("*.csv"))
    if not snaps:
        print(f"Brak migawek w {SNAP_DIR}")
        return 0
    print(f"Migawki w {SNAP_DIR}:")
    for s in snaps:
        d = read_csv(s)
        print(f"  {s.stem:<28} {len(d) if d else 0} lotow")
    return 0


def cmd_save(name: str):
    src = RESULTS / CSV_NAME
    if not src.exists():
        print(f"Brak {src}\nUruchom najpierw: "
              f"python telemetry_analysis/analyze_per_flight_6dof.py")
        return 1
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    dst = SNAP_DIR / f"{name}.csv"
    shutil.copy2(src, dst)
    print(f"Zapisano migawke: {dst}  ({len(read_csv(dst))} lotow)")
    return 0


def cmd_compare(name: str, do_plot: bool):
    base = read_csv(SNAP_DIR / f"{name}.csv")
    curr = read_csv(RESULTS / CSV_NAME)
    if base is None:
        print(f"Brak migawki '{name}'. Dostepne:"); cmd_list(); return 1
    if curr is None:
        print(f"Brak {RESULTS/CSV_NAME}\nUruchom najpierw: "
              f"python telemetry_analysis/analyze_per_flight_6dof.py")
        return 1

    print("=" * 84)
    print(f"WALIDACJA: '{name}' (przed)   vs   aktualny wynik (po)")
    print("=" * 84)

    summary = []
    for pred_key, act_key, label, unit in METRICS:
        eb = err_stats(base, pred_key, act_key)
        ec = err_stats(curr, pred_key, act_key)
        common = sorted(set(eb) & set(ec))
        if not common:
            print(f"\n{label}: brak wspolnych lotow z danymi — pomijam")
            continue

        print(f"\n{label} [{unit}]")
        print(f"  {'lot':>4} {'pomiar':>10} {'przed':>10} {'po':>10} "
              f"{'blad przed':>11} {'blad po':>10} {'zmiana':>9}")
        for fno in common:
            p0, a, e0 = eb[fno]
            p1, _, e1 = ec[fno]
            better = abs(e1) < abs(e0)
            mark = "lepiej" if better else ("gorzej" if abs(e1) > abs(e0) else "=")
            print(f"  {fno:>4} {a:>10.1f} {p0:>10.1f} {p1:>10.1f} "
                  f"{e0:>+11.1f} {e1:>+10.1f} {mark:>9}")

        rmse0 = float(np.sqrt(np.mean([eb[f][2] ** 2 for f in common])))
        rmse1 = float(np.sqrt(np.mean([ec[f][2] ** 2 for f in common])))
        mae0 = float(np.mean([abs(eb[f][2]) for f in common]))
        mae1 = float(np.mean([abs(ec[f][2]) for f in common]))
        chg = (rmse1 - rmse0) / rmse0 * 100.0 if rmse0 > 1e-12 else np.nan
        print(f"  {'RAZEM':>4} {'':>10} {'':>10} {'':>10} "
              f"RMSE {rmse0:>7.1f} -> {rmse1:<7.1f} ({chg:+.1f}%)")
        summary.append((label, unit, rmse0, rmse1, mae0, mae1, len(common)))

    print("\n" + "=" * 84)
    print("PODSUMOWANIE (RMSE bledu model-pomiar; mniej = lepiej)")
    print("=" * 84)
    print(f"  {'metryka':<12} {'n':>3} {'RMSE przed':>12} {'RMSE po':>10} "
          f"{'zmiana':>10}   ocena")
    for label, unit, r0, r1, m0, m1, n in summary:
        chg = (r1 - r0) / r0 * 100.0 if r0 > 1e-12 else np.nan
        verdict = ("POPRAWA" if chg < -1 else
                   "POGORSZENIE" if chg > 1 else "bez zmian")
        print(f"  {label:<12} {n:>3} {r0:>12.1f} {r1:>10.1f} {chg:>+9.1f}%   {verdict}")
    print("=" * 84)

    if do_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.4))
        if len(METRICS) == 1:
            axes = [axes]
        for ax, (pred_key, act_key, label, unit) in zip(axes, METRICS):
            eb, ec = err_stats(base, pred_key, act_key), err_stats(curr, pred_key, act_key)
            common = sorted(set(eb) & set(ec))
            if not common:
                ax.set_visible(False); continue
            x = np.arange(len(common)); w = 0.38
            ax.bar(x - w/2, [eb[f][2] for f in common], w, label=f"przed ({name})",
                   color="tab:gray")
            ax.bar(x + w/2, [ec[f][2] for f in common], w, label="po",
                   color="tab:blue")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels([str(f) for f in common], fontsize=8)
            ax.set_xlabel("lot"); ax.set_ylabel(f"blad model-pomiar [{unit}]")
            ax.set_title(label); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
        fig.suptitle(f"Wplyw zmiany modelu na walidacje  ('{name}' -> aktualny)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        p = RESULTS / f"validation_compare_{name}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"\nZapisano wykres: {p}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-baseline", metavar="NAZWA")
    ap.add_argument("--vs", metavar="NAZWA")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    if a.list:
        return cmd_list()
    if a.save_baseline:
        return cmd_save(a.save_baseline)
    if a.vs:
        return cmd_compare(a.vs, a.plot)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
