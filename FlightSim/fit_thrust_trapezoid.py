"""
fit_thrust_trapezoid.py
=======================
Pick engine CSVs in a file dialog (any number), simplify the thrust curve
to a polyline (Ramer-Douglas-Peucker, point count adapts to the curve shape), generate per-engine YAML configs, save a comparison plot and
the list of picked engines for run_engine_comparison.py.

Filename: <ENGINE>_<anything>_<mass>g.csv
  ENGINE = everything before the first "_", mass = number at the end [g].

Usage:
    python fit_thrust_trapezoid.py
    python fit_thrust_trapezoid.py --show
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "engine_plots"
CFG_DIR = ROOT / "configurations"
BASE_YAML = CFG_DIR / "rocket_70mm_WB500.yaml"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
SELECTED_FILE = OUT_DIR / "selected_engines.txt"

TOL_FRAC = 0.02      # max odchylka lamanej od CSV, jako ulamek ciagu maks.
IMPULSE_TOL = 0.01   # max blad impulsu calkowitego (1%)


def pick_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    files = filedialog.askopenfilenames(
        title="Wybierz pliki CSV silnikow (Ctrl/Shift = wiele)",
        initialdir=str(ENGINE_DIR),
        filetypes=[("CSV", "*.csv"), ("Wszystkie pliki", "*.*")])
    root.destroy()
    return [Path(f) for f in files]


def read_engine_csv(path):
    """Read Polish-locale CSV: semicolon sep, comma decimal."""
    t_list, F_list = [], []
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter=";")
        next(r)  # header
        for row in r:
            t_list.append(float(row[0].replace(",", ".")))
            F_list.append(float(row[2].replace(",", ".")))
    return np.array(t_list), np.array(F_list)


def parse_filename(name):
    """'WB700_ParametryNapedowe_masa2328g.csv' -> ('WB700', 2.328 kg)."""
    stem = Path(name).stem
    engine = stem.split("_", 1)[0]
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*g?$", stem, flags=re.IGNORECASE)
    if not engine or not m:
        raise ValueError(f"Cannot parse engine name/mass from filename: {name}")
    return engine, float(m.group(1).replace(",", ".")) / 1000.0


def rdp_indices(t, F, tol):
    """Ramer-Douglas-Peucker: indeksy punktow lamanej, ktora odbiega od
    krzywej o co najwyzej tol [N] (odleglosc pionowa — osie maja rozne
    jednostki, wiec liczy sie blad ciagu, nie odleglosc geometryczna)."""
    keep = {0, len(t) - 1}
    stack = [(0, len(t) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        seg = slice(i0 + 1, i1)
        line = F[i0] + (F[i1] - F[i0]) * (t[seg] - t[i0]) / (t[i1] - t[i0])
        dev = np.abs(F[seg] - line)
        k = int(np.argmax(dev))
        if dev[k] > tol:
            im = i0 + 1 + k
            keep.add(im)
            stack += [(i0, im), (im, i1)]
    return np.array(sorted(keep))


def fit_profile(t, F):
    """Uproszczony profil ciagu (lamana RDP) — liczba punktow dobiera sie
    sama: kilka dla trapezu, wiecej dla profilu piloksztaltnego.

    Start: t=0 tam, gdzie F po raz pierwszy >= 50 N (punkt [0, 50]).
    Koniec: ostatnia chwila F > 10 N (punkt [t_end, 0]).
    Tolerancja startowa TOL_FRAC*F_max; zaciesniana, az impuls calkowity
    rozni sie od CSV o mniej niz IMPULSE_TOL."""
    i_start = int(np.argmax(F >= 50.0)) if np.any(F >= 50.0) else 0
    i_end = int(np.where(F > 10.0)[0][-1]) if np.any(F > 10.0) else len(F) - 1
    t0 = t[i_start]
    tt, FF = t[i_start:i_end + 1] - t0, F[i_start:i_end + 1].copy()
    FF[0], FF[-1] = 50.0, 0.0
    I_csv = np.trapezoid(F[i_start:i_end + 1], tt)

    tol = TOL_FRAC * F.max()
    while True:
        idx = rdp_indices(tt, FF, tol)
        I_fit = np.trapezoid(FF[idx], tt[idx])
        err = (I_fit - I_csv) / I_csv
        if abs(err) <= IMPULSE_TOL or tol < 0.001 * F.max():
            break
        tol /= 2.0

    profile = [[round(float(tt[i]), 4), round(float(FF[i]), 1)] for i in idx]
    return profile, t0, err


def generate_yaml(engine_name, prop_mass_kg, thrust_profile, base_yaml_path):
    """Generate a new YAML config by reading the base and replacing key values."""
    text = base_yaml_path.read_text(encoding="utf-8")
    case_name = f"rocket_70mm_{engine_name}"

    # Read base to get empty mass
    with open(base_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    empty_mass = cfg["mass_model"]["empty"]["mass"]
    full_mass = round(empty_mass + prop_mass_kg, 3)
    base_full_mass = cfg["mass_model"]["full"]["mass"]

    # Replace name
    text = re.sub(r"^name:.*$", f"name: {case_name}", text, count=1, flags=re.MULTILINE)

    # Replace full mass
    text = text.replace(f"mass:   {base_full_mass}", f"mass:   {full_mass}")

    # Replace thrust profile
    lines = text.split("\n")
    new_lines = []
    in_thrust = False
    for line in lines:
        if "thrust_profile:" in line:
            in_thrust = True
            new_lines.append("  thrust_profile:")
            new_lines.append("    #  t [s]   F [N]")
            for pt in thrust_profile:
                new_lines.append(f"    - [{pt[0]},  {pt[1]}]")
            continue
        if in_thrust:
            if line.strip().startswith("-") or line.strip().startswith("#"):
                continue  # skip old profile lines
            else:
                in_thrust = False
                new_lines.append(line)
                continue
        new_lines.append(line)

    out_path = CFG_DIR / f"{case_name}.yaml"
    out_path.write_text("\n".join(new_lines), encoding="utf-8")
    return out_path


def main():
    picked = pick_files()
    if not picked:
        print("No files selected.")
        return

    parsed = []
    for p in sorted(picked):
        try:
            parsed.append((p, *parse_filename(p.name)))
        except ValueError as e:
            print(f"[SKIP] {e}")
    if not parsed:
        return

    n = len(parsed)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), squeeze=False)
    axes = axes.ravel()

    engines = []
    for i, (csv_path, engine_name, prop_mass) in enumerate(parsed):
        engines.append(engine_name)
        print(f"\n{'='*60}")
        print(f"Engine: {engine_name}, propellant mass: {prop_mass*1000:.0f} g")

        t, F = read_engine_csv(csv_path)
        profile, t0, i_err = fit_profile(t, F)

        print(f"Profile: {len(profile)} points, impulse error {i_err*100:+.2f}% "
              f"(t0 shifted by {t0:.4f} s):")
        for pt in profile:
            print(f"  t={pt[0]:.3f} s  F={pt[1]:.1f} N")

        # Generate YAML
        yaml_path = generate_yaml(engine_name, prop_mass, profile, BASE_YAML)
        print(f"YAML: {yaml_path.name}")

        # Plot with shifted time
        ax = axes[i]
        ax.plot(t - t0, F, "tab:blue", lw=0.5, alpha=0.7, label="CSV data")
        tp = [p[0] for p in profile]
        fp = [p[1] for p in profile]
        ax.plot(tp, fp, "r-o", lw=1.5, ms=4, label=f"fit: {len(profile)} pkt, impuls {i_err*100:+.1f}%")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("thrust [N]")
        ax.set_title(f"{engine_name} ({prop_mass*1000:.0f} g)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Engine thrust profiles — CSV vs simplified profile", fontweight="bold")
    fig.tight_layout()
    out_png = OUT_DIR / "thrust_trapezoid_fits.png"
    if out_png.exists():
        out_png.unlink()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved: {out_png}")

    SELECTED_FILE.write_text("\n".join(engines) + "\n", encoding="utf-8")
    print(f"Engines for run_engine_comparison.py: {engines}  ({SELECTED_FILE.name})")

    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
