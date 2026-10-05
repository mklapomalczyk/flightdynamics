"""
fit_thrust_trapezoid.py
=======================
Pick engine CSVs in a file dialog (any number), fit a 6-point trapezoid
thrust profile, generate per-engine YAML configs, save a comparison plot and
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


def fit_trapezoid(t, F):
    """Fit a 5-point trapezoid to the thrust curve.

    Points:
      0: [0.0, 50.0]          - start (50N to hold on rail)
      1: [t_ramp, F_ramp]     - end of ignition ramp (~10% of peak)
      2: [t_peak, F_peak]     - peak thrust
      3: [t_tail, F_tail]     - start of tail-off
      4: [t_end, 0.0]         - burnout

    Strategy: the curve is progressive (ramps up continuously), so
    the trapezoid captures ignition transient, main ramp, peak region,
    and tail-off.
    """
    F_max = np.max(F)

    # Time origin: first time F >= 50 N
    mask_50 = F >= 50.0
    if np.any(mask_50):
        t0 = t[mask_50][0]
    else:
        t0 = t[0]

    # Shift time so t0 becomes 0.0
    t = t - t0

    # ramp point: where F reaches 20% of peak (steep ramp)
    mask_ramp = F > 0.20 * F_max
    idx_ramp = np.argmax(mask_ramp)
    t_ramp = t[idx_ramp]
    F_ramp = F[idx_ramp]

    # knee point: top of steep ramp, where slope drops
    # Find where dF/dt is maximum, then the knee is where it drops to ~30%
    dt_arr = np.diff(t)
    dF_dt = np.diff(F) / np.where(dt_arr > 0, dt_arr, 1e-6)
    # smooth to avoid noise spikes
    win = min(200, len(dF_dt) // 10)
    if win > 1:
        kernel = np.ones(win) / win
        dF_smooth = np.convolve(dF_dt, kernel, mode="same")
    else:
        dF_smooth = dF_dt
    # search only in the rising part (before peak)
    idx_peak = np.argmax(F)
    rising = dF_smooth[:idx_peak]
    if len(rising) > 0:
        max_slope = np.max(rising)
        # knee: first point after max slope where slope drops below 30% of max
        idx_max_slope = np.argmax(rising)
        after_max = rising[idx_max_slope:]
        mask_knee = after_max < 0.30 * max_slope
        if np.any(mask_knee):
            idx_knee = idx_max_slope + np.argmax(mask_knee)
            t_knee = t[idx_knee]
            F_knee = F[idx_knee]
        else:
            t_knee = t[idx_ramp] + (t[idx_peak] - t[idx_ramp]) * 0.3
            F_knee = np.interp(t_knee, t, F)
    else:
        t_knee = t[idx_ramp] + (t[idx_peak] - t[idx_ramp]) * 0.3
        F_knee = np.interp(t_knee, t, F)

    # peak: max thrust
    t_peak = t[idx_peak]
    F_peak = F_max

    # tail-off start: after peak, where F drops below 50% of peak
    after_peak = F[idx_peak:]
    t_after = t[idx_peak:]
    mask_tail = after_peak < 0.50 * F_max
    if np.any(mask_tail):
        idx_tail = np.argmax(mask_tail)
        t_tail = t_after[idx_tail]
        F_tail = after_peak[idx_tail]
    else:
        t_tail = t_peak + 0.1
        F_tail = F_peak * 0.3

    # burnout: last time F > 10 N
    mask_burn = F > 10.0
    t_end = t[mask_burn][-1] if np.any(mask_burn) else t[-1]

    # Build 6-point profile
    profile = [
        [0.0, 50.0],
        [round(t_ramp, 3), round(F_ramp, 1)],
        [round(t_knee, 3), round(F_knee, 1)],
        [round(t_peak, 3), round(F_peak, 1)],
        [round(t_tail, 3), round(F_tail, 1)],
        [round(t_end, 3), 0.0],
    ]
    return profile, t0


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
        profile, t0 = fit_trapezoid(t, F)

        print(f"Trapezoid profile (t0 shifted by {t0:.4f} s):")
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
        ax.plot(tp, fp, "r-o", lw=2, ms=6, label="trapezoid fit")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("thrust [N]")
        ax.set_title(f"{engine_name} ({prop_mass*1000:.0f} g)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Engine thrust profiles — CSV vs trapezoid fit", fontweight="bold")
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
