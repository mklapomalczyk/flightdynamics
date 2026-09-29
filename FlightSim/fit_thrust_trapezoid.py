"""
fit_thrust_trapezoid.py
=======================
Read engine CSVs from engine_plots/, fit a 5-point trapezoid thrust profile,
generate per-engine YAML configs, and save a comparison plot.

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
    """Extract engine name and propellant mass from filename.
    E.g. 'WB700_ParametryNapedowe_masa2328g.csv' -> ('WB700', 2.328)
    """
    m = re.match(r"(WB\d+)_.*masa(\d+)g", name)
    if not m:
        raise ValueError(f"Cannot parse engine filename: {name}")
    return m.group(1), int(m.group(2)) / 1000.0  # kg


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

    # ramp point: where F reaches 20% of peak
    mask_ramp = F > 0.20 * F_max
    idx_ramp = np.argmax(mask_ramp)
    t_ramp = t[idx_ramp]
    F_ramp = F[idx_ramp]

    # peak: max thrust
    idx_peak = np.argmax(F)
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

    # Build 5-point profile, starting at [0.0, 50.0]
    profile = [
        [0.0, 50.0],
        [round(t_ramp, 3), round(F_ramp, 1)],
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
    csv_files = sorted(ENGINE_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in engine_plots/")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, csv_path in enumerate(csv_files):
        engine_name, prop_mass = parse_filename(csv_path.name)
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
        ax = axes[i] if i < len(axes) else axes[-1]
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

    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
