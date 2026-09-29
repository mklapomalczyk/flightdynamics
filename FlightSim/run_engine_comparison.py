"""
run_engine_comparison.py
========================
Run 6DOF simulation for each engine (WB700..WB1000) at two elevation sets
and generate comparison plots.

Usage:
    python run_engine_comparison.py
    python run_engine_comparison.py --show
"""

import sys
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.solver6 import run_simulation_6dof
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from models.launcher import LauncherConfig
from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import (build_mass_model, build_propulsion,
                                       build_geometry, build_initial_state)
from geo.geographic import MissionConfig
from models.force_logger import ForceLogger

# ============================================================
# USER PARAMETERS
# ============================================================

ENGINES = ["WB700", "WB800", "WB900", "WB1000"]

ELEVATIONS_LOW  = [30, 35, 40, 45]
ELEVATIONS_HIGH = [50, 55, 60, 65]

MISSION_FILE = "missions/mission_01.yaml"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================


def safe_savefig(fig, path, **kwargs):
    p = Path(path)
    if p.exists():
        p.unlink()
    fig.savefig(p, **kwargs)


def run_engine(engine_name, elevations, set_label):
    """Run simulation for one engine at given elevations, produce plots."""
    case_name = f"rocket_70mm_{engine_name}"
    cfg = load_config(f"configurations/{case_name}.yaml")
    mass = build_mass_model(cfg)
    prop = build_propulsion(cfg)
    geom = build_geometry(cfg)

    aero_model = get_aero_model(case_name, method="missile_datcom", force_rerun=True)
    atm = create_atmosphere("ISA")
    launcher = LauncherConfig(L_rail=3.0)

    print(f"\n{'='*60}")
    print(f"Engine: {engine_name} | elevations: {elevations} ({set_label})")
    print(f"Full mass: {cfg.mass_model['full']['mass']:.3f} kg")
    print(f"{'='*60}")

    for elev in elevations:
        mission = MissionConfig.from_yaml(MISSION_FILE)
        mission.elevation = elev
        logger = ForceLogger(case_name=case_name, log_dir="logs", enabled=True)

        force_model = ForceModel6DOF(
            atmosphere=atm, mass_model=mass, aero_model=aero_model,
            gravity=create_gravity("constant"), geometry=geom,
            propulsion=prop, launcher=launcher, logger=logger,
        )

        initial_state = build_initial_state(mission)
        result = run_simulation_6dof(
            force_model=force_model, initial_state=initial_state,
            t_max=100, dt_output=0.001,
        )
        logger.close()
        print(f"  {elev}°: range={result.max_range:.0f}m, "
              f"alt={result.max_altitude:.0f}m, "
              f"Vmax={result.max_speed:.0f}m/s, status={result.status}")

    # Read logs
    log_files = sorted(glob.glob(f"logs/{case_name}_*.csv"),
                       key=os.path.getmtime)
    logs = [pd.read_csv(f) for f in log_files[-len(elevations):]]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    m0 = logs[0]["mass"].values[0]
    fig.suptitle(f"Rakieta 70mm, silnik {engine_name}, "
                 f"masa startowa {m0:.2f} kg ({set_label})",
                 fontsize=13, fontweight="bold")

    # 1. Trajectory XZ
    ax = axes[0, 0]
    for df, elev in zip(logs, elevations):
        ax.plot(df["x"].values, -df["z"].values, lw=1.5, label=f"{elev}°")
    ax.set_xlabel("Zasięg x [m]")
    ax.set_ylabel("Wysokość z [m]")
    ax.set_title("Trajektoria (płaszczyzna XZ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.8)

    # 2. Altitude vs time
    ax = axes[0, 1]
    for df, elev in zip(logs, elevations):
        ax.plot(df["t"].values, -df["z"].values, lw=1.5, label=f"{elev}°")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Wysokość [m]")
    ax.set_title("Wysokość vs czas")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mach
    ax = axes[1, 0]
    for df, elev in zip(logs, elevations):
        mach_arr = np.array([atm.at(-z).mach(s) for z, s in
                             zip(-df["z"].values, df["speed"].values)])
        ax.plot(df["t"].values, mach_arr, lw=1.5, label=f"{elev}°")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="Ma=1")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Mach [-]")
    ax.set_title("Liczba Macha")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Acceleration
    ax = axes[1, 1]
    for df, elev in zip(logs, elevations):
        a_x = df["FX"].values / df["mass"].values
        a_y = df["FY"].values / df["mass"].values
        a_z = df["FZ"].values / df["mass"].values
        a_total = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        ax.plot(df["t"].values, a_total / 9.81, lw=1.5, label=f"{elev}°")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Przyspieszenie [g]")
    ax.set_title("Przyspieszenie vs czas")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_name = f"{engine_name}_{set_label}.png"
    safe_savefig(fig, OUT_DIR / png_name, dpi=130, bbox_inches="tight")
    print(f"  Plot: results/{png_name}")

    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)


def main():
    for engine in ENGINES:
        run_engine(engine, ELEVATIONS_LOW, "low")
        run_engine(engine, ELEVATIONS_HIGH, "high")

    print(f"\nAll plots saved in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
