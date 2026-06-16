"""
debug_motor.py — porownanie trajektorii silnikow 0.350, 0.370, 0.375
Uruchom z katalogu FlightSim: python debug_motor.py
"""
import sys, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import GeoModule, MissionConfig
import tempfile, os, yaml

CASE_NAME = "rocket_36mm_malarakieta_base"
IMPULSE   = 270.0
TIMES     = [0.350, 0.370, 0.375, 0.380]
COLORS    = ['blue', 'red', 'green', 'orange']

mission = MissionConfig.from_yaml("missions/mission_01.yaml")
geo     = GeoModule(mission)
aero    = get_aero_model(CASE_NAME, method="missile_datcom")

results = []
for t_burn in TIMES:
    thrust = IMPULSE / t_burn
    print(f"\nt_burn={t_burn}s  thrust={thrust:.1f}N")

    with open(f"configurations/{CASE_NAME}.yaml",
              encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())
    raw["propulsion"]["thrust_profile"] = [
        [0.0,    thrust],
        [t_burn, thrust],
        [round(t_burn+0.001,4), 0.0],
    ]
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    os.close(tmp_fd)
    with open(tmp_path, "w") as f:
        yaml.dump(raw, f)
    cfg = load_config(tmp_path)
    os.unlink(tmp_path)

    mass = build_mass_model(cfg)
    prop = build_propulsion(cfg)
    geom = build_geometry(cfg)
    geom.xcp = float(aero.xcp_table[len(aero.alpha_table)//2, 0])
    initial  = build_initial_state(mission)

    fm = ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"),
        mass_model=mass, aero_model=aero,
        gravity=create_gravity("constant"),
        geometry=geom, propulsion=prop,
    )
    result = run_simulation_6dof(fm, initial, t_max=60., z_ground=0.,
                                 max_step=0.05, rtol=1e-4, atol=1e-6)
    lf = geo.ned_to_lf(result)

    alpha_deg = np.abs(np.degrees(result.alpha))
    omega_qr  = np.sqrt(np.degrees(result.qr)**2 + np.degrees(result.r)**2)

    print(f"  status={result.status}")
    print(f"  zasieg={lf['x_lf'][-1]:.0f}m  alt_max={lf['alt'].max():.0f}m")
    print(f"  alpha_max={alpha_deg.max():.2f}°")
    print(f"  omega_qr_max={omega_qr.max():.1f}°/s")
    print(f"  V_max={result.speed.max():.1f}m/s  V_final={result.speed[-1]:.1f}m/s")

    results.append({
        "label": f"t={t_burn}s ({thrust:.0f}N)",
        "t": result.t, "alt": lf["alt"], "x": lf["x_lf"],
        "alpha": np.degrees(result.alpha),
        "omega": omega_qr, "V": result.speed,
    })

# Wykresy
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Diagnostyka sweep silnika — t=0.350/0.370/0.375/0.380s",
             fontsize=11, fontweight="bold")

for r, c in zip(results, COLORS):
    axes[0,0].plot(r["x"]/1000, r["alt"], color=c, lw=1.5, label=r["label"])
    axes[0,1].plot(r["t"], r["alpha"], color=c, lw=1.2, label=r["label"])
    axes[1,0].plot(r["t"], r["omega"], color=c, lw=1.2, label=r["label"])
    axes[1,1].plot(r["t"], r["V"],     color=c, lw=1.5, label=r["label"])

for ax, title, xlabel, ylabel in [
    (axes[0,0], "Trajektoria", "Zasieg [km]", "Wysokosc [m]"),
    (axes[0,1], "Kat natarcia", "Czas [s]", "Alpha [deg]"),
    (axes[1,0], "sqrt(q²+r²)", "Czas [s]", "[°/s]"),
    (axes[1,1], "Predkosc", "Czas [s]", "V [m/s]"),
]:
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

axes[1,0].axhline(500, color='gray', ls='--', lw=0.8, label='prog 500°/s')
plt.tight_layout()
plt.savefig("debug_motor.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nZapisano: debug_motor.png")
