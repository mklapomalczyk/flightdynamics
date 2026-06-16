"""
debug_cant.py — diagnostyka sweep cant angle
Uruchom z katalogu FlightSim: python debug_cant.py
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
import tempfile, os, yaml, shutil

CASE_NAME  = "rocket_36mm_malarakieta_base"
CANT_VALS  = [0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0]
COLORS     = plt.cm.viridis(np.linspace(0, 0.85, len(CANT_VALS)))
IMPULSE    = 270.0

mission = MissionConfig.from_yaml("missions/mission_01.yaml")
geo     = GeoModule(mission)

results = []
for cant, clr in zip(CANT_VALS, COLORS):
    label = f"cant={cant}°"

    with open(f"configurations/{CASE_NAME}.yaml",
              encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())
    for fin in raw.get("fins", []):
        fin["cant_angle"] = float(cant)
    case_tag = f"{CASE_NAME}_cant{str(cant).replace('.','p')}"
    with open(f"configurations/{case_tag}.yaml", "w") as f:
        yaml.dump(raw, f)

    try:
        aero = get_aero_model(case_tag, method="missile_datcom", force_rerun=False)
    except Exception:
        aero = get_aero_model(case_tag, method="missile_datcom", force_rerun=True)

    cfg     = load_config(f"configurations/{case_tag}.yaml")
    os.unlink(f"configurations/{case_tag}.yaml")

    mass = build_mass_model(cfg)
    geom = build_geometry(cfg)
    geom.xcp = float(aero.xcp_table[len(aero.alpha_table)//2, 0])
    initial  = build_initial_state(mission)

    fm = ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"),
        mass_model=mass, aero_model=aero,
        gravity=create_gravity("constant"),
        geometry=geom, propulsion=build_propulsion(cfg),
    )
    result = run_simulation_6dof(fm, initial, t_max=60., z_ground=0.,
                                 max_step=0.02, rtol=1e-5, atol=1e-7)
    lf = geo.ned_to_lf(result)

    alpha_abs = np.abs(np.degrees(result.alpha))
    omega_qr  = np.sqrt(np.degrees(result.qr)**2 + np.degrees(result.r)**2)
    p_deg     = np.degrees(result.p)

    print(f"{label}: status={result.status}  "
          f"zasieg={lf['x_lf'][-1]:.0f}m  "
          f"alt={lf['alt'].max():.0f}m  "
          f"y_max={np.max(np.abs(lf['y_lf'])):.1f}m  "
          f"p_eq={p_deg[-1]:.0f}°/s  "
          f"alpha_max={alpha_abs.max():.2f}°  "
          f"omega_max={omega_qr.max():.1f}°/s")

    results.append({
        "cant": cant, "label": label, "color": clr,
        "t": result.t, "alt": lf["alt"], "x": lf["x_lf"],
        "y": lf["y_lf"], "alpha": np.degrees(result.alpha),
        "p": p_deg, "omega": omega_qr, "V": result.speed,
    })

# Wykresy
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Diagnostyka sweep cant_angle", fontsize=12, fontweight="bold")

for r in results:
    c = r["color"]
    lbl = r["label"]
    axes[0,0].plot(r["x"]/1000, r["alt"], color=c, lw=1.5, label=lbl)
    axes[0,1].plot(r["t"], r["alpha"], color=c, lw=1.0, label=lbl)
    axes[0,2].plot(r["t"], r["p"],     color=c, lw=1.2, label=lbl)
    axes[1,0].plot(r["t"], r["y"],     color=c, lw=1.2, label=lbl)
    axes[1,1].plot(r["t"], r["omega"], color=c, lw=1.2, label=lbl)
    axes[1,2].plot(r["t"], r["V"],     color=c, lw=1.5, label=lbl)

titles = ["Trajektoria", "Kat natarcia alpha", "Roll p [°/s]",
          "Odchylenie boczne y [m]", "sqrt(q²+r²) [°/s]", "Predkosc V [m/s]"]
xlabels = ["Zasieg [km]", "Czas [s]", "Czas [s]",
           "Czas [s]",    "Czas [s]", "Czas [s]"]
ylabels = ["Wysokosc [m]", "Alpha [deg]", "p [°/s]",
           "y [m]",        "[°/s]",       "V [m/s]"]

for ax, t, xl, yl in zip(axes.flat, titles, xlabels, ylabels):
    ax.set_title(t); ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

axes[0,1].axhline(0, color='gray', lw=0.5, ls=':')
axes[1,0].axhline(0, color='gray', lw=0.5, ls=':')

plt.tight_layout()
plt.savefig("debug_cant.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nZapisano: debug_cant.png")
