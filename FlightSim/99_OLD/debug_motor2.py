"""
debug_motor2.py — porownanie tolerancji dla t=0.370s
Uruchom z katalogu FlightSim: python debug_motor2.py
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

# Testujemy trzy zestawy tolerancji dla t=0.370s
CONFIGS = [
    {"max_step": 0.05,  "rtol": 1e-4, "atol": 1e-6,  "label": "max_step=0.05 rtol=1e-4"},
    {"max_step": 0.02,  "rtol": 1e-5, "atol": 1e-7,  "label": "max_step=0.02 rtol=1e-5"},
    {"max_step": 0.01,  "rtol": 1e-6, "atol": 1e-8,  "label": "max_step=0.01 rtol=1e-6"},
]

# Testujemy tez sasiednie punkty z najdokladniejszymi tolerancjami
TIMES_REF = [0.350, 0.360, 0.365, 0.370, 0.375, 0.380]

mission = MissionConfig.from_yaml("missions/mission_01.yaml")
geo     = GeoModule(mission)
aero    = get_aero_model(CASE_NAME, method="missile_datcom")

def make_fm(t_burn):
    thrust = IMPULSE / t_burn
    with open(f"configurations/{CASE_NAME}.yaml",
              encoding='utf-8', errors='replace') as f:
        raw = yaml.safe_load(f.read())
    raw["propulsion"]["thrust_profile"] = [
        [0.0, thrust], [t_burn, thrust],
        [round(t_burn+0.001,4), 0.0],
    ]
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    os.close(tmp_fd)
    with open(tmp_path, "w") as f:
        yaml.dump(raw, f)
    cfg = load_config(tmp_path)
    os.unlink(tmp_path)
    mass = build_mass_model(cfg)
    geom = build_geometry(cfg)
    geom.xcp = float(aero.xcp_table[len(aero.alpha_table)//2, 0])
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"),
        mass_model=mass, aero_model=aero,
        gravity=create_gravity("constant"),
        geometry=geom, propulsion=build_propulsion(cfg),
    )

# Test 1: rozne tolerancje dla t=0.370s
print("=== Test tolerancji dla t=0.370s ===")
for cfg_t in CONFIGS:
    fm = make_fm(0.370)
    res = run_simulation_6dof(
        fm, build_initial_state(mission),
        t_max=60., z_ground=0.,
        max_step=cfg_t["max_step"],
        rtol=cfg_t["rtol"], atol=cfg_t["atol"],
    )
    lf = geo.ned_to_lf(res)
    print(f"  {cfg_t['label']}: zasieg={lf['x_lf'][-1]:.0f}m  alt={lf['alt'].max():.0f}m  "
          f"V_max={res.speed.max():.1f}m/s  n_steps={len(res.t)}")

# Test 2: gestsza siatka czasow z najdokladniejszymi tolerancjami
print("\n=== Gestsza siatka czasow (max_step=0.02) ===")
results_ref = []
for t_burn in TIMES_REF:
    fm  = make_fm(t_burn)
    res = run_simulation_6dof(
        fm, build_initial_state(mission),
        t_max=60., z_ground=0.,
        max_step=0.02, rtol=1e-5, atol=1e-7,
    )
    lf = geo.ned_to_lf(res)
    print(f"  t={t_burn:.3f}s ({IMPULSE/t_burn:.0f}N): "
          f"zasieg={lf['x_lf'][-1]:.0f}m  alt={lf['alt'].max():.0f}m  "
          f"V_max={res.speed.max():.1f}m/s")
    results_ref.append({
        "label": f"t={t_burn}s", "t": res.t,
        "alt": lf["alt"], "x": lf["x_lf"],
    })

# Wykres
fig, ax = plt.subplots(figsize=(10, 5))
CMAP = plt.cm.viridis(np.linspace(0, 0.85, len(TIMES_REF)))
for r, c in zip(results_ref, CMAP):
    ax.plot(r["x"]/1000, r["alt"], color=c, lw=1.5, label=r["label"])
ax.set_xlabel("Zasieg [km]"); ax.set_ylabel("Wysokosc [m]")
ax.set_title("Trajektorie — gestsza siatka (max_step=0.02, rtol=1e-5)")
ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("debug_motor2.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nZapisano: debug_motor2.png")
