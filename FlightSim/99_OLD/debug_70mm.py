"""Debug momentu dla rakiety 70mm — uruchom z katalogu FlightSim"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_geometry, build_mass_model, build_propulsion, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from geo.geographic import MissionConfig

CASE_NAME = "rocket_70mm_baseline"
cfg     = load_config(f"configurations/{CASE_NAME}.yaml")
mission = MissionConfig.from_yaml("missions/mission_01.yaml")
initial = build_initial_state(mission)
aero    = get_aero_model(CASE_NAME, method="missile_datcom")

print(f"xcp_table (alpha=0, Ma=0.3): {aero.xcp_table[len(aero.alpha_table)//2, 0]*1000:.1f}mm")
print(f"xcg_ref: {aero.xcg_ref*1000:.1f}mm")

mass = build_mass_model(cfg)
geom = build_geometry(cfg)
prop = build_propulsion(cfg)

fm = ForceModel6DOF(
    atmosphere=create_atmosphere("ISA"),
    mass_model=mass,
    aero_model=aero,
    gravity=create_gravity("constant"),
    geometry=geom,
    propulsion=prop,
)

# Sprawdz pochodne w t=0
x0 = initial.to_numpy(include_rail=True)
dx = fm.derivatives(0.0, x0)
print(f"\nPochodne t=0:")
print(f"  du/dt  = {dx[3]:.2f} m/s2")
print(f"  dqr/dt = {np.degrees(dx[11]):.4f} deg/s2")
print(f"  max|dx| = {np.max(np.abs(dx)):.2f}")

# Euler 10 krokow
dt = 0.001
x = x0.copy()
print(f"\nEuler 10 krokow (dt={dt}s):")
for i in range(10):
    dx = fm.derivatives(i*dt, x)
    x  = x + dx * dt
    speed = np.sqrt(x[3]**2+x[4]**2+x[5]**2)
    print(f"  t={i*dt:.3f}s: V={speed:.2f} dqr={np.degrees(dx[11]):.4f} max|dx|={np.max(np.abs(dx)):.2f}")
    if np.any(~np.isfinite(dx)):
        print("  NaN/Inf!")
        break
