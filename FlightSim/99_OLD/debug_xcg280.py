"""
Diagnostyka dla xcg=280mm — uruchom z katalogu FlightSim
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from geo.geographic import MissionConfig
import tempfile, os, yaml

CASE_NAME = "rocket_36mm_malarakieta_base"
xcg_test  = 0.280   # 280mm

# Zaladuj z modyfikacja xcg
with open(f"configurations/{CASE_NAME}.yaml", encoding='utf-8', errors='replace') as f:
    raw = yaml.safe_load(f.read())
raw["mass"]["xcg_ref"] = xcg_test
if "mass_model" in raw:
    raw["mass_model"]["full"]["xcg"]  = xcg_test
    raw["mass_model"]["empty"]["xcg"] = xcg_test

tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
os.close(tmp_fd)
with open(tmp_path, "w", encoding='utf-8') as f:
    yaml.dump(raw, f)
cfg = load_config(tmp_path)
os.unlink(tmp_path)

aero    = get_aero_model(CASE_NAME, method="missile_datcom")
mission = MissionConfig.from_yaml("missions/mission_01.yaml")
initial = build_initial_state(mission)

mass = build_mass_model(cfg)
geom = build_geometry(cfg)
geom.xcp = float(aero.xcp_table[len(aero.alpha_table)//2, 0])

fm = ForceModel6DOF(
    atmosphere=create_atmosphere("ISA"),
    mass_model=mass,
    aero_model=aero,
    gravity=create_gravity("constant"),
    geometry=geom,
    propulsion=build_propulsion(cfg),
)

print(f"xcg_ref w aero: {aero.xcg_ref*1000:.1f}mm")
print(f"xcg_test: {xcg_test*1000:.1f}mm")
print(f"xcp (Ma=0.3, a=0): {geom.xcp*1000:.1f}mm")
print(f"SM = {(geom.xcp - xcg_test)/0.036:.2f} kal")

# Sprawdz pochodne w t=0
x0 = initial.to_numpy(include_rail=True)
dx = fm.derivatives(0.0, x0)
print(f"\nPochodne t=0:")
print(f"  du/dt = {dx[3]:.2f} m/s2")
print(f"  dw/dt = {dx[5]:.2f} m/s2")
print(f"  dqr/dt = {np.degrees(dx[11]):.2f} deg/s2")

# Euler 50 krokow
dt = 0.01
x = x0.copy()
print(f"\nEuler 50 krokow (dt={dt}s):")
for i in range(50):
    dx = fm.derivatives(i*dt, x)
    x  = x + dx * dt
    if i % 10 == 0:
        speed = np.sqrt(x[3]**2+x[4]**2+x[5]**2)
        alpha = np.degrees(np.arctan2(abs(x[5]), abs(x[3])))
        print(f"  t={i*dt:.2f}s: V={speed:.1f} alpha={alpha:.2f} dqr={np.degrees(dx[11]):.2f}")
    if abs(np.degrees(dx[11])) > 1e6:
        print(f"  BLOW-UP at t={i*dt:.2f}s")
        break
