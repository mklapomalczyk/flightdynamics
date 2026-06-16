"""
main.py
========
Przykładowa symulacja 6DOF dla ARTEMIDY - rakiety 70mm.
"""

import numpy as np
import matplotlib.pyplot as plt

from core.state6 import State6DOF
from core.solver6 import run_simulation_6dof
from models.atmosphere import create_atmosphere
from models.mass6 import MassModel6DOF, ConstantIxx
from models.aerodynamics import ConstantAero
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF, RocketGeometry6DOF, PropulsionConfig6DOF
from models.launcher import LauncherConfig
from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from geo.geographic import GeoModule, MissionConfig

# ============================================================================
# Parametry rakiety
# ============================================================================

mission = MissionConfig.from_yaml("missions/mission_01.yaml")

cfg  = load_config("configurations/rocket_70mm_baseline.yaml")
mass = build_mass_model(cfg)
prop = build_propulsion(cfg)
geom = build_geometry(cfg)
geo     = GeoModule(mission)

### AERODYNAMICS from Fleeman book
# aero_model = aero = get_aero_model(
    # case_name = "rocket_70mm_baseline",
    # Cmq       = -500.,
# )

aero_model= get_aero_model("rocket_70mm_baseline", method="missile_datcom", force_rerun=True)

atm = create_atmosphere("ISA")

launcher = LauncherConfig(L_rail=3.0)   # 3m szyna

force_model = ForceModel6DOF(
    atmosphere  = atm,
    mass_model  = mass,
    aero_model  = aero_model,
    gravity     = create_gravity("constant"),
    geometry    = geom,
    propulsion  = prop,
    launcher    = launcher,
)

# ============================================================================
# Warunki startowe
# ============================================================================

initial_state = build_initial_state(mission)

print(f"Stan startowy: {initial_state}")
print(f"Model masy:    {mass}")
print()

# ============================================================================
# Symulacja
# ============================================================================

result = run_simulation_6dof(
    force_model   = force_model,
    initial_state = initial_state,
    t_max         = 100,
    dt_output     = 0.001,
)

print(result.summary())

ELEVATION_DEG = mission.elevation
azimuth_deg   = mission.azimuth

# # Statyczna mapa PNG
# geo.plot(result, output_path="trajectory.png")

# # Interaktywna mapa HTML
# geo.plot_interactive(result, output_path="trajectory.html")

# ============================================================================
# Wykresy
# ============================================================================

t = result.t
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle(f"Symulacja 6DOF — rakieta 70mm, elewacja {ELEVATION_DEG}°",
             fontsize=13, fontweight="bold")

# 1. Trajektoria XZ
ax = axes[0, 0]
ax.plot(result.x, -result.z, "b-", lw=1.5)
ax.set_xlabel("Zasięg x [m]")
ax.set_ylabel("Wysokość z [m]")
ax.set_title("Trajektoria (płaszczyzna XZ)")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", lw=0.8)

# 2. Trajektoria 3D (rzut XY)
ax = axes[0, 1]
lf = geo.ned_to_lf(result)
ax.plot(lf["x_lf"], lf["y_lf"], "g-", lw=1.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
ax.set_ylim(-1, 1)  # ograniczenie do ±1m
ax.set_xlabel("Zasięg x [m]")
ax.set_ylabel("Odchylenie boczne y [m]")
ax.set_title("Tor lotu (rzut poziomy)")
ax.grid(True, alpha=0.3)

# 3. Prędkość
ax = axes[0, 2]
ax.plot(t, result.speed, "r-", lw=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("V [m/s]")
ax.set_title("Prędkość")
ax.grid(True, alpha=0.3)

# 4. Kąty Eulera
euler = geo.euler_lf(result)
ax = axes[1, 0]
ax.plot(result.t, euler["psi_deg"],   'b-',  lw=1.5, label='ψ yaw (LF)')
ax.plot(result.t, euler["theta_deg"], 'g-',  lw=1.5, label='θ pitch (LF)')
ax.plot(result.t, euler["phi_deg"],   'r--', lw=1.5, label='φ roll (LF)')
ax.set_title("Kąty Eulera ZYX (Launch Frame)")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Kąt [°]")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. Kąty aerodynamiczne
ax = axes[1, 1]
ax.plot(t, np.degrees(result.alpha), "b-", lw=1.5, label="α (natarcia)")
ax.plot(t, np.degrees(result.beta),  "r-", lw=1.5, label="β (ślizgu)")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Kąt [°]")
ax.set_title("Kąty aerodynamiczne")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", lw=0.8, ls="--")

# 6. Prędkości kątowe
ax = axes[1, 2]
ax.plot(t, np.degrees(result.p),  "b-",  lw=1.5, label="p (roll)")
ax.plot(t, np.degrees(result.qr), "g-",  lw=1.5, label="q (pitch)")
ax.plot(t, np.degrees(result.r),  "r--", lw=1.5, label="r (yaw)")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("[°/s]")
ax.set_title("Prędkości kątowe")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", lw=0.8, ls="--")

# 7. Wysokość vs czas
ax = axes[2, 0]
ax.plot(t, -result.z, "b-", lw=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Wysokość [m]")
ax.set_title("Wysokość vs czas")
ax.grid(True, alpha=0.3)

# 8. Prędkości body frame
ax = axes[2, 1]
ax.plot(t, result.u, "b-", lw=1.5, label="u (wzdłużna)")
ax.plot(t, result.v, "r-", lw=1.5, label="v (boczna)")
ax.plot(t, result.w, "g-", lw=1.5, label="w (normalna)")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("[m/s]")
ax.set_title("Prędkości body frame")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 9a. Norma kwaterniona (kontrola dryftu)
# ax = axes[2, 2]
# q_norm = np.sqrt(result.q0**2 + result.q1**2 + result.q2**2 + result.q3**2)
# ax.plot(t, q_norm, "k-", lw=1.5)
# ax.set_xlabel("Czas [s]")
# ax.set_ylabel("|q|")
# ax.set_title("Norma kwaterniona (kontrola)")
# ax.axhline(1.0, color="r", lw=0.8, ls="--")
# ax.set_ylim(0.999, 1.001)
# ax.grid(True, alpha=0.3)

# 9b. Liczba Macha vs czas
mach_arr = np.array([atm.at(-z).mach(s) for z, s in zip(result.z, result.speed)])
ax_mach = axes[2, 2]  # podmień na właściwy indeks subplotu
ax_mach.plot(result.t, mach_arr, 'purple', lw=1.5)
ax_mach.axhline(1.0, color='gray', lw=0.8, ls='--', label='Ma=1')
ax_mach.set_xlabel("Czas [s]")
ax_mach.set_ylabel("Mach [-]")
ax_mach.set_title("Liczba Macha")
ax_mach.legend()
ax_mach.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("flight_6dof.png", dpi=120, bbox_inches="tight")
plt.show()
print("Wykres zapisany: flight_6dof.png")
