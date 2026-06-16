"""
MAIN.py
=======
Główny skrypt symulacji 6DOF rakiety.

Układ współrzędnych: Launch Frame
  X — wzdłuż azymutu strzału [m]
  Y — w prawo prostopadle    [m]
  Z — w dół                  [m]

Kąty Eulera ZYX odniesione do Launch Frame:
  ψ (yaw)   = 0° gdy nos skierowany wzdłuż X_LF
  θ (pitch) = elewacja na starcie
  φ (roll)  = 0° bez przechylenia

Użycie:
    python MAIN.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import (
    build_mass_model, build_propulsion,
    build_geometry, build_initial_state,
)
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import GeoModule, MissionConfig
from models.launcher import LauncherConfig
from models.force_logger import ForceLogger

# ============================================================================
# Konfiguracja — zmień tutaj
# ============================================================================
CASE_NAME    = "rocket_70mm_baseline"
MISSION_YAML = "missions/mission_01.yaml"
AERO_METHOD  = "missile_datcom"   # "barrowman" lub "missile_datcom"
USE_GEO      = False               # True = mapa i wykresy geograficzne

# ============================================================================
# Ładowanie
# ============================================================================
print(f"Konfiguracja: {CASE_NAME}")
print(f"Metoda aero:  {AERO_METHOD}")

cfg     = load_config(f"configurations/{CASE_NAME}.yaml")
mission = MissionConfig.from_yaml(MISSION_YAML)
geo     = GeoModule(mission) if USE_GEO else None
initial = build_initial_state(mission)

print(f"Misja: Az={mission.azimuth}° El={mission.elevation}° "
      f"({mission.lat:.4f}N, {mission.lon:.4f}E)")

mass = build_mass_model(cfg)
prop = build_propulsion(cfg)
geom = build_geometry(cfg)
aero = get_aero_model(CASE_NAME, method=AERO_METHOD, force_rerun=True)
atm  = create_atmosphere("ISA")
grv  = create_gravity("constant")
launcher = LauncherConfig(L_rail=2.0)   # L_rail = długość szyny startowej

logger = ForceLogger(
    case_name = CASE_NAME,
    log_dir   = "logs",
    enabled   = False,        # False = wyłączone
)

# ============================================================================
# Symulacja
# ============================================================================
print("\nUruchamianie symulacji 6DOF...")

fm = ForceModel6DOF(
    atmosphere = atm,
    mass_model = mass,
    aero_model = aero,
    gravity    = grv,
    geometry   = geom,
    propulsion = prop,
    launcher = launcher,
    logger = logger
)

result = run_simulation_6dof(
    force_model   = fm,
    initial_state = initial,
    t_max         = 120.0,
    z_ground      = -1.0,
    max_step      = 0.05,
)
logger.close()

# ============================================================================
# Wyniki podstawowe
# ============================================================================
lf      = geo.ned_to_lf(result) if geo else None
euler   = geo.euler_lf(result)  if geo else None
range_m = float(lf["x_lf"][-1]) if lf else float(result.x[-1])
alt_max = float(lf["alt"].max()) if lf else float(-result.z.min())

print(f"\n{'='*45}")
print(f"  Zasięg:      {range_m/1000:.3f} km")
print(f"  Wys. max:    {alt_max:.1f} m")
print(f"  Czas lotu:   {result.t[-1]:.2f} s")
print(f"  V_max:       {result.speed.max():.1f} m/s")
print(f"  V_upadek:    {result.speed[-1]:.1f} m/s")
print(f"{'='*45}")

# ============================================================================
# Wykresy 6DOF — 3×3
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(
    f"Symulacja 6DOF — {CASE_NAME}  [{AERO_METHOD}]  "
    f"El={mission.elevation}°  Az={mission.azimuth}°",
    fontsize=12, fontweight='bold'
)

# 1. Trajektoria — płaszczyzna strzału (LF)
ax = axes[0, 0]
x_plot = lf["x_lf"] if lf else result.x
h_plot = lf["alt"]  if lf else -result.z
ax.plot(x_plot / 1000, h_plot, 'b-', lw=2)
ax.plot(x_plot[0]  / 1000, h_plot[0],  'go', ms=8, label="Start")
ax.plot(x_plot[-1] / 1000, h_plot[-1], 'rx', ms=10, mew=2, label="Upadek")
ax.set_xlabel("Zasięg wzdłuż azymutu [km]")
ax.set_ylabel("Wysokość [m]")
ax.set_title("Trajektoria (płaszczyzna strzału, LF)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 2. Rzut poziomy — Launch Frame
ax = axes[0, 1]
x_plot = lf["x_lf"] if lf else result.x
y_plot = lf["y_lf"] if lf else result.y
ax.plot(x_plot, y_plot, 'g-', lw=1.5)
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel("Zasięg x_LF [m]")
ax.set_ylabel("Odchylenie boczne y_LF [m]")
ax.set_title("Tor lotu (rzut poziomy, LF)")
ax.grid(alpha=0.3)

# 3. Prędkość
ax = axes[0, 2]
ax.plot(result.t, result.speed, 'r-', lw=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("V [m/s]")
ax.set_title("Prędkość")
ax.grid(alpha=0.3)

# 4. Kąty Eulera ZYX — Launch Frame
ax = axes[1, 0]
if euler:
    ax.plot(result.t, euler["psi_deg"],   'b-',  lw=1.5, label='ψ yaw')
    ax.plot(result.t, euler["theta_deg"], 'g-',  lw=1.5, label='θ pitch')
    ax.plot(result.t, euler["phi_deg"],   'r--', lw=1.2, label='φ roll')
else:
    ax.plot(result.t, np.degrees(result.psi),   'b-',  lw=1.5, label='ψ yaw')
    ax.plot(result.t, np.degrees(result.theta), 'g-',  lw=1.5, label='θ pitch')
    ax.plot(result.t, np.degrees(result.phi),   'r--', lw=1.2, label='φ roll')
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Kąt [°]")
ax.set_title("Kąty Eulera ZYX (Launch Frame)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 5. Kąty aerodynamiczne
ax = axes[1, 1]
ax.plot(result.t, np.degrees(result.alpha), 'b-', lw=1.5, label='α (natarcia)')
ax.plot(result.t, np.degrees(result.beta),  'r-', lw=1.2, label='β (ślizgu)')
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Kąt [°]")
ax.set_title("Kąty aerodynamiczne")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 6. Prędkości kątowe
ax = axes[1, 2]
ax.plot(result.t, np.degrees(result.p),  'b-',  lw=1.5, label='p (roll)')
ax.plot(result.t, np.degrees(result.qr), 'g-',  lw=1.5, label='q (pitch)')
ax.plot(result.t, np.degrees(result.r),  'r--', lw=1.2, label='r (yaw)')
ax.set_xlabel("Czas [s]")
ax.set_ylabel("[°/s]")
ax.set_title("Prędkości kątowe")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 7. Wysokość vs czas
ax = axes[2, 0]
h_plot = lf["alt"] if lf else -result.z
ax.plot(result.t, h_plot, 'b-', lw=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Wysokość [m]")
ax.set_title("Wysokość vs czas")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)

# 8. Prędkości body frame
ax = axes[2, 1]
ax.plot(result.t, result.u, 'b-', lw=1.5, label='u (wzdłużna)')
ax.plot(result.t, result.v, 'r-', lw=1.2, label='v (boczna)')
ax.plot(result.t, result.w, 'g-', lw=1.2, label='w (normalna)')
ax.set_xlabel("Czas [s]")
ax.set_ylabel("[m/s]")
ax.set_title("Prędkości body frame")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 9. Liczba Macha
ax = axes[2, 2]
a_sound = np.array([
    atm.at(max(0., float(-result.z[i]))).speed_of_sound
    for i in range(len(result.t))
])
mach_arr = result.speed / a_sound
ax.plot(result.t, mach_arr, color='purple', lw=1.5)
ax.axhline(1.0, color='gray', lw=0.8, ls='--', label='Ma=1')
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Mach [-]")
ax.set_title("Liczba Macha")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("simulation_result.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: simulation_result.png")

# ============================================================================
# Moduł geograficzny
# ============================================================================
if USE_GEO and geo:
    print("\nGenerowanie map...")

    # Mapa statyczna PNG
    geo.plot(result, output_path="trajectory_map.png")

    # Mapa interaktywna HTML
    geo.plot_interactive(result, output_path="trajectory_map.html")

    print("Zapisano: trajectory_map.png, trajectory_map.html")

print("\nGotowe.")
