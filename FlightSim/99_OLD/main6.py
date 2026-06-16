"""
main6.py
========
Przykładowa symulacja 6DOF rakiety 70mm.
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

# ============================================================================
# Parametry rakiety
# ============================================================================

DIAMETER = 0.070
XCP      = 0.7

geometry = RocketGeometry6DOF.from_diameter(DIAMETER, XCP)

ixx_model  = ConstantIxx(Ixx=0.012)   # ~10% Iyy, typowe dla smukłej rakiety

mass_model = MassModel6DOF(
    m_full    = 6.0,
    m_empty   = 4.2,
    t_burn    = 1.8,
    xcg_full  = 0.42,
    xcg_empty = 0.38,
    Iyy_full  = 0.12,
    Iyy_empty = 0.09,
    ixx_model = ixx_model,
)

aero_model = ConstantAero(
    CA       = 0.35,
    CN_alpha = 5,    
    Cmq      = -1500,
    use_xcp_moment = True,
)

propulsion = PropulsionConfig6DOF(thrust=1800.0)

launcher = LauncherConfig(L_rail=3.0)   # 3m szyna

force_model = ForceModel6DOF(
    atmosphere  = create_atmosphere("ISA"),
    mass_model  = mass_model,
    aero_model  = aero_model,
    gravity     = create_gravity("constant"),
    geometry    = geometry,
    propulsion  = propulsion,
    launcher    = launcher,
)

# ============================================================================
# Warunki startowe
# ============================================================================

ELEVATION_DEG = 75.0
AZIMUTH_DEG   = 0.0

initial_state = State6DOF.initial(
    elevation_deg = ELEVATION_DEG,
    azimuth_deg   = AZIMUTH_DEG,
)

print(f"Stan startowy: {initial_state}")
print(f"Model masy:    {mass_model}")
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
ax.plot(result.x, result.y, "g-", lw=1.5)
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
ax = axes[1, 0]
ax.plot(t, np.degrees(result.psi),   "b-",  lw=1.5, label="ψ yaw")
ax.plot(t, np.degrees(result.theta), "g-",  lw=1.5, label="θ pitch")
ax.plot(t, np.degrees(result.phi),   "r--", lw=1.5, label="φ roll")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Kąt [°]")
ax.set_title("Kąty Eulera ZYX")
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

# 9. Norma kwaterniona (kontrola dryftu)
ax = axes[2, 2]
q_norm = np.sqrt(result.q0**2 + result.q1**2 + result.q2**2 + result.q3**2)
ax.plot(t, q_norm, "k-", lw=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("|q|")
ax.set_title("Norma kwaterniona (kontrola)")
ax.axhline(1.0, color="r", lw=0.8, ls="--")
ax.set_ylim(0.999, 1.001)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("flight_6dof.png", dpi=120, bbox_inches="tight")
plt.show()
print("Wykres zapisany: flight_6dof.png")
