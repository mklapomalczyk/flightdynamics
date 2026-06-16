"""
main.py
=======
Przykładowa symulacja lotu rakiety 70mm.

Parametry są celowo uproszczone (stałe wsp. aerodynamiczne,
liniowy model masy) — punkt startowy do kalibracji z danymi DATCOM.

Uruchomienie:
    python main.py

Wymagania:
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# Importy z pakietu FlightSim
from core.state import State3DOF
from core.solver import run_simulation
from models.atmosphere import create_atmosphere, AtmosphereModel
from models.mass import MassModel
from models.aerodynamics import ConstantAero
from models.gravity import create_gravity
from forces.force_model import ForceModel, RocketGeometry, PropulsionConfig


# ============================================================================
# Parametry rakiety 70mm (przykładowe — do kalibracji)
# ============================================================================

# Geometria
DIAMETER   = 0.070          # średnica [m]
XCP        = 0.55           # centrum parcia od nosa [m] — do kalibracji z DATCOM
geometry   = RocketGeometry.from_diameter(DIAMETER, XCP)

# Masa
mass_model = MassModel(
    m_full    = 6.0,         # masa startowa [kg]
    m_empty   = 4.2,         # masa po wypaleniu [kg]
    t_burn    = 1.8,         # czas palenia [s]
    xcg_full  = 0.42,        # xcg startowe od nosa [m]
    xcg_empty = 0.38,        # xcg końcowe od nosa [m]
    Iyy_full  = 0.12,        # moment bezwładności pitch, pełna [kg·m²]
    Iyy_empty = 0.09,        # moment bezwładności pitch, pusta [kg·m²]
)

# Aerodynamika (stałe wsp. — do zastąpienia danymi DATCOM)
aero_model = ConstantAero(
    CA       = 0.35,          # osiowy wsp. siły [-]
    CN_alpha = 5,           # pochodna normalnego wsp. siły [1/rad]
    Cmq      = -1500.0,         # tłumienie kątowe [1/rad]
    use_xcp_moment = True,    # moment z przesunięcia xcp-xcg
)

# Silnik
propulsion = PropulsionConfig(
    thrust = 1800.0,          # ciąg [N]
)

# Modele środowiskowe
atmosphere = create_atmosphere("ISA")
gravity    = create_gravity("constant")

# ============================================================================
# Złożenie modelu sił
# ============================================================================

force_model = ForceModel(
    atmosphere  = atmosphere,
    mass_model  = mass_model,
    aero_model  = aero_model,
    gravity     = gravity,
    geometry    = geometry,
    propulsion  = propulsion,
)

# ============================================================================
# Warunki startowe
# ============================================================================

ELEVATION_DEG = 75.0   # kąt elewacji wyrzutni [stopnie]

initial_state = State3DOF.initial(
    elevation_deg = ELEVATION_DEG,
    speed_0       = 0.0,          # rakieta startuje ze stojaka
)

print(f"Stan startowy: {initial_state}")
print(f"Model masy:    {mass_model}")
print(f"Model aero:    {aero_model}")
print()

# ============================================================================
# Symulacja
# ============================================================================

result = run_simulation(
    force_model   = force_model,
    initial_state = initial_state,
    t_max         = 100,
    dt_output     = 0.001,
    method        = "RK45",
)
# print(force_model)
# H=0.0

# while H < 86000:
    # atm = atmosphere.at(H)
    # T = atm.temperature
    # # p = atm.pressure
    # # print(T)
    # if T < 0:
        # print("Temp lower than 0 -> error")
    # H = H + 1

print(result.summary())

# ============================================================================
# Wykresy
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    f"Symulacja lotu 3DOF — rakieta 70mm, elewacja {ELEVATION_DEG}°",
    fontsize=13, fontweight="bold"
)

t = result.t

# 1. Trajektoria
ax = axes[0, 0]
ax.plot(result.x_pos, result.z_pos, "b-", linewidth=1.5)
ax.set_xlabel("Zasięg poziomy [m]")
ax.set_ylabel("Wysokość [m]")
ax.set_title("Trajektoria lotu")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
ax.axhline(0, color="k", linewidth=0.8)

# 2. Prędkość
ax = axes[0, 1]
ax.plot(t, result.speed, "r-", linewidth=1.5, label="V [m/s]")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Prędkość [m/s]")
ax.set_title("Prędkość")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Kąty
ax = axes[0, 2]
ax.plot(t, np.degrees(result.theta), "g-",  linewidth=1.5, label="θ — pochylenie")
ax.plot(t, np.degrees(result.alpha), "m--", linewidth=1.5, label="α — kąt natarcia")
ax.plot(t, np.degrees(result.gamma), "c:",  linewidth=1.5, label="γ — tor lotu")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Kąt [°]")
ax.set_title("Kąty")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Prędkości w body frame
ax = axes[1, 0]
ax.plot(t, result.u, "b-", linewidth=1.5, label="u (wzdłużna)")
ax.plot(t, result.w, "r-", linewidth=1.5, label="w (normalna)")
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Prędkość [m/s]")
ax.set_title("Prędkości w body frame")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Prędkość kątowa
ax = axes[1, 1]
ax.plot(t, np.degrees(result.q), "k-", linewidth=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("q [°/s]")
ax.set_title("Prędkość kątowa pitch")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

# 6. Wysokość vs czas
ax = axes[1, 2]
ax.plot(t, result.z_pos, "b-", linewidth=1.5)
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Wysokość [m]")
ax.set_title("Wysokość vs czas")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("flight_3dof.png", dpi=120, bbox_inches="tight")
plt.show()
print("Wykres zapisany: flight_3dof.png")
