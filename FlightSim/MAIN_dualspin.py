"""
MAIN_dualspin.py — symulacja 6DOF dual-spin
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aero import get_aero_model
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_mass_model, build_propulsion, build_geometry, build_initial_state
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import MissionConfig

# ============================================================================
CASE_NAME    = "rocket_70mm_dualspin"
MISSION_YAML = "missions/mission_01.yaml"
AERO_METHOD  = "missile_datcom"
# ============================================================================

print(f"Konfiguracja: {CASE_NAME}")
print(f"Metoda aero:  {AERO_METHOD}")

cfg     = load_config(f"configurations/{CASE_NAME}.yaml")
mission = MissionConfig.from_yaml(MISSION_YAML)
initial = build_initial_state(mission)

print(f"Misja: Az={mission.azimuth}° El={mission.elevation}°")

ds = cfg.dual_spin
if ds and ds.enabled:
    print(f"Dual-spin: bearing_x={ds.bearing_x}m  friction={ds.bearing_friction}")
    print(f"  Aft:  mass={ds.aft.mass}kg  Ixx={ds.aft.Ixx}  xcg={ds.aft.xcg}m")
    print(f"  Fwd:  mass={ds.forward.mass}kg  Ixx={ds.forward.Ixx}  xcg={ds.forward.xcg}m")

mass = build_mass_model(cfg)
prop = build_propulsion(cfg)
geom = build_geometry(cfg)
aero = get_aero_model("rocket_70mm_baseline", method=AERO_METHOD)
atm  = create_atmosphere("ISA")
grv  = create_gravity("constant")

print("\nUruchamianie symulacji 6DOF dual-spin...")

fm = ForceModel6DOF(
    atmosphere = atm,
    mass_model = mass,
    aero_model = aero,
    gravity    = grv,
    geometry   = geom,
    propulsion = prop,
    cfg        = cfg,
)

result = run_simulation_6dof(
    force_model   = fm,
    initial_state = initial,
    t_max         = 120.0,
    z_ground      = -1.0,
    max_step      = 0.05,
)

alt_max  = float((-result.z).max())
p_aft_eq = np.degrees(result.p[-1])
p_fwd_eq = np.degrees(result.p_fwd[-1]) if len(result.p_fwd) > 0 else 0.0

print(f"\n{'='*50}")
print(f"  Zasięg:       {result.x[-1]/1000:.3f} km")
print(f"  Wys. max:     {alt_max:.1f} m")
print(f"  Czas lotu:    {result.t[-1]:.2f} s")
print(f"  V_max:        {result.speed.max():.1f} m/s")
print(f"  p_aft końc.:  {p_aft_eq:.0f} °/s  ({p_aft_eq/360:.1f} obr/s)")
print(f"  p_fwd końc.:  {p_fwd_eq:.2f} °/s")
print(f"{'='*50}")

# ============================================================================
# Wykresy
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(
    f"Symulacja 6DOF dual-spin — {CASE_NAME}  [{AERO_METHOD}]  "
    f"El={mission.elevation}°  Az={mission.azimuth}°",
    fontsize=12, fontweight='bold'
)

# 1. Trajektoria
ax = axes[0, 0]
ax.plot(result.x/1000, -result.z, 'b-', lw=2)
ax.plot(result.x[0]/1000, 0, 'go', ms=8, label='Start')
ax.plot(result.x[-1]/1000, -result.z[-1], 'rx', ms=10, mew=2, label='Upadek')
ax.set_xlabel("Zasięg [km]"); ax.set_ylabel("Wysokość [m]")
ax.set_title("Trajektoria (płaszczyzna strzału)")
ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2. Rzut poziomy — Launch Frame
ax = axes[0, 1]
x_plot = result.x
y_plot = result.y
ax.plot(x_plot, y_plot, 'g-', lw=1.5)
ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel("Zasięg x_LF [m]")
ax.set_ylabel("Odchylenie boczne y_LF [m]")
ax.set_title("Tor lotu (rzut poziomy, LF)")
ax.grid(alpha=0.3)

# 3. Roll — AFT i FWD w obr/s
ax = axes[0, 2]
ax.plot(result.t, np.degrees(result.p)/360, 'b-', lw=2, label='p_aft')
if len(result.p_fwd) > 0:
    ax.plot(result.t, np.degrees(result.p_fwd)/360, 'r--', lw=1.8, label='p_fwd')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel("Czas [s]"); ax.set_ylabel("[obr/s]")
ax.set_title("Prędkości kątowe roll [obr/s]")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 4. Kąty aerodynamiczne
ax = axes[1, 0]
ax.plot(result.t, np.degrees(result.alpha), 'b-', lw=1.5, label='α (natarcia)')
ax.plot(result.t, np.degrees(result.beta),  'r-', lw=1.2, label='β (ślizgu)')
ax.set_xlabel("Czas [s]"); ax.set_ylabel("Kąt [°]")
ax.set_title("Kąty aerodynamiczne")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 5. Prędkości kątowe pitch i yaw
ax = axes[1, 1]
ax.plot(result.t, np.degrees(result.qr), 'g-',  lw=1.5, label='q (pitch)')
ax.plot(result.t, np.degrees(result.r),  'r--', lw=1.2, label='r (yaw)')
ax.set_xlabel("Czas [s]"); ax.set_ylabel("[°/s]")
ax.set_title("Prędkości kątowe pitch i yaw")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 6. Prędkość
ax = axes[1, 2]
ax.plot(result.t, result.speed, 'r-', lw=1.5)
ax.set_xlabel("Czas [s]"); ax.set_ylabel("V [m/s]")
ax.set_title("Prędkość"); ax.grid(alpha=0.3)

# 7. Wysokość vs czas
ax = axes[2, 0]
ax.plot(result.t, -result.z, 'b-', lw=1.5)
ax.set_xlabel("Czas [s]"); ax.set_ylabel("Wysokość [m]")
ax.set_title("Wysokość vs czas")
ax.set_ylim(bottom=0); ax.grid(alpha=0.3)

# 8. Różnica Δp = p_aft - p_fwd
ax = axes[2, 1]
if len(result.p_fwd) > 0:
    delta_p = np.degrees(result.p - result.p_fwd)
    ax.plot(result.t, delta_p, color='purple', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_ylabel("Δp [°/s]")
else:
    ax.text(0.5, 0.5, 'Brak danych p_fwd', ha='center', va='center',
            transform=ax.transAxes)
ax.set_xlabel("Czas [s]")
ax.set_title("Δp = p_aft - p_fwd")
ax.grid(alpha=0.3)

# 9. Mach
ax = axes[2, 2]
a_sound = np.array([atm.at(max(0., float(-result.z[i]))).speed_of_sound
                    for i in range(len(result.t))])
ax.plot(result.t, result.speed/a_sound, color='purple', lw=1.5)
ax.axhline(1.0, color='gray', lw=0.8, ls='--', label='Ma=1')
ax.set_xlabel("Czas [s]"); ax.set_ylabel("Mach [-]")
ax.set_title("Liczba Macha")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("simulation_dualspin.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Zapisano: simulation_dualspin.png")
