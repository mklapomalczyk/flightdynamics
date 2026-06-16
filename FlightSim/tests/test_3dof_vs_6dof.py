"""
tests/test_3dof_vs_6dof.py
==========================
Spójność 3DOF i 6DOF przy symetrycznym locie.

Konwencje:
  3DOF: Z_body w dół, CN_alpha > 0
  6DOF: Z_body w dół (kwaterniony), CN_alpha > 0, brak negacji alpha

Przy identycznych parametrach i locie symetrycznym (azymut=0)
trajektorie 3DOF i 6DOF powinny być zbieżne.

Uruchomienie:
    python tests/test_3dof_vs_6dof.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from core.state import State3DOF
from core.solver import run_simulation
from forces.force_model import ForceModel, RocketGeometry, PropulsionConfig

from core.state6 import State6DOF
from core.solver6 import run_simulation_6dof
from forces.force_model6 import ForceModel6DOF, RocketGeometry6DOF, PropulsionConfig6DOF

from models.atmosphere import create_atmosphere
from models.mass import MassModel
from models.mass6 import MassModel6DOF, ConstantIxx
from models.aerodynamics import ConstantAero
from models.gravity import create_gravity

PASS = "  [PASS]"; FAIL = "  [FAIL]"
results = []

def check(name, condition, info=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f"  ({info})" if info else ""))
    results.append(condition)
    return condition

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ============================================================================
# Parametry — identyczne dla obu modeli
# ============================================================================
ELEVATION = 75.0
AZIMUTH = 0.0
CN_ALPHA  = 5
CA        = 0.35
CMQ       = -1500.0
THRUST    = 1800.0
M_FULL    = 6.0;  M_EMPTY   = 4.2;  T_BURN = 1.8
XCG_FULL  = 0.42; XCG_EMPTY = 0.38
IYY_FULL  = 0.12; IYY_EMPTY = 0.09
DIAMETER  = 0.070; XCP = 0.55

section("Konfiguracja")
print(f"  Elewacja: {ELEVATION}°  |  CN_alpha: {CN_ALPHA}  |  Cmq: {CMQ}")
print(f"  Ciąg: {THRUST} N  |  t_burn: {T_BURN} s")

# ============================================================================
# Sanity check: znak momentu identyczny w obu modelach
# ============================================================================
section("Sanity check — znak momentu")

from core.quaternion import euler_zyx_to_quat

fm3 = ForceModel(
    atmosphere=create_atmosphere("ISA"),
    mass_model=MassModel(M_FULL, M_EMPTY, T_BURN, XCG_FULL, XCG_EMPTY,
                         IYY_FULL, IYY_EMPTY),
    aero_model=ConstantAero(CA=CA, CN_alpha=CN_ALPHA, Cmq=0., use_xcp_moment=True),
    gravity=create_gravity("constant"),
    geometry=RocketGeometry.from_diameter(DIAMETER, XCP),
    propulsion=PropulsionConfig(thrust=0.),
)
fm6 = ForceModel6DOF(
    atmosphere=create_atmosphere("ISA"),
    mass_model=MassModel6DOF(M_FULL, M_EMPTY, T_BURN, XCG_FULL, XCG_EMPTY,
                              IYY_FULL, IYY_EMPTY, ixx_model=ConstantIxx(0.012)),
    aero_model=ConstantAero(CA=CA, CN_alpha=CN_ALPHA, Cmq=0., use_xcp_moment=True),
    gravity=create_gravity("constant"),
    geometry=RocketGeometry6DOF.from_diameter(DIAMETER, XCP),
    propulsion=PropulsionConfig6DOF(thrust=0.),
)

# theta=0, u=200, w=+5 → alpha>0 (nos powyżej toru, Z_body w dół)
x3 = State3DOF.initial(0.).to_numpy(); x3[2]=200.; x3[3]=5.0
q0 = euler_zyx_to_quat(0,0,0)
x6 = np.array([0,0,0., 200.,0.,5., q0[0],q0[1],q0[2],q0[3], 0.,0.,0.])

dx3 = fm3.derivatives(5., x3)
dx6 = fm6.derivatives(5., x6)

check("3DOF: dq/dt < 0 przy alpha>0 (stabilizujący)",
    dx3[5] < 0, f"dq={np.degrees(dx3[5]):.2f} °/s²")
check("6DOF: dqr/dt < 0 przy alpha>0 (stabilizujący)",
    dx6[11] < 0, f"dqr={np.degrees(dx6[11]):.2f} °/s²")
check("Wartości momentów zbieżne (tol 5%)",
    abs(dx3[5] - dx6[11]) / abs(dx3[5]) < 0.05,
    f"3DOF={np.degrees(dx3[5]):.2f}, 6DOF={np.degrees(dx6[11]):.2f} °/s²")

# ============================================================================
# Symulacje
# ============================================================================
section("Symulacje")

fm3_full = ForceModel(
    atmosphere=create_atmosphere("ISA"),
    mass_model=MassModel(M_FULL, M_EMPTY, T_BURN, XCG_FULL, XCG_EMPTY,
                         IYY_FULL, IYY_EMPTY),
    aero_model=ConstantAero(CA=CA, CN_alpha=CN_ALPHA, Cmq=CMQ, use_xcp_moment=True),
    gravity=create_gravity("constant"),
    geometry=RocketGeometry.from_diameter(DIAMETER, XCP),
    propulsion=PropulsionConfig(thrust=THRUST),
)
fm6_full = ForceModel6DOF(
    atmosphere=create_atmosphere("ISA"),
    mass_model=MassModel6DOF(M_FULL, M_EMPTY, T_BURN, XCG_FULL, XCG_EMPTY,
                              IYY_FULL, IYY_EMPTY, ixx_model=ConstantIxx(0.012)),
    aero_model=ConstantAero(CA=CA, CN_alpha=CN_ALPHA, Cmq=CMQ, use_xcp_moment=True),
    gravity=create_gravity("constant"),
    geometry=RocketGeometry6DOF.from_diameter(DIAMETER, XCP),
    propulsion=PropulsionConfig6DOF(thrust=THRUST),
)

initial_state3 = State3DOF.initial(
    elevation_deg = ELEVATION,
    speed_0       = 0.0,          # rakieta startuje ze stojaka
)

initial_state6 = State6DOF.initial(
    elevation_deg = ELEVATION,
    azimuth_deg   = AZIMUTH,
)

print("  3DOF...", end=" ", flush=True)
r3 = run_simulation(    
    force_model   = fm3_full,
    initial_state = initial_state3,
    t_max         = 100,
    dt_output     = 0.001,
    method        = "RK45",
    )
print(f"OK  t={r3.t[-1]:.1f}s, z_max={r3.max_altitude:.0f}m")

print("  6DOF...", end=" ", flush=True)
r6 = run_simulation_6dof(    
    force_model   = fm6_full,
    initial_state = initial_state6,
    t_max         = 100,
    dt_output     = 0.001,
    )
print(f"OK  t={r6.t[-1]:.1f}s, z_max={r6.max_altitude:.0f}m")

# ============================================================================
# Porównanie
# ============================================================================
section("Porównanie trajektorii")

# Uwaga: 3DOF z = wysokość (Z w górę), 6DOF z = Z w dół
# 3DOF: z_pos = wysokość, x_pos = zasięg
# 6DOF: z = Z_launch (ujemne w górze), x = zasięg
t_end = min(r3.t[-1], r6.t[-1])
tc    = r3.t[r3.t <= t_end]

# Wysokości: 3DOF → z_pos, 6DOF → -z
alt3 = np.interp(tc, r3.t, -r3.z_pos)
alt6 = interp1d(r6.t, -r6.z, bounds_error=False, fill_value='extrapolate')(tc)

x3c  = np.interp(tc, r3.t, r3.x_pos)
x6c  = interp1d(r6.t, r6.x, bounds_error=False, fill_value='extrapolate')(tc)

V3c  = np.interp(tc, r3.t, r3.speed)
V6c  = interp1d(r6.t, r6.speed, bounds_error=False, fill_value='extrapolate')(tc)

th3c = np.interp(tc, r3.t, np.degrees(r3.theta))
th6c = interp1d(r6.t, np.degrees(r6.theta), bounds_error=False, fill_value='extrapolate')(tc)

err_alt = np.max(np.abs(alt6 - alt3)) / max(np.max(alt3), 1.)
err_x   = np.max(np.abs(x6c  - x3c))  / max(np.max(x3c),  1.)
err_V   = np.max(np.abs(V6c  - V3c))  / max(np.max(V3c),  1.)
err_th  = np.max(np.abs(th6c - th3c))

check("Czas lotu zbieżny (tol 2%)",
    abs(r6.t[-1] - r3.t[-1]) / r3.t[-1] < 0.02,
    f"3DOF={r3.t[-1]:.2f}s, 6DOF={r6.t[-1]:.2f}s")

check("Max wysokość zbieżna (tol 2%)",
    abs(r6.max_altitude - r3.max_altitude) / r3.max_altitude < 0.02,
    f"3DOF={r3.max_altitude:.0f}m, 6DOF={r6.max_altitude:.0f}m")

check("Trajektoria wysokości zbieżna (max błąd < 2%)",
    err_alt < 0.02, f"max|Δalt|/alt_max = {err_alt*100:.2f}%")

check("Prędkość V(t) zbieżna (max błąd < 2%)",
    err_V < 0.02, f"max|ΔV|/V_max = {err_V*100:.2f}%")

check("Kąt pochylenia θ(t) zbieżny (max błąd < 1°)",
    err_th < 1.0, f"max|Δθ| = {err_th:.3f}°")

check("Brak ruchu bocznego 6DOF (max|y| < 0.01 m)",
    np.max(np.abs(r6.y)) < 0.01, f"max|y|={np.max(np.abs(r6.y)):.4f} m")

q_norm = np.sqrt(r6.q0**2+r6.q1**2+r6.q2**2+r6.q3**2)
check("Norma kwaterniona ≈ 1 (dryft < 2e-6)",
    np.max(np.abs(q_norm-1.)) < 2e-6,
    f"max||q|-1|={np.max(np.abs(q_norm-1.)):.2e}")

# ============================================================================
# Wykres
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(f"3DOF vs 6DOF — elewacja {ELEVATION}°, CN_alpha={CN_ALPHA}", fontsize=12)

ax = axes[0,0]
ax.plot(r3.x_pos, -r3.z_pos, 'b-', lw=2, label='3DOF')
ax.plot(r6.x, -r6.z, 'r--', lw=1.5, label='6DOF')
ax.set_xlabel("x [m]"); ax.set_ylabel("Wysokość [m]")
ax.set_title("Trajektoria"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
ax.plot(r3.t, r3.speed, 'b-', lw=2, label='3DOF')
ax.plot(r6.t, r6.speed, 'r--', lw=1.5, label='6DOF')
ax.set_xlabel("t [s]"); ax.set_ylabel("V [m/s]")
ax.set_title("Prędkość"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,2]
ax.plot(r3.t, np.degrees(r3.theta), 'b-', lw=2, label='3DOF')
ax.plot(r6.t, np.degrees(r6.theta), 'r--', lw=1.5, label='6DOF')
ax.set_xlabel("t [s]"); ax.set_ylabel("θ [°]")
ax.set_title("Kąt pochylenia"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,0]
ax.plot(r3.t, -r3.z_pos, 'b-', lw=2, label='3DOF')
ax.plot(r6.t, -r6.z, 'r--', lw=1.5, label='6DOF')
ax.set_xlabel("t [s]"); ax.set_ylabel("Wysokość [m]")
ax.set_title("Wysokość vs czas"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.plot(r3.t, np.degrees(r3.alpha), 'b-', lw=2, label='3DOF')
ax.plot(r6.t, np.degrees(r6.alpha), 'r--', lw=1.5, label='6DOF')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel("t [s]"); ax.set_ylabel("α [°]")
ax.set_title("Kąt natarcia"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,2]
ax.plot(r6.t, q_norm - 1., 'k-', lw=1.5)
ax.axhline(0, color='r', lw=0.8, ls='--')
ax.set_xlabel("t [s]"); ax.set_ylabel("|q| - 1")
ax.set_title("Dryft normy kwaterniona (6DOF)"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("test_3dof_vs_6dof.png", dpi=120, bbox_inches="tight")
print("\n  Wykres: test_3dof_vs_6dof.png")

# ============================================================================
# Podsumowanie
# ============================================================================
n_pass = sum(results)
n_total = len(results)
print(f"\n{'='*60}")
print(f"  Wynik: {n_pass}/{n_total} testów zaliczonych")
print(f"  STATUS: {'OK — modele spójne' if n_pass==n_total else f'BLAD — {n_total-n_pass} niezaliczonych'}")
print('='*60)
sys.exit(0 if n_pass == n_total else 1)
