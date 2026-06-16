"""
tests/test_energy.py
====================
Walidacja energetyczna i przypadki graniczne — konwencja Z launch w dół.

Testy:
  1. Swobodny spadek pionowy — porównanie z rozwiązaniem analitycznym
  2. Zachowanie energii mechanicznej (bez aero, bez ciągu)
  3. Lot pionowy — symetria, brak ruchu bocznego
  4. Rzut ukośny — porównanie z parabolą analityczną
  5. Stabilność momentu — CN_alpha > 0 stabilizuje

Uruchomienie:
    python tests/test_energy.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.integrate import solve_ivp

from core.state6 import State6DOF
from core.solver6 import run_simulation_6dof
from core.quaternion import euler_zyx_to_quat
from models.atmosphere import create_atmosphere
from models.mass6 import MassModel6DOF, ConstantIxx
from models.aerodynamics import ConstantAero
from models.gravity import create_gravity
from forces.force_model6 import ForceModel6DOF, RocketGeometry6DOF, PropulsionConfig6DOF

G = 9.80665
PASS = "  [PASS]"; FAIL = "  [FAIL]"
results = []

def check(name, condition, info=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f"  ({info})" if info else ""))
    results.append(condition)
    return condition

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def make_fm(thrust=0., CN_alpha=0., CA=0., Cmq=0., m=5.0):
    """Model sił do testów — stała masa, brak spalania."""
    geo   = RocketGeometry6DOF.from_diameter(0.070, 0.55)
    mass  = MassModel6DOF(m+0.001, m, 0.001, 0.42, 0.42,
                          0.12, 0.12, ConstantIxx(0.012))
    aero  = ConstantAero(CA=CA, CN_alpha=CN_alpha, Cmq=Cmq, use_xcp_moment=True)
    prop  = PropulsionConfig6DOF(thrust=thrust)
    return ForceModel6DOF(create_atmosphere("ISA"), mass, aero,
                          create_gravity("constant"), geo, prop)

def simulate(fm, elevation, speed_0, t_max, z_ground=1.0, max_step=0.05):
    return run_simulation_6dof(fm, State6DOF.initial(elevation, speed_0=speed_0),
                               t_max=t_max, z_ground=z_ground, max_step=max_step)

# ============================================================================
# 1. Swobodny spadek pionowy — porównanie z analityką
# ============================================================================
section("1. Swobodny spadek pionowy (bez aero, bez ciągu)")

fm = make_fm()
# E=90°: u=V0 → leci pionowo w górę, z maleje (Z_launch w dół)
r  = simulate(fm, elevation=90., speed_0=100., t_max=25., z_ground=0.5)

t_end = r.t[-1]
# Analityczne: z(t) = -V0*t + 0.5*g*t² (Z w dół, nos w górę = -Z)
# u(t) = V0 - g*t
V0 = 100.
z_analytic = -V0 * t_end + 0.5 * G * t_end**2
u_analytic =  V0 - G * t_end

check("z końcowe vs analityczne (tol 0.5%)",
    abs(r.z[-1] - z_analytic) / max(abs(z_analytic), 1.) < 0.005,
    f"z_num={r.z[-1]:.3f}, z_analytic={z_analytic:.3f}")

check("u końcowe vs analityczne (tol 0.5%)",
    abs(r.u[-1] - u_analytic) / max(abs(u_analytic), 1.) < 0.005,
    f"u_num={r.u[-1]:.3f}, u_analytic={u_analytic:.3f}")

check("Brak ruchu bocznego y ≈ 0",
    np.max(np.abs(r.y)) < 0.01,
    f"max|y|={np.max(np.abs(r.y)):.2e} m")

# ============================================================================
# 2. Zachowanie energii
# ============================================================================
section("2. Zachowanie energii mechanicznej (bez aero, bez ciągu)")

fm = make_fm()
r  = simulate(fm, elevation=60., speed_0=200., t_max=15., z_ground=0.5)

m  = 5.0
# Energia: E = 0.5*m*V² + m*g*h, h = -z (Z w dół → h = -z)
E0 = 0.5 * m * r.speed[0]**2 + m * G * (-r.z[0])
E  = 0.5 * m * r.speed**2    + m * G * (-r.z)
dE = np.abs(E - E0) / E0

check("Energia stała (max błąd < 0.5%)",
    np.max(dE) < 0.005,
    f"max ΔE/E0 = {np.max(dE)*100:.3f}%")

check("Energia stała (średni błąd < 0.1%)",
    np.mean(dE) < 0.001,
    f"mean ΔE/E0 = {np.mean(dE)*100:.4f}%")

# ============================================================================
# 3. Lot pionowy — symetria
# ============================================================================
section("3. Lot pionowy (elewacja 90°) — symetria")

fm = make_fm()
r  = simulate(fm, elevation=90., speed_0=150., t_max=35., z_ground=0.5)

check("Brak ruchu poziomego x ≈ 0",
    np.max(np.abs(r.x)) < 0.01,
    f"max|x|={np.max(np.abs(r.x)):.2e} m")

check("Brak ruchu bocznego y ≈ 0",
    np.max(np.abs(r.y)) < 0.01,
    f"max|y|={np.max(np.abs(r.y)):.2e} m")

check("Theta ≈ 90° przez cały lot",
    np.max(np.abs(np.degrees(r.theta) - 90.)) < 0.01,
    f"max|θ-90°|={np.max(np.abs(np.degrees(r.theta)-90.)):.4f}°")

# Czas lotu analityczny: t = 2*V0/g
t_analytic = 2. * 150. / G
check("Czas lotu vs analityczne (tol 1%)",
    abs(r.t[-1] - t_analytic) / t_analytic < 0.01,
    f"t_num={r.t[-1]:.3f}s, t_analytic={t_analytic:.3f}s")

# ============================================================================
# 4. Rzut ukośny — porównanie z parabolą
# ============================================================================
section("4. Rzut ukośny 45° — porównanie z parabolą analityczną")

fm  = make_fm()
E_deg = 45.
r   = simulate(fm, elevation=E_deg, speed_0=100., t_max=20., z_ground=0.5)

t    = r.t
th   = np.deg2rad(E_deg)
V0   = 100.
# W konwencji Z w dół:
# vx0 = V0*cos(th), vz0 = -V0*sin(th) (w górę = -Z)
# x(t) = vx0*t
# z(t) = vz0*t + 0.5*g*t²
x_analytic = V0 * np.cos(th) * t
z_analytic = -V0 * np.sin(th) * t + 0.5 * G * t**2

err_x = np.max(np.abs(r.x - x_analytic))
err_z = np.max(np.abs(r.z - z_analytic))

check("Trajektoria x(t) vs parabola (max błąd < 1 m)",
    err_x < 1.0, f"max|Δx|={err_x:.4f} m")

check("Trajektoria z(t) vs parabola (max błąd < 1 m)",
    err_z < 1.0, f"max|Δz|={err_z:.4f} m")

# Zasięg analityczny: R = V0²*sin(2θ)/g
R_analytic = V0**2 * np.sin(2*th) / G
R_num      = r.x[-1]
check("Zasięg vs analityczny (tol 1%)",
    abs(R_num - R_analytic) / R_analytic < 0.01,
    f"R_num={R_num:.2f} m, R_analytic={R_analytic:.2f} m")

# ============================================================================
# 5. Stabilność momentu — CN_alpha > 0
# ============================================================================
section("5. Stabilność momentu (CN_alpha > 0 = stabilizujący)")

fm_stable   = make_fm(CN_alpha=+9.0, Cmq=-10.)
fm_unstable = make_fm(CN_alpha=-9.0, Cmq=-10.)

# Stan: theta=0, u=200, w=+5 (nos powyżej toru, alpha>0 w konwencji Z-dół)
q0 = euler_zyx_to_quat(0, 0, 0)
x  = np.array([0,0,0., 200.,0.,5., q0[0],q0[1],q0[2],q0[3], 0.,0.,0.])

dx_s = fm_stable.derivatives(5., x)
dx_u = fm_unstable.derivatives(5., x)

check("CN_alpha>0: dqr < 0 przy alpha>0 (stabilizujący)",
    dx_s[11] < 0,
    f"dqr={np.degrees(dx_s[11]):.2f} °/s²")

check("CN_alpha<0: dqr > 0 przy alpha>0 (destabilizujący)",
    dx_u[11] > 0,
    f"dqr={np.degrees(dx_u[11]):.2f} °/s²")

# ============================================================================
# 6. Norma kwaterniona — brak dryftu
# ============================================================================
section("6. Norma kwaterniona — brak dryftu numerycznego")

fm = make_fm(CN_alpha=9.0, Cmq=-10.)
r  = simulate(fm, elevation=75., speed_0=100., t_max=10., z_ground=0.5)

q_norm = np.sqrt(r.q0**2 + r.q1**2 + r.q2**2 + r.q3**2)
drift  = np.max(np.abs(q_norm - 1.0))

check("Norma kwaterniona ≈ 1 (dryft < 2e-6)",
    drift < 2e-6,
    f"max||q|-1| = {drift:.2e}")

# ============================================================================
# Podsumowanie
# ============================================================================
n_pass = sum(results)
n_total = len(results)
print(f"\n{'='*60}")
print(f"  Wynik: {n_pass}/{n_total} testów zaliczonych")
print(f"  STATUS: {'OK' if n_pass==n_total else f'BLAD — {n_total-n_pass} niezaliczonych'}")
print('='*60)
sys.exit(0 if n_pass == n_total else 1)
