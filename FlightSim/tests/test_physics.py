"""
tests/test_physics.py
=====================
Testy fizyczne modelu 6DOF — sanity checks.

Uruchomienie z katalogu FlightSim:
    python tests/test_physics.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.aerodynamics import ConstantAero
from models.mass6 import MassModel6DOF, ConstantIxx
from forces.force_model6 import ForceModel6DOF, RocketGeometry6DOF, PropulsionConfig6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF

results = []

def check(name, ok, got, expected, tol_pct=2.0):
    pct = abs(got - expected) / (abs(expected) + 1e-10) * 100
    tag = "PASS" if ok else "FAIL"
    results.append((name, ok))
    print(f"  [{tag}] {name}")
    print(f"         got={got:.4f}  expected={expected:.4f}  err={pct:.2f}%")

def make_fm(thrust=0., CA=0., Cmq=-500., m=4.2):
    """Model z zadanymi parametrami. m_empty=m-0.001 zeby przejsc walidacje."""
    return ForceModel6DOF(
        atmosphere = create_atmosphere("ISA"),
        mass_model = MassModel6DOF(
            m_full=m, m_empty=m-0.001, t_burn=0.001,
            xcg_full=0.71, xcg_empty=0.71,
            Iyy_full=0.38, Iyy_empty=0.38,
            ixx_model=ConstantIxx(0.003),
        ),
        aero_model = ConstantAero(CA=CA, CN_alpha=0., Cmq=Cmq,
                                  use_xcp_moment=False),
        gravity    = create_gravity("constant"),
        geometry   = RocketGeometry6DOF(
            S_ref=np.pi*(0.07/2)**2, d_ref=0.07, xcp=0.71,
        ),
        propulsion = PropulsionConfig6DOF(thrust=thrust),
    )

g = 9.80665
m = 4.2

# ============================================================================
# TEST 1: Swobodny spadek z h=100m
# t = sqrt(2h/g),  V = sqrt(2gh)
# ============================================================================
print("\n=== TEST 1: Swobodny spadek z h=100m ===")
h0  = 100.0
s0  = State6DOF(x=0., y=0., z=-h0, u=0., v=0., w=0.,
                q0=1., q1=0., q2=0., q3=0., p=0., qr=0., r=0.)
res = run_simulation_6dof(make_fm(), s0, t_max=10., z_ground=0., max_step=0.01)

t_exp = np.sqrt(2*h0/g)
v_exp = np.sqrt(2*g*h0)
check("Czas spadku",     abs(res.t[-1] - t_exp) < 0.05,     res.t[-1],     t_exp)
check("Predkosc upadku", abs(res.speed[-1] - v_exp) < 0.5,  res.speed[-1], v_exp)
check("x=0",             abs(res.x[-1]) < 0.01,             res.x[-1],     0.0)
check("y=0",             abs(res.y[-1]) < 0.01,             res.y[-1],     0.0)

# ============================================================================
# TEST 2: Rzut pionowy w gore bez oporu
# h_max = v0^2/(2g),  t_lot = 2*v0/g
# ============================================================================
print("\n=== TEST 2: Rzut pionowy w gore (v0=100 m/s, bez oporu) ===")
v0  = 100.0
s0  = State6DOF.initial(elevation_deg=90., speed_0=v0)
res = run_simulation_6dof(make_fm(), s0, t_max=30., z_ground=0., max_step=0.01)

h_max_exp = v0**2 / (2*g)
t_lot_exp = 2*v0 / g
check("Wysokosc max",  abs(float((-res.z).max()) - h_max_exp)/h_max_exp < 0.01,
      float((-res.z).max()), h_max_exp)
check("Czas lotu",     abs(res.t[-1] - t_lot_exp)/t_lot_exp < 0.01,
      res.t[-1], t_lot_exp)
check("Zasieg x~0",    abs(res.x[-1]) < 0.5, res.x[-1], 0.0)

# ============================================================================
# TEST 3: Rzut poziomy z h=500m, v0=200 m/s (bez oporu)
# t = sqrt(2h/g),  x = v0 * t
# ============================================================================
print("\n=== TEST 3: Rzut poziomy (h=500m, v0=200 m/s, bez oporu) ===")
v0  = 200.0
h0  = 500.0
s0  = State6DOF(x=0., y=0., z=-h0, u=v0, v=0., w=0.,
                q0=1., q1=0., q2=0., q3=0., p=0., qr=0., r=0.)
res = run_simulation_6dof(make_fm(), s0, t_max=20., z_ground=0., max_step=0.01)

t_exp = np.sqrt(2*h0/g)
x_exp = v0 * t_exp
check("Czas lotu",  abs(res.t[-1] - t_exp)/t_exp < 0.01,   res.t[-1], t_exp)
check("Zasieg",     abs(res.x[-1] - x_exp)/x_exp < 0.01,   res.x[-1], x_exp)
check("y=0",        abs(res.y[-1]) < 0.01,                  res.y[-1], 0.0)

# ============================================================================
# TEST 4: Symetria El=45 vs El=135 — ten sam zasieg (bez oporu)
# ============================================================================
print("\n=== TEST 4: Symetria rzutu — El=45 vs El=135 (bez oporu) ===")
v0 = 300.0
res45  = run_simulation_6dof(make_fm(),
                              State6DOF.initial(elevation_deg=45.,  speed_0=v0),
                              t_max=60., z_ground=0., max_step=0.02)
res135 = run_simulation_6dof(make_fm(),
                              State6DOF.initial(elevation_deg=135., speed_0=v0),
                              t_max=60., z_ground=0., max_step=0.02)

check("Zasieg 45~135",
      abs(res45.x[-1] - res135.x[-1]) / (res45.x[-1] + 1e-6) < 0.01,
      res45.x[-1], res135.x[-1])
check("hmax 45~hmax 135",
      abs(float((-res45.z).max()) - float((-res135.z).max())) /
          float((-res45.z).max()) < 0.01,
      float((-res45.z).max()), float((-res135.z).max()))

# ============================================================================
# TEST 5: Zachowanie energii — swobodny lot bez oporu
# E = 0.5*m*v^2 + m*g*h = const
# ============================================================================
print("\n=== TEST 5: Zachowanie energii (bez oporu, bez ciagu) ===")
s0  = State6DOF.initial(elevation_deg=60., speed_0=200.)
res = run_simulation_6dof(make_fm(), s0, t_max=30., z_ground=-5000., max_step=0.02)

h = -res.z
E = 0.5 * m * res.speed**2 + m * g * h
dE_pct = float(np.max(np.abs(E - E[0])) / E[0] * 100)
check("Zachowanie energii dE<0.5%", dE_pct < 0.5, E[-1], E[0], tol_pct=0.5)

# ============================================================================
# TEST 6: Ciag F=ma — predkosc po burnout
# a_netto = F/m - g,  v = a_netto * t_burn
# ============================================================================
print("\n=== TEST 6: Ciag F=ma — predkosc po burnout ===")
thrust  = 420.0
t_burn  = 2.0
a_netto = thrust/m - g
v_exp   = a_netto * t_burn

fm6 = ForceModel6DOF(
    atmosphere = create_atmosphere("ISA"),
    mass_model = MassModel6DOF(
        m_full=4.2, m_empty=4.199, t_burn=t_burn,
        xcg_full=0.71, xcg_empty=0.71,
        Iyy_full=0.38, Iyy_empty=0.38,
        ixx_model=ConstantIxx(0.003),
    ),
    aero_model = ConstantAero(CA=0., CN_alpha=0., Cmq=-500.,
                              use_xcp_moment=False),
    gravity    = create_gravity("constant"),
    geometry   = RocketGeometry6DOF(
        S_ref=np.pi*(0.07/2)**2, d_ref=0.07, xcp=0.71,
    ),
    propulsion = PropulsionConfig6DOF(thrust=thrust),
)
res = run_simulation_6dof(fm6, State6DOF.initial(elevation_deg=90.),
                           t_max=t_burn+0.01, z_ground=-10000., max_step=0.01)
check("Predkosc po burnout (+-5%)",
      abs(res.speed[-1] - v_exp)/v_exp < 0.05, res.speed[-1], v_exp, tol_pct=5.0)

# ============================================================================
# TEST 7: Brak odchylenia bocznego przy El=45, symetria
# ============================================================================
print("\n=== TEST 7: Brak odchylenia bocznego (El=45, bez wiatru) ===")
s0  = State6DOF.initial(elevation_deg=45., speed_0=300.)
res = run_simulation_6dof(make_fm(CA=0.3), s0, t_max=40., z_ground=0., max_step=0.05)
ymax = float(np.max(np.abs(res.y)))
check("y~0 przez caly lot (< 1m)", ymax < 1.0, ymax, 0.0, tol_pct=100.)

# ============================================================================
# Podsumowanie
# ============================================================================
print(f"\n{'='*50}")
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"Wynik: {n_pass}/{len(results)} testow PASS  ({n_fail} FAIL)")
print('='*50)
