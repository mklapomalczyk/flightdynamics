"""
tests/test_launcher.py
======================
Walidacja modelu szyny startowej.

Testy:
  1. Na szynie: v=w=0, p=qr=r=0 przez cały czas na szynie
  2. Na szynie: orientacja rakiety nie zmienia się (kwaternion stały)
  3. Droga na szynie: rail_dist rośnie poprawnie
  4. Po opuszczeniu szyny: rakieta leci swobodnie (qr może rosnąć)
  5. Bez szyny (L_rail=0): identyczne z lotem swobodnym od t=0
  6. Szyna pionowa (E=90°): rakieta leci pionowo w górę na szynie

Uruchomienie:
    python tests/test_launcher.py
"""

import sys
import os
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
from models.launcher import LauncherConfig, RailLauncher
from forces.force_model6 import (
    ForceModel6DOF, RocketGeometry6DOF, PropulsionConfig6DOF
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
results = []


def check(name, condition, info=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f"  ({info})" if info else ""))
    results.append(condition)
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def make_fm(L_rail=0.0, CN_alpha=9.0, Cmq=-25.0, thrust=2800.0, elevation=75.0):
    """Fabryka force_model z opcjonalną szyną."""
    geometry   = RocketGeometry6DOF.from_diameter(0.070, 0.55)
    mass_model = MassModel6DOF(
        m_full=6.0, m_empty=4.2, t_burn=1.8,
        xcg_full=0.42, xcg_empty=0.38,
        Iyy_full=0.12, Iyy_empty=0.09,
        ixx_model=ConstantIxx(0.012),
    )
    aero       = ConstantAero(CA=0.35, CN_alpha=CN_alpha, Cmq=Cmq,
                               use_xcp_moment=True)
    propulsion = PropulsionConfig6DOF(thrust=thrust)
    launcher   = LauncherConfig(L_rail=L_rail)

    return ForceModel6DOF(
        atmosphere  = create_atmosphere("ISA"),
        mass_model  = mass_model,
        aero_model  = aero,
        gravity     = create_gravity("constant"),
        geometry    = geometry,
        propulsion  = propulsion,
        launcher    = launcher,
    ), State6DOF.initial(elevation)


# ============================================================================
# 1. Na szynie: v=w=0 i p=qr=r=0
# ============================================================================
section("1. Ograniczenia na szynie (v=w=p=qr=r=0)")

fm, initial = make_fm(L_rail=3.0, elevation=75.0)
x0 = initial.to_numpy(include_rail=True)

# Kilka kroków Eulera — sprawdź czy v,w,p,qr,r pozostają zero
x = x0.copy()
dt = 0.001
max_v = 0.0; max_w = 0.0; max_qr = 0.0
for i in range(500):   # 0.5s — na szynie
    dx = fm.derivatives(i * dt, x)
    x  = x + dx * dt
    rail_dist = x[13]
    if rail_dist < 3.0:   # wciąż na szynie
        max_v  = max(max_v,  abs(x[4]))
        max_w  = max(max_w,  abs(x[5]))
        max_qr = max(max_qr, abs(x[11]))

check("Na szynie: v = 0",
    max_v < 1e-10,
    f"max|v| = {max_v:.2e} m/s")

check("Na szynie: w = 0",
    max_w < 1e-10,
    f"max|w| = {max_w:.2e} m/s")

check("Na szynie: qr = 0",
    max_qr < 1e-10,
    f"max|qr| = {np.degrees(max_qr):.2e} °/s")

# ============================================================================
# 2. Na szynie: kwaternion stały (orientacja nie zmienia się)
# ============================================================================
section("2. Orientacja stała na szynie")

fm, initial = make_fm(L_rail=3.0, elevation=75.0)
x0 = initial.to_numpy(include_rail=True)
q_initial = x0[6:10].copy()

x = x0.copy()
dt = 0.001
max_dq = 0.0
for i in range(500):
    dx = fm.derivatives(i * dt, x)
    x  = x + dx * dt
    if x[13] < 3.0:
        dq = np.max(np.abs(x[6:10] - q_initial))
        max_dq = max(max_dq, dq)

check("Na szynie: kwaternion stały (max zmiana < 1e-6)",
    max_dq < 1e-6,
    f"max|Δq| = {max_dq:.2e}")

# ============================================================================
# 3. Droga na szynie rośnie poprawnie
# ============================================================================
section("3. Całkowanie drogi na szynie")

fm, initial = make_fm(L_rail=10.0, elevation=75.0)
x0 = initial.to_numpy(include_rail=True)

x = x0.copy()
dt = 0.001
t_exit = None
for i in range(2000):   # do 2s
    t = i * dt
    dx = fm.derivatives(t, x)
    x  = x + dx * dt
    if x[13] >= 10.0 and t_exit is None:
        t_exit = t
        u_at_exit = x[3]

check("Droga na szynie osiąga L_rail",
    x[13] >= 10.0 or t_exit is not None,
    f"rail_dist po 2s = {x[13]:.3f} m")

check("Droga na szynie monotonicznie rośnie (u > 0 na szynie)",
    t_exit is not None,
    f"t_exit = {t_exit:.3f} s, u_exit = {u_at_exit:.1f} m/s" if t_exit else "nie opuścił szyny")

# ============================================================================
# 4. Po opuszczeniu szyny: qr może rosnąć (brak ograniczeń)
# ============================================================================
section("4. Swobodny lot po opuszczeniu szyny")

fm, initial = make_fm(L_rail=1.0, elevation=75.0, CN_alpha=9.0, Cmq=-25.0)
x0 = initial.to_numpy(include_rail=True)

x = x0.copy()
dt = 0.001
qr_on_rail = []
qr_off_rail = []

for i in range(500):   # 0.5s — wystarczy żeby ocenić zachowanie
    t = i * dt
    dx = fm.derivatives(t, x)
    x  = x + dx * dt
    if x[13] < 1.0:
        qr_on_rail.append(abs(x[11]))
    else:
        qr_off_rail.append(abs(x[11]))

max_qr_on  = max(qr_on_rail)  if qr_on_rail  else 0.0
max_qr_off = max(qr_off_rail) if qr_off_rail else 0.0

check("Po szynie: qr może być niezerowe",
    max_qr_off > 1e-6,
    f"max|qr| off_rail = {np.degrees(max_qr_off):.4f} °/s")

check("Na szynie: qr = 0, po szynie: qr > 0 (wyraźna różnica)",
    max_qr_off > max_qr_on * 100,
    f"on={np.degrees(max_qr_on):.2e} °/s, off={np.degrees(max_qr_off):.4f} °/s")

# ============================================================================
# 5. L_rail=0: identyczne z lotem bez szyny
# ============================================================================
section("5. L_rail=0 identyczne z lotem swobodnym")

fm_rail, initial = make_fm(L_rail=0.0, elevation=75.0)
fm_free, _       = make_fm(L_rail=0.0, elevation=75.0)

# Oba modele bez szyny — pochodne powinny być identyczne
x0_rail = initial.to_numpy(include_rail=True)
x0_free = initial.to_numpy(include_rail=False)

dx_rail = fm_rail.derivatives(0.1, x0_rail)
dx_free = fm_free.derivatives(0.1, x0_free)

# Porównaj pierwsze 13 składowych
err = np.max(np.abs(dx_rail[:13] - dx_free[:13]))
check("L_rail=0: pochodne identyczne z lotem swobodnym",
    err < 1e-10,
    f"max_err = {err:.2e}")

# ============================================================================
# 6. Szyna pionowa (E=90°): lot pionowy w górę
# ============================================================================
section("6. Szyna pionowa (elewacja 90°)")

fm, initial = make_fm(L_rail=2.0, elevation=90.0)
x0 = initial.to_numpy(include_rail=True)

x = x0.copy()
dt = 0.001
max_x = 0.0; max_y = 0.0
for i in range(500):
    dx = fm.derivatives(i * dt, x)
    x  = x + dx * dt
    if x[13] < 2.0:
        max_x = max(max_x, abs(x[0]))   # x launch
        max_y = max(max_y, abs(x[1]))   # y launch

check("Szyna pionowa: brak ruchu w x (tol 1e-6 m)",
    max_x < 1e-6,
    f"max|x| = {max_x:.2e} m")

check("Szyna pionowa: brak ruchu w y (tol 1e-6 m)",
    max_y < 1e-6,
    f"max|y| = {max_y:.2e} m")

# ============================================================================
# 7. Pełna symulacja z szyną przez solver
# ============================================================================
section("7. Pełna symulacja z szyną (solver RK45)")

fm, initial = make_fm(L_rail=3.0, elevation=75.0, CN_alpha=9.0, Cmq=-25.0)

from scipy.integrate import solve_ivp
x0 = initial.to_numpy(include_rail=True)

r = solve_ivp(fm.derivatives, (0, 5.0), x0,
              method='RK45', rtol=1e-4, atol=1e-4, max_step=0.01)

check("Solver zbiega z szyną",
    r.status >= 0,
    f"status={r.status}, t={r.t[-1]:.2f}s")

check("Pozycja z rozsądna (z < 0 po locie w górę)",
    r.y[2, -1] < 0,
    f"z_end = {r.y[2,-1]:.1f} m")

check("rail_dist >= L_rail po locie",
    r.y[13, -1] >= 3.0,
    f"rail_dist_end = {r.y[13,-1]:.3f} m")

# ============================================================================
# Podsumowanie
# ============================================================================
n_pass = sum(results)
n_total = len(results)
print(f"\n{'='*60}")
print(f"  Wynik: {n_pass}/{n_total} testów zaliczonych")
if n_pass == n_total:
    print("  STATUS: OK — model szyny działa poprawnie")
else:
    print(f"  STATUS: BLAD — {n_total - n_pass} testów niezaliczonych")
print('='*60)

sys.exit(0 if n_pass == n_total else 1)
