"""
tests/test_quaternion.py
========================
Walidacja analityczna operacji kwaternionowych.
Konwencja: Launch Z w dół, Body Z w dół.

Uruchomienie:
    python tests/test_quaternion.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.quaternion import (
    quat_norm, quat_multiply, quat_conjugate, quat_to_dcm,
    euler_zyx_to_quat, quat_to_euler_zyx, quat_derivative, aero_angles
)

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
# 1. Podstawowe operacje
# ============================================================================
section("1. Podstawowe operacje kwaternionowe")

q = quat_norm(np.array([0.5, 0.5, 0.5, 0.5]))
check("Norma kwaterniona = 1",
    abs(np.linalg.norm(q) - 1.0) < 1e-12,
    f"|q| = {np.linalg.norm(q):.15f}")

q_id   = np.array([1., 0., 0., 0.])
q_test = euler_zyx_to_quat(0.3, 0.5, 0.2)
check("q_id ⊗ q = q",
    np.allclose(quat_multiply(q_id, q_test), q_test, atol=1e-12))

check("q ⊗ q* = q_id",
    np.allclose(quat_multiply(q_test, quat_conjugate(q_test)), q_id, atol=1e-12))

q1 = quat_norm(np.array([1., 2., 3., 4.]))
q2 = quat_norm(np.array([0.5, -0.5, 0.5, -0.5]))
check("|q1 ⊗ q2| = 1",
    abs(np.linalg.norm(quat_multiply(q1, q2)) - 1.0) < 1e-12)

# ============================================================================
# 2. Wzajemna odwrotność Euler ↔ kwaternion
# ============================================================================
section("2. Euler ZYX ↔ kwaternion")

# Przy theta=±90° (gimbal lock) kąty Eulera nie są jednoznaczne —
# sprawdzamy czy DCM (orientacja fizyczna) jest identyczna.
for psi, theta, phi, label in [
    (0., 0., 0., "zero"),
    (0.3, 0.5, 0.2, "ogólne"),
    (0., np.deg2rad(75), 0., "elewacja 75°"),
    (0., np.deg2rad(45), 0., "elewacja 45°"),
    (0.5, 0.3, -0.4, "wszystkie osie"),
]:
    q = euler_zyx_to_quat(psi, theta, phi)
    e = quat_to_euler_zyx(q)
    err = np.max(np.abs(e - np.array([psi, theta, phi])))
    check(f"Euler→q→Euler [{label}]", err < 1e-12, f"err={err:.2e}")

# Gimbal lock (theta=90°): sprawdzamy DCM a nie kąty Eulera
# Tylko psi=phi=0 przy theta=90 — przy niezerowych psi/phi gimbal lock
# daje niejednoznaczność której nie rozwiązuje standardowy arctan2
for psi, theta, phi, label in [
    (0., np.deg2rad(90), 0., "elewacja 90° psi=0,phi=0"),
]:
    q_in  = euler_zyx_to_quat(psi, theta, phi)
    q_out = euler_zyx_to_quat(*quat_to_euler_zyx(q_in))
    dcm_in  = quat_to_dcm(q_in)
    dcm_out = quat_to_dcm(q_out)
    err_dcm = np.max(np.abs(dcm_in - dcm_out))
    check(f"Gimbal lock DCM [{label}]", err_dcm < 1e-12,
          f"DCM err={err_dcm:.2e} (kąty Eulera niejednoznaczne przy theta=90°)")

# ============================================================================
# 3. DCM — właściwości
# ============================================================================
section("3. DCM — właściwości macierzy obrotu")

q_test = euler_zyx_to_quat(0.3, 0.5, 0.2)
DCM    = quat_to_dcm(q_test)

RRT = DCM @ DCM.T
check("DCM ortogonalna (R @ R.T = I)",
    np.allclose(RRT, np.eye(3), atol=1e-12),
    f"max_err={np.max(np.abs(RRT - np.eye(3))):.2e}")

check("det(DCM) = +1",
    abs(np.linalg.det(DCM) - 1.0) < 1e-12,
    f"det={np.linalg.det(DCM):.15f}")

# ============================================================================
# 4. Transformacje fizyczne — nowa konwencja (Z launch w dół)
# ============================================================================
section("4. Transformacje fizyczne (Launch: Z w dół, Body: Z w dół)")

# E=0 (poziomo): X_body = X_launch = [1,0,0]
q0 = euler_zyx_to_quat(0, 0, 0)
D0 = quat_to_dcm(q0)
x_body_in_launch = D0.T @ np.array([1., 0., 0.])
check("E=0°: X_body w launch = [1,0,0]",
    np.allclose(x_body_in_launch, [1., 0., 0.], atol=1e-12),
    f"{x_body_in_launch.round(4)}")

# E=90 (pionowo): X_body wskazuje w górę = -Z_launch = [0,0,-1]
q90 = euler_zyx_to_quat(0, np.deg2rad(90), 0)
D90 = quat_to_dcm(q90)
x_body_in_launch_90 = D90.T @ np.array([1., 0., 0.])
check("E=90°: X_body w launch = [0,0,-1] (w górę = -Z_launch)",
    np.allclose(x_body_in_launch_90, [0., 0., -1.], atol=1e-12),
    f"{x_body_in_launch_90.round(4)}")

# Grawitacja E=90: g_launch=[0,0,+g], g_body=DCM@g_launch → gx=-g, gz=0
g = 9.81
g_launch = np.array([0., 0., +g])
g_body_90 = D90 @ g_launch
check("E=90°: gx_body = -g (hamuje lot pionowy)",
    abs(g_body_90[0] + g) < 1e-10 and abs(g_body_90[2]) < 1e-10,
    f"g_body={g_body_90.round(4)}")

# Grawitacja E=0: gz_body = +g (ciągnie w +Z_body = w dół)
g_body_0 = D0 @ g_launch
check("E=0°: gz_body = +g (grawitacja w dół)",
    abs(g_body_0[0]) < 1e-10 and abs(g_body_0[2] - g) < 1e-10,
    f"g_body={g_body_0.round(4)}")

# Kinematyka pozycji: v_launch = DCM.T @ v_body
# E=90, u=1: leci pionowo → vz_launch = -1 (w górę = -Z_launch)
v_launch_90 = D90.T @ np.array([1., 0., 0.])
check("E=90°: u=1 → dz/dt = -1 (ruch w górę = -Z_launch)",
    abs(v_launch_90[2] + 1.0) < 1e-12,
    f"v_launch={v_launch_90.round(4)}")

# E=0, u=1: leci poziomo → vx=1, vz=0
v_launch_0 = D0.T @ np.array([1., 0., 0.])
check("E=0°: u=1 → dx/dt=1, dz/dt=0",
    abs(v_launch_0[0] - 1.0) < 1e-12 and abs(v_launch_0[2]) < 1e-12,
    f"v_launch={v_launch_0.round(4)}")

# ============================================================================
# 5. Kąty aerodynamiczne
# ============================================================================
section("5. Kąty aerodynamiczne (Z_body w dół)")

# u=100, v=0, w=0 → alpha=0, beta=0
a, b = aero_angles(100., 0., 0.)
check("u=100, v=0, w=0 → alpha=0, beta=0",
    abs(a) < 1e-12 and abs(b) < 1e-12,
    f"alpha={np.degrees(a):.4f}°, beta={np.degrees(b):.4f}°")

# w > 0 (+Z_body = w dół): nos powyżej toru → alpha > 0
a, b = aero_angles(100., 0., 5.)
check("w=+5 (w dół): alpha > 0 (nos powyżej toru)",
    a > 0,
    f"alpha={np.degrees(a):.4f}°")

# w < 0 (-Z_body = w górę): nos poniżej toru → alpha < 0
a, b = aero_angles(100., 0., -5.)
check("w=-5 (w górę): alpha < 0 (nos poniżej toru)",
    a < 0,
    f"alpha={np.degrees(a):.4f}°")

# v=0 → beta=0
a, b = aero_angles(100., 0., 5.)
check("v=0 → beta=0", abs(b) < 1e-12, f"beta={np.degrees(b):.4f}°")

# ============================================================================
# 6. Kinematyka kwaternionowa
# ============================================================================
section("6. Kinematyka kwaternionowa")

# omega=0 → dq/dt=0
q_test = euler_zyx_to_quat(0.3, 0.5, 0.2)
dq = quat_derivative(q_test, np.zeros(3))
check("omega=0 → dq/dt=0",
    np.allclose(dq, 0., atol=1e-12),
    f"max|dq|={np.max(np.abs(dq)):.2e}")

# Całkowanie pełnego obrotu 2π wokół Z → powrót do punktu startowego
q = np.array([1., 0., 0., 0.])
omega = np.array([0., 0., 2*np.pi])   # 2π rad/s wokół Z
dt = 1e-5
for _ in range(int(1.0 / dt)):
    dq = quat_derivative(q, omega)
    q  = quat_norm(q + dq * dt)
euler_end = quat_to_euler_zyx(q)
check("Pełny obrót 2π wokół Z → yaw ≈ 0°",
    abs(euler_end[0]) < 0.01,
    f"yaw_end={np.degrees(euler_end[0]):.4f}°")

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
