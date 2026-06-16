"""
core/quaternion.py
==================
Operacje na kwaternionach jednostkowych dla kinematyki obrotowej.

Konwencja: q = [q0, q1, q2, q3] gdzie q0 to część skalarna (Hamilton).
Kwaternion jednostkowy: |q| = 1.

Kwaternion reprezentuje obrót z launch frame do body frame:
  v_body = q * v_launch * q_conj

Normalizacja wykonywana co krok solvera (w derivatives) — zapobiega
dryfowi numerycznemu bez osobnego callbacku.
"""

import numpy as np


# ============================================================================
# Podstawowe operacje
# ============================================================================

def quat_norm(q: np.ndarray) -> np.ndarray:
    """Normalizuje kwaternion do jednostkowego."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Sprzężenie kwaterniona: q* = [q0, -q1, -q2, -q3]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Iloczyn kwaternionów Hamilton: p ⊗ q.
    Kolejność: najpierw obrót q, potem p (prawostronna kompozycja).
    """
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Obraca wektor v kwaternionem q: v' = q ⊗ [0,v] ⊗ q*.
    Transformacja: launch frame → body frame.
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[1:]


# ============================================================================
# DCM ↔ Kwaternion
# ============================================================================

def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Kwaternion → Direction Cosine Matrix (body ← launch).
    R taki że v_body = R @ v_launch.
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2),  2*(q1*q2 + q0*q3),      2*(q1*q3 - q0*q2)     ],
        [2*(q1*q2 - q0*q3),      1 - 2*(q1**2 + q3**2),  2*(q2*q3 + q0*q1)     ],
        [2*(q1*q3 + q0*q2),      2*(q2*q3 - q0*q1),      1 - 2*(q1**2 + q2**2) ],
    ])


def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    """DCM → kwaternion (metoda Shepparda, numerycznie stabilna)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2,1] - R[1,2]) * s
        q2 = (R[0,2] - R[2,0]) * s
        q3 = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        q0 = (R[2,1] - R[1,2]) / s
        q1 = 0.25 * s
        q2 = (R[0,1] + R[1,0]) / s
        q3 = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        q0 = (R[0,2] - R[2,0]) / s
        q1 = (R[0,1] + R[1,0]) / s
        q2 = 0.25 * s
        q3 = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        q0 = (R[1,0] - R[0,1]) / s
        q1 = (R[0,2] + R[2,0]) / s
        q2 = (R[1,2] + R[2,1]) / s
        q3 = 0.25 * s
    return quat_norm(np.array([q0, q1, q2, q3]))


# ============================================================================
# Kwaternion ↔ Kąty Eulera ZYX
# ============================================================================

def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """
    Kwaternion → kąty Eulera ZYX (yaw-pitch-roll).

    Kolejność rotacji: najpierw roll (φ wokół X), potem pitch (θ wokół Y),
    potem yaw (ψ wokół Z). Konwencja ZYX = standard lotniczy.

    Returns
    -------
    np.ndarray [psi, theta, phi] : [rad]
        psi   — yaw   (ψ), obrót wokół Z [rad]
        theta — pitch (θ), obrót wokół Y [rad]
        phi   — roll  (φ), obrót wokół X [rad]

    Osobliwość (gimbal lock) przy theta = ±90°.
    """
    q0, q1, q2, q3 = q

    # Roll (φ)
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))

    # Pitch (θ) — clamp dla stabilności numerycznej
    sin_theta = 2*(q0*q2 - q3*q1)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    # Yaw (ψ)
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

    return np.array([psi, theta, phi])


def euler_zyx_to_quat(psi: float, theta: float, phi: float) -> np.ndarray:
    """
    Kąty Eulera ZYX → kwaternion.

    Parameters
    ----------
    psi   : float  yaw   [rad]
    theta : float  pitch [rad]
    phi   : float  roll  [rad]
    """
    cy, sy = np.cos(psi/2),   np.sin(psi/2)
    cp, sp = np.cos(theta/2), np.sin(theta/2)
    cr, sr = np.cos(phi/2),   np.sin(phi/2)

    q0 = cr*cp*cy + sr*sp*sy
    q1 = sr*cp*cy - cr*sp*sy
    q2 = cr*sp*cy + sr*cp*sy
    q3 = cr*cp*sy - sr*sp*cy

    return quat_norm(np.array([q0, q1, q2, q3]))


# ============================================================================
# Kinematyka kwaternionowa
# ============================================================================

def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Pochodna kwaterniona: dq/dt = 0.5 * q ⊗ [0, ω_body].

    Parameters
    ----------
    q : (4,) array
        Kwaternion orientacji (jednostkowy).
    omega : (3,) array
        Prędkość kątowa w body frame [p, q_rate, r] [rad/s].

    Returns
    -------
    dq/dt : (4,) array
    """
    p, q_rate, r = omega
    omega_quat = np.array([0.0, p, q_rate, r])
    dq = 0.5 * quat_multiply(q, omega_quat)
    return dq


# ============================================================================
# Kąt natarcia i ślizgu z prędkości w body frame
# ============================================================================

def aero_angles(u: float, v: float, w: float):
    """
    Kąt natarcia α i kąt ślizgu β z prędkości w body frame.

    Konwencja DATCOM (Z_body w dół):
      α = arctan2(w, u)   > 0 gdy nos powyżej wektora prędkości
      β = arcsin(v / V)   > 0 gdy prędkość boczna w prawo

    Parameters
    ----------
    u, v, w : float
        Prędkości wzdłużna, boczna, normalna w body frame [m/s].

    Returns
    -------
    (alpha, beta) : (float, float) [rad]
    """
    V = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v / V, -1.0, 1.0)) if V > 0.1 else 0.0
    return float(alpha), float(beta)
