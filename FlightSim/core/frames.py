"""
core/frames.py
==============
Transformacje między układami współrzędnych.

Zaimplementowane układy:
  B  - Body frame: oś X wzdłuż osi rakiety, oś Z w górę prostopadle do X
  L  - Launch frame: inercjalny, oś X poziomo, oś Z pionowo w górę, origin t=0
  A  - Aerodynamic frame: oś X wzdłuż wektora prędkości względem powietrza
  W  - Wind frame (velocity frame): tożsamy z A przy braku wiatru

Zaślepki architektoniczne (nie zaimplementowane, wołanie rzuca NotImplementedError):
  ECEF - Earth-Centered Earth-Fixed (do przyszłego modelu WGS84)
  ECI  - Earth-Centered Inertial

Konwencja kątów:
  theta - kąt pochylenia (pitch), dodatni nos w górę [rad]
  alpha - kąt natarcia, dodatni nos powyżej wektora prędkości [rad]
  beta  - kąt ślizgu (6DOF), nie używany w 3DOF [rad]

Wszystkie macierze obrotu są proper rotation matrices (det = +1, R^T = R^-1).
"""

import numpy as np
from typing import Tuple


# ============================================================================
# Macierze obrotu — elementy bazowe
# ============================================================================

def Ry(angle: float) -> np.ndarray:
    """
    Macierz obrotu o kąt `angle` wokół osi Y (obrót w płaszczyźnie XZ).
    Używana do pitch.

    Uwaga: obrót w prawo (right-hand rule), oś Y wskazuje "w ekran"
    dla płaszczyzny XZ.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c,  0.0,  s],
        [ 0.0, 1.0, 0.0],
        [-s,  0.0,  c],
    ])


# ============================================================================
# Transformacje Body ↔ Launch
# ============================================================================

def dcm_body_to_launch(theta: float) -> np.ndarray:
    """
    Direction Cosine Matrix: Body → Launch frame.
    Dla 3DOF (tylko pitch): obrót o kąt theta wokół osi Y.

    v_L = DCM_BL @ v_B

    Parameters
    ----------
    theta : float
        Kąt pochylenia [rad].
    """
    # Obrót dodatni theta obraca oś X body w górę (nos do góry)
    # co odpowiada obrót o +theta wokół osi Y (w prawo/right-hand)
    return Ry(theta)


def dcm_launch_to_body(theta: float) -> np.ndarray:
    """
    Direction Cosine Matrix: Launch → Body frame.
    v_B = DCM_LB @ v_L
    """
    return dcm_body_to_launch(theta).T


def body_to_launch(v_body: np.ndarray, theta: float) -> np.ndarray:
    """Transformuje wektor z body frame do launch frame."""
    return dcm_body_to_launch(theta) @ v_body


def launch_to_body(v_launch: np.ndarray, theta: float) -> np.ndarray:
    """Transformuje wektor z launch frame do body frame."""
    return dcm_launch_to_body(theta) @ v_launch


# ============================================================================
# Transformacje Body ↔ Aerodynamic
# ============================================================================

def dcm_aero_to_body(alpha: float, beta: float = 0.0) -> np.ndarray:
    """
    Direction Cosine Matrix: Aerodynamic → Body frame.
    v_B = DCM_AB @ v_A

    Parameters
    ----------
    alpha : float
        Kąt natarcia [rad].
    beta : float
        Kąt ślizgu [rad]. W 3DOF = 0.
    """
    # Pełna transformacja (gotowe na 6DOF):
    # najpierw obrót o -beta wokół Z, potem obrót o alpha wokół Y
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)

    return np.array([
        [ ca*cb, -ca*sb, -sa],
        [ sb,     cb,    0.0],
        [ sa*cb, -sa*sb,  ca],
    ])


def dcm_body_to_aero(alpha: float, beta: float = 0.0) -> np.ndarray:
    """Direction Cosine Matrix: Body → Aerodynamic frame."""
    return dcm_aero_to_body(alpha, beta).T


def aero_to_body(v_aero: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    """Transformuje wektor z aerodynamic frame do body frame."""
    return dcm_aero_to_body(alpha, beta) @ v_aero


def body_to_aero(v_body: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    """Transformuje wektor z body frame do aerodynamic frame."""
    return dcm_body_to_aero(alpha, beta) @ v_body


# ============================================================================
# Transformacje Launch ↔ Aerodynamic (przez body)
# ============================================================================

def aero_to_launch(v_aero: np.ndarray, theta: float,
                   alpha: float, beta: float = 0.0) -> np.ndarray:
    """Launch ← Aero: przez body frame jako pośredni."""
    v_body = aero_to_body(v_aero, alpha, beta)
    return body_to_launch(v_body, theta)


def launch_to_aero(v_launch: np.ndarray, theta: float,
                   alpha: float, beta: float = 0.0) -> np.ndarray:
    """Aero ← Launch."""
    v_body = launch_to_body(v_launch, theta)
    return body_to_aero(v_body, alpha, beta)


# ============================================================================
# Funkcje pomocnicze dla 3DOF
# ============================================================================

def velocity_body_to_launch(u: float, w: float, theta: float) -> Tuple[float, float]:
    """
    Przelicza składowe prędkości z body frame na launch frame.

    Konwencja osi (Z_body W DÓŁ, Z_launch W DÓŁ):
      Body frame:   X wzdłuż osi rakiety (nos), Z prostopadle (w dół)
      Launch frame: X poziomo (zasięg),          Z w dół

    Transformacja:
      vx_L =  u*cos(theta) + w*sin(theta)
      vz_L = -u*sin(theta) + w*cos(theta)

    Weryfikacja:
      u=1, w=0, theta=90°  → vx=0,    vz=-1  ✓ (nos w górę = -Z_launch)
      u=0, w=1, theta=0°   → vx=0,    vz=+1  ✓ (w>0 = w dół = +Z_launch)
      u=0, w=1, theta=90°  → vx=+1,   vz=0   ✓ (w dół przy th=90 → +X_launch)

    Parameters
    ----------
    u, w : float
        Prędkość wzdłużna (wzdłuż X_body) i normalna (wzdłuż Z_body) [m/s].
    theta : float
        Kąt pochylenia [rad].

    Returns
    -------
    (vx, vz) : Tuple[float, float]
        Składowe prędkości w launch frame [m/s].
    """
    c, s = np.cos(theta), np.sin(theta)
    vx = u * c + w * s
    vz = -u * s + w * c
    return float(vx), float(vz)


def compute_alpha(u: float, w: float) -> float:
    """
    Kąt natarcia z prędkości w body frame [rad].
    Konwencja DATCOM (Z_body w dół): α > 0 gdy nos powyżej wektora prędkości.
    Przy Z_body w dół: w > 0 oznacza składową prędkości w dół → nos powyżej toru.
    """
    return float(np.arctan2(w, u))


# ============================================================================
# Zaślepki architektoniczne — do przyszłej implementacji WGS84
# ============================================================================

def dcm_launch_to_ecef(*args, **kwargs) -> np.ndarray:
    """
    ZAŚLEPKA: Transformacja Launch → ECEF.
    Do implementacji przy rozszerzeniu na model WGS84.
    """
    raise NotImplementedError(
        "Transformacja Launch→ECEF nie jest jeszcze zaimplementowana. "
        "Wymaga pozycji geograficznej wyrzutni i czasu."
    )


def dcm_ecef_to_eci(*args, **kwargs) -> np.ndarray:
    """
    ZAŚLEPKA: Transformacja ECEF → ECI.
    Do implementacji przy rozszerzeniu na model WGS84.
    """
    raise NotImplementedError(
        "Transformacja ECEF→ECI nie jest jeszcze zaimplementowana. "
        "Wymaga czasu gwiazdowego (GMST)."
    )
