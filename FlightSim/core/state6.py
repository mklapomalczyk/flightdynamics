"""
core/state6.py
==============
Wektor stanu dla symulacji 6DOF.

Wektor stanu: x = [x, y, z, u, v, w, q0, q1, q2, q3, p, qr, r]  (13 składowych)

  x, y, z     — pozycja w launch frame [m]
                 oś X: poziomo w kierunku strzału
                 oś Y: poziomo w prawo
                 oś Z: pionowo w górę
  u, v, w     — prędkości w body frame [m/s]
                 u: wzdłużna (wzdłuż osi rakiety)
                 v: boczna (w prawo)
                 w: normalna (w dół, konwencja DATCOM)
  q0,q1,q2,q3 — kwaternion orientacji (body ← launch), część skalarna q0
  p, qr, r    — prędkości kątowe w body frame [rad/s]
                 p: roll (wokół X_body)
                 qr: pitch (wokół Y_body)
                 r: yaw (wokół Z_body)

Konwencja osi body (DATCOM):
  X_body: wzdłuż osi rakiety, w kierunku nosa
  Y_body: w prawo (patrząc z tyłu)
  Z_body: w dół (prostopadle do osi, w dół)
"""

import numpy as np
from dataclasses import dataclass, field

from core.quaternion import (
    quat_norm, quat_to_euler_zyx, euler_zyx_to_quat,
    aero_angles, quat_to_dcm
)

# Indeksy w wektorze stanu
IDX_X   = 0
IDX_Y   = 1
IDX_Z   = 2
IDX_U   = 3
IDX_V   = 4
IDX_W   = 5
IDX_Q0  = 6
IDX_Q1  = 7
IDX_Q2  = 8
IDX_Q3  = 9
IDX_P   = 10
IDX_QR  = 11   # pitch rate (qr żeby nie kolidować z q kwaternionem)
IDX_R         = 12
IDX_RAIL_DIST = 13   # rail_dist
IDX_PFWD      = 14   # p_fwd dual-spin

STATE_SIZE_6DOF      = 13
STATE_SIZE_6DOF_RAIL = 14


@dataclass
class State6DOF:
    """
    Wygodny wrapper na wektor stanu 6DOF.
    Kwaternion przechowywany jako znormalizowany.
    """
    # Pozycja
    x:  float = 0.0
    y:  float = 0.0
    z:  float = 0.0
    # Prędkość w body frame
    u:  float = 0.0
    v:  float = 0.0
    w:  float = 0.0
    # Kwaternion orientacji
    q0: float = 1.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    # Prędkości kątowe w body frame
    p:  float = 0.0   # roll rate
    qr: float = 0.0   # pitch rate
    r:  float = 0.0   # yaw rate
    p_fwd: float = 0.0   # roll rate — forward body (dual-spin)

    # ------------------------------------------------------------------ #
    #  Fabryki
    # ------------------------------------------------------------------ #

    @classmethod
    def from_numpy(cls, x: np.ndarray) -> "State6DOF":
        return cls(
            x=x[IDX_X], y=x[IDX_Y], z=x[IDX_Z],
            u=x[IDX_U], v=x[IDX_V], w=x[IDX_W],
            q0=x[IDX_Q0], q1=x[IDX_Q1], q2=x[IDX_Q2], q3=x[IDX_Q3],
            p=x[IDX_P], qr=x[IDX_QR], r=x[IDX_R],
            p_fwd=float(x[IDX_PFWD]) if len(x) > IDX_PFWD else 0.0,
        )

    @classmethod
    def initial(
        cls,
        elevation_deg: float,
        azimuth_deg:   float = 0.0,
        speed_0:       float = 0.0,
        roll_0:        float = 0.0,
    ) -> "State6DOF":
        """
        Stan startowy dla zadanego kąta elewacji.

        Układ Launch Frame: X wzdłuż azymutu, Y w prawo, Z w dół.
        Kwaternion opisuje orientację body względem LF — zawiera
        tylko elewację i roll. Azymut jest wbudowany w definicję
        osi X samego Launch Frame (nie w kwaternion).

        Parameters
        ----------
        elevation_deg : float
            Kąt elewacji [°]. 0 = poziomo, 90 = pionowo w górę.
        azimuth_deg : float
            Nieużywany w solverze — zachowany dla kompatybilności API.
            Azymut definiuje orientację Launch Frame względem NED
            i jest używany tylko w module geograficznym.
        speed_0 : float
            Prędkość początkowa wzdłuż osi rakiety [m/s].
        roll_0 : float
            Kąt przechylenia startowego [°].
        """
        theta0 = np.deg2rad(elevation_deg)
        phi0   = np.deg2rad(roll_0)

        # Kwaternion: tylko elewacja i roll — bez azymutu
        # psi=0 bo nos rakiety jest wzdłuż X_LF z definicji
        q = euler_zyx_to_quat(psi=0.0, theta=theta0, phi=phi0)

        return cls(
            x=0.0, y=0.0, z=0.0,
            u=speed_0, v=0.0, w=0.0,
            q0=q[0], q1=q[1], q2=q[2], q3=q[3],
            p=0.0, qr=0.0, r=0.0,
        )

    # ------------------------------------------------------------------ #
    #  Konwersja
    # ------------------------------------------------------------------ #

    def to_numpy(self, include_rail: bool = False, include_pfwd: bool = False) -> np.ndarray:
        base = np.array([
            self.x, self.y, self.z,
            self.u, self.v, self.w,
            self.q0, self.q1, self.q2, self.q3,
            self.p, self.qr, self.r,
        ], dtype=float)
        if include_rail:
            base = np.append(base, 0.0)   # rail_dist = 0 na starcie
        if include_pfwd:
            base = np.append(base, self.p_fwd)
        return base

    @property
    def quat(self) -> np.ndarray:
        return np.array([self.q0, self.q1, self.q2, self.q3])

    @property
    def omega(self) -> np.ndarray:
        """Wektor prędkości kątowej w body frame [p, qr, r]."""
        return np.array([self.p, self.qr, self.r])

    @property
    def velocity_body(self) -> np.ndarray:
        return np.array([self.u, self.v, self.w])

    # ------------------------------------------------------------------ #
    #  Kąty Eulera (tylko do outputu / wizualizacji)
    # ------------------------------------------------------------------ #

    @property
    def euler_zyx(self) -> np.ndarray:
        """[psi, theta, phi] w radianach."""
        return quat_to_euler_zyx(self.quat)

    @property
    def psi(self) -> float:
        return float(self.euler_zyx[0])

    @property
    def theta(self) -> float:
        return float(self.euler_zyx[1])

    @property
    def phi(self) -> float:
        return float(self.euler_zyx[2])

    @property
    def psi_deg(self) -> float:
        return float(np.degrees(self.psi))

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta))

    @property
    def phi_deg(self) -> float:
        return float(np.degrees(self.phi))

    # ------------------------------------------------------------------ #
    #  Kąty aerodynamiczne
    # ------------------------------------------------------------------ #

    @property
    def alpha(self) -> float:
        a, _ = aero_angles(self.u, self.v, self.w)
        return a

    @property
    def beta(self) -> float:
        _, b = aero_angles(self.u, self.v, self.w)
        return b

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.alpha))

    @property
    def beta_deg(self) -> float:
        return float(np.degrees(self.beta))

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.u**2 + self.v**2 + self.w**2))

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"State6DOF("
            f"pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) m | "
            f"V={self.speed:.1f} m/s | "
            f"α={self.alpha_deg:.2f}° β={self.beta_deg:.2f}° | "
            f"ψ={self.psi_deg:.1f}° θ={self.theta_deg:.1f}° φ={self.phi_deg:.1f}°)"
        )


# ============================================================================
# Wyniki symulacji 6DOF
# ============================================================================

@dataclass
class SimResult6DOF:
    """Wyniki symulacji 6DOF — wektory czasowe."""
    t:     np.ndarray
    # Pozycja
    x:     np.ndarray
    y:     np.ndarray
    z:     np.ndarray
    # Prędkości body
    u:     np.ndarray
    v:     np.ndarray
    w:     np.ndarray
    # Kwaternion
    q0:    np.ndarray
    q1:    np.ndarray
    q2:    np.ndarray
    q3:    np.ndarray
    # Prędkości kątowe
    p:     np.ndarray
    qr:    np.ndarray
    r:     np.ndarray
    p_fwd:  np.ndarray = field(default_factory=lambda: np.array([]))
    # Stany aktuatora (opcjonalne) — wychylenie [deg] i prędkość [deg/s] na kanał
    d_pitch_act: np.ndarray = field(default_factory=lambda: np.array([]))
    d_yaw_act:   np.ndarray = field(default_factory=lambda: np.array([]))
    d_roll_act:  np.ndarray = field(default_factory=lambda: np.array([]))
    d_pitch_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    d_yaw_rate:   np.ndarray = field(default_factory=lambda: np.array([]))
    d_roll_rate:  np.ndarray = field(default_factory=lambda: np.array([]))
    status: str        = field(default="ok")  # "ok" | "tumbling" | "blowup" | "timeout"

    # Wielkości pochodne
    speed: np.ndarray = field(init=False)
    alpha: np.ndarray = field(init=False)
    beta:  np.ndarray = field(init=False)
    psi:   np.ndarray = field(init=False)
    theta: np.ndarray = field(init=False)
    phi:   np.ndarray = field(init=False)

    def __post_init__(self):
        self.speed = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.alpha = np.arctan2(self.w, self.u)
        # beta wektorowo
        self.beta  = np.where(
            self.speed > 0.1,
            np.arcsin(np.clip(self.v / np.maximum(self.speed, 0.1), -1, 1)),
            0.0
        )
        # Kąty Eulera z kwaternionów
        n = len(self.t)
        psi_arr   = np.zeros(n)
        theta_arr = np.zeros(n)
        phi_arr   = np.zeros(n)
        for i in range(n):
            q = np.array([self.q0[i], self.q1[i], self.q2[i], self.q3[i]])
            euler = quat_to_euler_zyx(q)
            psi_arr[i]   = euler[0]
            theta_arr[i] = euler[1]
            phi_arr[i]   = euler[2]
        self.psi   = psi_arr
        self.theta = theta_arr
        self.phi   = phi_arr

    @classmethod
    def from_raw(cls, t: np.ndarray, y: np.ndarray,
                 status: str = "ok",
                 ctrl_idx: int = 0) -> "SimResult6DOF":
        n_ch = 0
        if ctrl_idx > 0 and y.shape[0] > ctrl_idx:
            n_ch = min((y.shape[0] - ctrl_idx) // 2, 3)
        empty = np.zeros_like(t)
        return cls(
            t=t,
            x=y[IDX_X],  y=y[IDX_Y],  z=y[IDX_Z],
            u=y[IDX_U],  v=y[IDX_V],  w=y[IDX_W],
            q0=y[IDX_Q0], q1=y[IDX_Q1], q2=y[IDX_Q2], q3=y[IDX_Q3],
            p=y[IDX_P],  qr=y[IDX_QR], r=y[IDX_R],
            p_fwd=y[IDX_PFWD] if y.shape[0] > IDX_PFWD else empty.copy(),
            d_pitch_act=y[ctrl_idx]     if n_ch >= 1 else empty.copy(),
            d_yaw_act=y[ctrl_idx + 2]   if n_ch >= 2 else empty.copy(),
            d_roll_act=y[ctrl_idx + 4]  if n_ch >= 3 else empty.copy(),
            d_pitch_rate=y[ctrl_idx + 1] if n_ch >= 1 else empty.copy(),
            d_yaw_rate=y[ctrl_idx + 3]   if n_ch >= 2 else empty.copy(),
            d_roll_rate=y[ctrl_idx + 5]  if n_ch >= 3 else empty.copy(),
            status=status,
        )

    @property
    def max_altitude(self) -> float:
        # Z launch w dół: wysokość = -z (z ujemne = w górze)
        return float(-np.min(self.z))

    @property
    def max_range(self) -> float:
        return float(np.max(np.sqrt(self.x**2 + self.y**2)))

    @property
    def max_speed(self) -> float:
        return float(np.max(self.speed))

    def summary(self) -> str:
        return (
            f"=== Wyniki symulacji 6DOF ===\n"
            f"  Czas lotu:         {self.t[-1]:.2f} s\n"
            f"  Max. wysokość:     {self.max_altitude:.1f} m  (Z-dół: -min(z))\n"
            f"  Max. zasięg:       {self.max_range:.1f} m\n"
            f"  Max. prędkość:     {self.max_speed:.1f} m/s\n"
            f"  Max. |alpha|:      {float(np.max(np.abs(np.degrees(self.alpha)))):.2f} °\n"
            f"  Max. |beta|:       {float(np.max(np.abs(np.degrees(self.beta)))):.2f} °\n"
        )
