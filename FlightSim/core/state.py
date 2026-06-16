"""
core/state.py
=============
Definicja wektora stanu dla symulacji 3DOF (płaszczyzna pitch).

Wektor stanu: x = [x_pos, z_pos, u, w, theta, q]
  x_pos  - pozycja pozioma w launch frame [m]
  z_pos  - pozycja pionowa w launch frame [m]
  u      - prędkość wzdłużna w body frame [m/s]
  w      - prędkość normalna w body frame [m/s]
  theta  - kąt pochylenia (pitch) [rad]
  q      - prędkość kątowa pochylenia [rad/s]

Konwencja układów:
  Launch frame (L): inercjalny, oś X poziomo, oś Z pionowo w górę,
                    origin = pozycja rakiety w t=0
  Body frame (B):   oś X wzdłuż osi rakiety (w kierunku lotu),
                    oś Z prostopadle w górę względem rakiety
  Aerodynamic frame (A): oś X wzdłuż wektora prędkości względem powietrza
"""

from dataclasses import dataclass, field
import numpy as np


# Indeksy w wektorze stanu numpy — używaj tych stałych zamiast magic numbers
IDX_X     = 0   # pozycja pozioma [m]
IDX_Z     = 1   # pozycja pionowa [m]
IDX_U     = 2   # prędkość wzdłużna body [m/s]
IDX_W     = 3   # prędkość normalna body [m/s]
IDX_THETA = 4   # kąt pochylenia [rad]
IDX_Q     = 5   # prędkość kątowa pitch [rad/s]

STATE_SIZE = 6


@dataclass
class State3DOF:
    """
    Wygodny wrapper na wektor stanu. Pozwala odczytywać składowe
    po nazwie zamiast po indeksie. Konwersja do/z numpy jest tania.
    """
    x_pos:  float = 0.0   # pozycja pozioma, launch frame [m]
    z_pos:  float = 0.0   # pozycja pionowa, launch frame [m]
    u:      float = 0.0   # prędkość wzdłużna, body frame [m/s]
    w:      float = 0.0   # prędkość normalna, body frame [m/s]
    theta:  float = 0.0   # kąt pochylenia [rad]
    q:      float = 0.0   # prędkość kątowa pitch [rad/s]

    # ------------------------------------------------------------------ #
    #  Fabryki
    # ------------------------------------------------------------------ #

    @classmethod
    def from_numpy(cls, x: np.ndarray) -> "State3DOF":
        """Tworzy State3DOF z wektora numpy (kolejność jak IDX_*)."""
        return cls(
            x_pos  = float(x[IDX_X]),
            z_pos  = float(x[IDX_Z]),
            u      = float(x[IDX_U]),
            w      = float(x[IDX_W]),
            theta  = float(x[IDX_THETA]),
            q      = float(x[IDX_Q]),
        )

    @classmethod
    def initial(cls, elevation_deg: float, speed_0: float = 0.0) -> "State3DOF":
        """
        Tworzy stan startowy dla zadanego kąta elewacji wyrzutni.

        Parameters
        ----------
        elevation_deg : float
            Kąt elewacji wyrzutni [stopnie], 0 = poziomo, 90 = pionowo.
        speed_0 : float
            Prędkość początkowa wzdłuż osi rakiety [m/s].
            Domyślnie 0 (rakieta startuje ze stojaka).
        """
        theta0 = np.deg2rad(elevation_deg)
        return cls(
            x_pos = 0.0,
            z_pos = 0.0,
            u     = speed_0,
            w     = 0.0,
            theta = theta0,
            q     = 0.0,
        )

    # ------------------------------------------------------------------ #
    #  Konwersja
    # ------------------------------------------------------------------ #

    def to_numpy(self) -> np.ndarray:
        """Zwraca wektor stanu jako 1D numpy array."""
        return np.array([
            self.x_pos,
            self.z_pos,
            self.u,
            self.w,
            self.theta,
            self.q,
        ], dtype=float)

    # ------------------------------------------------------------------ #
    #  Właściwości pochodne
    # ------------------------------------------------------------------ #

    @property
    def speed(self) -> float:
        """Prędkość względem powietrza (brak wiatru → prędkość bezwzględna) [m/s]."""
        return float(np.sqrt(self.u**2 + self.w**2))

    @property
    def alpha(self) -> float:
        """Kąt natarcia [rad]. Konwencja: dodatni gdy nos skierowany w górę względem V."""
        return float(np.arctan2(self.w, self.u))

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.alpha))

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta))

    @property
    def gamma(self) -> float:
        """Kąt toru lotu (flight path angle) w launch frame [rad]."""
        # gamma = theta - alpha  (dla 3DOF bez ślizgu)
        return self.theta - self.alpha

    @property
    def gamma_deg(self) -> float:
        return float(np.degrees(self.gamma))

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"State3DOF("
            f"x={self.x_pos:.1f} m, z={self.z_pos:.1f} m, "
            f"V={self.speed:.1f} m/s, α={self.alpha_deg:.2f}°, "
            f"θ={self.theta_deg:.2f}°, q={np.degrees(self.q):.3f} °/s)"
        )


@dataclass
class SimResult:
    """
    Wynik symulacji — wektory czasowe wszystkich wielkości.
    Przechowuje wszystko jako numpy arrays dla wygodnej analizy.
    """
    t:      np.ndarray   # czas [s]
    x_pos:  np.ndarray   # pozycja pozioma [m]
    z_pos:  np.ndarray   # pozycja pionowa [m]
    u:      np.ndarray   # prędkość wzdłużna body [m/s]
    w:      np.ndarray   # prędkość normalna body [m/s]
    theta:  np.ndarray   # kąt pochylenia [rad]
    q:      np.ndarray   # prędkość kątowa [rad/s]

    # wielkości pochodne (obliczane przy budowie)
    speed:  np.ndarray = field(init=False)
    alpha:  np.ndarray = field(init=False)
    gamma:  np.ndarray = field(init=False)

    def __post_init__(self):
        self.speed = np.sqrt(self.u**2 + self.w**2)
        self.alpha = np.arctan2(self.w, self.u)
        self.gamma = self.theta - self.alpha

    @classmethod
    def from_raw(cls, t: np.ndarray, y: np.ndarray) -> "SimResult":
        """
        Buduje SimResult z surowych danych solve_ivp.

        Parameters
        ----------
        t : (N,) array
        y : (STATE_SIZE, N) array  — układ solve_ivp (wiersze = składowe stanu)
        """
        return cls(
            t      = t,
            x_pos  = y[IDX_X],
            z_pos  = y[IDX_Z],
            u      = y[IDX_U],
            w      = y[IDX_W],
            theta  = y[IDX_THETA],
            q      = y[IDX_Q],
        )

    @property
    def max_altitude(self) -> float:
        return float(np.max(-self.z_pos))

    @property
    def max_range(self) -> float:
        return float(np.max(self.x_pos))

    @property
    def max_speed(self) -> float:
        return float(np.max(self.speed))

    def summary(self) -> str:
        return (
            f"=== Wyniki symulacji ===\n"
            f"  Czas lotu:         {self.t[-1]:.2f} s\n"
            f"  Max. wysokość:     {self.max_altitude:.1f} m\n"
            f"  Max. zasięg (x):   {self.max_range:.1f} m\n"
            f"  Max. prędkość:     {self.max_speed:.1f} m/s\n"
        )
