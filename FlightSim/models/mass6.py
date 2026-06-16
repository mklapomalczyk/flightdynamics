"""
models/mass6.py
===============
Model masy dla 6DOF — rozszerzenie mass.py o moment bezwładności roll (Ixx).

Konwencja momentów bezwładności:
  Ixx — moment wokół osi X_body (roll). Dla rakiety osiowosymetrycznej
        znacznie mniejszy niż Iyy/Izz. Zależy od układu sterowania
        (canards, płetwy obrotowe itp.).
  Iyy = Izz — symetria osiowa (domyślnie). Dla konfiguracji asymetrycznych
              można podać osobne wartości.

Interfejs Ixx jest celowo wymienny (IxxModel Protocol) — różne układy
sterowania będą miały różny profil Ixx(t).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol, Callable


# ============================================================================
# Interfejs modelu Ixx — do podmiany dla różnych układów sterowania
# ============================================================================

class IxxModel(Protocol):
    """Interfejs modelu momentu bezwładności roll."""
    def at(self, t: float, mass: float) -> float:
        """Zwraca Ixx w chwili t dla aktualnej masy [kg·m²]."""
        ...


class ConstantIxx:
    """Stały Ixx — uproszczenie startowe."""
    def __init__(self, Ixx: float):
        self._Ixx = float(Ixx)

    def at(self, t: float, mass: float) -> float:
        return self._Ixx

    def __repr__(self) -> str:
        return f"ConstantIxx({self._Ixx} kg·m²)"


class ProportionalIxx:
    """
    Ixx proporcjonalny do masy: Ixx(t) = Ixx_ref * (m(t) / m_ref).
    Prosta aproksymacja gdy Ixx zależy głównie od masy paliwa.
    """
    def __init__(self, Ixx_full: float, m_full: float):
        self._Ixx_full = float(Ixx_full)
        self._m_full   = float(m_full)

    def at(self, t: float, mass: float) -> float:
        return self._Ixx_full * (mass / self._m_full)

    def __repr__(self) -> str:
        return f"ProportionalIxx(Ixx_full={self._Ixx_full})"


# ============================================================================
# Stan masowy 6DOF
# ============================================================================

@dataclass
class MassState6DOF:
    """Stan masowy w chwili t — pełny tensor bezwładności."""
    mass:       float
    xcg:        float
    Ixx:        float   # roll
    Iyy:        float   # pitch
    Izz:        float   # yaw  (= Iyy dla symetrii osiowej)
    Ixy:        float = 0.0   # zaślepka produktów bezwładności (6DOF asymetryczny)
    Ixz:        float = 0.0
    Iyz:        float = 0.0
    is_burning: bool  = False


# ============================================================================
# Model masy 6DOF
# ============================================================================

class LinearIxx:
    """
    Ixx liniowo interpolowany między full i empty w czasie spalania.
    """
    def __init__(self, Ixx_full: float, Ixx_empty: float, t_burn: float):
        self.Ixx_full  = float(Ixx_full)
        self.Ixx_empty = float(Ixx_empty)
        self.t_burn    = float(t_burn)

    def at(self, t_since_ignition: float, mass: float = 0.0) -> float:
        if t_since_ignition <= 0.0:
            return self.Ixx_full
        if t_since_ignition >= self.t_burn:
            return self.Ixx_empty
        frac = t_since_ignition / self.t_burn
        return self.Ixx_full + frac * (self.Ixx_empty - self.Ixx_full)


class MassModel6DOF:
    """
    Model masy rakiety dla 6DOF.

    Rozszerza logikę MassModel o:
      - Ixx przez wymienialny IxxModel
      - Izz = Iyy (symetria osiowa, domyślnie)
      - Furtka na produkty bezwładności (Ixy, Ixz, Iyz) — zaślepki = 0

    Parameters
    ----------
    m_full, m_empty : float
        Masy startowa i końcowa [kg].
    t_burn : float
        Czas palenia [s].
    xcg_full, xcg_empty : float
        Pozycja xcg od nosa [m].
    Iyy_full, Iyy_empty : float
        Moment bezwładności pitch (= Izz dla symetrii) [kg·m²].
    ixx_model : IxxModel
        Model Ixx — domyślnie ConstantIxx.
    t_ignition : float
        Czas zapłonu [s].
    """

    def __init__(
        self,
        m_full:     float,
        m_empty:    float,
        t_burn:     float,
        xcg_full:   float,
        xcg_empty:  float,
        Iyy_full:   float,
        Iyy_empty:  float,
        ixx_model:  Optional[IxxModel] = None,
        t_ignition: float = 0.0,
    ):
        if m_empty >= m_full:
            raise ValueError("m_empty musi być < m_full.")
        if t_burn <= 0:
            raise ValueError("t_burn musi być > 0.")

        self.m_full     = float(m_full)
        self.m_empty    = float(m_empty)
        self.t_burn     = float(t_burn)
        self.xcg_full   = float(xcg_full)
        self.xcg_empty  = float(xcg_empty)
        self.Iyy_full   = float(Iyy_full)
        self.Iyy_empty  = float(Iyy_empty)
        self.t_ignition = float(t_ignition)

        self.m_propellant = self.m_full - self.m_empty
        self.mass_flow    = self.m_propellant / self.t_burn

        # Domyślny model Ixx: stały, ~10% Iyy (typowe dla smukłej rakiety)
        if ixx_model is None:
            Ixx_default = 0.1 * Iyy_full
            self.ixx_model = ConstantIxx(Ixx_default)
        else:
            self.ixx_model = ixx_model

    def at(self, t: float) -> MassState6DOF:
        """Zwraca pełny stan masowy w chwili t."""
        t_since = t - self.t_ignition

        if t_since < 0.0:
            mass = self.m_full
            xcg  = self.xcg_full
            Iyy  = self.Iyy_full
            burning = False
        elif t_since <= self.t_burn:
            xi   = t_since / self.t_burn
            mass = self.m_full - self.m_propellant * xi
            xcg  = self.xcg_full + (self.xcg_empty - self.xcg_full) * xi
            Iyy  = self.Iyy_full + (self.Iyy_empty - self.Iyy_full) * xi
            burning = True
        else:
            mass = self.m_empty
            xcg  = self.xcg_empty
            Iyy  = self.Iyy_empty
            burning = False

        Ixx = self.ixx_model.at(t, mass)

        return MassState6DOF(
            mass       = mass,
            xcg        = xcg,
            Ixx        = Ixx,
            Iyy        = Iyy,
            Izz        = Iyy,   # symetria osiowa
            is_burning = burning,
        )

    def __repr__(self) -> str:
        return (
            f"MassModel6DOF(m_full={self.m_full:.3f} kg, "
            f"m_empty={self.m_empty:.3f} kg, "
            f"t_burn={self.t_burn:.2f} s, "
            f"ixx={self.ixx_model})"
        )
