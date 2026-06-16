"""
models/mass.py
==============
Model masy rakiety.

Obsługuje:
  - Fazę napędzaną: masa zmienia się liniowo między m_full a m_empty
    w czasie t_burn. Jednocześnie zmienia się środek ciężkości.
  - Fazę balistyczną: masa stała = m_empty, xcg stałe.

Interfejs jest przygotowany na rozszerzenie:
  - Nielinearny profil masy (tabela T(t) → import z pliku)
  - Nieosiowość ciągu (offsety xcg w przyszłym modelu 6DOF)

Konwencja pozycji xcg:
  Mierzona od nosa rakiety wzdłuż osi X body [m], wartości dodatnie.
  xcp (centrum parcia aerodynamicznego) — dostarczane z zewnątrz (DATCOM),
  nie jest własnością modelu masy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MassState:
    """Stan masowy w danej chwili t."""
    mass:       float   # masa całkowita [kg]
    xcg:        float   # pozycja środka ciężkości od nosa [m]
    Iyy:        float   # moment bezwładności wokół osi Y (pitch) [kg·m²]
    is_burning: bool    # czy silnik pracuje


class MassModel:
    """
    Model masy rakiety z liniowym spalaniem paliwa.

    Parameters
    ----------
    m_full : float
        Masa startowa (z paliwem) [kg].
    m_empty : float
        Masa końcowa (po wypaleniu) [kg].
    t_burn : float
        Czas palenia silnika [s].
    xcg_full : float
        Pozycja xcg na starcie, od nosa [m].
    xcg_empty : float
        Pozycja xcg po wypaleniu, od nosa [m].
    Iyy_full : float
        Moment bezwładności pitch na starcie [kg·m²].
        Jeśli None → szacowany automatycznie (model walca).
    Iyy_empty : float
        Moment bezwładności pitch po wypaleniu [kg·m²].
        Jeśli None → szacowany automatycznie.
    t_ignition : float
        Czas zapłonu od początku symulacji [s]. Domyślnie 0.
    """

    def __init__(
        self,
        m_full:    float,
        m_empty:   float,
        t_burn:    float,
        xcg_full:  float,
        xcg_empty: float,
        Iyy_full:  Optional[float] = None,
        Iyy_empty: Optional[float] = None,
        t_ignition: float = 0.0,
    ):
        if m_empty >= m_full:
            raise ValueError(
                f"Masa końcowa ({m_empty} kg) musi być mniejsza od startowej ({m_full} kg)."
            )
        if t_burn <= 0:
            raise ValueError(f"Czas palenia musi być > 0, podano: {t_burn}.")

        self.m_full    = float(m_full)
        self.m_empty   = float(m_empty)
        self.t_burn    = float(t_burn)
        self.xcg_full  = float(xcg_full)
        self.xcg_empty = float(xcg_empty)
        self.t_ignition = float(t_ignition)

        self._Iyy_full  = Iyy_full
        self._Iyy_empty = Iyy_empty

        self.m_propellant = self.m_full - self.m_empty
        self.mass_flow    = self.m_propellant / self.t_burn   # [kg/s]

    # ------------------------------------------------------------------ #
    #  Główna metoda
    # ------------------------------------------------------------------ #

    def at(self, t: float) -> MassState:
        """
        Zwraca stan masowy w chwili t.

        Parameters
        ----------
        t : float
            Czas od początku symulacji [s].
        """
        t_since_ignition = t - self.t_ignition

        if t_since_ignition < 0.0:
            # Przed zapłonem
            return MassState(
                mass       = self.m_full,
                xcg        = self.xcg_full,
                Iyy        = self._iyy(self.m_full, self._Iyy_full),
                is_burning = False,
            )
        elif t_since_ignition <= self.t_burn:
            # Faza napędzana — liniowy ubytek masy
            xi = t_since_ignition / self.t_burn   # postęp palenia [0, 1]
            mass = self.m_full - self.m_propellant * xi
            xcg  = self.xcg_full + (self.xcg_empty - self.xcg_full) * xi
            Iyy  = self._interpolate_iyy(xi)
            return MassState(
                mass       = mass,
                xcg        = xcg,
                Iyy        = Iyy,
                is_burning = True,
            )
        else:
            # Faza balistyczna
            return MassState(
                mass       = self.m_empty,
                xcg        = self.xcg_empty,
                Iyy        = self._iyy(self.m_empty, self._Iyy_empty),
                is_burning = False,
            )

    # ------------------------------------------------------------------ #
    #  Moment bezwładności
    # ------------------------------------------------------------------ #

    def _iyy(self, mass: float, Iyy_override: Optional[float]) -> float:
        """Zwraca Iyy: podany wprost lub obliczony z modelu."""
        if Iyy_override is not None:
            return float(Iyy_override)
        # Automatyczne szacowanie — nadpisz tą metodą gdy masz geometrię
        return self._estimate_iyy(mass)

    def _estimate_iyy(self, mass: float) -> float:
        """
        Szacowanie Iyy z proporcji masy.
        Prosta interpolacja liniowa między pełną a pustą.
        Zastąp modelem geometrycznym gdy znana jest geometria.
        """
        if self._Iyy_full is not None and self._Iyy_empty is not None:
            xi = (self.m_full - mass) / self.m_propellant
            return self._Iyy_full + (self._Iyy_empty - self._Iyy_full) * xi
        # Brak danych — zwróć NaN żeby błąd był widoczny
        return float("nan")

    def _interpolate_iyy(self, xi: float) -> float:
        """Interpoluje Iyy dla postępu palenia xi ∈ [0, 1]."""
        Iyy_f = self._iyy(self.m_full,  self._Iyy_full)
        Iyy_e = self._iyy(self.m_empty, self._Iyy_empty)
        return Iyy_f + (Iyy_e - Iyy_f) * xi

    # ------------------------------------------------------------------ #
    #  Repr / info
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"MassModel("
            f"m_full={self.m_full:.3f} kg, "
            f"m_empty={self.m_empty:.3f} kg, "
            f"m_prop={self.m_propellant:.3f} kg, "
            f"t_burn={self.t_burn:.2f} s, "
            f"mass_flow={self.mass_flow:.4f} kg/s)"
        )
