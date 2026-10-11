"""
control/types.py
================
Obiekty danych przekazywane wzdluz lancucha sterowania:

    command -> actuator -> effector(moment) -> model 6DOF

To jest KONTRAKT modulu sterowania — wszystko inne (konkretne komendy,
serwa, efektory) jest wymienne, dopoki trzyma sie tych trzech typow.

Uklad osi: X-przod, Y-prawo, Z-dol (body frame), zgodnie z reszta modelu
(patrz forces/force_model6.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class FlightState:
    """
    Zamrozony "snapshot" stanu lotu podawany w dol lancucha sterowania.

    Powod istnienia: odcina efektory od wnetrza ForceModel6DOF. Dzieki temu
    dodanie nowego typu sterowania nie zmienia zadnej sygnatury po drodze.

    Pole `thrust` jest tu celowo, mimo ze sterowanie aerodynamiczne go nie
    uzywa — bez niego przyszly efektor TVC wymagalby przebudowy lancucha.
    """
    t:       float
    alpha:   float = 0.0     # [rad]
    beta:    float = 0.0     # [rad]
    mach:    float = 0.0
    q_dyn:   float = 0.0     # cisnienie dynamiczne [Pa]
    speed:   float = 0.0     # [m/s]
    p:       float = 0.0     # predkosc kątowa roll [rad/s]
    q:       float = 0.0     # pitch [rad/s]
    r:       float = 0.0     # yaw [rad/s]
    m:       float = 0.0     # masa [kg]
    Ixx:     float = 0.0
    Iyy:     float = 0.0
    Izz:     float = 0.0
    xcg:     float = 0.0     # [m od nosa]
    thrust:  float = 0.0     # [N]
    on_rail: bool  = False
    rho:     float = 0.0     # [kg/m^3]


@dataclass(frozen=True)
class ControlCommand:
    """
    Zadanie sterowania — GENERYCZNY wektor kanalow, nie sztywna trojka.

    v1 (canardy): channels = ("d_pitch", "d_yaw", "d_roll"), u w STOPNIACH
    wychylenia powierzchni (decyzja projektowa: komenda JEST wychyleniem).

    Przyszlosc bez zmian w rurociagu:
      TVC : channels = ("gimbal_y", "gimbal_z"), u w stopniach
      RCS : channels = ("thr_pitch", ...),       u jako wypelnienie 0..1
    """
    t:        float
    u:        np.ndarray
    channels: Tuple[str, ...]

    def get(self, name: str, default: float = 0.0) -> float:
        """Wartosc kanalu po nazwie (bezpieczne gdy kanal nie istnieje)."""
        if name in self.channels:
            return float(self.u[self.channels.index(name)])
        return default


@dataclass
class ControlWrench:
    """
    Sily i momenty sterowania w ukladzie ciala [N], [N*m].

    To jest "wspolna waluta" calego modulu: canardy, TVC i RCS roznia sie
    sposobem liczenia, ale wszystkie zwracaja wlasnie to. Dlatego dodanie
    nowego typu sterowania nie dotyka niczego poza katalogiem effectors/.

    `diag` trafia do logu sil (models/force_logger.py).
    """
    F:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    M:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    diag: Dict[str, float] = field(default_factory=dict)

    def __add__(self, other: "ControlWrench") -> "ControlWrench":
        """Sumowanie wielu efektorow (np. canardy + TVC) jest darmowe."""
        if not isinstance(other, ControlWrench):
            return NotImplemented
        d = dict(self.diag)
        d.update(other.diag)
        return ControlWrench(F=self.F + other.F, M=self.M + other.M, diag=d)

    @property
    def Mx(self) -> float:  # roll
        return float(self.M[0])

    @property
    def My(self) -> float:  # pitch
        return float(self.M[1])

    @property
    def Mz(self) -> float:  # yaw
        return float(self.M[2])
