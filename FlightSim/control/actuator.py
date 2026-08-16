"""
control/actuator.py
===================
Blok 2 lancucha: SERWO (aktuator).

v1 celowo BEZ dynamiki i BEZ ograniczen — komenda przechodzi na wylot.
Struktura jest jednak juz przygotowana na serwo 2. rzedu:

    n_states       ile dodatkowych stanow ODE potrzebuje aktuator
    derivatives()  pochodne tych stanow
    initial_state()warunek poczatkowy

Dzieki temu przejscie na dynamike 2. rzedu bedzie zmiana LOKALNA
(nowa klasa + rozszerzenie wektora stanu w solver6.py), bez ruszania
Commandera, efektorow ani modelu sil.

Bezstanowosc: output() nie moze modyfikowac self — patrz uwaga w commander.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .types import ControlCommand, FlightState


class Actuator(ABC):
    """Serwo: zadanie -> rzeczywiste wychylenie."""

    #: liczba dodatkowych stanow ODE (0 => aktuator bezstanowy)
    n_states: int = 0

    @abstractmethod
    def output(self, cmd: ControlCommand, fs: FlightState,
               xa: Optional[np.ndarray] = None) -> np.ndarray:
        """Rzeczywiste wychylenia [deg], w kolejnosci cmd.channels."""
        raise NotImplementedError

    def derivatives(self, cmd: ControlCommand, fs: FlightState,
                    xa: Optional[np.ndarray] = None) -> np.ndarray:
        """Pochodne stanow aktuatora. Pusty wektor dla serwa bezstanowego."""
        return np.zeros(0)

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.n_states)


class PassthroughActuator(Actuator):
    """
    Serwo idealne: wyjscie = zadanie. Bez opoznienia, bez ograniczenia
    predkosci i wychylenia. Zgodnie z zalozeniem "pierwszego strzalu".
    """

    n_states = 0

    def output(self, cmd: ControlCommand, fs: FlightState,
               xa: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(cmd.u, dtype=float)


class SecondOrderActuator(Actuator):
    """
    ZAPLANOWANE, NIEZAIMPLEMENTOWANE — serwo 2. rzedu.

        delta_ddot = wn^2 * (delta_cmd - delta) - 2*zeta*wn*delta_dot

    z ograniczeniem predkosci (rate_limit) i wychylenia (pos_limit).

    Klasa istnieje, zeby zamrozic KSZTALT interfejsu (n_states = 2 na kanal,
    stany trafiaja do wektora ODE od indeksu 15 — wzorem IDX_PFWD=14 w
    core/state6.py). Implementacja wymaga rozszerzenia solver6.py i
    SimResult6DOF, co jest poza zakresem pierwszej wersji.
    """

    def __init__(self, n_channels: int, wn: float = 60.0, zeta: float = 0.7,
                 rate_limit_deg_s: float = 400.0, pos_limit_deg: float = 15.0):
        self.n_channels = int(n_channels)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.rate_limit_deg_s = float(rate_limit_deg_s)
        self.pos_limit_deg = float(pos_limit_deg)
        self.n_states = 2 * self.n_channels

    def output(self, cmd, fs, xa=None):
        raise NotImplementedError(
            "SecondOrderActuator: wymaga stanow ODE (indeksy >=15) i zmian w "
            "solver6.py/state6.py. Poza zakresem v1 — uzyj PassthroughActuator.")

    def derivatives(self, cmd, fs, xa=None):
        raise NotImplementedError(
            "SecondOrderActuator: patrz output().")
