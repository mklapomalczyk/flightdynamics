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
    Serwo 2. rzedu z ograniczeniami predkosci i wychylenia.

        delta_ddot = wn^2 * (delta_cmd - delta) - 2*zeta*wn*delta_dot

    Stany ODE (2 na kanal): [delta_0, delta_dot_0, delta_1, delta_dot_1, ...]
    Trafiaja do wektora stanu solvera od indeksu 15+ (za IDX_PFWD=14).
    """

    def __init__(self, n_channels: int, wn: float = 60.0, zeta: float = 0.7,
                 rate_limit_deg_s: float = 400.0, pos_limit_deg: float = 15.0):
        self.n_channels = int(n_channels)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.rate_limit = float(rate_limit_deg_s)
        self.pos_limit = float(pos_limit_deg)
        self.n_states = 2 * self.n_channels

    def output(self, cmd: ControlCommand, fs: FlightState,
               xa: Optional[np.ndarray] = None) -> np.ndarray:
        if xa is None or len(xa) < self.n_states:
            return np.asarray(cmd.u, dtype=float)
        return np.array([xa[2 * i] for i in range(self.n_channels)])

    def derivatives(self, cmd: ControlCommand, fs: FlightState,
                    xa: Optional[np.ndarray] = None) -> np.ndarray:
        if xa is None or len(xa) < self.n_states:
            return np.zeros(self.n_states)
        u_cmd = np.asarray(cmd.u, dtype=float)
        dxa = np.zeros(self.n_states)
        for i in range(self.n_channels):
            delta = xa[2 * i]
            delta_dot = xa[2 * i + 1]
            cmd_i = float(np.clip(u_cmd[i], -self.pos_limit, self.pos_limit))
            ddot = (self.wn ** 2) * (cmd_i - delta) - 2.0 * self.zeta * self.wn * delta_dot
            delta_dot_clamped = float(np.clip(delta_dot, -self.rate_limit, self.rate_limit))
            if abs(delta) >= self.pos_limit and delta * delta_dot_clamped > 0:
                delta_dot_clamped = 0.0
            dxa[2 * i] = delta_dot_clamped
            dxa[2 * i + 1] = ddot
        return dxa
