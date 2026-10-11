"""
control/commander.py
====================
Blok 1 lancucha: KOMENDA.

W v1 sluzy wylacznie do testow (komenda skokowa) — to nie jest regulator.
Przyszle prawa sterowania (PID, naprowadzanie) wchodza w to samo miejsce,
implementujac Commander.command().

WAZNE — bezstanowosc: derivatives() modelu sil jest wolane w punktach
posrednich RK45, niemonotonicznie w czasie i wielokrotnie na krok. Commander
MUSI byc czysta funkcja FlightState. Pamiec regulatora (calka PID, stan
serwa) idzie do wektora stanu ODE, nie do atrybutow obiektu.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .types import ControlCommand, FlightState

# Standardowe kanaly sterowania aerodynamicznego.
DEFAULT_CHANNELS: Tuple[str, ...] = ("d_pitch", "d_yaw", "d_roll")


class Commander(ABC):
    """Zrodlo zadania sterowania."""

    channels: Tuple[str, ...] = DEFAULT_CHANNELS

    @abstractmethod
    def command(self, fs: FlightState) -> ControlCommand:
        """Zwraca zadane wychylenia [deg] dla self.channels."""
        raise NotImplementedError


class ZeroCommander(Commander):
    """Brak sterowania. Domyslny — trajektoria identyczna jak bez modulu."""

    def __init__(self, channels: Tuple[str, ...] = DEFAULT_CHANNELS):
        self.channels = channels

    def command(self, fs: FlightState) -> ControlCommand:
        return ControlCommand(t=fs.t, u=np.zeros(len(self.channels)),
                              channels=self.channels)


class StepCommander(Commander):
    """
    Komenda skokowa lub IMPULSOWA — narzedzie testowe do sprawdzenia lancucha.

    duration_s = None  -> skok trwaly:  0 przed t_step, amplitude_deg po nim.
    duration_s = T     -> impuls prostokatny: amplitude_deg tylko w przedziale
                          [t_step, t_step + T), potem powrot do 0.

    Impuls jest mocniejszym testem niz trwaly skok: sprawdza nie tylko, ze
    sterowanie DZIALA, ale i ze przestaje dzialac po zdjeciu komendy —
    moment wraca do zera, a wywolana zmiana orientacji zostaje (bo jest
    calka momentu). Trwaly skok nie odroznilby tych dwoch rzeczy.

    Uwaga numeryczna: kazde zbocze to nieciaglosc f(t); RK45 przechodzi przez
    nia skracajac krok (kontrola bledu to zalatwia). Impuls ma dwa zbocza,
    wiec dwa takie miejsca. Nie dodajemy obslugi zdarzen w v1 — wystarczy
    wiedziec o tym przy czytaniu logow.
    """

    def __init__(self, t_step: float = 3.0, amplitude_deg: float = 2.0,
                 channel: str = "d_pitch",
                 channels: Tuple[str, ...] = DEFAULT_CHANNELS,
                 duration_s: Optional[float] = None):
        if channel not in channels:
            raise ValueError(f"kanal {channel!r} spoza {channels}")
        if duration_s is not None and duration_s <= 0.0:
            raise ValueError("duration_s musi byc > 0 albo None (skok trwaly)")
        self.t_step = float(t_step)
        self.amplitude_deg = float(amplitude_deg)
        self.channel = channel
        self.channels = channels
        self.duration_s = None if duration_s is None else float(duration_s)

    @property
    def t_end(self) -> float:
        """Koniec impulsu (inf dla skoku trwalego)."""
        return (float("inf") if self.duration_s is None
                else self.t_step + self.duration_s)

    def command(self, fs: FlightState) -> ControlCommand:
        u = np.zeros(len(self.channels))
        if self.t_step <= fs.t < self.t_end:
            u[self.channels.index(self.channel)] = self.amplitude_deg
        return ControlCommand(t=fs.t, u=u, channels=self.channels)


class ConstantCommander(Commander):
    """Stale wychylenie — do testow trymu/rownowagi."""

    def __init__(self, deflections_deg: dict,
                 channels: Tuple[str, ...] = DEFAULT_CHANNELS):
        self.channels = channels
        self._u = np.array([float(deflections_deg.get(c, 0.0)) for c in channels])

    def command(self, fs: FlightState) -> ControlCommand:
        return ControlCommand(t=fs.t, u=self._u.copy(), channels=self.channels)
