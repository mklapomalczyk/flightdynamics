"""
control/effectors/base.py
=========================
Blok 3 lancucha: EFEKTOR — zamienia wychylenie/zadanie na sily i momenty.

To jest zawias rozszerzalnosci calego modulu. Kazdy typ sterowania
(aerodynamiczne, TVC, RCS) implementuje ten jeden interfejs i zwraca
ControlWrench w ukladzie ciala. Dzieki temu dodanie nowego typu sterowania
NIE DOTYKA niczego poza katalogiem effectors/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..types import ControlWrench, FlightState


class Effector(ABC):
    """Wychylenia -> sily i momenty w ukladzie ciala."""

    channels: Tuple[str, ...] = ()

    @abstractmethod
    def wrench(self, u: np.ndarray, fs: FlightState) -> ControlWrench:
        """u — rzeczywiste wychylenia [deg] w kolejnosci self.channels."""
        raise NotImplementedError
