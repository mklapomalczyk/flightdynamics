"""
control/system.py
=================
Spina lancuch: command -> actuator -> effector(y) -> ControlWrench.

To JEDYNY obiekt, o ktorym wie ForceModel6DOF.

Bezstanowosc jest wymogiem, nie stylem: derivatives() modelu sil jest wolane
w punktach posrednich RK45 (niemonotonicznie w czasie, kilka razy na krok),
wiec compute() musi byc czysta funkcja (fs, xa). Zaden obiekt w control/ nie
modyfikuje swoich atrybutow poza __init__.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .actuator import Actuator, PassthroughActuator
from .commander import Commander, ZeroCommander
from .effectors.base import Effector
from .types import ControlWrench, FlightState


class ControlSystem:
    """Lancuch sterowania."""

    def __init__(self, commander: Optional[Commander] = None,
                 actuator: Optional[Actuator] = None,
                 effectors: Optional[Sequence[Effector]] = None,
                 enabled: bool = True):
        self.commander = commander if commander is not None else ZeroCommander()
        self.actuator  = actuator  if actuator  is not None else PassthroughActuator()
        self.effectors: List[Effector] = list(effectors or [])
        self.enabled = bool(enabled)

    @property
    def n_states(self) -> int:
        """Dodatkowe stany ODE (0 dla serwa bezstanowego)."""
        return int(self.actuator.n_states)

    def compute(self, fs: FlightState,
                xa: Optional[np.ndarray] = None) -> ControlWrench:
        if not self.enabled or not self.effectors:
            return ControlWrench()
        cmd = self.commander.command(fs)
        u   = self.actuator.output(cmd, fs, xa)
        out = ControlWrench()
        for eff in self.effectors:
            out = out + eff.wrench(u, fs)
        return out
