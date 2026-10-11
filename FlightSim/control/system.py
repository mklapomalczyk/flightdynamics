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

from .actuator import Actuator, PassthroughActuator, SecondOrderActuator
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
        g = lambda i: float(cmd.u[i]) if len(cmd.u) > i else 0.0
        out.diag["d_pitch_cmd"] = g(0)
        out.diag["d_yaw_cmd"] = g(1)
        out.diag["d_roll_cmd"] = g(2)
        return out

    def actuator_derivatives(self, fs: FlightState,
                             xa: Optional[np.ndarray] = None) -> np.ndarray:
        """Pochodne stanow aktuatora (pusty wektor gdy bezstanowy)."""
        if not self.enabled or self.n_states == 0:
            return np.zeros(0)
        cmd = self.commander.command(fs)
        return self.actuator.derivatives(cmd, fs, xa)


def build_actuator_from_config(act_cfg, n_channels: int = 3) -> Actuator:
    """Tworzy aktuator z ActuatorConfig (datcom_io/config_reader.py)."""
    if act_cfg is None or act_cfg.type == "zero_order":
        return PassthroughActuator()
    if act_cfg.type == "second_order":
        return SecondOrderActuator(
            n_channels       = n_channels,
            wn               = act_cfg.wn,
            zeta             = act_cfg.zeta,
            rate_limit_deg_s = act_cfg.rate_limit_deg_s,
            pos_limit_deg    = act_cfg.pos_limit_deg,
        )
    raise ValueError(f"Nieznany typ aktuatora: {act_cfg.type!r} "
                     f"(dozwolone: 'zero_order', 'second_order')")
