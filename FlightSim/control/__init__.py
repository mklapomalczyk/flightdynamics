"""
control — modul sterowania dla modelu 6DOF.

Lancuch:  command -> actuator -> effector(moment) -> model 6DOF

Przyklad:
    from control import ControlSystem, StepCommander, PassthroughActuator
    from control.effectors import AeroSurfaceEffector

    cs = ControlSystem(
        commander=StepCommander(t_step=3.0, amplitude_deg=2.0),
        actuator=PassthroughActuator(),
        effectors=[AeroSurfaceEffector(table)],
    )
    force_model = ForceModel6DOF(..., control=cs)
"""

from .types import ControlCommand, ControlWrench, FlightState
from .commander import Commander, ZeroCommander, StepCommander, ConstantCommander
from .actuator import Actuator, PassthroughActuator, SecondOrderActuator
from .effectors import Effector, AeroSurfaceEffector
from .derivatives import ControlDerivTable, PANEL_PATTERNS, fit_derivative_from_sweep
from .system import ControlSystem, build_actuator_from_config

__all__ = [
    "ControlCommand", "ControlWrench", "FlightState",
    "Commander", "ZeroCommander", "StepCommander", "ConstantCommander",
    "Actuator", "PassthroughActuator", "SecondOrderActuator",
    "Effector", "AeroSurfaceEffector",
    "ControlDerivTable", "PANEL_PATTERNS", "fit_derivative_from_sweep",
    "ControlSystem", "build_actuator_from_config",
]
