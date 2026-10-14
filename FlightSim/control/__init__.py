"""
control — modul sterowania dla modelu 6DOF.

Lancuch:  command -> actuator -> effector(moment) -> model 6DOF

Dwa typy sterowania (YAML: control_type):
  canards        — ControlSystem + AeroSurfaceEffector (ciagle, przez ODE)
  reaction_jets  — ReactionJetSystem + ReactionJetEffector (dyskretne
                   odpalenia, calkowanie segmentami w core/solver6.py)

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
from .effectors import (Effector, AeroSurfaceEffector, JetLayout, JetRow,
                        JetThrustProfile, ReactionJetEffector)
from .jets import (JetCommander, JetEvent, JetPulseRequest, JetTickContext,
                   LF_DIRECTIONS, NoJetCommander, ReactionJetSystem,
                   RollAwareJetCommander, ScheduleJetCommander,
                   build_reaction_jet_system)
from .derivatives import ControlDerivTable, PANEL_PATTERNS, fit_derivative_from_sweep
from .system import ControlSystem, build_actuator_from_config

__all__ = [
    "ControlCommand", "ControlWrench", "FlightState",
    "Commander", "ZeroCommander", "StepCommander", "ConstantCommander",
    "Actuator", "PassthroughActuator", "SecondOrderActuator",
    "Effector", "AeroSurfaceEffector",
    "JetLayout", "JetRow", "JetThrustProfile", "ReactionJetEffector",
    "JetCommander", "JetEvent", "JetTickContext", "NoJetCommander",
    "ReactionJetSystem", "ScheduleJetCommander", "build_reaction_jet_system",
    "JetPulseRequest", "RollAwareJetCommander", "LF_DIRECTIONS",
    "ControlDerivTable", "PANEL_PATTERNS", "fit_derivative_from_sweep",
    "ControlSystem", "build_actuator_from_config",
]
