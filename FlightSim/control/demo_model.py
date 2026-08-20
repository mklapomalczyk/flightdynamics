"""
control/demo_model.py
=====================
Wspolny model demonstracyjny 70mm uzywany przez wykresy i testy sterowania.

Jedno miejsce definicji, zeby wykres i test nie rozjechaly sie po cichu —
inaczej "test przechodzi, a wykres pokazuje co innego" bylo tylko kwestia czasu.

To NIE jest model lotny do walidacji — brak ciagu, masa staly, uproszczone
tabele aero. Sluzy do pokazania i sprawdzenia LANCUCHA sterowania.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from forces.force_model6 import (ForceModel6DOF, PropulsionConfig6DOF,
                                 RocketGeometry6DOF)
from models.aerodynamics import TableAero
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from models.mass6 import ConstantIxx, MassModel6DOF

ALPHA_GRID = np.deg2rad(np.array([-20., -10., 0., 10., 20.]))
MACH_GRID = np.array([0.1, 0.5, 1.0, 2.0, 3.0])


def demo_aero(with_roll_damping: bool = True) -> TableAero:
    """
    Uproszczona, ale statycznie stabilna aerodynamika.

    with_roll_damping: Clp_table jest tu ISTOTNE dla kanalu roll. Bez niego
    moment tocznny nie ma sie o co oprzec i po impulsie predkosc obrotowa
    zostaje na stale (nic jej nie hamuje), co wyglada jak blad modulu
    sterowania, a jest po prostu brakiem tlumienia w modelu demo.
    Prawdziwa rakieta ma Clp z DATCOM ($RLLO) albo z przyblizenia analitycznego.
    """
    o = np.ones((len(ALPHA_GRID), len(MACH_GRID)))
    return TableAero(
        alpha_table=ALPHA_GRID,
        mach_table=MACH_GRID,
        CA_table=0.45 * o,
        CN_table=np.array([[-4.] * 5, [-2.] * 5, [0.] * 5, [2.] * 5, [4.] * 5]),
        Cm_table=np.array([[1.2] * 5, [0.6] * 5, [0.] * 5, [-0.6] * 5, [-1.2] * 5]),
        Clp_table=(-2.0 * o) if with_roll_damping else None,
        xcg_ref=0.71,
    )


def build_demo_model(control=None, with_roll_damping: bool = True) -> ForceModel6DOF:
    mass = MassModel6DOF(m_full=4.2, m_empty=4.19, t_burn=1.0,
                         xcg_full=0.71, xcg_empty=0.71,
                         Iyy_full=0.646, Iyy_empty=0.646,
                         ixx_model=ConstantIxx(0.0032))
    return ForceModel6DOF(
        atmosphere=create_atmosphere("ISA"), mass_model=mass,
        aero_model=demo_aero(with_roll_damping),
        gravity=create_gravity("constant"),
        geometry=RocketGeometry6DOF.from_diameter(0.070, xcp=0.95),
        propulsion=PropulsionConfig6DOF(thrust=0.0),
        launcher=LauncherConfig(L_rail=0.0), control=control)


def fly_demo(control=None, t_max: float = 15.0, dt_output: float = 0.01,
             elevation_deg: float = 80.0, speed_0: float = 250.0,
             with_roll_damping: bool = True):
    """Lot demonstracyjny. z_ground bardzo nisko — nie przerywamy o grunt."""
    st = State6DOF.initial(elevation_deg=elevation_deg, azimuth_deg=0.0,
                           speed_0=speed_0)
    return run_simulation_6dof(build_demo_model(control, with_roll_damping),
                               st, t_max=t_max, dt_output=dt_output,
                               z_ground=-1e9)
