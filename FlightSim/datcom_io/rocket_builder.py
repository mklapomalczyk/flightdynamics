"""
datcom_io/rocket_builder.py
===========================
Funkcje fabryczne — budują obiekty modelu dynamiki z RocketConfig.

Użycie:
    from datcom_io.rocket_builder import build_mass_model, build_propulsion
    from datcom_io.config_reader import load_config

    cfg  = load_config("configurations/rocket_70mm_baseline.yaml")
    mass = build_mass_model(cfg)
    prop = build_propulsion(cfg)
"""

import numpy as np
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datcom_io.config_reader import RocketConfig, MassModelConfig, PropulsionConfig
from models.mass6 import MassModel6DOF, ConstantIxx, LinearIxx
from forces.force_model6 import PropulsionConfig6DOF


# ============================================================================
# Model masowy
# ============================================================================

def build_mass_model(cfg: RocketConfig) -> MassModel6DOF:
    """
    Buduje MassModel6DOF z sekcji mass_model w YAML.

    Liniowo interpoluje masę, xcg, Iyy i Ixx między stanem
    full (zapłon) i empty (burnout).

    Parameters
    ----------
    cfg : RocketConfig
        Konfiguracja wczytana z YAML — musi zawierać mass_model i propulsion.

    Returns
    -------
    MassModel6DOF
    """
    if cfg.mass_model is None:
        raise ValueError(
            "Brak sekcji 'mass_model' w pliku YAML.\n"
            "Dodaj parametry full/empty do konfiguracji."
        )
    if cfg.propulsion is None:
        raise ValueError(
            "Brak sekcji 'propulsion' w pliku YAML.\n"
            "Model masowy potrzebuje t_burn z profilu ciągu."
        )

    mm  = cfg.mass_model
    t_b = cfg.propulsion.t_burn

    return MassModel6DOF(
        m_full    = mm.full.mass,
        m_empty   = mm.empty.mass,
        t_burn    = t_b,
        xcg_full  = mm.full.xcg,
        xcg_empty = mm.empty.xcg,
        Iyy_full  = mm.full.Iyy,
        Iyy_empty = mm.empty.Iyy,
        ixx_model = LinearIxx(
            Ixx_full  = mm.full.Ixx,
            Ixx_empty = mm.empty.Ixx,
            t_burn    = t_b,
        ),
        t_ignition = cfg.propulsion.t_ignition,
    )


# ============================================================================
# Propulsja z profilem ciągu
# ============================================================================

class ThrustProfile:
    """
    Interpoluje ciąg z tabeli [(t, F)].

    t=0 to moment zapłonu.
    Po ostatnim punkcie: F=0.
    Przed pierwszym punktem: F=0.
    """

    def __init__(self, profile: list):
        """
        Parameters
        ----------
        profile : list of [t, F]
            Tabela par [czas od zapłonu [s], ciąg [N]].
        """
        arr = np.array(profile, dtype=float)
        self.t_table = arr[:, 0]
        self.F_table = arr[:, 1]
        self.t_end   = float(self.t_table[-1])

    def thrust_at(self, t_since_ignition: float) -> float:
        """
        Zwraca ciąg [N] dla czasu t od zapłonu.

        Parameters
        ----------
        t_since_ignition : float
            Czas od zapłonu [s].
        """
        if t_since_ignition < 0.0 or t_since_ignition > self.t_end:
            return 0.0
        return float(np.interp(t_since_ignition, self.t_table, self.F_table))

    @property
    def t_burn(self) -> float:
        return self.t_end

    @property
    def max_thrust(self) -> float:
        return float(np.max(self.F_table))

    def summary(self) -> str:
        lines = [
            f"Profil ciągu:",
            f"  t_burn    = {self.t_end:.3f} s",
            f"  F_max     = {self.max_thrust:.1f} N",
            f"  Impuls    = {np.trapezoid(self.F_table, self.t_table):.1f} N·s",
            f"  Punkty    = {len(self.t_table)}",
        ]
        return "\n".join(lines)


def build_propulsion(cfg: RocketConfig) -> "PropulsionConfig6DOFDynamic":
    """
    Buduje obiekt propulsji z profilem ciągu z YAML.

    Returns
    -------
    PropulsionConfig6DOFDynamic
        Obiekt z metodą thrust_at(t) zamiast stałego ciągu.
    """
    if cfg.propulsion is None:
        raise ValueError("Brak sekcji 'propulsion' w pliku YAML.")

    profile = ThrustProfile(cfg.propulsion.thrust_profile)
    t_ign   = cfg.propulsion.t_ignition

    return PropulsionConfig6DOFDynamic(
        thrust_profile  = profile,
        t_ignition      = t_ign,
        nozzle_diameter = cfg.propulsion.nozzle_diameter,
    )


# ============================================================================
# Stan początkowy
# ============================================================================

def build_initial_state(mission) -> "State6DOF":
    """
    Buduje stan początkowy 6DOF z parametrów misji.

    Parameters
    ----------
    mission : MissionConfig
        Konfiguracja misji z elewacją i azymutem.

    Returns
    -------
    State6DOF
    """
    from core.state6 import State6DOF
    return State6DOF.initial(
        elevation_deg = mission.elevation,
        azimuth_deg   = mission.azimuth,
    )


# ============================================================================
# Geometria
# ============================================================================

def build_geometry(cfg: RocketConfig):
    """
    Buduje RocketGeometry6DOF z sekcji body w YAML.

    xcp jest pobierane z mass.xcg_ref jako przybliżenie startowe —
    przy TableAero jest nadpisywane tabelą xcp(alpha, Mach).
    """
    from forces.force_model6 import RocketGeometry6DOF

    import math
    fin = cfg.fins[0] if cfg.fins else None
    cant_rad  = math.radians(fin.cant_angle) if fin else 0.0
    n_fins    = int(fin.count) if fin else 0
    span_ref  = float(fin.span) if fin else 0.0

    return RocketGeometry6DOF(
        S_ref          = math.pi * (cfg.body.diameter / 2.0) ** 2,
        d_ref          = cfg.body.diameter,
        xcp            = cfg.mass.xcg_ref,
        cant_angle_rad = cant_rad,
        n_fins         = n_fins,
        span_ref       = span_ref,
        CNA_fins_ref   = 0.2,    # przybliżenie — nadpisane przez DATCOM gdy dostępne
        CY_magnus      = 2.0,
    )


class PropulsionConfig6DOFDynamic:
    """
    Dynamiczny profil ciągu — zastępuje PropulsionConfig6DOF(thrust=stały).

    Kompatybilny z ForceModel6DOF — podmień propulsion na ten obiekt.
    ForceModel6DOF wywołuje: thrust = self.prop.thrust_at(t, mass_state)
    """

    def __init__(
        self,
        thrust_profile:  ThrustProfile,
        t_ignition:      float = 0.0,
        nozzle_diameter: float = 0.0,
        thrust_offset_y: float = 0.0,
        thrust_offset_z: float = 0.0,
    ):
        self.profile         = thrust_profile
        self.t_ignition      = t_ignition
        self.nozzle_diameter = nozzle_diameter
        self.offset_y        = thrust_offset_y
        self.offset_z        = thrust_offset_z

    def thrust_at(self, t: float) -> float:
        """Ciąg [N] dla czasu symulacji t."""
        return self.profile.thrust_at(t - self.t_ignition)

    # Kompatybilność wsteczna z PropulsionConfig6DOF
    @property
    def thrust(self) -> float:
        """Zwraca maksymalny ciąg — tylko do diagnostyki."""
        return self.profile.max_thrust
