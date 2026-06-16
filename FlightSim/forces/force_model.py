"""
forces/force_model.py
=====================
Agregator sił — serce architektury.

Zbiera wszystkie siły i momenty działające na rakietę i zwraca
jeden wektor pochodnych stanu dla solvera ODE.

Równania ruchu 3DOF w body frame (płaszczyzna pitch):

  du/dt = (FX_total / m) - g*sin(θ) + q*w
  dw/dt = (FZ_total / m) - g*cos(θ) - q*u        ← tu minus! (konwencja)
  dθ/dt = q
  dq/dt = MA_yy / Iyy

  dx_pos/dt = u*cos(θ) + w*sin(θ)                 ← prędkość w launch frame
  dz_pos/dt = -u*sin(θ) + w*cos(θ)                ← lub: velocity_body_to_launch

Uwagi o konwencji:
  - q*w i -q*u to człony Coriolisa (obrót układu body względem inercjalnego)
  - g*sin(θ) i g*cos(θ) to projekcje grawitacji na osie body
  - dz_pos/dt: oś Z launch wskazuje W GÓRĘ, oś Z body "w górę prostopadle
    do osi rakiety", więc transformacja jest nieklasyczna — sprawdź signs!
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.state import State3DOF, STATE_SIZE
from core.frames import velocity_body_to_launch, compute_alpha
from models.atmosphere import AtmosphereModel
from models.mass import MassModel
from models.aerodynamics import AeroModel
from models.gravity import GravityModel


@dataclass
class PropulsionConfig:
    """
    Konfiguracja układu napędowego.

    Parameters
    ----------
    thrust : float
        Ciąg silnika [N]. Stały (na razie).
        Docelowo: thrust = f(t) jako callable lub tabela.
    thrust_offset_z : float
        Nieosiowość ciągu — przesunięcie siły względem osi X body [m].
        = 0 dla osiowego ciągu. Gotowe na rozszerzenie 6DOF.
    """
    thrust:          float = 0.0
    thrust_offset_z: float = 0.0   # zaślepka nieosiowości


@dataclass
class RocketGeometry:
    """
    Geometria referencyjna rakiety.

    Parameters
    ----------
    S_ref : float
        Przekrój referencyjny [m²]. Dla rakiet = π*(d/2)².
    d_ref : float
        Średnica referencyjna [m].
    xcp : float
        Położenie centrum parcia od nosa [m].
        Docelowo: xcp = f(alpha, Mach) z DATCOM.
    """
    S_ref: float
    d_ref: float
    xcp:   float   # centrum parcia, od nosa [m]

    @classmethod
    def from_diameter(cls, diameter_m: float, xcp: float) -> "RocketGeometry":
        """Tworzy geometrię z podanej średnicy i xcp."""
        S_ref = np.pi * (diameter_m / 2.0) ** 2
        return cls(S_ref=S_ref, d_ref=diameter_m, xcp=xcp)


class ForceModel:
    """
    Agregator sił i momentów.

    Używa wstrzykiwania zależności (dependency injection) — każdy
    model fizyczny jest podmienny bez zmiany kodu solvera.
    """

    def __init__(
        self,
        atmosphere:  AtmosphereModel,
        mass_model:  MassModel,
        aero_model:  AeroModel,
        gravity:     GravityModel,
        geometry:    RocketGeometry,
        propulsion:  PropulsionConfig,
    ):
        self.atmosphere  = atmosphere
        self.mass_model  = mass_model
        self.aero        = aero_model
        self.gravity     = gravity
        self.geom        = geometry
        self.prop        = propulsion

    def derivatives(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Funkcja pochodnych stanu — wołana przez solver ODE w każdym kroku.

        Parameters
        ----------
        t : float
            Czas [s].
        x : np.ndarray, shape (STATE_SIZE,)
            Wektor stanu: [x_pos, z_pos, u, w, theta, q].

        Returns
        -------
        np.ndarray, shape (STATE_SIZE,)
            Pochodne stanu: [dx, dz, du, dw, dtheta, dq].
        """
        state = State3DOF.from_numpy(x)

        # ---- Stan masowy i atmosferyczny --------------------------------- #
        mass_state = self.mass_model.at(t)
        atm        = self.atmosphere.at(-state.z_pos)

        m   = mass_state.mass
        Iyy = mass_state.Iyy
        xcg = mass_state.xcg

        g   = self.gravity.g(-state.z_pos)

        # ---- Kinematyka -------------------------------------------------- #
        alpha = compute_alpha(state.u, state.w)
        speed = state.speed
        mach  = atm.mach(speed)
        q_dyn = 0.5 * atm.density * speed**2

        # ---- Siły aerodynamiczne ----------------------------------------- #
        aero_forces = self.aero.compute(
            alpha  = alpha,
            mach   = mach,
            q_dyn  = q_dyn,
            q_rate = state.q,
            speed  = speed,
            xcg    = xcg,
            xcp    = self.geom.xcp,
            S_ref  = self.geom.S_ref,
            d_ref  = self.geom.d_ref,
        )

        # ---- Ciąg -------------------------------------------------------- #
        thrust = self.prop.thrust if mass_state.is_burning else 0.0

        # ---- Sumy sił w body frame --------------------------------------- #
        theta = state.theta
        q     = state.q
        u     = state.u
        w     = state.w

        # Projekcja grawitacji na osie body
        #   Oś X body: wzdłuż rakiety
        #   Oś Z body: prostopadle (w górę w płaszczyźnie lotu)
        # Wektor grawitacji w launch frame: [0, 0, -g] (oś Z launch w górę)
        # Transformacja launch→body:
        #   gx_body = -g * (-sin(theta)) = g*sin(theta) ... ale sprawdź znaki!
        #
        # Grawitacja w launch frame: g_L = [0, 0, -g]
        # g_body = R_LB @ g_L = R_LB @ [0, 0, -g]
        # R_LB = Ry(theta)^T = Ry(-theta)
        # g_body_x = -g * sin(theta)   ← siła grawitacji wzdłuż X body (hamuje przy θ>0)
        # g_body_z = -g * cos(theta)   ← siła grawitacji wzdłuż Z body

        gx_body = -g * np.sin(theta)
        gz_body = g * np.cos(theta)
        # print(gx_body,gz_body)
        # Siły całkowite w body (X i Z)
        FX = aero_forces.FA_x + thrust + m * gx_body
        FZ = -aero_forces.FA_z          + m * gz_body

        # Moment całkowity (tylko aerodynamiczny + tłumienie — ciąg osiowy)
        # Nieosiowość ciągu: M_thrust = thrust * offset_z (zaślepka)
        M_thrust = thrust * self.prop.thrust_offset_z
        MY = aero_forces.MA_yy + M_thrust

        # ---- Równania ruchu --------------------------------------------- #
        # Newton w body frame z uwzględnieniem ruchu obrotowego układu:
        #   m*(du/dt) = FX + m*q*w      (człon Coriolisa: q×v w osi X)
        #   m*(dw/dt) = FZ - m*q*u      (człon Coriolisa: q×v w osi Z)
        # Stąd:
        du_dt = FX / m + q * w
        dw_dt = FZ / m + q * u

        dtheta_dt = q
        dq_dt     = MY / Iyy if abs(Iyy) > 1e-10 else 0.0

        # ---- Kinematyka pozycji (body → launch) ------------------------- #
        dx_dt, dz_dt = velocity_body_to_launch(u, w, theta)

        return np.array([
            dx_dt,
            dz_dt,
            du_dt,
            dw_dt,
            dtheta_dt,
            dq_dt,
        ])
