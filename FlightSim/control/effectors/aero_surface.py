"""
control/effectors/aero_surface.py
==================================
Efektor sterowania AERODYNAMICZNEGO (canardy, stery ogonowe).

Momenty liczone z pochodnych C*_delta zmierzonych w DATCOM (sweep wychylen,
patrz control/derivatives.py oraz aero.py::get_control_derivatives).

KONWENCJA ZNAKOW — wyprowadzona z kodu, nie z pamieci
-----------------------------------------------------
Uklad osi: X-przod, Y-prawo, Z-dol.
  alpha = atan2(w, u)                        (forces/force_model6.py)
  dw_dt = FZ/m + q*u - p*v   => dodatnie q zwieksza w => zwieksza alpha
  dq_dt = MY/Iyy + ...       => dodatnie MY zwieksza alpha ("nos w gore")

Stad przyjeta konwencja v1:
  + d_pitch -> + MY -> alpha rosnie
  + d_yaw   -> + MZ -> r rosnie ("nos w prawo"); UWAGA: dbeta/dt ~ -r,
               wiec beta wtedy MALEJE. To ta sama inwersja recznosci, ktora
               jest opisana w forces/force_model6.py przy MA_yaw — i
               najczestsze zrodlo bledu znaku w tym projekcie.
  + d_roll  -> + MX -> p rosnie (prawe skrzydlo w dol)

Znak POCHODNYCH nie jest tu narzucany — bierzemy go wprost z pomiaru DATCOM.
Weryfikacja (a nie zalozenie) w tests/test_control_signs.py:
  1. canardy sa PRZED srodkiem ciezkosci, wiec Cm_delta musi miec znak
     PRZECIWNY do Cm_alpha (efekt destabilizujacy),
  2. test na zamrozonym stanie: +d_pitch => dq_dt > 0 itd.,
  3. kontrola symetrii Cn_delta ~ -Cm_delta dla ukladu krzyzowego.
Jesli ktorykolwiek nie przechodzi — poprawiamy PANEL_PATTERNS, nie znak tutaj.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..derivatives import ControlDerivTable
from ..types import ControlWrench, FlightState
from .base import Effector

CHANNELS: Tuple[str, ...] = ("d_pitch", "d_yaw", "d_roll")


class AeroSurfaceEffector(Effector):
    """
    Powierzchnie sterowe o skutecznosci z tablic DATCOM.

    M = C*_delta(alpha, Mach) * delta[rad] * q_dyn * S_ref * d_ref
    F = C*_delta(alpha, Mach) * delta[rad] * q_dyn * S_ref
    """

    channels = CHANNELS

    def __init__(self, table: ControlDerivTable,
                 S_ref: float | None = None, d_ref: float | None = None,
                 name: str = "canards"):
        """
        S_ref / d_ref: MUSZA byc te same, ktorych uzywa pasywna aerodynamika
        w models/aerodynamics.py (czyli geom.S_ref i geom.d_ref), inaczej
        moment sterowania i moment przywracajacy licza sie w dwoch roznych
        skalach i porownywanie ich nie ma sensu.

        DLACZEGO TO PULAPKA: DATCOM normalizuje CM przez LREF, a generator
        podaje mu LREF = dlugosc kadluba (1.285 m dla 70mm). Tymczasem model
        6DOF mnozy wspolczynniki momentu przez geom.d_ref = SREDNICA (0.070 m).
        Roznica to czynnik ~18. Domyslne wziecie table.d_ref (czyli LREF z
        DATCOM) dawalo moment sterowania 18x wiekszy niz moment przywracajacy
        liczony w konwencji modelu — 5 deg canardow "przewracalo" rakiete.
        Dlatego domyslnie bierzemy geometrie modelu, a nie tablicy.

        UWAGA (osobna sprawa, poza zakresem modulu sterowania): to znaczy, ze
        pasywna aerodynamika stosuje wspolczynniki DATCOM-a znormalizowane
        przez 1.285 m mnozac je przez 0.070 m. Jesli to blad, dotyczy on
        ZWALIDOWANEGO modelu pasywnego, nie sterowania — nie zmieniamy tego
        tutaj. Modul sterowania jedynie trzyma sie TEJ SAMEJ konwencji, co
        reszta modelu, zeby obie strony bilansu byly porownywalne.
        """
        self.table = table
        self.S_ref = float(S_ref if S_ref is not None else table.S_ref)
        self.d_ref = float(d_ref if d_ref is not None else table.d_ref)
        self.name = name
        if self.S_ref <= 0.0 or self.d_ref <= 0.0:
            raise ValueError("AeroSurfaceEffector: S_ref i d_ref musza byc > 0")

    @classmethod
    def from_geometry(cls, table: ControlDerivTable, geom,
                      name: str = "canards") -> "AeroSurfaceEffector":
        """
        Zalecany sposob tworzenia: bierze S_ref/d_ref z geometrii modelu 6DOF,
        czyli dokladnie te wartosci, ktorymi posluguje sie pasywna aero.
        """
        return cls(table, S_ref=geom.S_ref, d_ref=geom.d_ref, name=name)

    def wrench(self, u: np.ndarray, fs: FlightState) -> ControlWrench:
        u = np.asarray(u, dtype=float)
        # Brak cisnienia dynamicznego => brak sterowania aerodynamicznego.
        # (Np. na szynie przy v=0 albo po apogeum przy bardzo malej predkosci.)
        if fs.q_dyn <= 0.0 or u.size == 0:
            return ControlWrench(diag=self._diag(u, 0.0, 0.0, 0.0))

        d = np.deg2rad(u)
        c = self.table.interp(fs.alpha, fs.mach)

        qS  = fs.q_dyn * self.S_ref
        qSd = qS * self.d_ref

        d_pitch = d[0] if len(d) > 0 else 0.0
        d_yaw   = d[1] if len(d) > 1 else 0.0
        d_roll  = d[2] if len(d) > 2 else 0.0

        Mx = c["Cl_delta"] * d_roll  * qSd
        My = c["Cm_delta"] * d_pitch * qSd
        Mz = c["Cn_delta"] * d_yaw   * qSd

        # Sily: CN dziala wzdluz +Z_body przy dodatnim wychyleniu pitch
        # (ten sam kierunek co CN aerodynamiczne, patrz FA_z w force_model6).
        Fz = c["CN_delta"] * d_pitch * qS
        Fy = c["CY_delta"] * d_yaw   * qS

        return ControlWrench(
            F=np.array([0.0, Fy, Fz]),
            M=np.array([Mx, My, Mz]),
            diag=self._diag(u, Mx, My, Mz),
        )

    @staticmethod
    def _diag(u, Mx, My, Mz) -> dict:
        g = lambda i: float(u[i]) if len(u) > i else 0.0
        return {
            "d_pitch_act": g(0), "d_yaw_act": g(1), "d_roll_act": g(2),
            "M_ctrl_roll": float(Mx), "M_ctrl": float(My),
            "M_ctrl_yaw": float(Mz),
        }
