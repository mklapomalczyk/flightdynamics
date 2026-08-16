"""
forces/force_model6.py
======================
Agregator sił i równania ruchu 6DOF z modelem szyny startowej.

Konwencja układów współrzędnych:
  Launch frame: X "na północ" (kierunek strzału), Y w prawo, Z w dół
  Body frame:   X wzdłuż nosa, Y w prawo, Z w dół

  g_launch = [0, 0, +g]
  g_body   = DCM @ g_launch

  Alpha: w > 0 (+Z_body = w dół) gdy nos powyżej toru → alpha > 0 (DATCOM)

Wektor stanu rozszerzony o drogę na szynie:
  x = [x, y, z, u, v, w, q0, q1, q2, q3, p, qr, r, rail_dist]  (14 składowych)

  rail_dist — droga przebyta wzdłuż szyny od zapłonu [m].
  d(rail_dist)/dt = max(u, 0)  gdy na szynie, 0 po opuszczeniu.

Model szyny:
  Gdy rail_dist < L_rail: zeruj v, w, p, qr, r i ich pochodne.
  Rakieta porusza się wyłącznie wzdłuż X_body.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.state6 import State6DOF
from core.quaternion import quat_norm, quat_to_dcm, quat_derivative, aero_angles
from models.atmosphere import AtmosphereModel
from models.mass6 import MassModel6DOF
from models.aerodynamics import AeroModel
from models.gravity import GravityModel
from models.launcher import LauncherConfig, RailLauncher
from models.wind import WindModel

# Rozmiar wektora stanu z rail_dist
STATE_SIZE_6DOF_RAIL = 14
IDX_RAIL_DIST = 13

# Zerowy wektor sil/momentow sterowania — read-only, wspoldzielony.
# Unikamy alokacji np.zeros(3) przy kazdym wywolaniu derivatives() (gorący tor:
# kilka razy na krok RK45) w najczestszym przypadku control=None.
_ZERO3 = np.zeros(3)
_ZERO3.flags.writeable = False


@dataclass
class PropulsionConfig6DOF:
    """Konfiguracja układu napędowego."""
    thrust:   float = 0.0
    offset_y: float = 0.0   # nieosiowość: moment yaw [m]
    offset_z: float = 0.0   # nieosiowość: moment pitch [m]


@dataclass
class RocketGeometry6DOF:
    """Geometria referencyjna rakiety."""
    S_ref: float
    d_ref: float
    xcp:            float
    cant_angle_rad: float = 0.0
    n_fins:         int   = 4
    span_ref:       float = 0.0
    CNA_fins_ref:   float = 0.2
    CY_magnus:      float = 2.0

    @classmethod
    def from_diameter(cls, diameter_m: float, xcp: float) -> "RocketGeometry6DOF":
        return cls(
            S_ref = np.pi * (diameter_m / 2.0) ** 2,
            d_ref = diameter_m,
            xcp   = xcp,
        )


class ForceModel6DOF:

    def __init__(
        self,
        atmosphere:  AtmosphereModel,
        mass_model:  MassModel6DOF,
        aero_model:  AeroModel,
        gravity:     GravityModel,
        geometry:    RocketGeometry6DOF,
        propulsion:  PropulsionConfig6DOF,
        launcher:    Optional[LauncherConfig] = None,
        logger:      Optional[object]         = None,
        cfg:         Optional[object]         = None,
        wind_model:  Optional[WindModel]      = None,
        control:     Optional[object]         = None,
    ):
        self.atmosphere = atmosphere
        self.mass_model = mass_model
        self.aero       = aero_model
        self.gravity    = gravity
        self.geom       = geometry
        self.prop       = propulsion
        self.launcher   = RailLauncher(
            launcher if launcher is not None else LauncherConfig(L_rail=0.0)
        )
        self.logger     = logger
        self.wind_model = wind_model
        # Modul sterowania (control/system.py::ControlSystem) — opcjonalny,
        # wzorem wind_model: trzymany surowo, sprawdzany w miejscu uzycia.
        # None => model zachowuje sie dokladnie jak przed dodaniem sterowania.
        self.control    = control
        # Dual-spin — opcjonalnie z cfg
        self._dual_spin = None
        if cfg is not None and getattr(cfg, 'dual_spin', None) is not None:
            ds = cfg.dual_spin
            if ds.enabled:
                self._dual_spin = ds

    def derivatives(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Pochodne stanu 6DOF + rail_dist.

        Parameters
        ----------
        x : (14,) array
            [x,y,z, u,v,w, q0,q1,q2,q3, p,qr,r, rail_dist]
        """
        # ---- Stan podstawowy -------------------------------------------- #
        state = State6DOF.from_numpy(x[:13])

        # ---- Szyna startowa --------------------------------------------- #
        # rail_dist jest 14. elementem tylko gdy L_rail > 0
        if not self.launcher.is_disabled() and len(x) > 13:
            rail_dist = float(x[IDX_RAIL_DIST])
        else:
            rail_dist = 0.0
        on_rail = self.launcher.on_rail(rail_dist)

        # ---- Normalizacja kwaterniona ------------------------------------ #
        q = quat_norm(state.quat)

        # ---- Modele środowiskowe i masowe ------------------------------- #
        ms  = self.mass_model.at(t)
        atm = self.atmosphere.at(-state.z)  # Z_launch w dół: wysokość = -z

        m   = ms.mass
        Ixx = ms.Ixx
        Iyy = ms.Iyy
        Izz = ms.Izz
        xcg = ms.xcg
        g   = self.gravity.g(-state.z)  # Z_launch w dół: wysokość = -z

        # ---- Prędkości -------------------------------------------------- #
        u = state.u
        # Na szynie: v=w=p=qr=r=0 (zeruj aby zachować spójność obliczeń)
        if on_rail:
            v, w  = 0.0, 0.0
            p, qr, r = 0.0, 0.0, 0.0
        else:
            v, w  = state.v, state.w
            p, qr, r = state.p, state.qr, state.r

        # ---- Wiatr -> predkosc wzgledem powietrza (tylko do aero) ------- #
        # Wiatr aktywny TYLKO poza szyna (jak v,w,p,qr,r). u_air/v_air/w_air
        # uzywane WYLACZNIE do katow aero/mach/q_dyn — dynamika (du_dt itd.)
        # i kinematyka pozycji (vel_launch) zawsze uzywaja u,v,w wzgledem ziemi.
        DCM = quat_to_dcm(q)
        if not on_rail and self.wind_model is not None:
            wind_lf   = self.wind_model.velocity_launch_frame(t, -state.z)
            wind_body = DCM @ wind_lf
            u_air = u - wind_body[0]
            v_air = v - wind_body[1]
            w_air = w - wind_body[2]
        else:
            u_air, v_air, w_air = u, v, w

        # ---- Kąty aerodynamiczne ---------------------------------------- #
        alpha, beta = aero_angles(u_air, v_air, w_air)
        speed = float(np.sqrt(u_air**2 + v_air**2 + w_air**2))
        mach  = atm.mach(speed)
        q_dyn = 0.5 * atm.density * speed**2

        # Calkowity kat natarcia (alpha+beta) — kat miedzy osia X_body a
        # wektorem predkosci, niezalezny od plaszczyzny. Uzyty tylko do CA
        # (opor bryly osiowosymetrycznej), patrz models/aerodynamics.py.
        if speed > 1e-6:
            alpha_total = float(np.arccos(np.clip(u_air / speed, -1.0, 1.0)))
        else:
            alpha_total = 0.0

        # ---- Siły aerodynamiczne ---------------------------------------- #
        aero = self.aero.compute(
            alpha       = alpha,
            mach        = mach,
            q_dyn       = q_dyn,
            q_rate      = qr,
            speed       = speed,
            xcg         = xcg,
            xcp         = self.geom.xcp,
            S_ref       = self.geom.S_ref,
            d_ref       = self.geom.d_ref,
            alpha_total = alpha_total,
            beta        = beta,
            r_rate      = r,
            powered     = bool(getattr(ms, "is_burning", False)),
        )

        FA_x = aero.FA_x
        FA_z = -aero.FA_z   # negacja: aero.py liczy dla Z_body w górę

        # ---- Siła boczna od CYβ ----------------------------------------- #
        # CYβ [1/deg] — przelicz na [1/rad] i oblicz siłę boczną
        CYB = 0.0
        if hasattr(self.aero, 'CYB_table') and self.aero.CYB_table is not None:
            CYB = self.aero._interp(self.aero.CYB_table, alpha, mach)
        elif hasattr(self.aero, 'CYB'):
            CYB = float(self.aero.CYB)
        CYB_rad   = CYB * (180.0 / np.pi)   # [1/deg] → [1/rad]
        FA_y_aero = CYB_rad * beta * q_dyn * self.geom.S_ref

        # ---- Momenty aerodynamiczne --------------------------------------- #
        # Sterowanie NIE wchodzi tutaj — jest doliczane do sum MX/MY/MZ nizej,
        # zeby MA_pitch/MA_yaw/MA_roll dalej znaczyly "czysta aerodynamika"
        # (na tym opieraja sie skrypty analityczne i log sil).
        MA_pitch = aero.MA_yy
        # MA_yaw: restoring moment od slizgu (beta) + tlumienie (Cnr=Cmq),
        # patrz models/aerodynamics.py -- |Cn_beta|=|Cm_alpha| ale ze
        # znakiem przeciwnym (konwencja osi cial X-przod/Y-prawo/Z-dol
        # odwraca rcznosc miedzy plaszczyzna pitch i yaw: dbeta/dt~-r,
        # dalpha/dt~+q), wiec aero.MA_zz juz ma poprawny znak przywracajacy.
        # Wczesniej hardcoded 0.0, co dawalo nietlumiony/nieprzywracany
        # ruch yaw pod wiatrem (spurious "tumbling" post-apogeum). Pierwsza
        # implementacja (commit 01bbfa5) kopiowala znak Cm wprost do Cn,
        # co dawalo NIEstabilna (dodatnio sprzezona) petle beta/r zamiast
        # przywracajacej -- naprawione znakiem minus w aerodynamics.py.
        MA_yaw   = aero.MA_zz

        # ---- Moment toczący od zaklinowania płetw (z CLL_table DATCOM) ---- #
        MA_roll_cant = 0.0
        if hasattr(self.aero, 'CLL_table') and self.aero.CLL_table is not None:
            # CLL z DATCOM — bezpośrednio moment toczący [-]
            CLL = self.aero._interp(self.aero.CLL_table, alpha, mach)
            MA_roll_cant = CLL * q_dyn * self.geom.S_ref * self.geom.d_ref
        elif self.geom.cant_angle_rad != 0.0 and speed > 1.0:
            # Fallback — wzór analityczny gdy brak tabeli DATCOM
            CNA_fins     = getattr(self.geom, 'CNA_fins_ref', 0.2)
            r_fin_mid    = self.geom.d_ref / 2.0 + self.geom.span_ref / 2.0
            Cl_cant      = (self.geom.n_fins * CNA_fins *
                            self.geom.cant_angle_rad *
                            (r_fin_mid / self.geom.d_ref))
            MA_roll_cant = Cl_cant * q_dyn * self.geom.S_ref * self.geom.d_ref

        # ---- Tłumienie toczenia Clp z DATCOM ----------------------------- #
        Clp_rad = 0.0
        if hasattr(self.aero, 'Clp_table') and self.aero.Clp_table is not None:
            # Clp_table jest w [1/rad] — obliczone analitycznie z CNA_fins_rad
            Clp_rad = self.aero._interp(self.aero.Clp_table, alpha, mach)
        elif hasattr(self.aero, 'Clp'):
            # Fallback — stała wartość Clp [1/rad]
            Clp_rad = float(self.aero.Clp)
        MA_roll_damp = 0.0
        if speed > 1.0:
            MA_roll_damp = (Clp_rad * (p * self.geom.d_ref / (2.0 * speed)) *
                            q_dyn * self.geom.S_ref * self.geom.d_ref)

        # ---- Stały moment toczący niezależny od zaklinowania (diagnostyka) #
        # Cl0_manufacturing: opcjonalny atrybut aero (domyślnie 0.0, brak
        # wpływu na dotychczasowe zachowanie) -- do testowania hipotezy, że
        # dominujące realne zrodlo toczenia NIE jest proporcjonalne do
        # cant_angle (np. asymetria silnika/platform, tolerancja produkcyjna
        # wspolna dla calej floty), patrz check_roll_factors_all_flights.py
        # i fit_roll_common_torque.py -- loty z cant=0 pokazuja realny roll
        # porownywalny do lotow z cant!=0, czego czysto cant-driven CLL_table
        # nie moze wyjasnic.
        Cl0_manufacturing = getattr(self.aero, 'Cl0_manufacturing', 0.0)
        MA_roll_const = Cl0_manufacturing * q_dyn * self.geom.S_ref * self.geom.d_ref

        MA_roll = MA_roll_cant + MA_roll_damp + MA_roll_const

        # ---- Efekt Magnusa ----------------------------------------------- #
        FA_y_magnus = 0.0
        if abs(p) > 0.01 and speed > 1.0:
            CY_magnus   = getattr(self.geom, 'CY_magnus', 2.0)
            FA_y_magnus = (CY_magnus * p * self.geom.d_ref / (2.0 * speed) *
                           q_dyn * self.geom.S_ref)

        FA_y = FA_y_aero + FA_y_magnus

        # ---- Ciąg ------------------------------------------------------- #
        # Obsługuje zarówno stały ciąg (PropulsionConfig6DOF)
        # jak i dynamiczny profil (PropulsionConfig6DOFDynamic)
        if hasattr(self.prop, 'thrust_at'):
            thrust = self.prop.thrust_at(t)
        else:
            thrust = self.prop.thrust if ms.is_burning else 0.0
        M_thrust_pitch = thrust * self.prop.offset_z
        M_thrust_yaw   = thrust * self.prop.offset_y

        # ---- Sterowanie (opcjonalne) ------------------------------------- #
        # Celowo PO obliczeniu ciagu: efektor TVC bedzie potrzebowal thrust,
        # wiec FlightState musi go juz zawierac. Przy control=None caly blok
        # jest zerowy i model zachowuje sie identycznie jak wczesniej.
        ctrl_F = _ZERO3
        ctrl_M = _ZERO3
        ctrl_diag = None
        if self.control is not None:
            from control.types import FlightState as _FS
            fs = _FS(
                t=t, alpha=alpha, beta=beta, mach=mach, q_dyn=q_dyn,
                speed=speed, p=p, q=qr, r=r, m=m,
                Ixx=Ixx, Iyy=Iyy, Izz=Izz, xcg=xcg,
                thrust=thrust, on_rail=bool(on_rail), rho=atm.density,
            )
            # UWAGA: nie nazywac tego 'w' — 'w' to skladowa Z predkosci w ukladzie
            # ciala, uzywana nizej w rownaniach translacji.
            _wr = self.control.compute(fs)
            ctrl_F, ctrl_M, ctrl_diag = _wr.F, _wr.M, _wr.diag

        # ---- Grawitacja w body frame ------------------------------------ #
        g_launch = np.array([0.0, 0.0, +g])
        g_body   = DCM @ g_launch
        gx_body, gy_body, gz_body = g_body

        # ---- Sumy sił --------------------------------------------------- #
        FX = FA_x + thrust + m * gx_body + ctrl_F[0]
        FY = FA_y          + m * gy_body + ctrl_F[1]
        FZ = FA_z          + m * gz_body + ctrl_F[2]

        # ---- Równania translacji ---------------------------------------- #
        du_dt = FX / m + r*v  - qr*w

        if on_rail:
            # Na szynie: tylko ruch wzdłuż osi X_body
            # Reakcja szyny znosi FY, FZ — liczymy tylko du_dt
            dv_dt = 0.0
            dw_dt = 0.0
        else:
            dv_dt = FY / m + p*w  - r*u
            dw_dt = FZ / m + qr*u - p*v

        # ---- Sumy momentów ---------------------------------------------- #
        # Sterowanie dokladane do sum (wszystkie TRZY osie — wczesniej istnial
        # tylko martwy placeholder w pitch, a yaw i roll nie mialy go wcale).
        MX = MA_roll  + ctrl_M[0]
        MY = MA_pitch + M_thrust_pitch + ctrl_M[1]
        MZ = MA_yaw   + M_thrust_yaw   + ctrl_M[2]

        # ---- Równania rotacji ------------------------------------------- #
        if on_rail:
            # Na szynie: brak obrotu
            dp_dt  = 0.0
            dqr_dt = 0.0
            dr_dt  = 0.0
        else:
            dp_dt  = MX / Ixx if Ixx > 1e-10 else 0.0
            dqr_dt = (MY + (Izz - Ixx) * p * r)  / Iyy if Iyy > 1e-10 else 0.0
            dr_dt  = (MZ + (Ixx - Iyy) * p * qr) / Izz if Izz > 1e-10 else 0.0

        # ---- Kinematyka pozycji ----------------------------------------- #
        vel_launch = DCM.T @ np.array([u, v, w])
        dx_dt, dy_dt, dz_dt = vel_launch

        # ---- Kinematyka orientacji -------------------------------------- #
        omega = np.array([p, qr, r])
        dq    = quat_derivative(q, omega)
        dq   += -0.5 * (np.dot(q, q) - 1.0) * q
        dq0_dt, dq1_dt, dq2_dt, dq3_dt = dq

        # ---- Droga na szynie -------------------------------------------- #
        # Gdy L_rail=0: zwracamy 13-elementowy wektor (bez rail_dist)
        # Gdy L_rail>0: zwracamy 14-elementowy wektor (z rail_dist)
        # rail_dist: rośnie tylko gdy szyna aktywna i rakieta jedzie do przodu
        d_rail_dist_dt = max(u, 0.0) if not self.launcher.is_disabled() else 0.0

        # ---- Logger sił i momentów ------------------------------------ #
        if self.logger is not None and self.logger.enabled:
            from core.quaternion import quat_to_euler_zyx
            euler = quat_to_euler_zyx(q)
            speed_safe = max(speed, 1e-6)
            self.logger.log({
                "t": t,
                # Stan
                "x": state.x, "y": state.y, "z": state.z,
                "u": u, "v": v, "w": w, "speed": speed,
                "alpha_deg": np.degrees(alpha),
                "beta_deg":  np.degrees(beta),
                "mach": mach, "q_dyn": q_dyn,
                "psi_deg":   np.degrees(euler[0]),
                "theta_deg": np.degrees(euler[1]),
                "phi_deg":   np.degrees(euler[2]),
                # Masa
                "mass": m, "xcg": xcg,
                # Współczynniki aero
                "CN":  aero.CN, "CA": aero.CA, "Cm": aero.Cm,
                "xcp": self.geom.xcp,
                "Cmq_eff": (self.aero._interp(self.aero.Cmq_table, alpha, mach)
                             if (hasattr(self.aero, 'Cmq_table') and
                                 self.aero.Cmq_table is not None)
                             else float(getattr(self.aero, 'Cmq', 0.0))),
                "Clp_eff": float(Clp_rad),
                "CYB_eff": float(CYB_rad),
                # Siły składowe [N]
                "FA_x": FA_x, "FA_z": FA_z,
                "FA_y_aero":   FA_y_aero,
                "FA_y_magnus": FA_y_magnus,
                "F_ctrl":  float(ctrl_F[2]),
                "F_ctrl_y": float(ctrl_F[1]),
                "F_thrust": thrust,
                "Fg_x": m * gx_body,
                "Fg_y": m * gy_body,
                "Fg_z": m * gz_body,
                "FX": FX, "FY": FY, "FZ": FZ,
                # Momenty składowe [N·m]
                "MA_pitch": MA_pitch, "MA_yaw": MA_yaw,
                "MA_roll_cant": MA_roll_cant,
                "MA_roll_damp": MA_roll_damp,
                "MA_roll":  MA_roll,
                "M_ctrl":      float(ctrl_M[1]),
                "M_ctrl_roll": float(ctrl_M[0]),
                "M_ctrl_yaw":  float(ctrl_M[2]),
                "M_thrust_pitch": M_thrust_pitch,
                "M_thrust_yaw":   M_thrust_yaw,
                "MX": MX, "MY": MY, "MZ": MZ,
                **(ctrl_diag or {}),
            })

        # ---- Dual-spin — równania dp_aft_dt i dp_fwd_dt ---------------- #
        dp_fwd_dt = 0.0
        if self._dual_spin is not None:
            ds    = self._dual_spin
            p_fwd = float(x[14]) if len(x) > 14 else 0.0

            # Moment tarcia łożyska [N·m] — przenosi moment między sekcjami
            # M_bearing > 0 gdy p_aft > p_fwd (hamuje aft, napędza fwd)
            M_bearing = ds.bearing_friction * (p - p_fwd)

            # --- Sekcja tylna (aft) ---
            # Momenty: aerodynamiczne (cant + tłumienie) minus tarcie łożyska
            # Tłumienie Clp dla sekcji tylnej — z tabeli DATCOM lub stała
            if hasattr(self.aero, 'Clp_table') and self.aero.Clp_table is not None:
                Clp_aft = self.aero._interp(self.aero.Clp_table, alpha, mach)
            elif abs(ds.aft.Clp) > 1e-10:
                # Clp zdefiniowane w YAML dla sekcji tylnej
                Clp_aft = ds.aft.Clp
            else:
                # Fallback na Clp z modelu aerodynamicznego
                Clp_aft = float(getattr(self.aero, 'Clp', -1.5))
            MA_damp_aft = 0.0
            if speed > 1.0:
                MA_damp_aft = (Clp_aft * (p * self.geom.d_ref / (2.0 * speed))
                               * q_dyn * self.geom.S_ref * self.geom.d_ref)

            M_aft  = MA_roll_cant + MA_damp_aft - M_bearing
            Ixx_aft = ds.aft.Ixx
            dp_dt  = M_aft / Ixx_aft if (Ixx_aft > 1e-10 and not on_rail) else 0.0

            # --- Sekcja przednia (fwd) ---
            # Momenty: canard roll (placeholder=0) + tłumienie + tarcie łożyska
            MA_roll_canard = 0.0   # placeholder — canard CLL gdy dodane

            Clp_fwd = float(getattr(ds.forward, 'Clp', 0.0))
            MA_damp_fwd = 0.0
            if speed > 1.0 and abs(Clp_fwd) > 1e-10:
                MA_damp_fwd = (Clp_fwd * (p_fwd * self.geom.d_ref / (2.0 * speed))
                               * q_dyn * self.geom.S_ref * self.geom.d_ref)

            M_fwd   = MA_roll_canard + MA_damp_fwd + M_bearing
            Ixx_fwd = ds.forward.Ixx
            dp_fwd_dt = M_fwd / Ixx_fwd if Ixx_fwd > 1e-10 else 0.0

        base = np.array([
            dx_dt, dy_dt, dz_dt,
            du_dt, dv_dt, dw_dt,
            dq0_dt, dq1_dt, dq2_dt, dq3_dt,
            dp_dt, dqr_dt, dr_dt,
            d_rail_dist_dt,
        ])
        if self._dual_spin is not None:
            return np.append(base, dp_fwd_dt)
        return base
