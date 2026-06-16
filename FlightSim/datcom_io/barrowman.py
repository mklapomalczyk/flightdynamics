"""
datcom_io/barrowman.py
======================
Metody aerodynamiczne z "Tactical Missile Design" (Fleeman)
rozszerzone o metody Barrowmana dla CNα i xcp.

Obliczane wielkości:
  CN(alpha, Mach)   — współczynnik siły normalnej
  CA(Mach)          — osiowy współczynnik siły (opór)
  xcp(alpha, Mach)  — centrum parcia od nosa [m]

Zakresy:
  CN:  alpha 0-90°, Mach 0-5
  CA:  Mach 0-5
  xcp: wynika z CN_kadlub i CN_stateczniki

Parametry geometryczne:
  Wszystkie w metrach i radianach.
  Konfiguracja wczytywana z RocketConfig (plik YAML).
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from datcom_io.config_reader import RocketConfig, FinSet


# ============================================================================
# Wyniki
# ============================================================================

@dataclass
class AeroResult:
    """Wyniki aerodynamiczne dla jednego (alpha, Mach)."""
    alpha:   float   # kąt natarcia [rad]
    mach:    float
    CN:      float   # współczynnik siły normalnej [-]
    CA:      float   # osiowy współczynnik siły [-]
    xcp:     float   # centrum parcia od nosa [m]
    # Składniki diagnostyczne
    CN_body: float = 0.0
    CN_fins: float = 0.0
    CA_friction_body:  float = 0.0
    CA_friction_fins:  float = 0.0
    CA_wave_body:      float = 0.0
    CA_wave_fins:      float = 0.0
    CA_base:           float = 0.0


# ============================================================================
# Kalkulator
# ============================================================================

class FleemanCalculator:
    """
    Kalkulator aerodynamiczny wg Fleeman "Tactical Missile Design".

    Parameters
    ----------
    config : RocketConfig
        Konfiguracja rakiety wczytana z YAML.
    """

    def __init__(self, config: RocketConfig):
        self.cfg   = config
        self.body  = config.body
        self.fins  = config.fins

        # Geometria globalna
        self.d     = config.body.diameter          # średnica max [m]
        self.r     = self.d / 2.0
        self.l     = config.body.length            # długość całkowita [m]
        self.l_N   = config.body.nose.length       # długość nosa [m]
        self.A_ref = np.pi * self.r**2             # pole referencyjne [m²]
        self.R_N   = getattr(config.body.nose, 'bluntness_radius', 0.0)  # promień zaokrąglenia czubka [m]

        # Średnica końcowa (boattail lub brak)
        bt = config.body.boattail
        self.d_k = bt.diameter if bt is not None else self.d
        self.d_e = getattr(config.body, 'nozzle_diameter', 0.0)  # średnica dyszy [m]

    # ------------------------------------------------------------------ #
    #  Główna metoda
    # ------------------------------------------------------------------ #

    def compute(self, alpha: float, mach: float) -> AeroResult:
        """
        Oblicza CN, CA, xcp dla podanego (alpha, Mach).

        Parameters
        ----------
        alpha : float  Kąt natarcia [rad]
        mach  : float  Liczba Macha [-]
        """
        # ---- CN --------------------------------------------------------- #
        CN_body, xcp_body = self._CN_body(alpha, mach)
        CN_fins, xcp_fins = self._CN_fins(alpha, mach)

        CN_total = CN_body + CN_fins

        if CN_total > 1e-10:
            xcp = (CN_body * xcp_body + CN_fins * xcp_fins) / CN_total
        else:
            xcp = self.l / 2.0

        # ---- CA --------------------------------------------------------- #
        CA_fb, CA_ff, CA_wb, CA_wf, CA_base = self._CA(mach)
        CA_total = CA_fb + CA_ff + CA_wb + CA_wf + CA_base

        return AeroResult(
            alpha   = alpha,
            mach    = mach,
            CN      = CN_total,
            CA      = CA_total,
            xcp     = xcp,
            CN_body = CN_body,
            CN_fins = CN_fins,
            CA_friction_body = CA_fb,
            CA_friction_fins = CA_ff,
            CA_wave_body     = CA_wb,
            CA_wave_fins     = CA_wf,
            CA_base          = CA_base,
        )

    # ------------------------------------------------------------------ #
    #  CN kadłuba — Fleeman
    # ------------------------------------------------------------------ #

    def _CN_body(self, alpha: float, mach: float):
        """
        CN kadłuba wg Fleeman:
        CN_kadlub = sin(2α)cos(α/2) + 2(l-l_N)/d * sin²(α)

        xcp kadłuba wg Fleeman (analityczny wzór):
        xsp = (1 - num/den) * l
        gdzie:
          num = 0.733 + 0.667α*(l_N/d)*((l/l_N)²-1)
          den = (l/l_N)*(1.57 + 1.334α*(l_N/d)*((l/l_N)²-1))
        """
        a   = abs(alpha)   # wzor dla alpha>=0, sign przeniesiony do CN
        l   = self.l
        l_N = self.l_N
        d   = self.d

        CN_sign = np.sign(alpha) if alpha != 0 else 1.0
        CN = CN_sign * (np.sin(2*abs(alpha)) * np.cos(abs(alpha)/2.0) +
              2.0 * (l - l_N) / d * np.sin(abs(alpha))**2)

        # xsp kadluba — wzor Fleemana
        ratio_lN_d  = l_N / d
        ratio_l_lN  = l / l_N
        inner       = ratio_lN_d * (ratio_l_lN**2 - 1.0)

        num = 0.733 + 0.667 * a * inner
        den = ratio_l_lN * (1.57 + 1.334 * a * inner)

        xcp = (1.0 - num / den) * l

        return CN, xcp

    # ------------------------------------------------------------------ #
    #  CN stateczników — Fleeman
    # ------------------------------------------------------------------ #

    def _CN_fins(self, alpha: float, mach: float):
        """
        CN stateczników wg Fleeman.

        Dla M ≤ sqrt(1 + (8/πΛ)²):
            CN_stat = S_stat/S_ref * (πΛ/2 * sin(α)cos(α) + 2sin²(α))

        Dla M > sqrt(1 + (8/πΛ)²):
            CN_stat = S_stat/S_ref * (4sin(α)cos(α)/sqrt(M²-1) + 2sin²(α))

        S_stat = pole jednej pary stateczników.
        Dla N stateczników w układzie symetrycznym:
          - układ +: efektywna 1 para → mnożnik = 1
          - układ X: obie pary pod 45° → mnożnik = √2
        """
        if not self.fins:
            return 0.0, self.l - 0.25 * self.fins[0].root_chord if self.fins else self.l

        CN_fins_total = 0.0
        xcp_fins_num  = 0.0

        for fin in self.fins:
            CN_f, xcp_f = self._CN_single_fin(fin, alpha, mach)
            CN_fins_total += CN_f
            xcp_fins_num  += CN_f * xcp_f

        xcp_fins = (xcp_fins_num / CN_fins_total
                    if CN_fins_total > 1e-10
                    else self.l - self.fins[0].root_chord / 4.0)

        return CN_fins_total, xcp_fins

    def _CN_single_fin(self, fin: FinSet, alpha: float, mach: float):
        """CN jednego zestawu płetw."""
        Cr    = fin.root_chord
        Ct    = fin.tip_chord
        s     = fin.span          # rozpiętość od korpusu [m]
        N     = fin.count

        # Pole jednej płetwy [m²]
        S_one = 0.5 * (Cr + Ct) * s

        # Dla N płetw w układzie symetrycznym:
        # Efektywna liczba par w płaszczyźnie pitch
        # układ +: 1 para efektywna, układ X: √2 pary efektywne
        # Przybliżenie: zawsze bierzemy N/2 par
        n_pairs = N / 2.0
        S_stat  = S_one * 2.0   # pole jednej pary (2 płetwy)

        # Wydłużenie: Λ = b²/SCA gdzie b = rozpiętość całkowita pary
        # SCA = średnia cięciwa aerodynamiczna (MAC)
        b_pair = 2.0 * (self.r + s)   # od końca do końca przez oś
        SCA    = (2.0/3.0) * (Cr + Ct - Cr*Ct/(Cr+Ct))  # MAC [m]
        Lambda = b_pair**2 / SCA

        # Próg Macha
        M_crit = np.sqrt(1.0 + (8.0 / (np.pi * Lambda))**2)

        if mach <= M_crit:
            # Poddźwiękowy / umiarkowanie naddźwiękowy
            CN_pair = (S_stat / self.A_ref) * (
                (np.pi * Lambda / 2.0) * np.sin(alpha) * np.cos(alpha) +
                2.0 * np.sin(alpha)**2
            )
        else:
            # Naddźwiękowy
            beta_s = np.sqrt(max(mach**2 - 1.0, 0.01))
            CN_pair = (S_stat / self.A_ref) * (
                (4.0 * np.sin(alpha) * np.cos(alpha)) / beta_s +
                2.0 * np.sin(alpha)**2
            )

        # n_pairs par stateczników
        CN_total = CN_pair * n_pairs

        # xcp statecznika wzgledem krawedzi natarcia nasady (Fleeman)
        # M <= 0.7: xsp = 0.25 * SCA
        # M >= 2.0: xsp = SCA * (A*sqrt(M^2-1) - 0.67) / (2*A*sqrt(M^2-1) - 1)
        # 0.7 < M < 2.0: interpolacja liniowa
        SCA   = (2.0/3.0) * (Cr + Ct - Cr*Ct/(Cr+Ct)) if (Cr+Ct) > 0 else Cr  # MAC
        b_pair = 2.0 * (self.r + s)
        Lambda = b_pair**2 / SCA   # wydluzenie

        if mach <= 0.7:
            xsp_local = 0.25 * SCA
        elif mach >= 2.0:
            sq = np.sqrt(max(mach**2 - 1.0, 0.01))
            xsp_local = SCA * (Lambda*sq - 0.67) / (2.0*Lambda*sq - 1.0)
        else:
            # Interpolacja liniowa miedzy M=0.7 a M=2.0
            sq2 = np.sqrt(max(4.0 - 1.0, 0.01))   # sqrt(2^2 - 1)
            xsp_M2  = SCA * (Lambda*sq2 - 0.67) / (2.0*Lambda*sq2 - 1.0)
            xsp_M07 = 0.25 * SCA
            A_lin = (xsp_M2 - xsp_M07) / 1.3
            B_lin = xsp_M07 - 0.7 * A_lin
            xsp_local = A_lin * mach + B_lin

        # Pozycja w ukladzie rakiety:
        # x_CGstat = pozycja krawedzi natarcia nasady + 0.5*SCA (srodek cieciwy)
        x_CG_stat = fin.position + 0.5 * SCA
        # xsp_stat w ukladzie rakiety (od nosa):
        x_ac = x_CG_stat + 0.5 * SCA - xsp_local

        return CN_total, x_ac

    # ------------------------------------------------------------------ #
    #  CA — składniki oporu
    # ------------------------------------------------------------------ #

    def _CA(self, mach: float):
        """
        Zwraca (CA_friction_body, CA_friction_fins,
                CA_wave_body, CA_wave_fins, CA_base).
        """
        CA_fb   = self._CA_friction_body(mach)
        CA_ff   = self._CA_friction_fins(mach)
        CA_wb   = self._CA_wave_body(mach)
        CA_wf   = self._CA_wave_fins(mach)
        CA_base = self._CA_base(mach)
        return CA_fb, CA_ff, CA_wb, CA_wf, CA_base

    def _CA_friction_body(self, mach: float, altitude_m: float = 1000.0) -> float:
        """
        Opór tarcia kadłuba — Fleeman (Jerger reference, turbulent BL):
        CD_f = 0.053 * (l/d) * (M/q)^0.2
        gdzie q w psf (lb/ft²), l w ft.

        Przeliczamy q z SI na psf i l z m na ft.
        Dla M < 0.2: używamy wartości z M=0.2.
        """
        M2FT  = 1.0 / 0.3048
        PA2PSF = 1.0 / 47.880   # Pa → psf

        # ISA przybliżone na danej wysokości
        T0, rho0, g, R, L = 288.15, 1.225, 9.81, 287.0, 0.0065
        T   = T0 - L * altitude_m
        rho = rho0 * (T / T0) ** (g / (R * L) - 1)
        a   = np.sqrt(1.4 * R * T)

        m_eff = max(mach, 0.2)
        V     = m_eff * a
        q_Pa  = 0.5 * rho * V**2
        q_psf = q_Pa * PA2PSF

        l_ft = self.l * M2FT

        # Fleeman: CD_f = 0.053 * (l/d) * (M/(q*l))^0.2
        # q w psf, l w ft
        return 0.053 * (self.l / self.d) * (m_eff / (q_psf * l_ft))**0.2

    def _CA_friction_fins(self, mach: float, altitude_m: float = 1000.0) -> float:
        """
        Opór tarcia stateczników — Fleeman:
        CD_f = 0.0227 * (M/q)^0.2 * (2*S_stat/S_ref)
        gdzie q w psf.
        """
        if not self.fins:
            return 0.0

        M2FT   = 1.0 / 0.3048
        PA2PSF = 1.0 / 47.880

        T0, rho0, g, R, L = 288.15, 1.225, 9.81, 287.0, 0.0065
        T   = T0 - L * altitude_m
        rho = rho0 * (T / T0) ** (g / (R * L) - 1)
        a   = np.sqrt(1.4 * R * T)

        m_eff = max(mach, 0.2)
        V     = m_eff * a
        q_psf = 0.5 * rho * V**2 * PA2PSF

        CA_f = 0.0
        for fin in self.fins:
            S_one   = 0.5 * (fin.root_chord + fin.tip_chord) * fin.span
            S_stat  = S_one * 2.0    # jedna para (2 płetwy)
            n_pairs = fin.count / 2.0

            # Fleeman eq 2.42: CD_f = n * 0.0133 * (M/(q*C_mac))^0.2 * (2S/S_ref)
            # q w psf, C_mac w ft
            M2FT  = 1.0 / 0.3048
            C_mac = (2.0/3.0)*(fin.root_chord+fin.tip_chord - fin.root_chord*fin.tip_chord/(fin.root_chord+fin.tip_chord)) * M2FT  # MAC w ft
            CA_f += n_pairs * 0.0133 * (m_eff / (q_psf * C_mac))**0.2 * (2.0 * S_stat / self.A_ref)

        return CA_f

    def _CA_wave_body(self, mach: float) -> float:
        """
        Falowy opór kadłuba — Fleeman:
        M < 0.8:  CD_w = 0
        0.8≤M<1:  CD_w = 3.42 * atan(d/2l_N)^1.69 * (1 - 4R_N²/d²
                          + 0.665*(4R_N²/d²)) * (25M²-40M+16)
        M > 1:    CD_w = (1.586 + 1.834/M²) * atan(d/2l_N)^1.69
                          * (1 - 4R_N²/d² + 0.665*(4R_N²/d²))
        """
        d    = self.d
        l_N  = self.l_N
        R_N  = self.R_N

        # Człon geometryczny wspólny
        angle_term = np.arctan(d / (2.0 * l_N))**1.69
        blunt_term = (1.0 - 4.0*R_N**2/d**2 + 0.665*(4.0*R_N**2/d**2))

        if mach < 0.8:
            return 0.0
        elif mach < 1.0:
            return 3.42 * angle_term * blunt_term * (25*mach**2 - 40*mach + 16)
        else:
            return (1.586 + 1.834/mach**2) * angle_term * blunt_term

    def _CA_wave_fins(self, mach: float) -> float:
        """
        Falowy opór stateczników — Fleeman:
        M < 0.8:  CD_w = 0
        0.8≤M<1:  CD_w = 1.276 * sin²(ψ)*cos(η)*δ*b * (1/S_ref)*(25M_η²-40M_η+16)
        M > 1:    CD_w = (1.429/M_η²)*((16.829*M_η^7/(2.8*M_η²-0.4)^2.5)-1)
                          * sin(ψ)*cos(η)*δ*b*(1/S_ref)
        """
        if not self.fins:
            return 0.0

        CA_wf = 0.0
        for fin in self.fins:
            Cr    = fin.root_chord
            Ct    = fin.tip_chord
            s     = fin.span
            t     = fin.thickness
            eta   = np.deg2rad(fin.sweep_le)     # kąt skosu krawędzi natarcia [rad]
            N     = fin.count

            # ψ — kąt półstożkowy profilu (half-angle)
            # dla profilu hex: ψ = arctan(t/2 / (Cr/2)) = arctan(t/Cr)
            psi   = np.arctan(t / Cr) if Cr > 0 else 0.0

            # b — rozpiętość od osi do końca (jedna strona)
            b_half = self.r + s

            # δ — grubość statecznika [m]
            delta = t

            # S_ref — pole referencyjne (jedno skrzydło/płetwa)
            S_one  = 0.5 * (Cr + Ct) * s
            n_pairs = N / 2.0

            if mach < 0.8:
                cd_w = 0.0
            elif mach < 1.0:
                cd_w = (1.276 * np.sin(psi)**2 * np.cos(eta) * delta * b_half
                        * (1.0 / self.A_ref)
                        * (25*mach**2 - 40*mach + 16))
            else:
                M_eta = mach   # M_η = Mach (uproszczenie dla eta=0)
                factor = ((16.829 * M_eta**7) /
                          (2.8 * M_eta**2 - 0.4)**2.5 - 1.0)
                cd_w = ((1.429 / M_eta**2) * factor
                        * np.sin(psi) * np.cos(eta) * delta * b_half
                        * (1.0 / self.A_ref))

            CA_wf += n_pairs * 2.0 * cd_w   # obie strony pary

        return max(CA_wf, 0.0)

    def _CA_base(self, mach: float) -> float:
        """
        Opór denny (base drag) — Fleeman:
        ξ_s = 1 - d_e²/(ξ_k * d²)
        ξ_k = d_k²/d²
        M < 1: CD_p = (0.12 + 0.13*M²) * ξ_s * ξ_k
        M > 1: CD_p = (0.25/M) * ξ_s * ξ_k
        """
        d    = self.d
        d_k  = self.d_k
        d_e  = self.d_e

        xi_k = (d_k / d)**2
        # Zabezpieczenie przed dzieleniem przez zero
        if xi_k > 1e-10:
            xi_s = 1.0 - (d_e / d)**2 / xi_k
        else:
            xi_s = 1.0

        xi_s = max(0.0, min(1.0, xi_s))   # ogranicz do [0,1]

        if mach < 1.0:
            return (0.12 + 0.13 * mach**2) * xi_s * xi_k
        else:
            return (0.25 / mach) * xi_s * xi_k


# ============================================================================
# Wynik dla tabeli (stałe alpha, wiele Mach)
# ============================================================================

@dataclass
class BarrowmanResult:
    """Alias dla kompatybilności z barrowman_aero.py."""
    mach:   float
    CNa:    float   # CNα = CN/alpha przy małym alpha [1/rad]
    xcp:    float   # centrum parcia [m]
    CA:     float
    CNa_nose:  float = 0.0
    CNa_body:  float = 0.0
    CNa_fins:  float = 0.0
    xcp_nose:  float = 0.0
    xcp_fins:  float = 0.0


class BarrowmanCalculator:
    """
    Wrapper dla FleemanCalculator z interfejsem zgodnym z barrowman_aero.py.
    Oblicza CNα przez różniczkowanie CN po alpha przy małym alpha.
    """

    def __init__(self, config: RocketConfig):
        self.fleeman = FleemanCalculator(config)
        self.cfg     = config

    def compute(self, mach: float) -> BarrowmanResult:
        """Oblicza CNα, xcp, CA dla podanej liczby Macha."""
        # CNα przez różniczkę centralną przy alpha=5°
        da    = np.deg2rad(5.0)
        r_pos = self.fleeman.compute(+da, mach)
        r_neg = self.fleeman.compute(-da, mach)
        CNa   = (r_pos.CN - r_neg.CN) / (2.0 * da)

        # xcp przy małym dodatnim alpha
        r_ref = self.fleeman.compute(da, mach)

        return BarrowmanResult(
            mach      = mach,
            CNa       = CNa,
            xcp       = r_ref.xcp,
            CA        = r_ref.CA,
            CNa_body  = r_pos.CN_body / da,
            CNa_fins  = r_pos.CN_fins / da,
        )
