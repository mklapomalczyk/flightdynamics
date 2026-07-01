"""
models/aerodynamics.py
======================
Model aerodynamiczny.

Architektura oparta na interfejsie (Protocol), co pozwala na wymianę
modelu bez zmiany kodu solvera:

  AeroModel (Protocol)
    ├── ConstantAero     — stałe współczynniki (do prac koncepcyjnych)
    ├── TableAero        — tabele Cx, CN, Cm = f(alpha, Mach) z pliku/pamięci
    └── DatcomAero       — wrapper na wyniki DATCOM (io/datcom_reader.py)

Siły aerodynamiczne obliczane są w układzie aerodynamicznym (wind frame),
a następnie transformowane do body frame przez core/frames.py.

Konwencja współczynników (zgodna z DATCOM / NATO STANAG):
  CA  - osiowy współczynnik siły (Axial force), wzdłuż osi X body, opór w osi
  CN  - normalny współczynnik siły (Normal force), wzdłuż osi Z body
  Cm  - współczynnik momentu pochylającego (Pitching moment), wokół xcg
  Cmq - pochylna pochodna tłumienia (pitch damping derivative)

Siły obliczane z:
  F = q_dyn * S_ref * C
  M = q_dyn * S_ref * d_ref * Cm
gdzie:
  q_dyn = 0.5 * rho * V²   — ciśnienie dynamiczne [Pa]
  S_ref                     — pole powierzchni referencyjnej [m²] (przekrój poprzeczny)
  d_ref                     — średnica referencyjna [m]

Calkowity kat natarcia (alpha_total):
  Dla bryly osiowosymetrycznej opor wzdluz osi (FA_x, body X) zalezy od
  KATA MIEDZY osia a wektorem predkosci w 3D — nie tylko od jego
  skladowej w plaszczyznie pitch. Czyste slizganie (beta!=0, alpha=0)
  MUSI wiec generowac opor tak samo jak rownowazny alpha w plaszczyznie
  pitch. compute() przyjmuje opcjonalny `alpha_total` (domyslnie None ->
  uzyte zostaje `alpha`, zachowanie jak dawniej); gdy podane, CA i CN
  uzyte do FA_x (oporu osiowego) sa interpolowane przy `alpha_total`
  (zarowno w members cos/sin jak i w lookupie tabeli) — FA_z/Cm/moment
  pochylajacy ZOSTAJA przy plaszczyznie pitch (`alpha`), zgodnie z
  istniejaca/zwalidowana dynamika pitch. Sila boczna od slizgania
  (CYB*beta, w force_model6.py) jest osobnym, liniowym mechanizmem i nie
  jest tu dotykana.
  Wolajacy (forces/force_model6.py) liczy:
      alpha_total = arccos(clip(u_air/speed, -1, 1))
"""

import numpy as np
from dataclasses import dataclass
from typing import Protocol, Callable, Optional


# ============================================================================
# Struktury danych
# ============================================================================

@dataclass
class AeroForces:
    """
    Siły i momenty aerodynamiczne w body frame.

    Uwaga: siły w body frame, nie aerodynamicznym.
    FA_x, FA_z to składowe siły aerodynamicznej wzdłuż osi X i Z body.
    """
    FA_x:   float   # siła aerodynamiczna wzdłuż osi X body [N] (ujemna = opór)
    FA_z:   float   # siła aerodynamiczna wzdłuż osi Z body [N]
    MA_yy:  float   # moment aerodynamiczny wokół osi Y (pitch) [N·m]
    MA_zz:  float = 0.0   # moment aerodynamiczny wokół osi Z (yaw) [N·m]

    # Wielkości pomocnicze (do logowania / debugowania)
    q_dyn:  float = 0.0   # ciśnienie dynamiczne [Pa]
    CA:     float = 0.0   # osiowy współczynnik siły [-]
    CN:     float = 0.0   # normalny współczynnik siły [-]
    Cm:     float = 0.0   # współczynnik momentu pochylającego [-]
    Cn:     float = 0.0   # współczynnik momentu odchylającego (yaw) [-]
    alpha:  float = 0.0   # kąt natarcia [rad]
    mach:   float = 0.0   # liczba Macha [-]


# ============================================================================
# Interfejs
# ============================================================================

class AeroModel(Protocol):
    """
    Interfejs modelu aerodynamicznego.
    Każdy model musi implementować metodę `compute`.
    """

    def compute(
        self,
        alpha:       float,      # kąt natarcia [rad]
        mach:        float,      # liczba Macha [-]
        q_dyn:       float,      # ciśnienie dynamiczne [Pa]
        q_rate:      float,      # prędkość kątowa pitch [rad/s] (do Cmq)
        speed:       float,      # prędkość [m/s] (do normalizacji Cmq)
        xcg:         float,      # pozycja xcg od nosa [m]
        xcp:         float,      # pozycja centrum parcia od nosa [m]
        S_ref:       float,      # pole referencyjne [m²]
        d_ref:       float,      # średnica referencyjna [m]
        alpha_total: Optional[float] = None,  # calkowity kat natarcia [rad]
                                               # (alpha+beta, do CA — patrz docstring modulu)
        beta:        float = 0.0,  # kąt ślizgu [rad] (do momentu odchylającego, symetria z alpha)
        r_rate:      float = 0.0,  # prędkość kątowa yaw [rad/s] (do Cnr, symetria z Cmq)
    ) -> AeroForces: ...


# ============================================================================
# Model stałych współczynników
# ============================================================================

class ConstantAero:
    """
    Model aerodynamiczny ze stałymi współczynnikami.
    Używany do prac koncepcyjnych i walidacji kodu solvera.

    Zakłada liniową zależność CN od alpha (małe kąty):
      CN = CN_alpha * alpha
    oraz stały CA (brak wpływu Macha i alpha na opór osi).

    Parameters
    ----------
    CA : float
        Osiowy współczynnik siły [-]. Typowo 0.1–0.5 dla rakiet.
    CN_alpha : float
        Pochodna normalnego wsp. siły po kącie natarcia [1/rad].
        Typowo 5–15 dla rakiet z płetwami (zależnie od konfiguracji).
    Cm_alpha : float
        Pochodna wsp. momentu po kącie natarcia [1/rad].
        Nie używana jeśli Cm liczymy z xcp-xcg (zalecane).
    Cmq : float
        Pochodna tłumienia kątowego [1/rad]. Typowo ujemna (tłumi), ~-10 do -100.
    use_xcp_moment : bool
        Jeśli True (domyślnie), moment liczy się z przesunięcia xcp-xcg.
        Jeśli False, używa Cm_alpha * alpha.
    """

    def __init__(
        self,
        CA:              float = 0.3,
        CN_alpha:        float = 10.0,
        Cm_alpha:        float = 0.0,
        Cmq:             float = -20.0,
        use_xcp_moment:  bool  = True,
    ):
        self.CA             = float(CA)
        self.CN_alpha       = float(CN_alpha)
        self.Cm_alpha       = float(Cm_alpha)
        self.Cmq            = float(Cmq)
        self.Cmq_table      = None   # brak tabeli dla ConstantAero
        self.use_xcp_moment = use_xcp_moment

    def compute(
        self,
        alpha:       float,
        mach:        float,
        q_dyn:       float,
        q_rate:      float,
        speed:       float,
        xcg:         float,
        xcp:         float,
        S_ref:       float,
        d_ref:       float,
        alpha_total: Optional[float] = None,
        beta:        float = 0.0,
        r_rate:      float = 0.0,
        powered:     bool = False,
    ) -> AeroForces:

        # Współczynniki
        # CA stale (brak tabeli) -> alpha_total nie ma tu wplywu, ani
        # rozroznienia powered/coast (brak CA-BASE); parametry przyjete
        # dla zgodnosci sygnatury z TableAero.
        CA = self.CA
        CN = self.CN_alpha * alpha

        # Moment pochylający
        if self.use_xcp_moment:
            # Moment = siła normalna × ramię (xcg - xcp)
            # Konwencja: xcg, xcp mierzone od nosa, xcg > xcp → rakieta niestabilna
            # xcg < xcp → stabilna (cp za cg)
            arm = (xcg - xcp)   # [m], ujemne = stabilne
            Cm = CN * arm / d_ref
        else:
            Cm = self.Cm_alpha * alpha

        # Moment odchylający (yaw) — bryla osiowosymetryczna, wiec |Cn_beta|
        # = |Cm_alpha|, ALE ze znakiem przeciwnym (Cn_beta = -Cm_alpha) wzgledem
        # statycznego/przywracajacego czlonu — konwencja osi cial (X w przod,
        # Y w prawo, Z w dol) odwraca rcznosc przy przejsciu z plaszczyzny
        # pitch (X-Z) do plaszczyzny yaw (X-Y): dbeta/dt ~ -r, podczas gdy
        # dalpha/dt ~ +q. Bez tego odwrocenia znaku petla beta/r jest
        # NIEstabilna (dodatnie sprzezenie zwrotne) zamiast oscylatora
        # przywracajacego — patrz docstring modulu, sekcja o MA_yaw w
        # force_model6.py. Tlumienie (Cnr=Cmq, ponizej) NIE zmienia znaku —
        # to standardowa relacja dla bryl osiowosymetrycznych.
        CN_beta = self.CN_alpha * beta
        if self.use_xcp_moment:
            Cn = -(CN_beta * arm / d_ref)
        else:
            Cn = -(self.Cm_alpha * beta)

        # Tłumienie kątowe (pitch/yaw damping) — Cmq=Cnr dla bryly
        # osiowosymetrycznej (brak osobnej tabeli Cnr, patrz docstring).
        # Normalizacja: Cmq * (q * d_ref / (2 * V))
        if speed > 1.0:
            # Użyj tabeli Cmq jeśli dostępna, inaczej stałej wartości
            if self.Cmq_table is not None:
                cmq = self._interp(self.Cmq_table, alpha, mach)
                cnr = self._interp(self.Cmq_table, beta, mach)
            else:
                cmq = self.Cmq
                cnr = self.Cmq
            Cm_damping = cmq * (q_rate * d_ref / (2.0 * speed))
            Cn_damping = cnr * (r_rate * d_ref / (2.0 * speed))
        else:
            Cm_damping = 0.0
            Cn_damping = 0.0

        Cm_total = Cm + Cm_damping
        Cn_total = Cn + Cn_damping

        # Siły w body frame
        # Konwencja body: X wzdłuż osi, Z prostopadle (w górę)
        # CA działa przeciw prędkości (opór osi) → ujemne FA_x
        # CN działa prostopadle → FA_z
        # Obrót z aero do body dla małych kątów:
        #   FA_x ≈ -CA*cos(alpha) - CN*sin(alpha) ≈ -CA - CN*alpha
        #   FA_z ≈  CN*cos(alpha) - CA*sin(alpha) ≈  CN - CA*alpha
        # Dla dokładności używamy pełnych wyrażeń:
        # FA_x (opor osiowy) uzywa calkowitego kata natarcia — patrz
        # docstring modulu. FA_z zostaje w plaszczyznie pitch (alpha).
        alpha_total_eff = alpha if alpha_total is None else alpha_total
        ca, sa = np.cos(alpha), np.sin(alpha)
        ca_t, sa_t = np.cos(alpha_total_eff), np.sin(alpha_total_eff)
        CN_x = self.CN_alpha * alpha_total_eff
        FA_x = q_dyn * S_ref * (-CA * ca_t - CN_x * sa_t)
        FA_z = q_dyn * S_ref * ( CN * ca - CA * sa)

        MA_yy = q_dyn * S_ref * d_ref * Cm_total
        MA_zz = q_dyn * S_ref * d_ref * Cn_total

        return AeroForces(
            FA_x  = FA_x,
            FA_z  = FA_z,
            MA_yy = MA_yy,
            MA_zz = MA_zz,
            q_dyn = q_dyn,
            CA    = CA,
            CN    = CN,
            Cm    = Cm_total,
            Cn    = Cn_total,
            alpha = alpha,
            mach  = mach,
        )

    def __repr__(self) -> str:
        return (
            f"ConstantAero(CA={self.CA}, CN_alpha={self.CN_alpha}, "
            f"Cmq={self.Cmq})"
        )


# ============================================================================
# Model tabelaryczny
# ============================================================================

class TableAero:
    """
    Model aerodynamiczny z tabelami współczynników CA, CN, Cm
    jako funkcji alpha i Mach.

    Interpolacja biliniowa. Gotowy na dane z DATCOM lub CFD.

    Parameters
    ----------
    alpha_table : array (N,)
        Kąty natarcia w tabeli [rad].
    mach_table : array (M,)
        Liczby Macha w tabeli [-].
    CA_table : array (N, M)
        Tabela CA[i_alpha, i_mach].
    CN_table : array (N, M)
        Tabela CN[i_alpha, i_mach].
    Cm_table : array (N, M)  lub None
        Tabela Cm[i_alpha, i_mach]. Jeśli None → obliczany z xcp-xcg.
    xcp_table : array (N, M) lub None
        Tabela xcp od nosa [m]. Jeśli None i Cm_table=None → błąd.
    Cmq : float
        Stały współczynnik tłumienia (na razie uproszczenie).
    """

    def __init__(
        self,
        alpha_table: np.ndarray,
        mach_table:  np.ndarray,
        CA_table:    np.ndarray,
        CN_table:    np.ndarray,
        Cm_table:    Optional[np.ndarray] = None,
        xcp_table:   Optional[np.ndarray] = None,
        Cmq:         float = -20.0,
        Cmq_table:   Optional[np.ndarray] = None,
        Clp_table:   Optional[np.ndarray] = None,
        CYB_table:   Optional[np.ndarray] = None,
        CLL_table:   Optional[np.ndarray] = None,
        xcg_ref:     float = 0.0,
        CA_base_table: Optional[np.ndarray] = None,
    ):
        self.alpha_table = np.asarray(alpha_table, dtype=float)
        self.mach_table  = np.asarray(mach_table,  dtype=float)
        self.CA_table    = np.asarray(CA_table,    dtype=float)
        # Opor denny (skladowa CA) — do modelu z napędem: podczas spalania
        # plomien silnika wypelnia den i opor denny ~0; podczas lotu
        # balistycznego (coast) obowiazuje pelny opor denny. None -> brak
        # rozroznienia (zachowanie jak dawniej: pelny CA zawsze).
        self.CA_base_table = (np.asarray(CA_base_table, dtype=float)
                              if CA_base_table is not None else None)
        self.CN_table    = np.asarray(CN_table,    dtype=float)
        self.Cm_table    = np.asarray(Cm_table,    dtype=float) if Cm_table  is not None else None
        self.xcp_table   = np.asarray(xcp_table,   dtype=float) if xcp_table is not None else None
        self.Cmq_table   = np.asarray(Cmq_table,   dtype=float) if Cmq_table is not None else None
        self.Clp_table   = np.asarray(Clp_table,   dtype=float) if Clp_table is not None else None
        self.CYB_table   = np.asarray(CYB_table,   dtype=float) if CYB_table is not None else None
        self.CLL_table   = np.asarray(CLL_table, dtype=float) if CLL_table is not None else None
        self.Cmq         = float(Cmq)   # fallback gdy brak tabeli
        self.xcg_ref     = float(xcg_ref)  # xcg uzyte w DATCOM [m od nosa]

        if self.Cm_table is None and self.xcp_table is None:
            raise ValueError(
                "TableAero wymaga albo Cm_table albo xcp_table do obliczenia momentu."
            )

    def _interp(self, table: np.ndarray, alpha: float, mach: float) -> float:
        """Interpolacja biliniowa w tabeli (alpha, mach)."""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.alpha_table, self.mach_table),
            table,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        result = interp([[alpha, mach]])
        return float(result.ravel()[0])

    def compute(
        self,
        alpha:       float,
        mach:        float,
        q_dyn:       float,
        q_rate:      float,
        speed:       float,
        xcg:         float,
        xcp:         float,
        S_ref:       float,
        d_ref:       float,
        alpha_total: Optional[float] = None,
        beta:        float = 0.0,
        r_rate:      float = 0.0,
        powered:     bool = False,
    ) -> AeroForces:

        # Ogranicz alpha do zakresu tabeli — poza nim DATCOM nie ma sensu
        # Przy tumblingu alpha moze przekroczyc 90 stopni co daje blow-up
        alpha_max = float(self.alpha_table[-1])
        alpha_min = float(self.alpha_table[0])
        alpha_clip = float(np.clip(alpha, alpha_min, alpha_max))
        beta_clip  = float(np.clip(beta,  alpha_min, alpha_max))

        CA = self._interp(self.CA_table, alpha_clip, mach)
        CN = self._interp(self.CN_table, alpha_clip, mach)
        CN_beta = self._interp(self.CN_table, beta_clip, mach)

        # Opor osiowy (FA_x) dla bryly osiowosymetrycznej zalezy od calkowitego
        # kata natarcia (alpha+beta), nie tylko od plaszczyzny pitch — patrz
        # docstring modulu. FA_z/Cm zostaja przy CA/CN w plaszczyznie pitch
        # (powyzej) — zwalidowana dynamika pitch sie nie zmienia.
        alpha_total_clip = (alpha_clip if alpha_total is None
                             else float(np.clip(alpha_total, alpha_min, alpha_max)))
        CA_x = self._interp(self.CA_table, alpha_total_clip, mach)
        CN_x = self._interp(self.CN_table, alpha_total_clip, mach)

        # Korekcja oporu dennego podczas pracy silnika: plomien wylotowy
        # wypelnia i doszczelnia den, wiec opor denny (CA-BASE z DATCOM,
        # liczony dla lotu balistycznego / zamknietego dna) praktycznie
        # znika. Odejmujemy go od CA (obie osie) tylko gdy powered=True.
        # Coast (powered=False) -> pelny opor denny, jak dawniej.
        if powered and self.CA_base_table is not None:
            ca_base   = self._interp(self.CA_base_table, alpha_clip,       mach)
            ca_base_x = self._interp(self.CA_base_table, alpha_total_clip, mach)
            CA   = max(CA   - ca_base,   0.0)
            CA_x = max(CA_x - ca_base_x, 0.0)

        if self.Cm_table is not None:
            Cm_datcom = self._interp(self.Cm_table, alpha_clip, mach)
            if (self.xcp_table is not None
                    and abs(self.xcg_ref) > 1e-6
                    and abs(xcg - self.xcg_ref) > 1e-6):
                # Korekcja na zmiane xcg przez skalowanie wzgledem xcp:
                # Cm_corr = Cm_datcom * (xcg - xcp) / (xcg_ref - xcp)
                # Zachowuje Cm_datcom przy xcg=xcg_ref
                # Daje Cm=0 przy xcg=xcp (granica stabilnosci)
                xcp_interp = self._interp(self.xcp_table, alpha_clip, mach)
                denom = self.xcg_ref - xcp_interp
                if abs(denom) > 1e-4:
                    Cm = Cm_datcom * (xcg - xcp_interp) / denom
                else:
                    Cm = Cm_datcom
            else:
                Cm = Cm_datcom
        elif self.xcp_table is not None:
            xcp_interp = self._interp(self.xcp_table, alpha_clip, mach)
            Cm = CN * (xcg - xcp_interp) / d_ref
        else:
            Cm = 0.0

        # Moment odchylajacy (yaw) — bryla osiowosymetryczna: |Cn_beta| =
        # |Cm_alpha|, ALE ze znakiem przeciwnym (Cn_beta = -Cm_alpha) dla
        # czlonu statycznego/przywracajacego — konwencja osi cial (X w przod,
        # Y w prawo, Z w dol) odwraca rcznosc przy przejsciu z plaszczyzny
        # pitch (X-Z) do plaszczyzny yaw (X-Y): dbeta/dt ~ -r, podczas gdy
        # dalpha/dt ~ +q. Bez tego odwrocenia znaku petla beta/r jest
        # NIEstabilna (dodatnie sprzezenie zwrotne) zamiast oscylatora
        # przywracajacego — patrz docstring modulu, sekcja o MA_yaw w
        # force_model6.py. Tlumienie (Cnr=Cmq, ponizej) NIE zmienia znaku —
        # to standardowa relacja dla bryl osiowosymetrycznych.
        if self.Cm_table is not None:
            Cn_datcom = self._interp(self.Cm_table, beta_clip, mach)
            if (self.xcp_table is not None
                    and abs(self.xcg_ref) > 1e-6
                    and abs(xcg - self.xcg_ref) > 1e-6):
                xcp_interp_b = self._interp(self.xcp_table, beta_clip, mach)
                denom_b = self.xcg_ref - xcp_interp_b
                if abs(denom_b) > 1e-4:
                    Cn = Cn_datcom * (xcg - xcp_interp_b) / denom_b
                else:
                    Cn = Cn_datcom
            else:
                Cn = Cn_datcom
        elif self.xcp_table is not None:
            xcp_interp_b = self._interp(self.xcp_table, beta_clip, mach)
            Cn = CN_beta * (xcg - xcp_interp_b) / d_ref
        else:
            Cn = 0.0
        Cn = -Cn

        if speed > 1.0:
            # Użyj tabeli Cmq jeśli dostępna, inaczej stałej wartości.
            # Cnr = Cmq (ta sama tabela przy beta) — bryla osiowosymetryczna,
            # brak osobnej tabeli Cnr z DATCOM.
            if self.Cmq_table is not None:
                cmq = self._interp(self.Cmq_table, alpha_clip, mach)
                cnr = self._interp(self.Cmq_table, beta_clip, mach)
            else:
                cmq = self.Cmq
                cnr = self.Cmq
            Cm_damping = cmq * (q_rate * d_ref / (2.0 * speed))
            Cn_damping = cnr * (r_rate * d_ref / (2.0 * speed))
        else:
            Cm_damping = 0.0
            Cn_damping = 0.0

        Cm_total = Cm + Cm_damping
        Cn_total = Cn + Cn_damping

        ca, sa = np.cos(alpha), np.sin(alpha)
        ca_t, sa_t = np.cos(alpha_total_clip), np.sin(alpha_total_clip)
        FA_x = q_dyn * S_ref * (-CA_x * ca_t - CN_x * sa_t)
        FA_z = q_dyn * S_ref * ( CN * ca - CA * sa)
        MA_yy = q_dyn * S_ref * d_ref * Cm_total
        MA_zz = q_dyn * S_ref * d_ref * Cn_total

        return AeroForces(
            FA_x  = FA_x,
            FA_z  = FA_z,
            MA_yy = MA_yy,
            MA_zz = MA_zz,
            q_dyn = q_dyn,
            CA    = CA,
            CN    = CN,
            Cm    = Cm_total,
            Cn    = Cn_total,
            alpha = alpha,
            mach  = mach,
        )
