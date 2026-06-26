"""
models/atmosphere.py
====================
Model atmosfery standardowej ISA (International Standard Atmosphere).

Zaimplementowane:
  - ISA do 86 km (troposfera + stratosfera dolna + stratosfera górna)
  - Gęstość, ciśnienie, temperatura, prędkość dźwięku jako funkcje wysokości
  - Lepkość dynamiczna (wzór Sutherlanda) — potrzebna do liczby Re

Zaślepka architektoniczna:
  - WGS84: rzeczywiste odwzorowanie wysokości geometrycznej na geopotencjalną
    (różnica ~0.3% przy h=10km, pomijalna dla rakiet 70mm ale
    architektura jest gotowa na przyszłość)

Konwencja:
  Wysokość h jest zawsze wysokością geometryczną nad poziomem morza [m].
  Przy prostej geometrii (płaska ziemia) h = z_pos z launch frame,
  przy WGS84 będzie wymagana korekcja.
"""

import numpy as np
from dataclasses import dataclass
from typing import Protocol


# ============================================================================
# Stałe ISA
# ============================================================================

R_AIR   = 287.058   # stała gazu dla powietrza [J/(kg·K)]
GAMMA   = 1.4        # współczynnik adiabatyczny powietrza [-]
G0      = 9.80665    # standardowe przyspieszenie grawitacyjne [m/s²]

# Warunki na poziomie morza (ISA MSL)
T0   = 288.15    # temperatura [K]
P0   = 101325.0  # ciśnienie [Pa]
RHO0 = 1.225     # gęstość [kg/m³]

# Warstwy ISA: (h_base [m], T_base [K], L [K/m])
# L > 0 → wzrost temperatury z wysokością (inwersja)
# L = 0 → izotermia
# L < 0 → spadek temperatury (troposfera)
_ISA_LAYERS = [
    (0,      288.15, -0.0065),   # troposfera
    (11000,  216.65,  0.0),      # dolna stratosfera (izotermiczna)
    (20000,  216.65,  0.001),    # środkowa stratosfera
    (32000,  228.65,  0.0028),   # górna stratosfera
    (47000,  270.65,  0.0),      # mezosfera dolna
    (51000,  270.65, -0.0028),   # mezosfera górna
    (71000,  214.65, -0.002),    # górna mezosfera
    (86000,  186.87,  0.0),      # granica (wartownik)
]


# ============================================================================
# Silnik ISA
# ============================================================================

def _isa_layer(h: float):
    """Zwraca parametry warstwy ISA dla danej wysokości."""
    h = np.clip(h, 0.0, 86000.0)
    for i in range(len(_ISA_LAYERS) - 1):
        h_top = _ISA_LAYERS[i + 1][0]
        if h <= h_top:
            return _ISA_LAYERS[i]
    return _ISA_LAYERS[-2]


def _pressure_at_base(h_base: float) -> float:
    """Ciśnienie na dolnej granicy każdej warstwy [Pa]."""
    p = P0
    for i in range(len(_ISA_LAYERS) - 1):
        h0, T0_layer, L = _ISA_LAYERS[i]
        h1 = _ISA_LAYERS[i + 1][0]
        if h_base <= h0:
            break
        dh = min(h_base, h1) - h0
        if abs(L) < 1e-10:
            p *= np.exp(-G0 * dh / (R_AIR * T0_layer))
        else:
            p *= (T0_layer / (T0_layer + L * dh)) ** (G0 / (R_AIR * L))
    return p


@dataclass
class AtmosphereState:
    """Stan atmosfery w danym punkcie."""
    h:          float   # wysokość geometryczna [m]
    temperature: float  # temperatura [K]
    pressure:   float   # ciśnienie [Pa]
    density:    float   # gęstość [kg/m³]
    speed_of_sound: float  # prędkość dźwięku [m/s]
    dynamic_viscosity: float  # lepkość dynamiczna [Pa·s]

    def mach(self, speed: float) -> float:
        """Liczba Macha dla zadanej prędkości [m/s]."""
        return speed / self.speed_of_sound if self.speed_of_sound > 0 else 0.0


class AtmosphereModel(Protocol):
    """Interfejs modelu atmosfery — każdy model musi implementować tę metodę."""
    def at(self, h: float) -> AtmosphereState: ...


# ============================================================================
# Model ISA
# ============================================================================

class ISAAtmosphere:
    """
    Model ISA (International Standard Atmosphere).
    Ważny do h = 86 000 m. Powyżej zwraca wartości dla h = 86 km.
    """
    def __init__(self):
        self._cache_h = None
        self._cache_result = None
        
    def at(self, h: float) -> AtmosphereState:
        """
        Oblicza parametry atmosfery na wysokości h.

        Parameters
        ----------
        h : float
            Wysokość geometryczna [m]. Wartości ujemne → h = 0.
        """
        h = max(0.0, float(h))
        if self._cache_h is not None and abs(h - self._cache_h) < 0.1:
            return self._cache_result
        h0, T_base, L = _isa_layer(h)
        p_base = _pressure_at_base(h0)
        dh = h - h0
        T_base_layer = _ISA_LAYERS[_ISA_LAYERS.index(
            next(l for l in _ISA_LAYERS if l[0] == h0)
        )][1]

        # Temperatura
        T = T_base_layer + L * dh

        # Ciśnienie
        if abs(L) < 1e-10:
            P = p_base * np.exp(-G0 * dh / (R_AIR * T_base_layer))
        else:
            P = p_base * (T_base_layer / T) ** (G0 / (R_AIR * L))

        # Gęstość
        rho = P / (R_AIR * T)

        # Prędkość dźwięku
        a = np.sqrt(GAMMA * R_AIR * T)

        # Lepkość dynamiczna — wzór Sutherlanda
        mu = self._sutherland(T)
        
        # self._cache_h = h
        # self._cache_result = result

        return AtmosphereState(
            h=h,
            temperature=T,
            pressure=P,
            density=rho,
            speed_of_sound=a,
            dynamic_viscosity=mu,
        )

    @staticmethod
    def _sutherland(T: float) -> float:
        """
        Wzór Sutherlanda na lepkość dynamiczną powietrza [Pa·s].
        Ważny w zakresie 170–2000 K.
        """
        T_ref = 273.15   # [K]
        mu_ref = 1.716e-5  # [Pa·s]
        S = 110.4          # stała Sutherlanda [K]
        return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


# ============================================================================
# Zaślepka WGS84
# ============================================================================

class WGS84Atmosphere:
    """
    ZAŚLEPKA: Atmosfera ISA z korekcją WGS84.

    Konwersja wysokości geometrycznej na geopotencjalną:
        h_geop = h_geom * R_earth / (R_earth + h_geom)
    gdzie R_earth = 6356766 m (promień ziemi wg WGS84 w kierunku biegunowym,
    ISA używa tej wartości).

    Różnica: ~0.3% przy h = 10 km, ~1% przy h = 30 km.
    Dla rakiet 70mm (maks. kilka km) pomijalnie mała.
    """

    def __init__(self):
        self._isa = ISAAtmosphere()
        self._R_earth = 6356766.0  # [m]

    def at(self, h_geom: float) -> AtmosphereState:
        """
        Oblicza parametry atmosfery z korekcją WGS84.

        Parameters
        ----------
        h_geom : float
            Wysokość geometryczna (rzeczywista) [m].
        """
        # Konwersja na geopotencjalną
        h_geop = h_geom * self._R_earth / (self._R_earth + h_geom)
        state = self._isa.at(h_geop)
        # Zwróć z oryginalną wysokością geometryczną
        return AtmosphereState(
            h=h_geom,
            temperature=state.temperature,
            pressure=state.pressure,
            density=state.density,
            speed_of_sound=state.speed_of_sound,
            dynamic_viscosity=state.dynamic_viscosity,
        )


# ============================================================================
# ISA zakotwiczona w zmierzonych warunkach naziemnych (dzień startu)
# ============================================================================

def _saturation_vapor_pressure_hpa(T_C: float) -> float:
    """Wzór Magnusa — ciśnienie pary wodnej nasyconej [hPa] dla temp. w [°C]."""
    return 6.1094 * np.exp(17.625 * T_C / (243.04 + T_C))


class ISALaunchSiteAtmosphere:
    """
    ISA zakotwiczona w zmierzonych warunkach naziemnych (T, p, wilgotność
    względna) w momencie startu, zamiast standardowych warunków MSL
    (288.15 K, 101325 Pa). h=0 odpowiada poziomowi startu (tak jak w
    pozostałych modelach atmosfery w tym pliku — z=0 to wyrzutnia, nie
    rzeczywista wysokość AMSL).

    Założenia uproszczające (efekt rzędu <1% gęstości, drugorzędny wobec
    przesunięcia T/p, ale liczony jawnie skoro mamy zmierzoną wilgotność):
      - profil temperatury = standardowy profil ISA (te same warstwy/lapse
        rate) przesunięty o stałą ΔT = T0_zmierzone - T0_ISA,
      - ciśnienie liczone barometrycznie od p0_zmierzonego przy h=0 z tym
        przesuniętym profilem T,
      - stosunek zmieszania pary wodnej (specific humidity) zakładany
        stały z wysokością (dobrze wymieszana warstwa przyziemna) —
        korekta gęstości przez temperaturę wirtualną Tv.
    """

    def __init__(self, T0_C: float, p0_hpa: float, RH_pct: float = 0.0):
        self.T0 = float(T0_C) + 273.15      # [K]
        self.p0 = float(p0_hpa) * 100.0      # [Pa]
        self.RH = float(RH_pct)
        self._delta_T = self.T0 - T0         # przesunięcie wzgl. standardowej ISA (288.15 K)

        e0_hpa = _saturation_vapor_pressure_hpa(float(T0_C)) * (self.RH / 100.0)
        # mieszanina wodorowo-powietrzna: stosunek zmieszania r = 0.622*e/(p-e), zakładany const z h
        self._mixing_ratio = 0.622 * e0_hpa / max(p0_hpa - e0_hpa, 1e-6)

        self._cache_h = None
        self._cache_result = None

    def _layer_base_T(self, h0: float) -> float:
        base = next(l for l in _ISA_LAYERS if l[0] == h0)
        return base[1] + self._delta_T

    def _pressure_at_base(self, h_base: float) -> float:
        p = self.p0
        for i in range(len(_ISA_LAYERS) - 1):
            h0, T_base_std, L = _ISA_LAYERS[i]
            h1 = _ISA_LAYERS[i + 1][0]
            if h_base <= h0:
                break
            T_base = T_base_std + self._delta_T
            dh = min(h_base, h1) - h0
            if abs(L) < 1e-10:
                p *= np.exp(-G0 * dh / (R_AIR * T_base))
            else:
                p *= (T_base / (T_base + L * dh)) ** (G0 / (R_AIR * L))
        return p

    def at(self, h: float) -> AtmosphereState:
        h = max(0.0, float(h))
        if self._cache_h is not None and abs(h - self._cache_h) < 0.1:
            return self._cache_result

        h0, T_base_std, L = _isa_layer(h)
        T_base = T_base_std + self._delta_T
        p_base = self._pressure_at_base(h0)
        dh = h - h0

        T = T_base + L * dh
        if abs(L) < 1e-10:
            P = p_base * np.exp(-G0 * dh / (R_AIR * T_base))
        else:
            P = p_base * (T_base / T) ** (G0 / (R_AIR * L))

        # gęstość z korekcją wilgotności (temperatura wirtualna, r = const z h)
        Tv = T * (1.0 + 0.61 * self._mixing_ratio)
        rho = P / (R_AIR * Tv)

        a = np.sqrt(GAMMA * R_AIR * T)
        mu = ISAAtmosphere._sutherland(T)

        result = AtmosphereState(
            h=h, temperature=T, pressure=P, density=rho,
            speed_of_sound=a, dynamic_viscosity=mu,
        )
        self._cache_h, self._cache_result = h, result
        return result


# ============================================================================
# Fabryka — wybór modelu przez parametr
# ============================================================================

def create_atmosphere(model: str = "ISA", **kwargs) -> AtmosphereModel:
    """
    Fabryka modelu atmosfery.

    Parameters
    ----------
    model : str
        "ISA"        → standardowa atmosfera ISA (domyślna)
        "WGS84"      → ISA z korekcją WGS84 wysokości
        "ISA_LAUNCH" → ISA zakotwiczona w zmierzonych warunkach naziemnych;
                       wymaga kwargs: T0_C, p0_hpa, opcjonalnie RH_pct
    """
    if model == "ISA":
        return ISAAtmosphere()
    elif model == "WGS84":
        return WGS84Atmosphere()
    elif model == "ISA_LAUNCH":
        return ISALaunchSiteAtmosphere(**kwargs)
    else:
        raise ValueError(f"Nieznany model atmosfery: '{model}'. Wybierz 'ISA', 'WGS84' lub 'ISA_LAUNCH'.")
