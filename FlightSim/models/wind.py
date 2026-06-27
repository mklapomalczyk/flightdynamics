"""
models/wind.py
===============
Model wiatru — wektor predkosci powietrza wzgledem ziemi w Launch Frame
(X wzdluz azymutu strzalu, Y w prawo, Z w dol), uzywany w
forces/force_model6.py do liczenia katow aerodynamicznych (alpha,
beta) i predkosci wzgledem powietrza (mach, q_dyn) — NIE do kinematyki
pozycji (ta zawsze uzywa predkosci wzgledem ziemi).

Wiatr jest aktywny TYLKO poza szyna startowa — ForceModel6DOF zeruje
go (tak jak v,w,p,qr,r) gdy on_rail=True.

Zaimplementowane modele:
  1. HorizontalWind     — staly wektor wiatru, bez zmiany z wysokoscia
  2. PowerLawWind       — predkosc rosnie z wysokoscia wg prawa
                          potegowego V(h) = V_ref * (h/h_ref)^alpha
                          (standardowy model warstwy granicznej, patrz
                          np. ESDU 82026 / Eurocode 1 — alpha~0.14-0.16
                          dla otwartego terenu, do 0.4 dla terenu
                          zabudowanego)
  3. PowerLawGustWind   — PowerLawWind + sinusoidalny podmuch (gust)
                          superponowany na skladowa srednia, do testow
                          czasowo-zmiennego wiatru (wymaga >1 strzalu
                          do walidacji fazy/okresu wzgledem rzeczywistych
                          podmuchow — patrz docstring main()).

Kierunek wiatru podaje sie jako kierunek METEOROLOGICZNY (skad wiatr
wieje, konwencja kompasowa 0=N, 90=E, 180=S, 270=W), a azymut_deg to
azymut strzalu (oba w stopniach, mierzone od polnocy w prawo/zgodnie
z ruchem wskazowek zegara) — model sam przelicza na Launch Frame.
"""

import numpy as np
from typing import Protocol


class WindModel(Protocol):
    """Interfejs modelu wiatru."""
    def velocity_launch_frame(self, t: float, altitude_m: float) -> np.ndarray:
        """Wektor predkosci wiatru [vx, vy, vz] w Launch Frame [m/s]."""
        ...


def _wind_vector_launch_frame(speed_mps: float, dir_from_deg: float,
                               azimuth_deg: float) -> np.ndarray:
    """
    Przelicza wiatr o predkosci speed_mps, wiejacy OD kierunku
    dir_from_deg (konwencja meteorologiczna, kompas), na wektor w
    Launch Frame (X wzdluz azymutu strzalu azimuth_deg, Y w prawo, Z w
    dol). Wiatr jest poziomy -> vz = 0.
    """
    if speed_mps == 0.0:
        return np.zeros(3)

    # Kierunek, w ktory wiatr WIEJE (przeciwny do "skad wieje")
    heading_to_rad = np.radians((dir_from_deg + 180.0) % 360.0)
    wind_north = speed_mps * np.cos(heading_to_rad)
    wind_east  = speed_mps * np.sin(heading_to_rad)

    az = np.radians(azimuth_deg)
    # Rzutowanie NE -> Launch Frame (X = wzdluz azymutu, Y = w prawo od X)
    vx = wind_north * np.cos(az) + wind_east * np.sin(az)
    vy = -wind_north * np.sin(az) + wind_east * np.cos(az)
    return np.array([vx, vy, 0.0])


class HorizontalWind:
    """
    Staly wektor wiatru, niezalezny od wysokosci i czasu — najprostszy
    model, do oszacowania czy STALY wiatr danej predkosci moze
    wyjasnic obserwowany deficyt apogeum.
    """

    def __init__(self, speed_mps: float, dir_from_deg: float, azimuth_deg: float):
        self.speed_mps    = speed_mps
        self.dir_from_deg = dir_from_deg
        self.azimuth_deg  = azimuth_deg
        self._vec = _wind_vector_launch_frame(speed_mps, dir_from_deg, azimuth_deg)

    def velocity_launch_frame(self, t: float, altitude_m: float) -> np.ndarray:
        return self._vec

    def __repr__(self) -> str:
        return (f"HorizontalWind(speed={self.speed_mps:.1f}m/s, "
                f"from={self.dir_from_deg:.0f}°, az={self.azimuth_deg:.0f}°)")


class PowerLawWind:
    """
    Predkosc wiatru rosnie z wysokoscia wg prawa potegowego:
        V(h) = V_ref * (h / h_ref)^alpha_exp

    Kierunek staly (taki sam na wszystkich wysokosciach — uproszczenie;
    rzeczywisty wiatr moze skrecac z wysokoscia, ale dla pierwszych
    kilkuset metrow i krotkiego czasu wznoszenia (~15-20s) to
    rozsadne przyblizenie).

    Parameters
    ----------
    speed_ref_mps : float
        Predkosc wiatru na wysokosci referencyjnej h_ref_m [m/s]
        (np. pomiar anemometrem na wysokosci 2-10m).
    dir_from_deg : float
        Kierunek meteorologiczny (skad wieje) [°].
    azimuth_deg : float
        Azymut strzalu tego lotu [°].
    h_ref_m : float
        Wysokosc referencyjna pomiaru [m]. Domyslnie 10m (standard WMO).
    alpha_exp : float
        Wykladnik prawa potegowego. ~0.14-0.16 dla otwartego terenu,
        ~0.2-0.25 dla terenu z pojedynczymi przeszkodami, do ~0.4 dla
        terenu silnie zabudowanego. Domyslnie 0.16.
    h_min_m : float
        Wysokosc minimalna do liczenia profilu — poniżej niej V(h)
        jest zamrozone na V(h_min) zamiast spadac do V=0 przy h->0
        (unika osobliwosci numerycznej tuz po opuszczeniu szyny).
    """

    def __init__(self, speed_ref_mps: float, dir_from_deg: float, azimuth_deg: float,
                 h_ref_m: float = 10.0, alpha_exp: float = 0.16, h_min_m: float = 0.5):
        self.speed_ref_mps = speed_ref_mps
        self.dir_from_deg  = dir_from_deg
        self.azimuth_deg   = azimuth_deg
        self.h_ref_m       = h_ref_m
        self.alpha_exp     = alpha_exp
        self.h_min_m       = h_min_m
        self._dir_vec = _wind_vector_launch_frame(1.0, dir_from_deg, azimuth_deg)

    def speed_at(self, altitude_m: float) -> float:
        h = max(altitude_m, self.h_min_m)
        return self.speed_ref_mps * (h / self.h_ref_m) ** self.alpha_exp

    def velocity_launch_frame(self, t: float, altitude_m: float) -> np.ndarray:
        return self.speed_at(altitude_m) * self._dir_vec

    def __repr__(self) -> str:
        return (f"PowerLawWind(V_ref={self.speed_ref_mps:.1f}m/s@{self.h_ref_m:.0f}m, "
                f"alpha={self.alpha_exp:.2f}, from={self.dir_from_deg:.0f}°, "
                f"az={self.azimuth_deg:.0f}°)")


class PowerLawGustWind(PowerLawWind):
    """
    PowerLawWind + sinusoidalny podmuch superponowany na skladowa
    srednia: predkosc(t,h) = V_powerlaw(h) * (1 + gust_amp * sin(2*pi*t/T + phase)).

    UWAGA: walidacja fazy/okresu podmuchu wzgledem rzeczywistego
    wiatru wymaga danych z >1 lotu (np. roznicy miedzy lotami 30 min
    od siebie) — do tego czasu uzywac jako narzedzie do oszacowania
    GORNEJ GRANICY wplywu wiatru zmiennego w czasie, nie jako
    dopasowany model konkretnego lotu.

    Parameters
    ----------
    gust_amp : float
        Amplituda podmuchu jako frakcja sredniej predkosci (np. 0.3 =
        podmuchy +/-30% sredniej).
    gust_period_s : float
        Okres podmuchu [s].
    gust_phase_rad : float
        Faza startowa podmuchu [rad] (t=0 = zaplon).
    """

    def __init__(self, speed_ref_mps: float, dir_from_deg: float, azimuth_deg: float,
                 h_ref_m: float = 10.0, alpha_exp: float = 0.16, h_min_m: float = 0.5,
                 gust_amp: float = 0.3, gust_period_s: float = 5.0,
                 gust_phase_rad: float = 0.0):
        super().__init__(speed_ref_mps, dir_from_deg, azimuth_deg,
                          h_ref_m=h_ref_m, alpha_exp=alpha_exp, h_min_m=h_min_m)
        self.gust_amp       = gust_amp
        self.gust_period_s  = gust_period_s
        self.gust_phase_rad = gust_phase_rad

    def velocity_launch_frame(self, t: float, altitude_m: float) -> np.ndarray:
        base = self.speed_at(altitude_m)
        gust_factor = 1.0 + self.gust_amp * np.sin(
            2.0 * np.pi * t / self.gust_period_s + self.gust_phase_rad
        )
        return max(base * gust_factor, 0.0) * self._dir_vec

    def __repr__(self) -> str:
        return (f"PowerLawGustWind(V_ref={self.speed_ref_mps:.1f}m/s@{self.h_ref_m:.0f}m, "
                f"alpha={self.alpha_exp:.2f}, gust_amp={self.gust_amp:.2f}, "
                f"T={self.gust_period_s:.1f}s, from={self.dir_from_deg:.0f}°, "
                f"az={self.azimuth_deg:.0f}°)")


def create_wind(model: str = "none", **kwargs) -> "WindModel | None":
    """
    Fabryka modelu wiatru.

    Parameters
    ----------
    model : str
        "none"          -> brak wiatru (None)
        "horizontal"     -> HorizontalWind(**kwargs)
        "power_law"      -> PowerLawWind(**kwargs)
        "power_law_gust" -> PowerLawGustWind(**kwargs)
    """
    if model == "none":
        return None
    elif model == "horizontal":
        return HorizontalWind(**kwargs)
    elif model == "power_law":
        return PowerLawWind(**kwargs)
    elif model == "power_law_gust":
        return PowerLawGustWind(**kwargs)
    else:
        raise ValueError(f"Nieznany model wiatru: '{model}'.")
