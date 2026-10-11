"""
control/effectors/reaction_jet.py
=================================
Efektor REACTION JETS — zestaw jednorazowych silnikow na paliwo stale,
dzialajacych (domyslnie) prostopadle do osi rakiety.

Ten plik zawiera wylacznie FIZYKE: geometrie silnikow, profil ciagu i
zamiane "ktore silniki pala sie i od kiedy" na sily/momenty w ukladzie
ciala. Logika "kiedy odpalic" (komendy, pamiec zuzytych silnikow) jest w
control/jets.py, a dokladne calkowanie impulsow w core/solver6.py.

GEOMETRIA I KONWENCJE (body frame: X-przod, Y-prawo, Z-dol)
-----------------------------------------------------------
  * Rzad i: polozenie osiowe x_i [m od nosa], n silnikow rowno po obwodzie,
    przesuniecie fazowe phase_i [deg].
  * Silnik k w rzedzie i: kat obwodowy
        phi = phase_i + k * 360/n
    mierzony od osi +Z (dol) w strone +Y (prawo).
    phi=0 -> silnik na +Z, phi=90 -> silnik na +Y.
  * Wersor promieniowy (od osi na zewnatrz):   e_r = (0, sin phi, cos phi)
  * Wersor obwodowy w sensie DODATNIEGO roll:  e_t = (0, -cos phi, sin phi)
    (e_r x e_t = +X, wiec sila wzdluz +e_t daje dodatni moment Mx).
  * Dysza skierowana na zewnatrz => sila dziala DO OSI (-e_r).
  * Pochylenia (domyslnie 0):
      tilt_axial_deg      > 0 -> skladowa sily w strone nosa (+X)
      tilt_tangential_deg > 0 -> skladowa sily wzdluz +e_t (dodatni roll)
    Kierunek sily (wersor):
        d = -cos(a)cos(t) e_r + cos(a)sin(t) e_t + sin(a) e_x

  * Moment wzgledem srodka masy: M = r x F,
        r = (xcg - x_i, R sin phi, R cos phi)
    (xcg - x_i > 0 gdy rzad PRZED srodkiem masy).

  Kontrola znaku: silnik na +Z (phi=0) przed SC pcha nos w strone -Z (w gore)
  -> My > 0 -> alpha rosnie ("nos w gore"), zgodnie z konwencja
  control/effectors/aero_surface.py.

PROFIL CIAGU
------------
  * "rectangular" (domyslny): F = thrust przez burn_time.
  * tabela znormalizowana [[t/t_burn, F/F_nom], ...] od 0 do 1: ksztalt jest
    przeskalowany tak, aby IMPULS = thrust * burn_time (impuls silnika jest
    zachowany, ksztalt decyduje tylko o rozkladzie w czasie). Szczyt moze
    wiec przekroczyc thrust.

Pominiete swiadomie (v1): ubytek masy silnikow, interakcja strugi z
oplywem (jet interaction), zaleznosc ciagu od cisnienia otoczenia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..types import ControlWrench, FlightState
from .base import Effector


# ============================================================================
# Profil ciagu jednego silnika
# ============================================================================

class JetThrustProfile:
    """Ciag jednego silnika w funkcji czasu od zaplonu tau [s]."""

    def __init__(self, thrust: float, burn_time: float,
                 shape: Union[str, Sequence[Sequence[float]], None] = "rectangular"):
        if thrust <= 0.0:
            raise ValueError(f"reaction jets: thrust musi byc > 0 (jest {thrust})")
        if burn_time <= 0.0:
            raise ValueError(f"reaction jets: burn_time musi byc > 0 (jest {burn_time})")
        self.thrust = float(thrust)
        self.burn_time = float(burn_time)

        if shape is None or (isinstance(shape, str) and shape == "rectangular"):
            self.is_rectangular = True
            self._tau_n = np.array([0.0, 1.0])
            self._f_n = np.array([1.0, 1.0])
        elif isinstance(shape, str):
            raise ValueError(f"reaction jets: nieznany profil {shape!r} "
                             f"(dozwolone: 'rectangular' albo tabela [[t/t_burn, F/F_nom], ...])")
        else:
            arr = np.asarray(shape, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
                raise ValueError("reaction jets: profil musi byc lista par [t/t_burn, F/F_nom]")
            tau_n, f_n = arr[:, 0], arr[:, 1]
            if abs(tau_n[0]) > 1e-9 or abs(tau_n[-1] - 1.0) > 1e-9:
                raise ValueError("reaction jets: profil musi zaczynac sie w t/t_burn=0 i konczyc w 1")
            if np.any(np.diff(tau_n) <= 0.0):
                raise ValueError("reaction jets: t/t_burn w profilu musi rosnac")
            if np.any(f_n < 0.0):
                raise ValueError("reaction jets: F/F_nom w profilu nie moze byc ujemne")
            # Metoda trapezow recznie — np.trapz/np.trapezoid zaleza od wersji numpy
            area = float(np.sum(np.diff(tau_n) * (f_n[1:] + f_n[:-1]) / 2.0))
            if area <= 0.0:
                raise ValueError("reaction jets: profil ma zerowy impuls")
            self.is_rectangular = False
            self._tau_n = tau_n
            # Normalizacja: srednia ksztaltu = 1  =>  impuls = thrust * burn_time
            self._f_n = f_n / area

    @property
    def impulse(self) -> float:
        """Impuls calkowity [N*s]."""
        return self.thrust * self.burn_time

    @property
    def knots(self) -> np.ndarray:
        """Wezly profilu w czasie od zaplonu [s] (punkty zalaman, 0..burn_time)."""
        return self._tau_n * self.burn_time

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        """
        Ciag [N] dla tau = t - t_ign, ZACISKANEGO do [0, burn_time].

        Zaciskanie jest celowe: o tym, CZY silnik pracuje, decyduje maska
        aktywnosci ustawiana przez solver na caly segment (patrz
        ReactionJetEffector.wrench). Dzieki temu krok RK45 konczacy sie
        dokladnie na wygaszeniu nie "widzi" skoku do zera.
        """
        tau = np.clip(np.asarray(tau, dtype=float), 0.0, self.burn_time)
        if self.is_rectangular:
            return np.full_like(tau, self.thrust)
        return self.thrust * np.interp(tau / self.burn_time, self._tau_n, self._f_n)


# ============================================================================
# Uklad silnikow
# ============================================================================

@dataclass(frozen=True)
class JetRow:
    """Jeden rzad silnikow."""
    x:         float          # polozenie osiowe [m od nosa]
    n_engines: int
    phase_deg: float = 0.0


class JetLayout:
    """
    Geometria wszystkich silnikow: polozenia, kierunki sil i nazwy.

    Silniki sa ponumerowane plasko: rzad po rzedzie, w rzedzie po rosnacym phi.
    Nazwy sa 1-indeksowane: "R1E1", "R1E2", ..., "R2E1", ...
    """

    def __init__(self, rows: Sequence[JetRow], radius: float,
                 tilt_axial_deg: float = 0.0, tilt_tangential_deg: float = 0.0):
        if not rows:
            raise ValueError("reaction jets: brak rzedow silnikow")
        if radius <= 0.0:
            raise ValueError(f"reaction jets: radius musi byc > 0 (jest {radius})")
        self.rows = list(rows)
        self.radius = float(radius)
        self.tilt_axial_deg = float(tilt_axial_deg)
        self.tilt_tangential_deg = float(tilt_tangential_deg)

        names: List[str] = []
        row_idx, xs, phis = [], [], []
        for i, row in enumerate(self.rows):
            if row.n_engines < 1:
                raise ValueError(f"reaction jets: rzad {i + 1} ma n_engines < 1")
            for k in range(row.n_engines):
                names.append(f"R{i + 1}E{k + 1}")
                row_idx.append(i)
                xs.append(float(row.x))
                phis.append(np.deg2rad(row.phase_deg + k * 360.0 / row.n_engines))

        self.names: Tuple[str, ...] = tuple(names)
        self.row_index = np.array(row_idx, dtype=int)
        self.x = np.array(xs)                    # [m od nosa]
        self.phi = np.array(phis)                # [rad]
        self._name_to_idx = {n: i for i, n in enumerate(self.names)}

        s, c = np.sin(self.phi), np.cos(self.phi)
        zeros = np.zeros_like(self.phi)
        self.e_r = np.column_stack([zeros, s, c])
        self.e_t = np.column_stack([zeros, -c, s])
        e_x = np.column_stack([np.ones_like(self.phi), zeros, zeros])

        a = np.deg2rad(self.tilt_axial_deg)
        t = np.deg2rad(self.tilt_tangential_deg)
        # Wersory kierunku sily (n, 3) — jednostkowe z konstrukcji.
        self.direction = (-np.cos(a) * np.cos(t) * self.e_r
                          + np.cos(a) * np.sin(t) * self.e_t
                          + np.sin(a) * e_x)
        # Polozenie poprzeczne (y, z) — czesc osiowa zalezy od xcg(t).
        self.r_lateral = self.radius * self.e_r

    @property
    def n_engines(self) -> int:
        return len(self.names)

    def index(self, engine: Union[int, str]) -> int:
        """Indeks plaski z nazwy ("R2E5") albo z liczby (0-indeksowanej)."""
        if isinstance(engine, (int, np.integer)):
            i = int(engine)
            if not 0 <= i < self.n_engines:
                raise ValueError(f"reaction jets: indeks silnika {i} poza 0..{self.n_engines - 1}")
            return i
        key = str(engine).strip().upper()
        if key not in self._name_to_idx:
            raise ValueError(f"reaction jets: nieznany silnik {engine!r} "
                             f"(dostepne: {self.names[0]}..{self.names[-1]})")
        return self._name_to_idx[key]

    def arms(self, xcg: float) -> np.ndarray:
        """Ramiona r (n, 3) wzgledem srodka masy w ukladzie ciala."""
        r = self.r_lateral.copy()
        r[:, 0] = xcg - self.x
        return r


# ============================================================================
# Efektor
# ============================================================================

class ReactionJetEffector(Effector):
    """
    Czasy zaplonu -> sily i momenty.

    Wejscie u (wg kontraktu Effector.wrench) to wektor czasow zaplonu [s]
    kazdego silnika; np.inf = silnik nieodpalony.
    """

    channels: Tuple[str, ...] = ()

    def __init__(self, layout: JetLayout, profile: JetThrustProfile,
                 name: str = "reaction_jets"):
        self.layout = layout
        self.profile = profile
        self.name = name
        self.channels = layout.names

    def active_mask(self, t: float, t_ign: np.ndarray) -> np.ndarray:
        """Silniki pracujace w chwili t (przedzial [t_ign, t_ign + burn))."""
        dt = t - np.asarray(t_ign, dtype=float)
        return (dt >= 0.0) & (dt < self.profile.burn_time)

    def thrusts(self, t: float, t_ign: np.ndarray,
                active: Optional[np.ndarray] = None) -> np.ndarray:
        """Ciag kazdego silnika [N] w chwili t."""
        t_ign = np.asarray(t_ign, dtype=float)
        if active is None:
            active = self.active_mask(t, t_ign)
        T = np.zeros(self.layout.n_engines)
        if np.any(active):
            T[active] = self.profile(t - t_ign[active])
        return T

    def wrench(self, u: np.ndarray, fs: FlightState,
               active: Optional[np.ndarray] = None) -> ControlWrench:
        """
        active: maska pracujacych silnikow ustalona na caly segment calkowania
        (przekazuje ja ReactionJetSystem). None -> liczona z czasu fs.t.
        """
        T = self.thrusts(fs.t, u, active)
        n_on = int(np.count_nonzero(T))
        if n_on == 0:
            return ControlWrench(diag=self._diag(0, np.zeros(3), np.zeros(3)))
        F_each = T[:, None] * self.layout.direction            # (n, 3)
        F = F_each.sum(axis=0)
        M = np.cross(self.layout.arms(fs.xcg), F_each).sum(axis=0)
        return ControlWrench(F=F, M=M, diag=self._diag(n_on, F, M))

    @staticmethod
    def _diag(n_on: int, F: np.ndarray, M: np.ndarray) -> dict:
        return {
            "jet_n_active": float(n_on),
            "jet_Fx": float(F[0]), "jet_Fy": float(F[1]), "jet_Fz": float(F[2]),
            "jet_Mx": float(M[0]), "jet_My": float(M[1]), "jet_Mz": float(M[2]),
        }
