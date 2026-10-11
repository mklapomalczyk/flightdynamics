"""
control/jets.py
===============
Sterowanie REACTION JETS: komenda (ktory silnik odpalic) -> zaplon ->
impuls ciagu -> odpowiedz modelu 6DOF.

Lancuch:
    JetCommander.decide()  -> lista silnikow do odpalenia w danym "ticku"
    ReactionJetSystem      -> pamiec zuzytych silnikow, czasy zaplonu,
                              opoznienie zaplonu, log zdarzen
    ReactionJetEffector    -> sily/momenty (control/effectors/reaction_jet.py)
    core/solver6.py        -> calkowanie segmentami z granicami dokladnie na
                              tickach, zaplonach i wygaszeniach

DLACZEGO DYSKRETNIE, A NIE W WEKTORZE STANU ODE
-----------------------------------------------
Silnik na paliwo stale odpala sie raz. "Ktory silnik jest zuzyty i kiedy sie
zapalil" to stan, ktory zmienia sie SKOKOWO, a nie jest calkowany. Pochodne
modelu sa wolane w punktach posrednich RK45 (niemonotonicznie w czasie),
wiec decyzja o odpaleniu nie moze zapasc wewnatrz derivatives().

Rozwiazanie (uklad hybrydowy):
  * Stan dyskretny (czasy zaplonu, maska pracujacych silnikow) jest STALY
    w obrebie jednego segmentu solve_ivp — derivatives() pozostaje czysta
    funkcja (t, x).
  * Zmienia sie wylacznie MIEDZY segmentami: w tickach sterownika
    (tick()) i na granicach impulsow (begin_segment()).
  * Ciag silnika i liczy sie dokladnie jako profil(t - t_ign[i]), wiec
    zaden "pamietany przebieg" nie jest potrzebny.

CZASY ZAPLONU NIE MUSZA BYC ZNANE Z GORY
----------------------------------------
Sterownik decyduje w tickach (co 1/control_rate_hz). W chwili decyzji znany
jest czas zaplonu (t_tick + ignition_delay) i wygaszenia (+ burn_time), wiec
solver od razu wstawia je jako granice segmentow. Harmonogram testowy
(ScheduleJetCommander) dodatkowo podaje swoje czasy z gory (planned_times),
dzieki czemu nie potrzebuje regularnych tickow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .effectors.reaction_jet import (JetLayout, JetRow, JetThrustProfile,
                                     ReactionJetEffector)
from .types import ControlWrench, FlightState

EngineRef = Union[int, str]

# Tolerancja porownan czasow granic segmentow [s].
T_EPS = 1e-9


# ============================================================================
# Kontekst ticku i sterowniki
# ============================================================================

@dataclass(frozen=True)
class JetTickContext:
    """
    Wszystko, co sterownik widzi w chwili decyzji.

    x jest ZAAKCEPTOWANYM stanem solvera w chwili t (nie punktem posrednim
    RK45). Pelny wektor stanu -> sterownik ma dostep do orientacji
    (kwaternion), predkosci i predkosci katowych; patrz core/state6.py.
    """
    t:       float                 # chwila ticku [s]
    t_prev:  float                 # poprzedni tick [s] (-inf dla pierwszego)
    x:       np.ndarray            # wektor stanu solvera (kopia)
    t_ign:   Tuple[float, ...]     # czasy zaplonu wszystkich silnikow (inf = nieodpalony)
    names:   Tuple[str, ...]       # nazwy silnikow ("R1E1", ...)

    def is_fired(self, engine: EngineRef) -> bool:
        i = engine if isinstance(engine, (int, np.integer)) else self.names.index(str(engine).upper())
        return bool(np.isfinite(self.t_ign[int(i)]))

    @property
    def available(self) -> Tuple[str, ...]:
        """Nazwy silnikow jeszcze nieodpalonych."""
        return tuple(n for n, ti in zip(self.names, self.t_ign) if not np.isfinite(ti))

    @property
    def state(self):
        """Stan jako State6DOF (katy Eulera, alpha, beta, ...)."""
        from core.state6 import State6DOF
        return State6DOF.from_numpy(self.x[:13])


class JetCommander(ABC):
    """
    Zrodlo komend odpalenia. Wywolywany przez solver TYLKO w tickach,
    monotonicznie w czasie, z zaakceptowanym stanem — moze wiec (inaczej niz
    Commander z commander.py) trzymac wlasna pamiec, np. calke regulatora.

    rate_hz: czestotliwosc tickow [Hz]; None = tylko planned_times().
    """

    rate_hz: Optional[float] = None

    def planned_times(self) -> Sequence[float]:
        """Chwile znane z gory, w ktorych trzeba wywolac decide()."""
        return ()

    def reset(self) -> None:
        """Wywolywane na poczatku kazdej symulacji (dla sterownikow z pamiecia)."""

    @abstractmethod
    def decide(self, ctx: JetTickContext) -> Sequence[EngineRef]:
        """Silniki do odpalenia TERAZ (nazwy "R1E3" lub indeksy 0..n-1)."""
        raise NotImplementedError


class NoJetCommander(JetCommander):
    """Nic nie odpala — trajektoria identyczna jak bez systemu."""

    def decide(self, ctx: JetTickContext) -> Sequence[EngineRef]:
        return ()


class ScheduleJetCommander(JetCommander):
    """
    Harmonogram testowy: [(t, "R1E3"), (t, ["R1E1", "R2E5"]), ...].

    Wpis jest wydawany w ticku t_k gdy t_prev < t_wpisu <= t_k. Poniewaz
    czasy wpisow sa podawane jako planned_times(), solver robi tick dokladnie
    w tych chwilach — bez kwantyzacji do siatki tickow.
    """

    def __init__(self, schedule: Sequence[Tuple[float, Union[EngineRef, Sequence[EngineRef]]]],
                 rate_hz: Optional[float] = None):
        entries: List[Tuple[float, Tuple[EngineRef, ...]]] = []
        for t, eng in schedule:
            if isinstance(eng, (str, int, np.integer)):
                eng = (eng,)
            entries.append((float(t), tuple(eng)))
        self.schedule = sorted(entries, key=lambda e: e[0])
        self.rate_hz = rate_hz

    def planned_times(self) -> Sequence[float]:
        return [t for t, _ in self.schedule]

    def decide(self, ctx: JetTickContext) -> Sequence[EngineRef]:
        out: List[EngineRef] = []
        for t, engines in self.schedule:
            if ctx.t_prev + T_EPS < t <= ctx.t + T_EPS:
                out.extend(engines)
        return out


# ============================================================================
# Log zdarzen
# ============================================================================

@dataclass(frozen=True)
class JetEvent:
    """Jedna komenda odpalenia i jej skutek."""
    engine:  str
    t_cmd:   float            # chwila komendy [s]
    t_ign:   float            # chwila zaplonu [s] (nan gdy odrzucona)
    t_end:   float            # koniec pracy [s] (nan gdy odrzucona)
    status:  str              # "fired" | "rejected_spent"


# ============================================================================
# System
# ============================================================================

class ReactionJetSystem:
    """
    Obiekt przekazywany do ForceModel6DOF jako `control=` (ten sam slot co
    ControlSystem dla canardow). ForceModel wola tylko compute(fs, xa);
    reszta metod jest dla solvera segmentowego (core/solver6.py).
    """

    # Flaga dla solvera: wymagane calkowanie segmentami.
    is_discrete = True
    n_states = 0

    def __init__(self, effector: ReactionJetEffector,
                 commander: Optional[JetCommander] = None,
                 ignition_delay: float = 0.0,
                 control_rate_hz: Optional[float] = None,
                 enabled: bool = True):
        if ignition_delay < 0.0:
            raise ValueError("reaction jets: ignition_delay musi byc >= 0")
        self.effector = effector
        self.commander = commander if commander is not None else NoJetCommander()
        self.ignition_delay = float(ignition_delay)
        # Czestotliwosc tickow: z sterownika, a jesli ten nie podaje — z YAML.
        self.control_rate_hz = control_rate_hz
        self.enabled = bool(enabled)
        self.reset()

    # ---- Wlasnosci ------------------------------------------------------ #
    @property
    def layout(self) -> JetLayout:
        return self.effector.layout

    @property
    def profile(self) -> JetThrustProfile:
        return self.effector.profile

    @property
    def n_engines(self) -> int:
        return self.layout.n_engines

    @property
    def t_ign(self) -> np.ndarray:
        """Czasy zaplonu (kopia; inf = nieodpalony)."""
        return self._t_ign.copy()

    @property
    def tick_rate_hz(self) -> Optional[float]:
        r = getattr(self.commander, "rate_hz", None)
        return r if r is not None else self.control_rate_hz

    # ---- Stan dyskretny -------------------------------------------------- #
    def reset(self) -> None:
        """Wszystkie silniki niezuzyte; wolane przez solver na starcie."""
        self._t_ign = np.full(self.n_engines, np.inf)
        self._active: Optional[np.ndarray] = None
        self.events: List[JetEvent] = []
        if hasattr(self.commander, "reset"):
            self.commander.reset()

    def fire(self, engines: Sequence[EngineRef], t_cmd: float) -> List[JetEvent]:
        """Wykonuje komende odpalenia. Zuzyty silnik -> odrzucenie (log)."""
        new: List[JetEvent] = []
        for ref in engines:
            i = self.layout.index(ref)
            name = self.layout.names[i]
            if np.isfinite(self._t_ign[i]):
                ev = JetEvent(name, float(t_cmd), float("nan"), float("nan"),
                              "rejected_spent")
            else:
                t_ign = float(t_cmd) + self.ignition_delay
                self._t_ign[i] = t_ign
                ev = JetEvent(name, float(t_cmd), t_ign,
                              t_ign + self.profile.burn_time, "fired")
            self.events.append(ev)
            new.append(ev)
        return new

    def tick(self, t: float, t_prev: float, x: np.ndarray) -> List[JetEvent]:
        """Wywolanie sterownika w ticku i wykonanie jego komend."""
        if not self.enabled:
            return []
        ctx = JetTickContext(t=float(t), t_prev=float(t_prev),
                             x=np.array(x, dtype=float, copy=True),
                             t_ign=tuple(self._t_ign.tolist()),
                             names=self.layout.names)
        engines = list(self.commander.decide(ctx) or ())
        return self.fire(engines, t) if engines else []

    # ---- Granice segmentow --------------------------------------------- #
    def switch_times(self) -> np.ndarray:
        """Wszystkie chwile nieciaglosci ciagu: zaplony, wezly profilu, wygaszenia."""
        fired = self._t_ign[np.isfinite(self._t_ign)]
        if fired.size == 0:
            return np.zeros(0)
        return np.unique((fired[:, None] + self.profile.knots[None, :]).ravel())

    def next_switch_after(self, t: float) -> float:
        st = self.switch_times()
        st = st[st > t + T_EPS]
        return float(st[0]) if st.size else float("inf")

    def begin_segment(self, t0: float, t1: float) -> None:
        """
        Ustala, ktore silniki pracuja w segmencie [t0, t1] — po srodku
        segmentu, bo jego granice sa dokladnie na zaplonach/wygaszeniach.
        Maska jest stala w calym segmencie: krok RK45 konczacy sie na
        wygaszeniu dalej widzi ciag (a nie skok do zera na koncu kroku).
        """
        self._active = self.effector.active_mask(0.5 * (t0 + t1), self._t_ign)

    def end_segments(self) -> None:
        """Powrot do maski liczonej z czasu (poza solverem segmentowym)."""
        self._active = None

    def segment_max_step(self, max_step: float) -> float:
        """Przy nieprostokatnym profilu ograniczamy krok, zeby rozdzielic ksztalt."""
        if self._active is not None and np.any(self._active) and not self.profile.is_rectangular:
            return min(max_step, self.profile.burn_time / 20.0)
        return max_step

    # ---- Interfejs dla ForceModel6DOF --------------------------------- #
    def compute(self, fs: FlightState, xa: Optional[np.ndarray] = None) -> ControlWrench:
        if not self.enabled:
            return ControlWrench()
        return self.effector.wrench(self._t_ign, fs, active=self._active)

    def actuator_derivatives(self, fs: FlightState,
                             xa: Optional[np.ndarray] = None) -> np.ndarray:
        return np.zeros(0)

    # ---- Postprocessing ------------------------------------------------- #
    def thrust_history(self, t: np.ndarray) -> np.ndarray:
        """Ciag kazdego silnika [N] wzdluz osi czasu, ksztalt (len(t), n)."""
        t = np.asarray(t, dtype=float)
        out = np.zeros((t.size, self.n_engines))
        for k, tk in enumerate(t):
            out[k] = self.effector.thrusts(float(tk), self._t_ign)
        return out


# ============================================================================
# Budowanie z konfiguracji YAML
# ============================================================================

def build_reaction_jet_system(rj_cfg, body_diameter: Optional[float] = None,
                              commander: Optional[JetCommander] = None) -> ReactionJetSystem:
    """
    rj_cfg: ReactionJetsConfig (datcom_io/config_reader.py) albo RocketConfig
    (wtedy bierzemy cfg.reaction_jets i srednice z cfg.body).
    """
    if hasattr(rj_cfg, "reaction_jets"):           # podano caly RocketConfig
        if body_diameter is None:
            body_diameter = rj_cfg.body.diameter
        rj_cfg = rj_cfg.reaction_jets
    if rj_cfg is None:
        raise ValueError("Konfiguracja nie zawiera sekcji reaction_jets")

    radius = rj_cfg.radius
    if radius is None:
        if body_diameter is None:
            raise ValueError("reaction jets: podaj radius albo srednice korpusu")
        radius = 0.5 * float(body_diameter)

    eng = rj_cfg.engine
    layout = JetLayout(
        rows=[JetRow(x=r.x, n_engines=r.n_engines, phase_deg=r.phase_deg)
              for r in rj_cfg.rows],
        radius=radius,
        tilt_axial_deg=eng.tilt_axial_deg,
        tilt_tangential_deg=eng.tilt_tangential_deg,
    )
    profile = JetThrustProfile(eng.thrust, eng.burn_time, eng.profile)
    return ReactionJetSystem(
        ReactionJetEffector(layout, profile),
        commander=commander,
        ignition_delay=eng.ignition_delay,
        control_rate_hz=rj_cfg.control_rate_hz,
    )
