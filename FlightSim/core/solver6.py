"""
core/solver6.py
===============
Wrapper solve_ivp dla symulacji 6DOF.

Konwencja Z launch w dół:
  Rakieta startuje z z=0, leci w górę → z ujemne.
  Lądowanie: z wraca do 0. Ground event: z >= z_ground.

Wektor stanu ma 14 składowych gdy launcher.L_rail > 0
(dodatkowy element rail_dist na końcu).
"""

import numpy as np
from scipy.integrate import solve_ivp

from core.state6 import (State6DOF, SimResult6DOF,
                         IDX_U, IDX_V, IDX_W, IDX_P, IDX_QR, IDX_R)


def run_simulation_6dof(
    force_model,
    initial_state:  State6DOF,
    t_max:          float = 60.0,
    dt_output:      float = 0.01,
    method:         str   = "RK45",
    rtol:           float = 1e-6,
    atol:           float = 1e-8,
    z_ground:       float = 1.0,
    max_step:       float = 0.05,
) -> SimResult6DOF:
    """
    Uruchamia symulację 6DOF.

    Parameters
    ----------
    z_ground : float
        Poziom gruntu w konwencji Z-dół [m].
        Domyślnie 1.0 — lekko powyżej zera, żeby nie wyzwalać na starcie.
    """
    _dual_spin_active = getattr(force_model, "_dual_spin", None) is not None
    x0 = initial_state.to_numpy(include_rail=True, include_pfwd=_dual_spin_active)
    _n_ctrl = int(getattr(force_model, "_n_ctrl_states", 0))
    if _n_ctrl > 0:
        ctrl_x0 = force_model.control.actuator.initial_state()
        x0 = np.append(x0, ctrl_x0)
    if _dual_spin_active:
        rtol = max(rtol, 1e-4)
        atol = max(atol, 1e-6)

    t_span = (0.0, t_max)
    t_eval = np.arange(0.0, t_max, dt_output)

    # ---- Zdarzenie 1: lądowanie ---------------------------------------- #
    def ground_event(t, x):
        return x[2] - z_ground   # IDX_Z = 2

    ground_event.terminal  = True
    ground_event.direction = +1

    # ---- Zdarzenie 2: burnout ------------------------------------------ #
    t_ignition = getattr(force_model.mass_model, 't_ignition', 0.0)
    t_burn     = getattr(force_model.mass_model, 't_burn',     np.inf)
    t_burnout  = t_ignition + t_burn

    def burnout_event(t, x):
        return t - t_burnout

    burnout_event.terminal  = False
    burnout_event.direction = 1

    # ---- Zdarzenie 3: blow-up prędkości translacyjnej ------------------ #
    # V_max jest osiągana na końcu fazy napędowej — po burnout V tylko maleje
    # (lub rośnie nieznacznie w swobodnym spadku, nigdy nie przekroczy V_max)
    # Śledzimy V_max_seen i flagujemy gdy V > V_max_seen * 1.05 po burnout
    _state = {"V_max_seen": 0.0, "after_burnout": False,
              "omega_qr_counter": 0.0, "dt_last": 0.0, "t_last": 0.0}

    def blowup_event(t, x):
        u, v, w = x[IDX_U], x[IDX_V], x[IDX_W]
        V = float(np.sqrt(u**2 + v**2 + w**2))
        if not _state["after_burnout"]:
            _state["V_max_seen"] = max(_state["V_max_seen"], V)
            if t > t_burnout:
                _state["after_burnout"] = True
            return 1.0  # nie wyzwalaj podczas napędu
        # Po burnout: wyzwól gdy V > 1.05 * V_max_seen
        return _state["V_max_seen"] * 1.05 - V

    blowup_event.terminal  = True
    blowup_event.direction = -1   # gdy V spada przez próg (ujemny gradient)

    # ---- Zdarzenie 4: tumbling — sqrt(q²+r²) > 500°/s przez 5s -------- #
    # Używamy zmiennej stanu zamiast event (event nie ma pamięci)
    # Tumbling wykrywany przez wrapper pochodnych

    # ---- Zdarzenie 3: opuszczenie szyny -------------------------------- #
    events = [ground_event, burnout_event, blowup_event]

    L_rail = getattr(getattr(force_model, "launcher", None), "L_rail", 0.0)
    if L_rail > 0.0:
        def rail_exit_event(t, x):
            return x[13] - L_rail

        rail_exit_event.terminal  = False
        rail_exit_event.direction = +1
        events.append(rail_exit_event)

    # ---- Wrapper z projekcją wiązań szyny ------------------------------ #
    # solve_ivp (RK45) całkuje etapami pośrednimi — bez projekcji stanu
    # po każdym kroku, v/w/p/qr/r mogą narastać numerycznie na szynie.
    # Rozwiązanie: wrapper fun który przed zwróceniem pochodnych zeruje
    # składowe poprzeczne gdy na szynie.

    # Próg tumblingu: sqrt(q²+r²) > 45°/s przez 3s
    # 45°/s to ~5x typowe omega_qr stabilnej rakiety 36mm (~7°/s)
    _TUMBLE_THRESHOLD_RAD = np.radians(45.)
    _TUMBLE_DURATION      = 3.0
    _tumble_state = {"t_above": 0.0, "triggered": False}

    def derivatives_projected(t, x):
        """Pochodne z projekcją wiązań szyny i detekcją tumblingu."""
        rail_dist = float(x[13]) if L_rail > 0.0 else 0.0
        on_rail   = (L_rail > 0.0) and (rail_dist < L_rail)
        if on_rail:
            x = x.copy()
            x[IDX_V]  = 0.0
            x[IDX_W]  = 0.0
            x[IDX_P]  = 0.0
            x[IDX_QR] = 0.0
            x[IDX_R]  = 0.0

        # Detekcja tumblingu — akumuluj czas powyżej progu
        if not on_rail and not _tumble_state["triggered"]:
            omega_qr = float(np.sqrt(x[IDX_QR]**2 + x[IDX_R]**2))
            dt = t - _state["t_last"] if _state["t_last"] > 0.0 else 0.0
            _state["t_last"] = t
            if omega_qr > _TUMBLE_THRESHOLD_RAD:
                _tumble_state["t_above"] += dt
            else:
                _tumble_state["t_above"] = 0.0  # reset gdy ponizej progu
            if _tumble_state["t_above"] >= _TUMBLE_DURATION:
                _tumble_state["triggered"] = True
        else:
            _state["t_last"] = t

        return force_model.derivatives(t, x)

    # ---- Integracja ----------------------------------------------------- #
    result = solve_ivp(
        fun          = derivatives_projected,
        t_span       = t_span,
        y0           = x0,
        method       = method,
        t_eval       = t_eval,
        events       = events,
        rtol         = rtol,
        atol         = atol,
        max_step     = max_step,
        dense_output = False,
    )

    # Określ status symulacji
    if not result.success and result.status != 1:
        raise RuntimeError(f"Solver 6DOF nie zbiegł: {result.message}")

    if _tumble_state["triggered"]:
        sim_status = "tumbling"
        print(f"[solver] Tumbling wykryty — sqrt(q²+r²) > 500°/s przez {_TUMBLE_DURATION}s")
    elif len(result.t_events[2]) > 0:   # blowup_event
        sim_status = "blowup"
        print(f"[solver] Blow-up prędkości wykryty w t={result.t_events[2][0]:.2f}s")
    else:
        sim_status = "ok"

    # ---- Projekcja wiązań szyny na zapisanych punktach t_eval ----------- #
    # solve_ivp interpoluje punkty t_eval z gęstego wyjścia — mogą zawierać
    # małe wartości v/w/qr. Zerujemy je post-hoc dla czystości wyników.
    y = result.y.copy()
    if L_rail > 0.0:
        rail_dist_arr = y[13, :]
        on_rail_mask  = rail_dist_arr < L_rail
        for idx in [IDX_V, IDX_W, IDX_P, IDX_QR, IDX_R]:
            y[idx, on_rail_mask] = 0.0

    # SimResult6DOF używa pierwszych 13 wierszy
    n_state = 15 if _dual_spin_active else 13
    return SimResult6DOF.from_raw(result.t, y[:n_state], status=sim_status)
