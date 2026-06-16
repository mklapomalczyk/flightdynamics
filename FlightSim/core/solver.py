"""
core/solver.py
==============
Wrapper na scipy.integrate.solve_ivp.

Odpowiada za:
  - Uruchomienie integracji ODE z zadanym modelem sił
  - Warunki stopu (uderzenie w ziemię, max czas)
  - Zebranie wyników do SimResult

Solver jest świadomie oddzielony od ForceModel — można podstawić
inny integrator (RK23, DOP853, LSODA) bez zmiany fizyki.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional

from core.state import State3DOF, SimResult, STATE_SIZE

t_history = []
y_history = []

def _ground_hit(t: float, x: np.ndarray, *args) -> float:
    """Zdarzenie stopu: rakieta uderza w ziemię (z_pos = 0) po starcie."""
    return x[1]  # z_pos

_ground_hit.terminal  = True    # przerwij całkowanie
_ground_hit.direction = -1      # tylko gdy z_pos spada (nie przy startowym z=0)

def plot_callback(t, y, state):
    t_data.append(t)
    y_data.append(y[0]) # Zakładamy śledzenie pierwszej zmiennej
    print(t_data, y_data)

def run_simulation(
    force_model,
    initial_state:  State3DOF,
    t_max:          float = 60.0,
    dt_output:      float = 0.01,
    method:         str   = "RK45",
    rtol:           float = 1e-5,
    atol:           float = 1e-5,
    z_min:          float = -1.0,
) -> SimResult:
    """
    Uruchamia symulację dynamiki lotu 3DOF.

    Parameters
    ----------
    force_model : ForceModel
        Agregator sił — musi implementować metodę `derivatives(t, x)`.
    initial_state : State3DOF
        Stan początkowy rakiety.
    t_max : float
        Maksymalny czas symulacji [s].
    dt_output : float
        Krok wyjściowy wyników [s]. Nie wpływa na krok integracji.
    method : str
        Metoda integracji scipy: "RK45" (domyślna), "RK23", "DOP853", "LSODA".
    rtol, atol : float
        Tolerancje względna i bezwzględna integratora.
    z_min : float
        Minimalna wysokość [m] przed wyzwoleniem zdarzenia stopu.
        Domyślnie -1 m (odrobinę poniżej zera, by nie przerywać na starcie).

    Returns
    -------
    SimResult
        Wyniki symulacji.
    """
    x0 = initial_state.to_numpy()
    t_span = (0.0, t_max)
    t_eval = np.arange(0.0, t_max, dt_output)

    # Zdarzenie stopu — dopasowanie z_min
    def ground_event(t, x):
        return x[1] - z_min

    ground_event.terminal  = True
    ground_event.direction = +1
    
    def plot_event(t, y):
        # Aktualizacja danych do wykresu
        t_history.append(t)
        y_history.append(y[0]) # Przykład dla pierwszej zmiennej
        
        # print(t, y)
        # print(t_history, y_history)
        
        # # Aktualizacja linii na wykresie
        # line.set_data(t_history, y_history)
        # ax.relim()
        # ax.autoscale_view()
        # plt.pause(0.001)
        
        return 1 # Zwraca wartość różną od zera, by event "trwał"

    plot_event.terminal = False  # Nie przerywaj symulacji
    plot_event.direction = 0     # Reaguj na każdą zmianę
    
    result = solve_ivp(
        fun       = force_model.derivatives,
        t_span    = t_span,
        y0        = x0,
        method    = method,
        t_eval    = t_eval,
        events    = [ground_event, plot_event],
        rtol      = rtol,
        atol      = atol,
        dense_output = False,
    )

    if not result.success and result.status != 1:
        # status=1 → zdarzenie stopu (normalne), status=0 → koniec t_max (ok)
        raise RuntimeError(
            f"Solver nie zbiegł: {result.message}"
        )

    return SimResult.from_raw(result.t, result.y)
