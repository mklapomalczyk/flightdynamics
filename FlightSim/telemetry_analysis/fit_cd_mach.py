"""
fit_cd_mach.py
==============
Identyfikacja parametryczna krzywej Cd(Mach) przez dopasowanie
trajektorii balistycznej do danych GPS (faza zniżania).

Idea:
  - Cd(Ma) opisane jako funkcja liniowo-odcinkowa na siatce Ma
    od Ma_min do Ma_max co dMa (domyslnie 0.20-0.50, co 0.025 = 13 punktow)
  - Model: rzut ciałem m_coast z oporem aerodynamicznym w 2D (V, h)
    z warunkow poczatkowych z GPS kilka sekund po apogeum
  - Kryterium: J = sum(w_h*(h_model-h_GPS)^2) + sum(w_v*(V_model-V_GPS)^2)
               + lambda * sum((Cd[i+1]-Cd[i])^2)  [regularyzacja]
  - Optymalizacja: Nelder-Mead (scipy), wielokrotny start z losowych punktow

Uzycie:
    python fit_cd_mach.py 13
    python fit_cd_mach.py 13 --n_starts 20 --lam 0.5
    python fit_cd_mach.py 13 14 17 18   # wiele lotow naraz (wspolna krzywa)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from telemetry_parser import parse_telemetry, get_data_dir
from imu_reconstruction import detect_ignition
from diag_drag import load_config, isa, calib_acc_scale, detect_events

G0    = 9.80665
D_CAL = 0.070
S     = np.pi * D_CAL**2 / 4.0

# --- siatka Mach ---
MA_MIN   = 0.20
MA_MAX   = 0.50
DMA      = 0.025
MA_NODES = np.arange(MA_MIN, MA_MAX + DMA*0.5, DMA)   # ~13 punktow
N_NODES  = len(MA_NODES)

CD_LO, CD_HI = 0.10, 0.80   # dopuszczalny zakres Cd w wezlach


# --------------------------------------------------------------------------
# Interpolacja liniowo-odcinkowa Cd(Ma)
# --------------------------------------------------------------------------
def cd_interp(Ma, cd_nodes):
    """Liniowa interpolacja Cd z wezlow na siatce MA_NODES."""
    return np.interp(Ma, MA_NODES, cd_nodes,
                     left=cd_nodes[0], right=cd_nodes[-1])


# --------------------------------------------------------------------------
# Model balistyczny 2D (plaszczyzna pionowa)
# --------------------------------------------------------------------------
def ballistic_rhs(t, state, m, S, cd_nodes):
    """
    Prawa strona rownan ruchu w 2D (plaszczyznie pionowej).
    state = [h, V_h, V_v]
      h   — wysokosc [m]
      V_h — predkosc pozioma [m/s]
      V_v — predkosc pionowa [m/s] (+ = w gore)
    """
    h, Vh, Vv = state
    h = max(h, 0.0)

    rho, a_snd, _, _ = isa(h)
    V   = np.sqrt(Vh**2 + Vv**2)
    Ma  = V / max(a_snd, 1.0)
    Cd  = cd_interp(Ma, cd_nodes)
    q   = 0.5 * rho * V**2
    D   = Cd * q * S          # sila oporu [N]

    # przyspieszenia
    if V > 0.01:
        ax = -D * Vh / (V * m)
        az = -D * Vv / (V * m) - G0
    else:
        ax = 0.0
        az = -G0

    return [Vv, ax, az]


def simulate(t_eval, state0, m, S, cd_nodes):
    """
    Propaguje trajektorie balistyczna od t_eval[0] do t_eval[-1].
    Zwraca (h, V) na siatce t_eval. Jesli rakieta uderzy w ziemie,
    wypelnia reszte NaN.
    """
    def hit_ground(t, y, *args): return y[0]
    hit_ground.terminal  = True
    hit_ground.direction = -1

    sol = solve_ivp(
        ballistic_rhs, [t_eval[0], t_eval[-1]], state0,
        args=(m, S, cd_nodes),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-5, atol=1e-6,
        events=hit_ground,
        dense_output=False,
    )

    n  = len(t_eval)
    h_out = np.full(n, np.nan)
    V_out = np.full(n, np.nan)

    if sol.y.shape[1] > 0:
        nn = sol.y.shape[1]
        h_out[:nn] = sol.y[0]
        Vh = sol.y[1]; Vv = sol.y[2]
        V_out[:nn]  = np.sqrt(Vh**2 + Vv**2)

    return h_out, V_out


# --------------------------------------------------------------------------
# Przygotowanie danych GPS dla jednego lotu
# --------------------------------------------------------------------------
def prepare_flight(tel, cfg, t_offset_apo=2.0, min_points=30):
    """
    Wyciaga dane GPS fazy zniżania.

    t_offset_apo: ile sekund po apogeum brac jako punkt startowy modelu
    Vv0 estymowane z dh/dt na dluzszym oknie (5s) dla stabilnosci.
    """
    t    = tel.time
    h    = tel.alt_onboard
    V    = tel.vel_onboard
    N    = len(t)

    t_ign = detect_ignition(tel)
    _, _, i_apo, _ = detect_events(tel, t_ign)
    t_apo = t[i_apo]

    t_start = t_apo + t_offset_apo
    i_start = int(np.argmin(np.abs(t - t_start)))
    if i_start >= N - min_points:
        i_start = i_apo + 5

    h0 = float(h[i_start])
    V0 = float(V[i_start])

    step  = 5
    i_ref = np.arange(i_start, N, step)
    valid = np.isfinite(h[i_ref]) & np.isfinite(V[i_ref]) & (V[i_ref] > 1.0)
    i_ref = i_ref[valid]

    if len(i_ref) < min_points:
        return None

    return dict(
        t0=float(t[i_start]), h0=h0, V0=V0,
        t_ref=t[i_ref], h_ref=h[i_ref], V_ref=V[i_ref],
        m_coast=cfg["m_coast"],
        flight_no=None,
        t_apo=t_apo, h_apo=float(h[i_apo]),
    )


# --------------------------------------------------------------------------
# Funkcja celu
# --------------------------------------------------------------------------
def cost(params, flights_data, w_h=1.0, w_v=1.0, lam=0.1):
    """
    Funkcja celu — znormalizowana przez wariancje h i V GPS.

    J = sum_loty [ mean((dh/sigma_h)^2) + w_v*mean((dV/sigma_V)^2) ]
        + lam * sum(dCd^2)

    Normalizacja przez sigma zapewnia ze bledy h i V maja porownywalne
    wagi niezaleznie od ich skal bezwzglednych.
    """
    n_f    = len(flights_data)
    cd_raw = params[:N_NODES]
    g0_raw = params[N_NODES:]

    cd_nodes = CD_LO + (CD_HI - CD_LO) * _sigmoid(cd_raw)
    gamma0s  = np.radians(-89 + 88 * _sigmoid(g0_raw))

    J = 0.0
    for k, fd in enumerate(flights_data):
        t_ref = fd['t_ref']
        h_ref = fd['h_ref']
        V_ref = fd['V_ref']
        m     = fd['m_coast']
        V0    = fd['V0']

        gamma0 = float(gamma0s[k])
        Vv0    = V0 * np.sin(gamma0)
        Vh0    = V0 * np.cos(gamma0)

        state0 = [fd['h0'], Vh0, Vv0]
        t_sim  = t_ref - t_ref[0]

        h_sim, V_sim = simulate(t_sim, state0, m, S, cd_nodes)

        ok = np.isfinite(h_sim) & np.isfinite(V_sim)
        if np.sum(ok) < 5:
            J += 1e6
            continue

        dh = h_sim[ok] - h_ref[ok]
        dV = V_sim[ok] - V_ref[ok]

        # normalizuj przez odchylenie standardowe danych referencyjnych
        sig_h = max(np.std(h_ref[ok]), 1.0)
        sig_v = max(np.std(V_ref[ok]), 0.1)

        J += w_h * np.mean((dh / sig_h)**2) + w_v * np.mean((dV / sig_v)**2)

    dCd = np.diff(cd_nodes)
    J  += lam * np.sum(dCd**2)

    return J


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_inv(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------
# Optymalizacja
# --------------------------------------------------------------------------
def fit_cd(flights_data, n_starts=10, lam=0.1, w_h=1.0, w_v=1.0,
           cd_init=None, verbose=True):
    """
    Wielokrotny Nelder-Mead z losowych startow.
    Wektor optymalizowany: [cd_nodes (N_NODES), gamma0_per_lot (n_lots)]
    gamma0 = kat toru w punkcie startowym — traktowany jako nieznany.
    Zwraca (best_cd, best_gamma0s, best_J, results).
    """
    n_f = len(flights_data)
    if cd_init is None:
        cd_init = np.full(N_NODES, 0.35)
    # gamma0 init: -30 stopni dla wszystkich
    g0_init = np.full(n_f, np.radians(-30))

    def pack(cd, g0):
        cd_raw = _sigmoid_inv((cd - CD_LO) / (CD_HI - CD_LO))
        # gamma0 w (-89°, -1°): mapuj przez sigmoid
        g0_norm = (np.degrees(g0) + 89) / 88   # -> (0,1)
        g0_raw  = _sigmoid_inv(np.clip(g0_norm, 1e-6, 1-1e-6))
        return np.concatenate([cd_raw, g0_raw])

    def unpack(params):
        cd  = CD_LO + (CD_HI - CD_LO) * _sigmoid(params[:N_NODES])
        g0  = np.radians(-89 + 88 * _sigmoid(params[N_NODES:]))
        return cd, g0

    best_J    = np.inf
    best_cd   = cd_init.copy()
    best_g0   = g0_init.copy()
    results   = []

    rng = np.random.default_rng(42)
    starts_cd = [cd_init.copy()]
    starts_g0 = [g0_init.copy()]
    for _ in range(n_starts - 1):
        cd_r = rng.uniform(CD_LO+0.05, CD_HI-0.05, N_NODES)
        cd_r = np.convolve(cd_r, [0.25,0.5,0.25], mode='same')
        cd_r = np.clip(cd_r, CD_LO+0.01, CD_HI-0.01)
        g0_r = np.radians(rng.uniform(-70, -10, n_f))
        starts_cd.append(cd_r)
        starts_g0.append(g0_r)

    for k in range(n_starts):
        x0  = pack(starts_cd[k], starts_g0[k])
        res = minimize(
            cost, x0,
            args=(flights_data, w_h, w_v, lam),
            method='Nelder-Mead',
            options=dict(maxiter=8000, xatol=1e-4, fatol=1e-2,
                         adaptive=True),
        )
        cd_opt, g0_opt = unpack(res.x)
        J_opt = res.fun
        results.append((J_opt, cd_opt.copy(), g0_opt.copy()))
        if verbose:
            g0_deg = np.degrees(g0_opt)
            print(f"  start {k+1:2d}/{n_starts}: J={J_opt:.1f}  "
                  f"Cd=[{cd_opt.min():.3f},{cd_opt.max():.3f}]  "
                  f"gamma0={g0_deg}  "
                  f"{'*' if J_opt < best_J else ''}")
        if J_opt < best_J:
            best_J  = J_opt
            best_cd = cd_opt.copy()
            best_g0 = g0_opt.copy()

    return best_cd, best_g0, best_J, results


# --------------------------------------------------------------------------
# Glowna analiza jednego lub wielu lotow
# --------------------------------------------------------------------------
def run(flight_nos, base=None, n_starts=15, lam=0.1, w_h=1.0, w_v=0.3,
        t_offset_apo=3.0, out_dir=None):

    if base is None:
        base = get_data_dir()
    if out_dir is None:
        out_dir = Path(base) / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flights_data = []
    tels         = {}

    for fno in flight_nos:
        fpath = Path(base) / f"ARTEMIDA_{fno}_LOT.txt"
        if not fpath.exists():
            print(f"[LOT {fno}] brak pliku — pomijam")
            continue
        cfg = load_config(fno, base)
        if cfg is None or cfg["m_rocket"] == 0:
            print(f"[LOT {fno}] brak configu — pomijam")
            continue

        tel = parse_telemetry(fpath, verbose=False)
        fd  = prepare_flight(tel, cfg, t_offset_apo=t_offset_apo)
        if fd is None:
            print(f"[LOT {fno}] za malo danych GPS — pomijam")
            continue

        fd['flight_no'] = fno
        flights_data.append(fd)
        tels[fno] = tel

        print(f"[LOT {fno}] m_coast={cfg['m_coast']:.3f}kg  "
              f"t0={fd['t0']:.1f}s  h0={fd['h0']:.0f}m  "
              f"V0={fd['V0']:.1f}m/s  n_ref={len(fd['t_ref'])}")

    if not flights_data:
        print("Brak danych do analizy.")
        return None

    print(f"\nOptymalizacja Cd(Ma) — {len(flights_data)} lot(ow), "
          f"{n_starts} startow, lambda={lam}")
    print(f"Siatka Ma: {MA_NODES[0]:.3f} .. {MA_NODES[-1]:.3f}  "
          f"({N_NODES} wezlow, dMa={DMA})")

    best_cd, best_g0, best_J, all_res = fit_cd(
        flights_data, n_starts=n_starts, lam=lam, w_h=1.0, w_v=1.0)

    print(f"\nNajlepsze J={best_J:.3f}")
    for k, fd in enumerate(flights_data):
        print(f"  Lot {fd['flight_no']}: gamma0={np.degrees(best_g0[k]):.1f}°")
    print(f"{'Mach':>6} {'Cd':>8}")
    for ma, cd in zip(MA_NODES, best_cd):
        print(f"{ma:6.3f} {cd:8.4f}")

    # ---------- wykresy ----------
    n_rows = 1 + len(flights_data)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(14, 4 + 4*len(flights_data)))
    axes = np.array(axes).reshape(n_rows, 2)

    ax0 = axes[0, 0]
    ax0.plot(MA_NODES, best_cd, 'b-o', lw=2, ms=6, label="Cd dopasowane")
    ax0.fill_between(MA_NODES, CD_LO, CD_HI, alpha=0.07, color='gray',
                     label=f"zakres ({CD_LO}-{CD_HI})")
    ax0.set_xlabel("Mach"); ax0.set_ylabel("Cd")
    ax0.set_title(f"Cd(Ma) — loty {flight_nos}  J={best_J:.2f}")
    ax0.grid(alpha=0.3); ax0.legend(fontsize=8)
    ax0.set_ylim(0, 0.9)

    ax1 = axes[0, 1]
    js = sorted([r[0] for r in all_res])
    ax1.semilogy(range(1, len(js)+1), js, 'ko-', ms=4)
    ax1.axhline(best_J, color='r', ls='--', lw=1)
    ax1.set_xlabel("Start #"); ax1.set_ylabel("J (log)")
    ax1.set_title("Zbieznosc — J per start"); ax1.grid(alpha=0.3)

    t_fine = np.linspace(0, 80, 4000)
    for row, (fd, g0) in enumerate(zip(flights_data, best_g0), start=1):
        fno  = fd['flight_no']
        V0   = fd['V0']
        Vv0  = V0 * np.sin(g0)
        Vh0  = V0 * np.cos(g0)
        state0 = [fd['h0'], Vh0, Vv0]
        h_sim, V_sim = simulate(t_fine, state0, fd['m_coast'], S, best_cd)
        t_abs = t_fine + fd['t0']

        ax_h = axes[row, 0]
        ax_h.plot(fd['t_ref'], fd['h_ref'], 'r.', ms=3, label="GPS")
        ok = np.isfinite(h_sim)
        ax_h.plot(t_abs[ok], h_sim[ok], 'b-', lw=1.5, label="model")
        ax_h.axvline(fd['t0'], color='k', ls=':', lw=1)
        ax_h.set_xlabel("Czas [s]"); ax_h.set_ylabel("h [m]")
        ax_h.set_title(f"Lot {fno} — h  (gamma0={np.degrees(g0):.1f}°)")
        ax_h.legend(fontsize=8); ax_h.grid(alpha=0.3)

        ax_v = axes[row, 1]
        ax_v.plot(fd['t_ref'], fd['V_ref'], 'r.', ms=3, label="GPS")
        ax_v.plot(t_abs[ok], V_sim[ok], 'b-', lw=1.5, label="model")
        ax_v.set_xlabel("Czas [s]"); ax_v.set_ylabel("V [m/s]")
        ax_v.set_title(f"Lot {fno} — V")
        ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3)

    plt.tight_layout()
    tag = "_".join(str(f) for f in flight_nos)
    out_png = out_dir / f"fit_cd_lot{tag}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")

    return best_cd, best_g0


# --------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    flight_nos = []
    n_starts   = 15
    lam        = 0.1

    i = 0
    while i < len(args):
        if args[i] == '--n_starts':
            n_starts = int(args[i+1]); i += 2
        elif args[i] == '--lam':
            lam = float(args[i+1]); i += 2
        else:
            try:
                flight_nos.append(int(args[i]))
            except ValueError:
                pass
            i += 1

    if not flight_nos:
        flight_nos = [13]

    run(flight_nos, n_starts=n_starts, lam=lam)


if __name__ == "__main__":
    main()
