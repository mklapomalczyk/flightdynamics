"""
diag_drag.py
============
Ekstrakcja Cd(Mach) z fazy balistycznej ZNIŻANIA telemetrii ARTEMIDA.

Rownanie ruchu w coastcie (T=0) wzdluz osi ciala X:
    m * acc_x_kal * G0 = -D - m * g * sin(gamma)
    D = m * (-acc_x_kal * G0 - G0 * sin(gamma))
    Cd = D / (q * S)

gamma wyznaczane z rownania ruchu wzdluz wektora predkosci:
    g * sin(gamma) = acc_x_kal * G0 - dV/dt
gdzie dV/dt z V_GPS (predkosc Dopplera, stabilna w coastcie).

Bias akcelerometru:
    Jesli jest staly bias b [g], to:
        D_meas = D_true - m * b * G0
        Cd_meas = Cd_true - m*b*G0 / (q*S) = Cd_true + C/q
    gdzie C = -m*b*G0/S.
    Estymujemy b z regresji: Cd_meas = Cd_true + C/q
    (Cd_true traktujemy jako stale w waskym oknie Mach).
    Fizycznie: przy q->inf bias zanika, przy malym q dominuje.

Faza analizy: TYLKO ZNIŻANIE (od apogeum do konca danych).
Faza wznosząca odrzucona — V_GPS niestabilne po burnoutcie.
Koniec fazy balistycznej = koniec danych (nie flaga spadochronu).

Uzycie:
    python diag_drag.py 13
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from telemetry_parser import parse_telemetry, resolve_data_file, get_data_dir
from imu_reconstruction import detect_ignition

G0    = 9.80665
R_AIR = 287.05
D_CAL = 0.070
S     = np.pi * D_CAL**2 / 4.0


# --------------------------------------------------------------------------
def load_config(fno, base=None):
    """Wczytuje konfiguracje lotu z configs.txt."""
    if base is None:
        base = get_data_dir()
    for p in (Path(base) / "configs.txt", Path("configs.txt")):
        if p.exists():
            break
    else:
        return None
    lines = open(p, encoding="utf-8", errors="replace").readlines()
    header = [h.strip() for h in lines[0].strip().split("\t")]
    for line in lines[1:]:
        parts = [x.strip() for x in line.strip().split("\t")]
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        try:
            if int(row.get("flight no", -1)) != fno:
                continue
        except ValueError:
            continue
        def f(k, default=0.0):
            try: return float(row.get(k, default))
            except ValueError: return default
        m_empty      = f("m_empty")
        m_full       = f("m_full")
        m_rocket     = f("m_rocket")
        m_propellant = m_full - m_empty
        m_coast      = m_rocket - m_propellant
        return dict(
            cant=f("cant angle"), azimuth=f("azimuth"),
            elevation=f("elevation"), head=row.get("head configuration", "?"),
            m_empty=m_empty, m_full=m_full, m_rocket=m_rocket,
            m_propellant=m_propellant, m_coast=m_coast,
        )
    return None


def isa(h):
    """ISA troposfera. h [m ASL] -> (rho, a_sound, T, p)."""
    T   = 288.15 - 0.0065 * h
    p   = 101325.0 * (T / 288.15) ** 5.2561
    rho = p / (R_AIR * T)
    a   = np.sqrt(1.4 * R_AIR * T)
    return rho, a, T, p


def calib_acc_scale(tel, t_ign):
    """Skala akcelerometru z |a|=1g w spoczynku (t < t_ign-0.5)."""
    rest = tel.time < (t_ign - 0.5)
    if np.sum(rest) < 5:
        rest = tel.time < t_ign
    a = np.sqrt(tel.acc_x[rest]**2 + tel.acc_y[rest]**2 + tel.acc_z[rest]**2)
    g_rest = np.nanmean(a)
    return 1.0 / g_rest, g_rest


def detect_burnout(tel, i_ign, t_max_burn=8.0,
                   press_thresh=2.0, press_drop_frac=0.05):
    """
    Burnout z cisnienia komory (fallback: szczyt acc_x).
    Nie uzywa V_GPS ani pierwszego acc_x<0.
    """
    t   = tel.time
    N   = len(t)
    i_end  = int(np.argmin(np.abs(t - (t[i_ign] + t_max_burn))))
    window = np.arange(i_ign, min(i_end, N))

    pc = tel.press_cham
    if pc is not None:
        pc_win = pc[window]
        pc_max = np.nanmax(pc_win)
        if pc_max > press_thresh:
            from numpy.lib.stride_tricks import sliding_window_view
            w = min(10, len(pc_win) // 2)
            if w >= 2:
                sm     = np.median(sliding_window_view(pc_win, w), axis=1)
                offset = w // 2
                drop   = np.where(sm < press_drop_frac * pc_max)[0]
                if len(drop):
                    return int(window[drop[0] + offset])

    ax_win = tel.acc_x[window]
    i_peak = int(np.argmax(ax_win))
    return min(int(window[i_peak]) + 3, N - 1)


def detect_events(tel, t_ign):
    """
    Zwraca indeksy: zaplon, burnout, apogeum, koniec_danych.
    Koniec fazy balistycznej = ostatnia probka (bez flagi spadochronu).
    """
    t     = tel.time
    i_ign = int(np.argmin(np.abs(t - t_ign)))
    i_bo  = detect_burnout(tel, i_ign)
    i_apo = int(np.nanargmax(tel.alt_onboard))
    i_end = len(t) - 1
    return i_ign, i_bo, i_apo, i_end


def safe_dVdt(V, t_arr, smooth_window=25):
    """
    dV/dt z gradientem centralnym, odporny na powtarzajace sie timestampy
    (dt=0 z zamrozonych GPS). Punkty z dt=0 interpolowane liniowo.
    """
    dt = np.diff(t_arr)
    # napraw dt=0: zastap medianą kroczacą
    dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.004
    dt_safe = np.where(dt <= 0, dt_med, dt)
    # odbuduj os czasu bez skokow (tylko do gradientu)
    t_safe = np.concatenate([[t_arr[0]], t_arr[0] + np.cumsum(dt_safe)])

    dV = np.gradient(V, t_safe)

    if smooth_window >= 2:
        kernel = np.ones(smooth_window) / smooth_window
        dV = np.convolve(dV, kernel, mode='same')
    return dV


def cd_on(tel, mask, m_coast, S, acc_scale):
    """
    Cd z fazy balistycznej (tylko zniżanie).

    gamma z rownania ruchu wzdluz toru: g*sin(gamma) = ax*G0 - dV/dt
    a_drag = dV/dt - ax*G0  (= -g*sin(gamma), zawiera bias jako staly offset)
    Cd_raw = m * a_drag / (q * S)   — zawiera blad biasu proporcjonalny 1/q
    """
    t_mask = tel.time[mask]
    V  = tel.vel_onboard[mask]
    h  = tel.alt_onboard[mask]
    rho, a_snd, _, _ = isa(h)
    q  = 0.5 * rho * V**2

    ax = tel.acc_x[mask] * acc_scale   # [g]
    ay = tel.acc_y[mask] * acc_scale
    az = tel.acc_z[mask] * acc_scale

    dV_dt = safe_dVdt(V, t_mask, smooth_window=25)

    # g*sin(gamma) = ax*G0 - dV_dt  =>  sin(gamma) = (ax*G0 - dV_dt)/G0
    sin_g = np.clip((ax * G0 - dV_dt) / G0, -1.0, 1.0)
    gamma = np.arcsin(sin_g)

    # D/m = -ax*G0 - G0*sin(gamma) = dV_dt - 2*ax*G0 + ax*G0 = ...
    # jawna forma:
    a_drag = -ax * G0 - G0 * sin_g   # [m/s²], dodatnia = hamowanie
    a_mag  = np.sqrt(ax**2 + ay**2 + az**2) * G0

    with np.errstate(divide='ignore', invalid='ignore'):
        Cd_axial = np.where(q > 0.5, (m_coast * a_drag) / (q * S), np.nan)
        Cd_mag   = np.where(q > 0.5, (m_coast * a_mag)  / (q * S), np.nan)

    M   = V / a_snd
    lat = np.sqrt(ay**2 + az**2) * G0
    return dict(t=t_mask, V=V, h=h, M=M, q=q,
                gamma=gamma, sin_g=sin_g,
                Cd_axial=Cd_axial, Cd_mag=Cd_mag,
                a_drag=a_drag, a_mag=a_mag, lat=lat)


def estimate_bias(D, m_coast, S, M_lo=0.28, M_hi=0.42):
    """
    Estymacja biasu akcelerometru z regresji Cd_meas = Cd0 + C/q.

    Model: w waskim pasmie Mach gdzie Cd_true ≈ const:
        Cd_meas = Cd_true + C/q,  C = -m*b*G0/S
        => b = -C*S/(m*G0)  [g]

    Dobre okno: wąskie Mach (Cd płaskie) + duży zakres q (bias obserwowalny).
    R² > 0.3 i |b| < 0.1g — dobra estymacja.
    R² < 0.1 — bias nieobserwowalny w tym oknie.
    """
    sel = (D['M'] >= M_lo) & (D['M'] < M_hi) & np.isfinite(D['Cd_axial'])
    if np.sum(sel) < 20:
        return None

    q_sel  = D['q'][sel]
    Cd_sel = D['Cd_axial'][sel]

    q_range = np.max(q_sel) / max(np.min(q_sel), 1e-6)
    if q_range < 1.3:
        return None   # za maly zakres q

    inv_q = 1.0 / q_sel
    A     = np.vstack([np.ones_like(inv_q), inv_q]).T
    coef, *_ = np.linalg.lstsq(A, Cd_sel, rcond=None)
    Cd0, C = coef

    pred   = A @ coef
    ss_res = np.sum((Cd_sel - pred)**2)
    ss_tot = np.sum((Cd_sel - np.mean(Cd_sel))**2)
    R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sigma  = np.std(Cd_sel - pred)
    b_est  = -C * S / (m_coast * G0)

    return dict(b_est=b_est, C=C, Cd0=Cd0, R2=R2, sigma=sigma,
                n=int(np.sum(sel)), M_lo=M_lo, M_hi=M_hi,
                q_range=q_range)


def apply_bias_correction(D, b_est, m_coast, S):
    """
    Koryguje Cd o oszacowany bias: Cd_corr = Cd_meas + m*b*G0/(q*S).
    Zwraca nowa tablice Cd_corrected.
    """
    correction = m_coast * b_est * G0 / (D['q'] * S)
    return D['Cd_axial'] + correction


def bin_stats(M, Cd, edges):
    """Mediana i odchylenie Cd w koszykach Mach."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (M >= lo) & (M < hi) & np.isfinite(Cd)
        if np.sum(sel) >= 5:
            out.append((0.5*(lo+hi), np.median(Cd[sel]),
                        np.std(Cd[sel]), int(np.sum(sel))))
    return out


# --------------------------------------------------------------------------
def analyze(tel, cfg, S, flight_no, out_png):
    """
    Analiza Cd z fazy zniżania. Estymuje bias akcelerometru.
    """
    m_coast      = cfg["m_coast"]
    m_rocket     = cfg["m_rocket"]
    m_propellant = cfg["m_propellant"]

    t_ign = detect_ignition(tel)
    acc_scale, g_rest = calib_acc_scale(tel, t_ign)
    i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign)
    t = tel.time

    print(f"\n=== LOT {flight_no} ===")
    print(f"  Masy: m_rocket={m_rocket:.3f}  m_propellant={m_propellant:.3f}  "
          f"m_coast={m_coast:.3f} kg")
    print(f"  Calib: |a|_rest={g_rest:.4f}g  scale={acc_scale:.4f}")
    print(f"  Zaplon   t={t[i_ign]:.3f}s")
    print(f"  Burnout  t={t[i_bo]:.3f}s")
    print(f"  Apogeum  t={t[i_apo]:.3f}s  h={tel.alt_onboard[i_apo]:.1f}m  "
          f"V={tel.vel_onboard[i_apo]:.1f}m/s")
    print(f"  Koniec   t={t[i_end]:.3f}s  (ostatnia probka)")

    n   = len(t)
    des = (np.arange(n) >= i_apo) & (np.arange(n) <= i_end)

    if np.sum(des) < 20:
        print("  [BLAD] Za malo probek w fazie znizania"); return None, None

    D = cd_on(tel, des, m_coast, S, acc_scale)

    # --- estymacja biasu ---
    bias = estimate_bias(D, m_coast, S, M_lo=0.25, M_hi=0.50)
    if bias:
        print(f"\n  Bias: b={bias['b_est']:+.4f}g  Cd0={bias['Cd0']:.3f}  "
              f"C={bias['C']:.2f}  R²={bias['R2']:.3f}  "
              f"σ={bias['sigma']:.3f}  n={bias['n']}  "
              f"M=[{bias['M_lo']},{bias['M_hi']}]")
        Cd_corr = apply_bias_correction(D, bias['b_est'], m_coast, S)
    else:
        print("  Bias: za malo danych do estymacji")
        Cd_corr = D['Cd_axial'].copy()

    # --- tabela Mach ---
    edges = np.arange(0.18, 0.68, 0.04)
    print(f"\n  {'Mach':>6} {'Cd_raw':>8} {'Cd_corr':>8} {'sd_raw':>7} {'n':>5}")
    bins_raw  = {round(b[0],3): b for b in bin_stats(D['M'], D['Cd_axial'], edges)}
    bins_corr = {round(b[0],3): b for b in bin_stats(D['M'], Cd_corr,       edges)}
    for c in sorted(bins_raw):
        r = bins_raw[c]; cr = bins_corr.get(c)
        cd_c = f"{cr[1]:8.3f}" if cr else "     ---"
        print(f"  {c:6.2f} {r[1]:8.3f} {cd_c} {r[2]:7.3f} {r[3]:5d}")

    ok = np.isfinite(D['Cd_axial'])
    ok_c = np.isfinite(Cd_corr)
    print(f"\n  Zniżanie: Cd_raw={np.nanmedian(D['Cd_axial'][ok]):.3f}  "
          f"Cd_corr={np.nanmedian(Cd_corr[ok_c]):.3f}  "
          f"M=[{D['M'][ok].min():.2f},{D['M'][ok].max():.2f}]  "
          f"gamma_med={np.degrees(np.nanmedian(D['gamma'])):.1f}°  "
          f"lat/drag={np.nanmedian(D['lat']/(np.abs(D['a_drag'])+1e-9)):.2f}")

    # ---------- wykresy ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"diag_drag — Lot {flight_no}  "
                 f"m_coast={m_coast:.2f}kg  m_rocket={m_rocket:.2f}kg  "
                 f"S={S*1e4:.2f}cm²",
                 fontweight="bold")

    # 1. Cd vs Mach (raw i corrected)
    ax0 = axes[0, 0]
    ax0.scatter(D['M'], D['Cd_axial'], s=5, c='tab:red',    alpha=0.3, label="Cd_raw")
    ax0.scatter(D['M'], Cd_corr,       s=5, c='tab:blue',   alpha=0.4, label="Cd_corr")
    ax0.scatter(D['M'], D['Cd_mag'],   s=4, c='tab:orange', alpha=0.2, marker='x', label="|a| mag")
    ax0.axhline(0, color='k', lw=0.5)
    ax0.set_xlabel("Mach"); ax0.set_ylabel("Cd"); ax0.set_ylim(-0.2, 1.5)
    ax0.set_title("Cd vs Mach (zniżanie)"); ax0.legend(fontsize=7); ax0.grid(alpha=0.3)

    # 2. Cd vs 1/q — liniowość biasu
    ax1 = axes[0, 1]
    inv_q = 1.0 / np.where(D['q'] > 0.1, D['q'], np.nan)
    ax1.scatter(inv_q, D['Cd_axial'], s=4, c='tab:red', alpha=0.3, label="Cd_raw")
    if bias:
        q_line = np.linspace(np.nanmin(D['q']), np.nanmax(D['q']), 100)
        cd_line = bias['Cd0'] + bias['C'] / q_line
        ax1.plot(1/q_line, cd_line, 'b-', lw=1.5,
                 label=f"fit: Cd0={bias['Cd0']:.3f} b={bias['b_est']:+.4f}g")
    ax1.set_xlabel("1/q [m²/N]"); ax1.set_ylabel("Cd_raw")
    ax1.set_title("Bias test: Cd vs 1/q  (liniowy = stały bias)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 3. gamma i a_drag vs czas
    ax2 = axes[1, 0]
    ax2.plot(D['t'], D['a_drag'], '-', lw=0.8, c='tab:red',  label="a_drag")
    ax2.plot(D['t'], D['lat'],    '-', lw=0.6, c='gray', alpha=0.6, label="|lat|")
    ax2_r = ax2.twinx()
    ax2_r.plot(D['t'], np.degrees(D['gamma']), '--', lw=0.8,
               c='tab:green', alpha=0.8, label="gamma")
    ax2_r.set_ylabel("gamma [°]", color='tab:green')
    ax2.set_xlabel("Czas [s]"); ax2.set_ylabel("a [m/s²]")
    ax2.set_title("a_drag i gamma vs czas"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # 4. h i V vs czas z zaznaczonym apogeum
    ax3 = axes[1, 1]
    ax3b = ax3.twinx()
    ax3.plot(t, tel.alt_onboard, 'b-', lw=1, label="h GPS")
    ax3b.plot(t, tel.vel_onboard, 'g-', lw=1, label="V GPS")
    ax3.axvline(t[i_apo], color='m', ls='--', lw=1.2, label="apogeum")
    ax3.axvline(t[i_bo],  color='k', ls='--', lw=1,   label="burnout")
    ax3.set_xlabel("Czas [s]"); ax3.set_ylabel("h [m]", color='b')
    ax3b.set_ylabel("V [m/s]", color='g')
    ax3.set_title("Trajektoria (fiolet=apo, czarny=burnout)")
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {out_png}")
    return D, bias


def main():
    flight_no = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    fname = resolve_data_file(f"ARTEMIDA_{flight_no}_LOT.txt")
    tel   = parse_telemetry(fname, verbose=False)
    cfg   = load_config(flight_no)
    if cfg is None:
        print(f"Brak configu dla lotu {flight_no}")
        return
    out = str(get_data_dir() / "results" / f"drag_flight_{flight_no}.png")
    analyze(tel, cfg, S, flight_no, out)


if __name__ == "__main__":
    main()
