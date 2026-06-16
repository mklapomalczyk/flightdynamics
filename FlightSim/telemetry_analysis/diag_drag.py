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


def smooth_gps_derivative(signal, t_arr, window_s=0.6):
    """
    Pochodna sygnalu GPS (schodkowego) przez srednia kroczaca.

    GPS aktualizuje sie z ~4-10 Hz wiec sygnal jest schodkowy przy 250 Hz
    telemetrii. Uzywamy sredniej kroczacej nad oknem ~0.6s (150 probek)
    przed roznickowaniem — to wyglądza skoki i daje stabilna pochodna.
    """
    dt = np.median(np.diff(t_arr))
    dt = dt if dt > 0 else 0.004
    win = max(3, int(round(window_s / dt)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    smoothed = np.convolve(signal, kernel, mode='same')

    # napraw krawedzie (convolve ze stala -> efekt brzegowy)
    half = win // 2
    for i in range(half):
        smoothed[i]      = np.mean(signal[:2*i+1]) if i > 0 else signal[0]
        smoothed[-i-1]   = np.mean(signal[-2*i-2:]) if i > 0 else signal[-1]

    # gradient centralny na wygladzonym sygnale
    dt_safe = np.where(np.diff(t_arr) > 0, np.diff(t_arr), dt)
    t_safe  = np.concatenate([[t_arr[0]], t_arr[0] + np.cumsum(dt_safe)])
    return np.gradient(smoothed, t_safe)


def cd_on(tel, mask, m_coast, S, acc_scale=None):
    """
    Cd z fazy balistycznej — metoda czysto GPS.

    Kolumna GPS 'Predkosc lotu' zawiera pozioma predkosc Vh (ground speed 2D).
    Predkosc pionowa Vz = dh/dt z wysokosci GPS.
    Predkosc calkowita: V_total = sqrt(Vh^2 + Vz^2).
    Dynamika pozioma (bez ciagu):
        m * dVh/dt = -D * (Vh/V_total)
    stad:
        D = -m * dVh/dt * V_total / Vh
        Cd = D / (q * S)  gdzie q = 0.5 * rho * V_total^2

    Nie uzywa IMU — brak bledu od wirowania rakiety / biasu akcelerometru.
    """
    t_m = tel.time[mask]
    Vh  = tel.vel_onboard[mask]      # pozioma predkosc GPS [m/s]
    h   = tel.alt_onboard[mask]      # wysokosc GPS (MSL) [m]

    # Predkosc pionowa z pochodnej wysokosci GPS
    Vz = smooth_gps_derivative(h, t_m, window_s=0.5)

    # Calkowita predkosc i kat toru
    V_total = np.sqrt(Vh**2 + Vz**2)
    sin_g   = np.where(V_total > 1.0, Vz / V_total, 0.0)
    gamma   = np.arcsin(np.clip(sin_g, -1.0, 1.0))

    # Atmosfera i cisnienie dynamiczne
    rho, a_snd, _, _ = isa(h)
    q  = 0.5 * rho * V_total**2
    Ma = V_total / np.maximum(a_snd, 1.0)

    # Pochodna poziomej predkosci GPS
    dVh_dt = smooth_gps_derivative(Vh, t_m, window_s=0.5)

    # Sila oporu z rownania poziomego: D = -m * dVh/dt * V_total / Vh
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.where(Vh > 1.0, -m_coast * dVh_dt * V_total / Vh, np.nan)
        Cd = np.where(q > 50.0, D / (q * S), np.nan)

    # Weryfikacja: Cd z rownania calkowitego (powinno sie zgadzac)
    dVtot_dt = smooth_gps_derivative(V_total, t_m, window_s=0.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        a_drag_tot = -dVtot_dt - G0 * sin_g
        Cd_check   = np.where(q > 50.0, m_coast * a_drag_tot / (q * S), np.nan)

    return dict(t=t_m, Vh=Vh, Vz=Vz, V_total=V_total, h=h, M=Ma, q=q,
                gamma=gamma, sin_g=sin_g,
                Cd=Cd, Cd_check=Cd_check, dVh_dt=dVh_dt)


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
    Analiza Cd z fazy zniżania — metoda GPS.

    GPS daje pozioma predkosc Vh i wysokosc h.
    Predkosc pionowa Vz = dh/dt, predkosc calkowita V_total = sqrt(Vh^2+Vz^2).
    Cd wyznaczane z rownania poziomego bez IMU.
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
          f"V_GPS={tel.vel_onboard[i_apo]:.1f}m/s")
    print(f"  Koniec   t={t[i_end]:.3f}s  (ostatnia probka)")

    n   = len(t)
    # Faza ballistyczna: od apogeum do konca, ale pomijamy ostatnie sekundy
    # gdzie moze byc wdrozenie spadochronu (anomalie predkosci)
    des = (np.arange(n) >= i_apo) & (np.arange(n) <= i_end)

    if np.sum(des) < 20:
        print("  [BLAD] Za malo probek w fazie znizania"); return None

    D = cd_on(tel, des, m_coast, S)

    # Odrzuc punkty przed stabilizacja (pierwsze 2s po apogeum)
    t_apo = t[i_apo]
    valid = (D['t'] > t_apo + 2.0) & np.isfinite(D['Cd']) & (D['q'] > 100)

    # Wykryj koniec fazy balistycznej: skoki predkosci (spadochron) => dVh/dt > 10
    # Szukamy pierwszego momentu gdzie dVh/dt > 10 m/s^2 (hamowanie spadochronem)
    chute_mask = np.abs(D['dVh_dt']) > 10.0
    if np.any(chute_mask[valid]):
        i_chute = np.where(valid & chute_mask)[0][0]
        t_chute = D['t'][i_chute]
        valid = valid & (D['t'] < t_chute)
        print(f"  Spadochron wykryty ok. t={t_chute:.1f}s — obcinanie danych")

    if np.sum(valid) < 10:
        print("  [BLAD] Za malo waznych probek Cd"); return None

    # --- tabela Mach ---
    edges = np.arange(0.20, 0.80, 0.04)
    Cd_v   = D['Cd'][valid]
    Ma_v   = D['M'][valid]
    bins   = bin_stats(Ma_v, Cd_v, edges)
    bins_check = bin_stats(Ma_v, D['Cd_check'][valid], edges)

    print(f"\n  Cd z metody GPS (Vh-only) vs weryfikacja (V_total):")
    print(f"  {'Mach':>6} {'Cd_Vh':>8} {'Cd_Vtot':>9} {'sd':>7} {'n':>5}")
    bins_dict  = {round(b[0],3): b for b in bins}
    binsck_dict = {round(b[0],3): b for b in bins_check}
    for c in sorted(bins_dict):
        r  = bins_dict[c]
        ck = binsck_dict.get(c)
        ck_str = f"{ck[1]:9.3f}" if ck else "      ---"
        print(f"  {c:6.2f} {r[1]:8.3f} {ck_str} {r[2]:7.3f} {r[3]:5d}")

    ok_Cd = np.isfinite(Cd_v)
    Ma_range = (Ma_v[ok_Cd].min(), Ma_v[ok_Cd].max())
    gamma_med = np.degrees(np.nanmedian(D['gamma'][valid]))
    print(f"\n  Zniżanie: Cd_med={np.nanmedian(Cd_v[ok_Cd]):.3f}  "
          f"Ma=[{Ma_range[0]:.2f},{Ma_range[1]:.2f}]  "
          f"gamma_med={gamma_med:.1f}°")
    print(f"  V_total_max={D['V_total'][valid].max():.1f}  "
          f"V_total_min={D['V_total'][valid].min():.1f} m/s")

    # ---------- wykresy ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"diag_drag (GPS) — Lot {flight_no}  "
                 f"m_coast={m_coast:.2f}kg  S={S*1e4:.2f}cm²",
                 fontweight="bold")

    # 1. Cd vs Mach
    ax0 = axes[0, 0]
    ax0.scatter(D['M'][valid], D['Cd'][valid], s=6, c='tab:blue',
                alpha=0.5, label="Cd (metoda Vh)")
    ax0.scatter(D['M'][valid], D['Cd_check'][valid], s=4, c='tab:orange',
                alpha=0.3, marker='x', label="Cd (weryfikacja V_total)")
    if bins:
        Ma_b  = [b[0] for b in bins]
        Cd_b  = [b[1] for b in bins]
        Cd_sd = [b[2] for b in bins]
        ax0.errorbar(Ma_b, Cd_b, yerr=Cd_sd, fmt='ro-', ms=6, lw=1.5,
                     capsize=3, label="mediana ± sd (koszyki)")
    ax0.axhline(0, color='k', lw=0.5)
    ax0.set_xlabel("Mach (V_total)"); ax0.set_ylabel("Cd")
    ax0.set_ylim(-0.1, 1.2)
    ax0.set_title("Cd vs Mach — faza zniżania (GPS)")
    ax0.legend(fontsize=7); ax0.grid(alpha=0.3)

    # 2. Predkosci vs czas
    ax1 = axes[0, 1]
    ax1.plot(D['t'], D['Vh'],      'b-',  lw=1,   label="Vh (GPS)")
    ax1.plot(D['t'], D['V_total'], 'r-',  lw=1.2, label="V_total = sqrt(Vh²+Vz²)")
    ax1.plot(D['t'], np.abs(D['Vz']), 'g--', lw=0.8, label="|Vz| = |dh/dt|")
    ax1.axvline(t_apo, color='m', ls='--', lw=1, label="apogeum")
    if 't_chute' in dir():
        ax1.axvline(t_chute, color='k', ls=':', lw=1, label="spadochron")
    ax1.set_xlabel("Czas [s]"); ax1.set_ylabel("V [m/s]")
    ax1.set_title("Predkosci GPS vs czas")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # 3. gamma i Cd vs czas
    ax2 = axes[1, 0]
    ax2.plot(D['t'][valid], D['Cd'][valid], 'b-', lw=1, label="Cd (GPS)")
    ax2_r = ax2.twinx()
    ax2_r.plot(D['t'][valid], np.degrees(D['gamma'][valid]),
               '--', lw=0.8, c='tab:green', alpha=0.8, label="gamma [°]")
    ax2_r.set_ylabel("gamma [°]", color='tab:green')
    ax2.set_xlabel("Czas [s]"); ax2.set_ylabel("Cd")
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_title("Cd i gamma vs czas")
    ax2.legend(fontsize=7, loc='upper left'); ax2.grid(alpha=0.3)

    # 4. h i V vs czas (pelny lot)
    ax3 = axes[1, 1]
    ax3b = ax3.twinx()
    ax3.plot(t, tel.alt_onboard, 'b-', lw=1, label="h GPS")
    ax3b.plot(t, tel.vel_onboard, 'g-', lw=1, alpha=0.7, label="Vh GPS")
    ax3.axvline(t[i_apo], color='m', ls='--', lw=1.2, label="apogeum")
    ax3.axvline(t[i_bo],  color='k', ls='--', lw=1,   label="burnout")
    ax3.set_xlabel("Czas [s]"); ax3.set_ylabel("h [m]", color='b')
    ax3b.set_ylabel("Vh [m/s]", color='g')
    ax3.set_title("Trajektoria (fiolet=apo, czarny=burnout)")
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {out_png}")
    return D


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
