"""
diag_survey.py
==============
Przeglad wszystkich lotow ARTEMIDA pod katem ekstrakcji Cd z faz balistycznych.
Masy czytane z configs.txt (m_empty, m_full = silnik; m_rocket = cala rakieta startowa).
m_coast = m_rocket - (m_full - m_empty)

Uzycie:
    python diag_survey.py                # loty 13-21 z field_test_data/
    python diag_survey.py 13 16 18       # wybrane loty
    python diag_survey.py --dir /sciezka # inny katalog z danymi
"""

import sys
import numpy as np
from pathlib import Path

from telemetry_parser import parse_telemetry, get_data_dir
from imu_reconstruction import detect_ignition
import diag_drag as dd

FLIGHTS_DEFAULT = [13, 14, 15, 16, 17, 18, 19, 20, 21]
D_CAL  = 0.070
S      = np.pi * D_CAL**2 / 4.0
SAT    = 1990.0   # prog nasycenia gyro_X [st/s]


def find_file(fno, base):
    for p in (Path(base) / f"ARTEMIDA_{fno}_LOT.txt",
              Path(f"ARTEMIDA_{fno}_LOT.txt")):
        if p.exists():
            return p
    return None


def survey_one(fno, base):
    print(f"\n{'='*68}\n=== LOT {fno} ===")

    fpath = find_file(fno, base)
    if fpath is None:
        print(f"  [pominieto] brak pliku ARTEMIDA_{fno}_LOT.txt")
        return

    cfg = dd.load_config(fno, base)
    if cfg is None:
        print(f"  [pominieto] brak wpisu w configs.txt"); return

    if cfg["m_rocket"] == 0.0:
        print(f"  [UWAGA] m_rocket=0 w configs.txt — uzupelnij kolumne m_rocket"); return

    try:
        tel = parse_telemetry(fpath, verbose=False)
    except Exception as e:
        print(f"  [BLAD parsowania] {e}"); return

    t = tel.time
    N = len(t)
    rate = 1.0 / tel.dt_mean if tel.dt_mean else 0

    rudder  = tel.raw_columns.get("rudder")
    steered = rudder is not None and np.any(np.abs(np.nan_to_num(rudder)) > 0.1)

    # kalibracja + zdarzenia
    t_ign = detect_ignition(tel)
    acc_scale, g_rest = dd.calib_acc_scale(tel, t_ign)
    rest = tel.time < (t_ign - 0.5)
    axial_rest  = np.nanmean(tel.acc_x[rest]) if np.sum(rest) > 3 else np.nan
    el_implied  = np.degrees(np.arcsin(np.clip(axial_rest / max(g_rest, 1e-6), -1, 1)))

    i_ign, i_bo, i_apo, i_end = dd.detect_events(tel, t_ign)

    def mach_at(i):
        _, a_s, _, _ = dd.isa(tel.alt_onboard[i])
        return tel.vel_onboard[i] / a_s

    Vmax  = np.nanmax(tel.vel_onboard)
    iVmax = int(np.nanargmax(tel.vel_onboard))
    Mmax  = mach_at(iVmax)
    V_apo = tel.vel_onboard[i_apo]
    M_apo = mach_at(i_apo)

    # nasycenie rolla (caly lot od zaplonu do konca)
    fl      = (np.arange(N) >= i_ign)
    gx      = tel.gyro_x
    sat_frac= np.mean(np.abs(gx[fl]) >= SAT) if np.sum(fl) else 0.0
    ax_roll = tel.raw_columns.get("gyro_ax")
    ax_max  = np.nanmax(np.abs(ax_roll[fl])) if (ax_roll is not None and np.sum(fl)) else np.nan

    # --- druk ---
    print(f"  cant={cfg['cant']}°  el={cfg['elevation']}°  az={cfg['azimuth']}°  "
          f"head={cfg['head']}  steered={'TAK' if steered else 'nie'}")
    print(f"  masy: m_rocket={cfg['m_rocket']:.3f}  m_propellant={cfg['m_propellant']:.3f}  "
          f"m_coast={cfg['m_coast']:.3f} kg")
    print(f"  parse: N={N} {rate:.0f}Hz  t=[{t[0]:.1f}..{t[-1]:.1f}]s  kol={len(tel.raw_columns)}")
    print(f"  zdarzenia: ign={t[i_ign]:.2f}s  burnout={t[i_bo]:.2f}s(V={tel.vel_onboard[i_bo]:.0f})  "
          f"apo={t[i_apo]:.1f}s/{tel.alt_onboard[i_apo]:.0f}m  "
          f"koniec={t[-1]:.1f}s")
    print(f"  mach:  Vmax={Vmax:.0f}m/s(M{Mmax:.2f})  V_apo={V_apo:.0f}m/s(M{M_apo:.2f})")
    print(f"  calib: |a|_rest={g_rest:.3f}g  scale={acc_scale:.3f}  "
          f"axial_rest={axial_rest:.3f}g (el~{el_implied:.0f}°)")
    print(f"  roll:  gyroX nasycony {sat_frac*100:.0f}% lotu  AX|max|={ax_max:.0f} st/s")

    # tylko zniżanie (apogeum -> koniec danych)
    des = (np.arange(N) >= i_apo)

    if np.sum(des) < 20:
        print("  [drag] za malo probek w fazie zniżania"); return

    m = cfg["m_coast"]
    D = dd.cd_on(tel, des, m, S, acc_scale)

    def med(X, lo, hi):
        sel = (X['M'] >= lo) & (X['M'] < hi) & np.isfinite(X['Cd_axial'])
        return (np.median(X['Cd_axial'][sel]), int(np.sum(sel))) if np.sum(sel) >= 5 else (np.nan, 0)

    # bias
    bias = dd.estimate_bias(D, m, S, M_lo=0.25, M_hi=0.50)
    if bias:
        Cd_corr = dd.apply_bias_correction(D, bias['b_est'], m, S)
        b_str = (f"bias b={bias['b_est']:+.4f}g  Cd0={bias['Cd0']:.3f}  "
                 f"R²={bias['R2']:.3f}  n={bias['n']}")
    else:
        Cd_corr = D['Cd_axial'].copy()
        b_str = "bias: za malo danych"

    mid_raw,  _ = med(D, 0.28, 0.55)
    mid_corr, _ = (np.nanmedian(Cd_corr[(D['M'] >= 0.28) & (D['M'] < 0.55) &
                                         np.isfinite(Cd_corr)]), 0)
    low_raw,  _ = med(D, 0.18, 0.28)

    ok = np.isfinite(D['Cd_axial'])
    lat_r = np.nanmedian(D['lat'] / np.abs(D['a_drag'] + 1e-9))
    gamma_med = np.degrees(np.nanmedian(D['gamma']))

    print(f"  drag:  Cd_raw[mid]={mid_raw:.3f}  Cd_corr[mid]={mid_corr:.3f}  "
          f"low(<0.28)={low_raw:.3f}  M=[{D['M'][ok].min():.2f},{D['M'][ok].max():.2f}]")
    print(f"  {b_str}")
    print(f"  jakosc: lat/drag={lat_r:.2f}(koning)  gamma_med={gamma_med:.1f}°")


def main():
    args = [a for a in sys.argv[1:]]
    base = get_data_dir()
    if "--dir" in args:
        i = args.index("--dir"); base = args[i+1]; del args[i:i+2]
    flights = [int(a) for a in args if a.isdigit()] or FLIGHTS_DEFAULT

    print(f"Katalog: {base}")
    print(f"Loty:    {flights}")
    print(f"S={S*1e4:.2f} cm²  (d={D_CAL*1000:.0f}mm)")
    for fno in flights:
        survey_one(fno, base)
    print(f"\n{'='*68}\nGotowe.")


if __name__ == "__main__":
    main()
