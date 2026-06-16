"""
gps_fusion.py
=============
Fuzja IMU + GPS metoda filtru komplementarnego.

IMU daje wysoka czestotliwosc i dokladnosc krotkoterminowa, ale dryfuje.
GPS daje pozycje bezwzgledna (bez dryfu) ale z niska czestotliwoscia
i ograniczeniami przy duzych przyspieszeniach (faza napedowa).

Filtr komplementarny laczy oba:
  pos_fused = pos_imu - alpha * (pos_imu - pos_gps)

gdzie alpha to waga korekcji GPS (wieksza = bardziej ufamy GPS).
Korekcja jest stosowana tylko gdy GPS jest wiarygodny (faza balistyczna,
male przyspieszenia osiowe).

Uklad: ENU lokalny, poczatek w punkcie startu.
"""

import numpy as np
from dataclasses import dataclass


# Promien Ziemi do konwersji lat/lon -> metry (lokalnie plaska aproksymacja)
R_EARTH = 6378137.0   # m (WGS84)


def latlon_to_enu(lat, lon, lat0, lon0):
    """
    Konwersja lat/lon [stopnie] -> lokalne ENU [m] wzgledem (lat0, lon0).
    Aproksymacja plaskiej Ziemi (dobra dla zasiegu < kilkadziesiat km).
    """
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east  = dlon * R_EARTH * np.cos(np.radians(lat0))
    north = dlat * R_EARTH
    return east, north


@dataclass
class FusedTrajectory:
    time:  np.ndarray
    pos:   np.ndarray   # (N,3) ENU [m]
    vel:   np.ndarray   # (N,3) ENU [m/s]
    speed: np.ndarray
    alt:   np.ndarray
    gps_valid: np.ndarray  # (N,) bool — gdzie GPS byl uzyty
    n_gps_rejected: int = 0  # liczba odfiltrowanych falszywych punktow GPS


def filter_gps_glitches(t_gps, e_gps, n_gps, u_gps,
                        v_max=400.0, alt_jump_max=150.0, freeze_max=0.5):
    """
    Wykrywa i odrzuca fałszywe wskazania GPS.

    Trzy rodzaje bledow (progi ustalone z danych ARTEMIDA):
    1. Przeskok pozycji — implikowana predkosc pozioma > v_max [m/s]
       (rakieta nie przekracza ~410 m/s, wiec skok dajacy wiecej = blad).
    2. Przeskok wysokosci > alt_jump_max [m] miedzy aktualizacjami GPS.
    3. Zamrozenie — pozycja niezmienna dluzej niz freeze_max [s]
       (utrata fixa; GPS aktualizuje sie ~10 Hz wiec >0.5s = problem).

    Zwraca maske bool (True = punkt wiarygodny) na siatce t_gps.
    Odrzucone punkty beda pominiete — fuzja propaguje IMU miedzy
    wiarygodnymi punktami.
    """
    n = len(t_gps)
    valid = np.ones(n, dtype=bool)

    # Punkty gdzie GPS faktycznie sie zmienia (telemetria 250 Hz, GPS 10 Hz)
    pos = np.column_stack([e_gps, n_gps, u_gps])
    changed = np.zeros(n, dtype=bool)
    changed[0] = True
    for k in range(1, n):
        if (abs(e_gps[k]-e_gps[k-1]) > 1e-6 or
            abs(n_gps[k]-n_gps[k-1]) > 1e-6 or
            abs(u_gps[k]-u_gps[k-1]) > 1e-6):
            changed[k] = True

    change_idx = np.where(changed)[0]

    # 1+2. Skoki pozycji/wysokosci miedzy kolejnymi aktualizacjami GPS
    last_good = change_idx[0]
    for idx in change_idx[1:]:
        dt = t_gps[idx] - t_gps[last_good]
        if dt <= 0:
            continue
        de = e_gps[idx] - e_gps[last_good]
        dn = n_gps[idx] - n_gps[last_good]
        da = u_gps[idx] - u_gps[last_good]
        v_horiz = np.sqrt(de**2 + dn**2) / dt
        # Odrzuc jesli skok niefizyczny
        if v_horiz > v_max or abs(da) > alt_jump_max:
            valid[idx] = False
            # nie aktualizuj last_good — porownuj nastepny do ostatniego dobrego
        else:
            last_good = idx

    # 3. Zamrozenia — dlugie ciagi bez zmiany pozycji
    freeze_samples = 0
    dt_med = np.median(np.diff(t_gps)) if n > 1 else 0.004
    freeze_limit = int(freeze_max / dt_med) if dt_med > 0 else 125
    for k in range(1, n):
        if not changed[k]:
            freeze_samples += 1
            if freeze_samples > freeze_limit:
                valid[k] = False
        else:
            freeze_samples = 0

    return valid


def fuse_imu_gps(traj, tel, t_ign,
                 alpha_pos=0.01, alpha_vel=0.005,
                 acc_gate=3.0):
    """
    Filtr komplementarny IMU + GPS.

    traj  — Trajectory z rekonstrukcji IMU
    tel   — Telemetry (zawiera lat/lon/alt_onboard z GPS)
    t_ign — czas zaplonu (poczatek okna)
    alpha_pos — waga korekcji pozycji GPS (0..1). Wieksza = silniejsza
                korekcja dryfu IMU. 0.15 zapobiega odjazdowi w dlugiej
                fazie opadania, zachowujac dynamike IMU w fazie napedowej.
    alpha_vel — waga korekcji predkosci
    acc_gate  — prog przyspieszenia [g]; powyzej GPS nieufny (bramkowanie)

    Zwraca FusedTrajectory.
    """
    t = traj.time
    n = len(t)

    # GPS w oknie rekonstrukcji
    mask_gps = (tel.time >= t[0]) & (tel.time <= t[-1])
    t_gps   = tel.time[mask_gps]
    lat_gps = tel.lat[mask_gps]
    lon_gps = tel.lon[mask_gps]
    alt_gps = tel.alt_onboard[mask_gps]

    # Punkt odniesienia = pierwsza pozycja GPS
    lat0, lon0 = lat_gps[0], lon_gps[0]
    alt0 = alt_gps[0]

    e_gps, n_gps = latlon_to_enu(lat_gps, lon_gps, lat0, lon0)
    u_gps = alt_gps - alt0

    # Odfiltruj falszywe wskazania GPS (przeskoki, zamrozenia)
    gps_ok = filter_gps_glitches(t_gps, e_gps, n_gps, u_gps)
    n_rejected = int(np.sum(~gps_ok))

    # Zostaw tylko wiarygodne punkty GPS do interpolacji.
    # Tam gdzie GPS odrzucony, interpolacja liniowa "przeskoczy" dziure —
    # czyli IMU efektywnie propaguje miedzy dobrymi punktami.
    t_good = t_gps[gps_ok]
    e_good = e_gps[gps_ok]
    n_good = n_gps[gps_ok]
    u_good = u_gps[gps_ok]

    # Interpoluj GPS (tylko dobre punkty) na siatke czasu IMU
    e_gps_i = np.interp(t, t_good, e_good)
    n_gps_i = np.interp(t, t_good, n_good)
    u_gps_i = np.interp(t, t_good, u_good)
    pos_gps = np.column_stack([e_gps_i, n_gps_i, u_gps_i])

    # Przyspieszenie osiowe (do bramkowania GPS) — z telemetrii
    acc_axial = np.interp(t, tel.time, np.abs(tel.acc_x))

    # Predkosc GPS — liczona z odfiltrowanych, wiarygodnych punktow GPS.
    # Bierzemy punkty gdzie GPS sie zmienia I jest oznaczony jako dobry.
    good_change = gps_ok & np.concatenate([[True], (np.abs(np.diff(u_gps)) > 1e-6)])
    gc_idx = np.where(good_change)[0]
    if len(gc_idx) > 2:
        t_gps_pts = t_gps[gc_idx]
        e_pts  = e_gps[gc_idx]
        n_pts  = n_gps[gc_idx]
        up_pts = u_gps[gc_idx]
        # Predkosc na wiarygodnych punktach zmian GPS
        ve = np.gradient(e_pts,  t_gps_pts)
        vn = np.gradient(n_pts,  t_gps_pts)
        vu = np.gradient(up_pts, t_gps_pts)
        vel_gps = np.column_stack([
            np.interp(t, t_gps_pts, ve),
            np.interp(t, t_gps_pts, vn),
            np.interp(t, t_gps_pts, vu),
        ])
    else:
        vel_gps = np.zeros((n, 3))

    # Filtr komplementarny — korekcja iteracyjna
    pos = traj.pos.copy()
    vel = traj.vel.copy()
    gps_valid = np.zeros(n, dtype=bool)

    # Estymowany dryf pozycji i predkosci IMU — korygowany przez GPS.
    pos_err = np.zeros(3)
    vel_err = np.zeros(3)

    # Predkosc GPS jest wiarygodna tylko w fazie balistycznej (male przysp.)
    # i tylko gdy |v_gps| jest fizycznie sensowna (GPS gubi sie przy duzej dynamice)
    v_gps_mag = np.linalg.norm(vel_gps, axis=1)
    v_gps_physical = v_gps_mag < 350.0   # cap fizyczny dla tej rakiety

    for k in range(1, n):
        valid_pos = acc_axial[k] < acc_gate
        valid_vel = valid_pos and v_gps_physical[k]
        gps_valid[k] = valid_pos

        pos_imu_corrected = traj.pos[k] - pos_err
        vel_imu_corrected = traj.vel[k] - vel_err

        if valid_pos:
            innov_pos = pos_imu_corrected - pos_gps[k]
            pos_err = pos_err + alpha_pos * innov_pos
        if valid_vel:
            innov_vel = vel_imu_corrected - vel_gps[k]
            vel_err = vel_err + alpha_vel * innov_vel

        pos[k] = traj.pos[k] - pos_err
        vel[k] = traj.vel[k] - vel_err

    speed = np.linalg.norm(vel, axis=1)

    return FusedTrajectory(
        time=t, pos=pos, vel=vel, speed=speed, alt=pos[:, 2],
        gps_valid=gps_valid, n_gps_rejected=n_rejected,
    )
