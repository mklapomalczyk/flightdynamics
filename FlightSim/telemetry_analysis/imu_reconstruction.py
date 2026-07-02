"""
imu_reconstruction.py
=====================
Rekonstrukcja trajektorii lotu z danych IMU (strapdown inertial navigation).

Algorytm:
1. Kalibracja — odejmij bias zyroskopu i znormalizuj skale akcelerometru
   na podstawie danych spoczynkowych (rakieta nieruchoma na wyrzutni).
2. Orientacja poczatkowa — z wektora grawitacji w spoczynku.
3. Strapdown INS:
   - calkuj predkosc katowa -> kwaternion orientacji
   - obroc przyspieszenie z ukladu ciala do ukladu ziemi (NED/ENU)
   - odejmij grawitacje
   - calkuj przyspieszenie -> predkosc -> pozycja

Uklad odniesienia: ENU (East-North-Up) lokalny, poczatek w punkcie startu.
Konwencja osi ciala: X podluzna (do przodu), ustalana z danych.
"""

import numpy as np
from dataclasses import dataclass

G0 = 9.80665   # m/s^2


# ---------------------------------------------------------------------------
# Kwaterniony
# ---------------------------------------------------------------------------
def reconstruct_roll_from_ax(tel, t_ignition, sat_threshold=1990.0):
    """
    Rekonstruuje predkosc rolla z kolumny 'Predkosc obrotowa AX'.

    Zwykly zyroskop gyro_X wysyca sie na ±2000 st/s, ale roll rakiety
    siega kilku tys. st/s. Kolumna AX to osobny high-range czujnik
    mierzacy MODUL |roll rate| (zawsze dodatni, w innych jednostkach).

    Procedura:
    1. Kalibracja offsetu: AX w spoczynku ma niezerowa wartosc (~150) —
       odejmujemy srednia spoczynkowa.
    2. Kalibracja skali: dopasowujemy |AX| do |gyro_X| w obszarze gdzie
       gyro_X NIE jest wysycony (automatycznie, mediana stosunku).
    3. Odzyskanie znaku: AX jest bez znaku; znak bierzemy z gyro_X tam
       gdzie niewysycony, a w obszarach wysycenia zachowujemy ostatni
       znany znak (roll rakiety rzadko zmienia kierunek w locie).

    Zwraca: roll_rate [st/s] ze znakiem, tej samej dlugosci co tel.time.
    Jesli brak kolumny AX, zwraca None.
    """
    ax = tel.raw_columns.get("gyro_ax")
    if ax is None:
        return None

    gx = tel.gyro_x

    # 1. Offset spoczynkowy AX
    rest = tel.time < (t_ignition - 0.5)
    if np.sum(rest) < 5:
        rest = tel.time < t_ignition
    ax_offset = np.nanmean(ax[rest]) if np.sum(rest) > 0 else 0.0
    ax_cal = ax - ax_offset

    # 2. Skala — dopasuj |AX| do |gyro_X| gdzie gyro_X niewysycony
    notsat = (np.abs(gx) < sat_threshold) & (np.abs(gx) > 100)
    if np.sum(notsat) > 20:
        scale = np.nanmedian(np.abs(gx[notsat]) / (np.abs(ax_cal[notsat]) + 1e-9))
    else:
        scale = 1.0
    ax_scaled = np.abs(ax_cal) * scale   # wielkosc rolla w st/s

    # 3. Znak z gyro_X (gdzie niewysycony), inaczej ostatni znany
    sign = np.sign(gx)
    sign[np.abs(gx) >= sat_threshold] = 0   # nieznany przy wysyceniu
    sign_filled = sign.copy()
    last = 1.0
    for i in range(len(sign)):
        if sign[i] == 0:
            sign_filled[i] = last
        else:
            last = sign[i]

    roll_rate = ax_scaled * sign_filled
    return roll_rate


def reconstruct_roll_from_ax_const_sign(tel, t_ignition, sat_threshold=1990.0,
                                          sign_ref_t_start=5.0, sign_ref_t_end=8.0):
    """
    Wariant reconstruct_roll_from_ax() ze STALYM znakiem zamiast
    "ostatniego znanego znaku" per-probka.

    Motywacja: |AX_scaled| (modul, PRZED przypisaniem znaku) jest sam w
    sobie GLADKI i fizycznie sensowny od t~1s po zaplonie (rosnie do
    szczytu w okolicy burnout, potem monotonicznie opada, plynnie
    laczac sie z segmentem t>=5s bez nieciaglosci) -- ZWERYFIKOWANE
    numerycznie dla lotu 19. Pozorna erratyczna oscylacja +/- w
    reconstruct_roll_from_ax() to WYLACZNIE artefakt kruchej logiki
    "ostatni znany znak" (gyro_X czesto wysyca sie w tym oknie, a
    krotkie, zaszumione zejscia ponizej progu wysycenia moga wstrzykiwac
    falszywe zmiany znaku). Rakieta ze stalym zaklinowaniem pletw nie
    odwraca kierunku obrotu w trakcie wznoszenia, wiec STALY znak
    (ustalony tam, gdzie gyro_X jest wiarygodny, tj. w segmencie
    [sign_ref_t_start, sign_ref_t_end] od zaplonu) jest fizycznie
    uzasadniony na cala probke.

    Zwraca: roll_rate [st/s] ze stalym znakiem, tej samej dlugosci co
    tel.time. None jesli brak kolumny AX lub nie da sie ustalic znaku
    referencyjnego.
    """
    ax = tel.raw_columns.get("gyro_ax")
    if ax is None:
        return None

    gx = tel.gyro_x
    t_rel = tel.time - t_ignition

    rest = tel.time < (t_ignition - 0.5)
    if np.sum(rest) < 5:
        rest = tel.time < t_ignition
    ax_offset = np.nanmean(ax[rest]) if np.sum(rest) > 0 else 0.0
    ax_cal = ax - ax_offset

    notsat = (np.abs(gx) < sat_threshold) & (np.abs(gx) > 100)
    if np.sum(notsat) > 20:
        scale = np.nanmedian(np.abs(gx[notsat]) / (np.abs(ax_cal[notsat]) + 1e-9))
    else:
        scale = 1.0
    ax_scaled = np.abs(ax_cal) * scale

    ref_mask = (t_rel >= sign_ref_t_start) & (t_rel < sign_ref_t_end)
    if np.sum(ref_mask) < 5:
        return None
    const_sign = float(np.sign(np.nanmedian(gx[ref_mask])))
    if const_sign == 0.0:
        const_sign = 1.0

    return ax_scaled * const_sign


def detect_ignition(tel):
    """
    Wykrywa moment zaplonu (start lotu).

    Kolejnosc preferencji (ustalone empirycznie na danych ARTEMIDA):
    1. FLAGA_LOT — najpewniejsza, ustawiana przez komputer pokladowy w
       momencie wykrycia startu (daje t~0.00s).
    2. acc_X > 8g — fallback gdy brak flagi. Prog 8g jednoznacznie
       odroznia start od drgan na wyrzutni (prog 3-5g lapie drgania).
    3. press_cham > 2bar — ostatecznosc; NIEWIARYGODNE (lapie szum
       cisnienia, daje czasy -2 do -4s przed startem).

    Zwraca czas zaplonu [s].
    """
    # 1. FLAGA LOT
    if tel.flag_flight is not None and np.any(tel.flag_flight > 0):
        idx = int(np.argmax(tel.flag_flight > 0))
        return float(tel.time[idx])
    # 2. Przyspieszenie osiowe > 8g
    if tel.acc_x is not None and np.any(tel.acc_x > 8.0):
        idx = int(np.argmax(tel.acc_x > 8.0))
        return float(tel.time[idx])
    # 3. Cisnienie komory (ostatecznosc)
    if tel.press_cham is not None and np.any(tel.press_cham > 2.0):
        idx = int(np.argmax(tel.press_cham > 2.0))
        return float(tel.time[idx])
    # Brak danych — zacznij od 0
    return 0.0


def quat_mult(q1, q2):
    """Iloczyn kwaternionow Hamiltona (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


def quat_to_dcm(q):
    """Kwaternion -> macierz obrotu (body->world). q=(w,x,y,z)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quat_from_two_vectors(v_from, v_to):
    """Kwaternion obracajacy v_from na v_to."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to   / np.linalg.norm(v_to)
    dot = np.dot(a, b)
    if dot > 0.999999:
        return np.array([1., 0., 0., 0.])
    if dot < -0.999999:
        # 180 stopni — znajdz dowolna os prostopadla
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        return np.array([0., axis[0], axis[1], axis[2]])
    axis = np.cross(a, b)
    w = 1.0 + dot
    q = np.array([w, axis[0], axis[1], axis[2]])
    return quat_normalize(q)


# ---------------------------------------------------------------------------
# Kalibracja
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    gyro_bias: np.ndarray     # [st/s] bias do odjecia (3,)
    acc_scale: float          # globalny mnoznik skali akcelerometru
    g_body_rest: np.ndarray   # [g] kierunek grawitacji w ukladzie ciala (spoczynek)
    n_rest: int


def calibrate(tel, t_rest_end=-1.0):
    """
    Kalibracja na podstawie danych spoczynkowych (t < t_rest_end).

    - Bias zyroskopu: srednia predkosc katowa w spoczynku (czujnik nieruchomy).
    - Skala akcelerometru: w spoczynku |a| powinno = 1g, korygujemy globalnie.
    - Kierunek grawitacji: znormalizowany sredni wektor przyspieszenia.
    """
    mask = tel.time < t_rest_end
    n_rest = int(np.sum(mask))

    gyro_bias = np.array([
        np.mean(tel.gyro_x[mask]),
        np.mean(tel.gyro_y[mask]),
        np.mean(tel.gyro_z[mask]),
    ])

    g_body = np.array([
        np.mean(tel.acc_x[mask]),
        np.mean(tel.acc_y[mask]),
        np.mean(tel.acc_z[mask]),
    ])
    g_mag = np.linalg.norm(g_body)
    acc_scale = 1.0 / g_mag

    return Calibration(
        gyro_bias=gyro_bias,
        acc_scale=acc_scale,
        g_body_rest=g_body / g_mag,
        n_rest=n_rest,
    )


# ---------------------------------------------------------------------------
# Orientacja poczatkowa
# ---------------------------------------------------------------------------
def initial_orientation(cal, azimuth_deg, elevation_deg, use_gravity=True):
    """
    Wyznacza kwaternion orientacji poczatkowej (body->world ENU).

    Dwa skladniki informacji:
    1. Azymut + elewacja okreslaja kierunek osi podluznej (X) w ENU.
    2. Wektor grawitacji zmierzony w spoczynku okresla orientacje wokol
       osi podluznej (roll) oraz koryguje rzeczywisty kat elewacji.

    use_gravity=True: buduje pelna orientacje uwzgledniajaca pomiar g.
    use_gravity=False: tylko z azymut+elewacja (wariant uproszczony).
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Kierunek osi podluznej rakiety w ENU (z azymut+elewacja)
    x_axis_world = np.array([
        np.sin(az) * np.cos(el),
        np.cos(az) * np.cos(el),
        np.sin(el),
    ])
    x_axis_world /= np.linalg.norm(x_axis_world)

    if not use_gravity:
        return quat_normalize(
            quat_from_two_vectors(np.array([1., 0., 0.]), x_axis_world))

    # --- Pelna orientacja z wykorzystaniem grawitacji ---
    # W spoczynku akcelerometr mierzy specific force = -g_world w ukladzie ciala.
    # g_body_rest to znormalizowany kierunek tego pomiaru (wskazuje "w gore"
    # rakiety, bo reakcja podpory jest skierowana ku gorze).
    # W ENU "gora" to [0,0,1]. Chcemy znalezc DCM (body->world) taka, ze:
    #   - os X ciala -> x_axis_world (kierunek lufy)
    #   - kierunek g_body_rest -> [0,0,1] (pion w gore)
    #
    # Budujemy uklad ortonormalny w obu ramkach i skladamy obrot.

    # Ramka swiata: x = kierunek lufy, z' = pion, y' = uzupelnienie
    xw = x_axis_world
    up_world = np.array([0., 0., 1.])
    # Skladowa "up" prostopadla do xw
    zw = up_world - np.dot(up_world, xw) * xw
    if np.linalg.norm(zw) < 1e-6:
        zw = np.array([0., 0., 1.])
    zw /= np.linalg.norm(zw)
    yw = np.cross(zw, xw)
    yw /= np.linalg.norm(yw)
    R_world = np.column_stack([xw, yw, zw])   # kolumny = osie w ENU

    # Ramka ciala: x = [1,0,0], "up" = g_body_rest (kierunek reakcji = gora)
    xb = np.array([1., 0., 0.])
    gb = cal.g_body_rest / np.linalg.norm(cal.g_body_rest)
    zb = gb - np.dot(gb, xb) * xb
    if np.linalg.norm(zb) < 1e-6:
        zb = np.array([0., 0., 1.])
    zb /= np.linalg.norm(zb)
    yb = np.cross(zb, xb)
    yb /= np.linalg.norm(yb)
    R_body = np.column_stack([xb, yb, zb])

    # DCM body->world: R = R_world @ R_body^T
    DCM = R_world @ R_body.T
    return dcm_to_quat(DCM)


def dcm_to_quat(R):
    """Macierz obrotu -> kwaternion (w,x,y,z). Metoda Sheppard'a."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z]))


# ---------------------------------------------------------------------------
# Strapdown INS
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    time:  np.ndarray
    pos:   np.ndarray   # (N,3) ENU [m]
    vel:   np.ndarray   # (N,3) ENU [m/s]
    acc_world: np.ndarray  # (N,3) ENU [m/s^2] po odjeciu g
    quat:  np.ndarray   # (N,4) orientacja body->world
    euler: np.ndarray   # (N,3) [roll, pitch, yaw] w stopniach
    speed: np.ndarray   # (N,) |v|
    alt:   np.ndarray   # (N,) wysokosc = pos[:,2]


def reconstruct(tel, cal, q0, t_start=0.0, t_end=None, use_ax_roll=True):
    """
    Strapdown INS — rekonstrukcja trajektorii z IMU.

    tel  — Telemetry
    cal  — Calibration
    q0   — kwaternion orientacji poczatkowej
    t_start, t_end — zakres czasu do rekonstrukcji (lot wlasciwy)
    use_ax_roll — jesli True i dostepna kolumna AX, uzyj zrekonstruowanego
                  rolla z high-range czujnika AX zamiast wysyconego gyro_X.
                  Krytyczne dla lotow z duzym rollem (gyro_X wysyca sie na
                  ±2000 st/s, a roll siega kilku tys.). Bez tego orientacja
                  — a wiec trajektoria — jest bledna.
    """
    if t_end is None:
        t_end = tel.time[-1]

    mask = (tel.time >= t_start) & (tel.time <= t_end)
    idx  = np.where(mask)[0]
    n    = len(idx)

    t  = tel.time[idx]
    # Przyspieszenia [g] -> skalowane
    ax = tel.acc_x[idx] * cal.acc_scale
    ay = tel.acc_y[idx] * cal.acc_scale
    az = tel.acc_z[idx] * cal.acc_scale
    # Predkosci katowe [st/s] -> [rad/s], odejmij bias
    # Roll: z AX (high-range) jesli dostepny, inaczej gyro_X
    roll_src = None
    if use_ax_roll:
        roll_src = reconstruct_roll_from_ax(tel, t_start)
    if roll_src is not None:
        gx = np.radians(roll_src[idx])   # AX juz ma odjety offset/skale
    else:
        gx = np.radians(tel.gyro_x[idx] - cal.gyro_bias[0])
    gy = np.radians(tel.gyro_y[idx] - cal.gyro_bias[1])
    gz = np.radians(tel.gyro_z[idx] - cal.gyro_bias[2])

    # Inicjalizacja
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    acc_world = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    quat[0] = q0

    g_world = np.array([0., 0., -G0])   # ENU: grawitacja w dol

    for k in range(1, n):
        dt = t[k] - t[k-1]
        if dt <= 0 or dt > 0.1:
            dt = cal_dt if (cal_dt := tel.dt_mean) else 0.004

        # 1. Aktualizacja orientacji — calkowanie predkosci katowej
        omega = np.array([gx[k], gy[k], gz[k]])
        omega_mag = np.linalg.norm(omega)
        if omega_mag > 1e-9:
            # Kwaternion przyrostu obrotu
            angle = omega_mag * dt
            axis  = omega / omega_mag
            dq = np.array([np.cos(angle/2),
                           axis[0]*np.sin(angle/2),
                           axis[1]*np.sin(angle/2),
                           axis[2]*np.sin(angle/2)])
        else:
            dq = np.array([1., 0., 0., 0.])
        q_new = quat_mult(quat[k-1], dq)
        quat[k] = quat_normalize(q_new)

        # 2. Przyspieszenie z ukladu ciala do ukladu swiata
        DCM = quat_to_dcm(quat[k])
        a_body = np.array([ax[k], ay[k], az[k]]) * G0   # [g]->[m/s^2]
        a_world_specific = DCM @ a_body

        # 3. Odejmij grawitacje (specific force = a_inertial - g)
        # a_measured = a_inertial - g_world  =>  a_inertial = a_measured + g
        a_inertial = a_world_specific + g_world
        acc_world[k] = a_inertial

        # 4. Calkowanie predkosci i pozycji (trapez)
        vel[k] = vel[k-1] + 0.5 * (acc_world[k] + acc_world[k-1]) * dt
        pos[k] = pos[k-1] + 0.5 * (vel[k] + vel[k-1]) * dt

    # Katy Eulera z kwaternionow
    euler = np.zeros((n, 3))
    for k in range(n):
        w, x, y, z = quat[k]
        # roll (X), pitch (Y), yaw (Z)
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        euler[k] = np.degrees([roll, pitch, yaw])

    speed = np.linalg.norm(vel, axis=1)

    return Trajectory(
        time=t, pos=pos, vel=vel, acc_world=acc_world,
        quat=quat, euler=euler, speed=speed, alt=pos[:, 2],
    )
