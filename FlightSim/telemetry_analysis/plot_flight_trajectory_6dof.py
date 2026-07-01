"""
plot_flight_trajectory_6dof.py
================================
Wykres trajektorii i predkosci PELNEGO przebiegu czasowego (nie tylko
apogeum/impact), model 6DOF (wariant 'adjusted' -- rzeczywisty profil
ciagu tego lotu, masa/elewacja/azymut/atmosfera tego lotu, jak w
analyze_per_flight_6dof.py) vs dane polowe, dla wybranych lotow.

4 panele per lot: wysokosc AGL(t), downrange(t), crossrange(t),
predkosc(t). Predkosc rzeczywista to ta sama hybryda co w
validate_trajectory.py (spalanie: calkowanie akcelerometru; coast:
GPS Vh + rozniczka baro dla Vz) -- NIE predkosc onboard wprost.

Uzycie (lokalnie, z prawdziwym DATCOM):
    python plot_flight_trajectory_6dof.py 16 17

Sanity-check w kontenerze (bez DATCOM):
    python plot_flight_trajectory_6dof.py --no-rerun-datcom 16 17

Sweep okresu podmuchu (diagnostyka "tumbling" pod wiatrem -- sprawdza
czy utrata stabilnosci po apogeum jest rezonansem z arbitralnie
przyjetym GUST_PERIOD_S=3.0s (NIE zwalidowanym wzgledem realnego
podmuchu, patrz analyze_per_flight_6dof.py), czy realnym deficytem
tlumienia aero modelu przy niskim cisnieniu dynamicznym po apogeum):
    python plot_flight_trajectory_6dof.py --gust-period-sweep 1,2,3,5,8 19

Sweep fazy podmuchu (przy stalym okresie -- sprawdza czy utrata
stabilnosci jest kwestia konkretnej fazy/timingu podmuchu, co
wzmacnia hipoteze marginalnego/za slabego tlumienia post-apogeum,
zamiast wezkopasmowego rezonansu z jednym okresem):
    python plot_flight_trajectory_6dof.py --gust-phase-sweep 0,90,180,270 19
"""

import sys
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, parse_telemetry, resolve_data_file
from run_6dof_cant_montecarlo import get_aero_for_cant
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import build_geometry
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from imu_reconstruction import detect_ignition
from diag_drag import detect_events, baro_altitude, calib_acc_scale, smooth_gps_derivative, G0
from gps_fusion import latlon_to_enu
from models.wind import PowerLawGustWind, PowerLawGustPulseWind
from analyze_wind_sensitivity import read_measured_wind

from analyze_per_flight_6dof import (
    read_flights, build_flight_thrust, build_scaled_mass, downrange_crossrange,
    GUST_PERIOD_S, GUST_PHASE_RAD,
)


def build_wind(base, fno, azimuth_deg, gust_period_s=GUST_PERIOD_S, gust_phase_rad=GUST_PHASE_RAD):
    """PowerLawGustWind z faktycznie zmierzonego wiatru dla tego lotu
    (Open-Meteo, field_test_data/results/launch_weather_openmeteo.csv) --
    gust_amp = gust/mean - 1. Zwraca None jesli brak pliku/wiersza dla
    tego lotu. gust_period_s/gust_phase_rad nadpisywalne (domyslnie
    GUST_PERIOD_S/GUST_PHASE_RAD z analyze_per_flight_6dof.py) -- do
    sweepu okresu podmuchu, patrz plot_gust_period_sweep()."""
    mw = read_measured_wind(base, fno)
    if mw is None or mw["mean_speed_mps"] <= 0:
        return None
    dir_from_deg = (azimuth_deg + mw["rel_az_deg"]) % 360.0
    gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
    return PowerLawGustWind(
        speed_ref_mps=mw["mean_speed_mps"], dir_from_deg=dir_from_deg, azimuth_deg=azimuth_deg,
        h_ref_m=10.0, alpha_exp=0.16,
        gust_amp=gust_amp, gust_period_s=gust_period_s, gust_phase_rad=gust_phase_rad,
    )


def build_wind_pulse(base, fno, azimuth_deg, t_center_s=5.0, sigma_s=2.0):
    """PowerLawGustPulseWind z faktycznie zmierzonego wiatru dla tego lotu
    -- POJEDYNCZY zlokalizowany w czasie impuls gaussowski (gust_amp =
    gust/mean - 1 w szczycie) zamiast sinusoidy trwajacej caly lot (patrz
    build_wind()/PowerLawGustWind). t_center_s/sigma_s nadpisywalne -- do
    przeszukania KIEDY (w jakim momencie lotu) pojedynczy podmuch
    realistycznie mogl wystapic, zamiast zakladac ciagla oscylacje."""
    mw = read_measured_wind(base, fno)
    if mw is None or mw["mean_speed_mps"] <= 0:
        return None
    dir_from_deg = (azimuth_deg + mw["rel_az_deg"]) % 360.0
    gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
    return PowerLawGustPulseWind(
        speed_ref_mps=mw["mean_speed_mps"], dir_from_deg=dir_from_deg, azimuth_deg=azimuth_deg,
        h_ref_m=10.0, alpha_exp=0.16,
        gust_amp=gust_amp, t_center_s=t_center_s, sigma_s=sigma_s,
    )


def actual_time_series(base, fno, azimuth_deg, t_burn, elev_deg):
    """Szeregi czasowe rzeczywiste (od zaplonu, AGL): t, h, downrange,
    crossrange, V -- ta sama hybryda predkosci co simulate_ascent() w
    validate_trajectory.py (spalanie: akcelerometr; coast: GPS Vh +
    rozniczka baro Vz), na pelnym zakresie i_ign..i_end."""
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    tel = parse_telemetry(fpath, verbose=False)
    t_ign_abs = detect_ignition(tel)
    i_ign, _, i_apo, i_end = detect_events(tel, t_ign_abs)

    h_baro = baro_altitude(tel, i_ign)
    h0 = h_baro[i_ign]
    t = tel.time
    seg = slice(i_ign, i_end + 1)
    t_rel = t[seg] - t[i_ign]
    h = h_baro[seg] - h0

    lat0, lon0 = tel.lat[i_ign], tel.lon[i_ign]
    e, n = latlon_to_enu(tel.lat[seg], tel.lon[seg], lat0, lon0)
    downrange, crossrange = downrange_crossrange(e, n, azimuth_deg)

    elev = math.radians(elev_deg)
    acc_scale, _ = calib_acc_scale(tel, t_ign_abs)
    a_meas = tel.acc_x[seg] * acc_scale * G0
    a_kin = a_meas - G0 * np.sin(elev)
    V_acc = np.concatenate([[0.0],
        np.cumsum(0.5 * (a_kin[1:] + a_kin[:-1]) * np.diff(t_rel))])
    V_acc = np.abs(V_acc)

    Vz_baro = smooth_gps_derivative(h_baro, t, window_s=0.15)[seg]
    Vh_gps = tel.vel_onboard[seg]
    V_gps = np.sqrt(Vh_gps ** 2 + Vz_baro ** 2)
    dh_step = np.abs(np.diff(h_baro[seg], prepend=h_baro[seg][0]))
    half_win = int(0.15 / 0.004)
    bad = np.convolve(dh_step > 5.0, np.ones(2 * half_win + 1), mode='same') > 0
    V_gps = np.where(bad, np.nan, V_gps)

    i_switch = int(np.argmin(np.abs(t_rel - (t_burn + 0.3))))
    V = np.concatenate([V_acc[:i_switch], V_gps[i_switch:]])

    return dict(t=t_rel, h=h, downrange=downrange, crossrange=crossrange, V=V,
                t_apo=t[i_apo] - t[i_ign], h_apo=h_baro[i_apo] - h0)


def model_time_series(aero, geom, atm, gravity, launcher, mass, prop, initial_state, wind_model=None):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
        wind_model=wind_model,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-6, atol=1e-8, max_step=0.05)
    qr_deg_s = np.degrees(np.sqrt(result.qr ** 2 + result.r ** 2))
    p_deg_s = np.degrees(result.p)
    return dict(t=result.t, h=-result.z, downrange=result.x, crossrange=result.y,
                V=result.speed, status=result.status, qr_deg_s=qr_deg_s, p_deg_s=p_deg_s)


def coast_velocity_oscillation(t, h, V, t_apo):
    """Amplituda oscylacji predkosci w fazie coast (po apogeum): odejmuje
    od V(t) wygladzony (low-pass, okno 2s) trend i zwraca odchylenie
    standardowe reszty na odcinku t > t_apo. Uzywane do porownania
    'szumu' predkosci modelu (sztuczne oscylacje pod wiatrem) vs danych
    polowych (oczekiwany gladki spadek -- patrz analiza lotu 19)."""
    mask = (t > t_apo) & np.isfinite(V)
    if mask.sum() < 10:
        return float("nan")
    tt, VV = t[mask], V[mask]
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 0.02
    win = max(3, int(round(2.0 / max(dt, 1e-6))))
    if win % 2 == 0:
        win += 1
    if win >= len(VV):
        return float("nan")
    kernel = np.ones(win) / win
    trend = np.convolve(VV, kernel, mode="same")
    resid = VV - trend
    edge = win // 2
    if len(resid) <= 2 * edge:
        return float("nan")
    return float(np.std(resid[edge:-edge]))


def plot_flight(fno, model, actual, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(model["t"], model["h"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["h"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.scatter([actual["t_apo"]], [actual["h_apo"]], marker='x', s=60, c='k', zorder=5,
               label="apogeum GPS")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("wysokosc AGL [m]")
    ax.set_title("Trajektoria wertykalna"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(model["t"], model["downrange"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["downrange"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("downrange [m]")
    ax.set_title("Downrange(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(model["t"], model["crossrange"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["crossrange"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("crossrange [m]")
    ax.set_title("Crossrange(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(model["t"], model["V"], color="tab:blue", lw=1.5, label="model 6DOF")
    ax.plot(actual["t"], actual["V"], color="tab:orange", lw=1.2, label="dane polowe")
    ax.set_xlabel("czas od zaplonu [s]"); ax.set_ylabel("predkosc [m/s]")
    ax.set_title("Predkosc(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Lot {fno}: model 6DOF (adjusted+wiatr, profil ciagu tego lotu + zmierzony "
                 f"wiatr Open-Meteo) vs dane polowe (status modelu: {model['status']})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")

    osc_model = coast_velocity_oscillation(model["t"], model["h"], model["V"], actual["t_apo"])
    osc_actual = coast_velocity_oscillation(actual["t"], actual["h"], actual["V"], actual["t_apo"])
    print(f"Lot {fno}: oscylacja predkosci coast (std reszty po odjeciu trendu 2s) -- "
          f"model={osc_model:.1f} m/s, dane polowe={osc_actual:.1f} m/s "
          f"(stosunek model/dane={osc_model / osc_actual if osc_actual > 1e-6 else float('nan'):.1f}x)")


def plot_gust_sweep(fno, labels, model_runs, out_png, sweep_name, title_extra):
    """Wspolny wykres dla sweepu okresu/fazy podmuchu: predkosc(t) i
    |qr|=sqrt(q^2+r^2) (deg/s) po apogeum dla tej samej konfiguracji
    lotu, rozne wartosci parametru podmuchu -- patrz
    plot_gust_period_sweep()/plot_gust_phase_sweep()."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    for label_val, model, col in zip(labels, model_runs, colors):
        if model is None:
            continue
        label = f"{label_val} (status={model['status']})"
        axes[0].plot(model["t"], model["V"], color=col, lw=1.3, label=label)
        axes[1].plot(model["t"], model["qr_deg_s"], color=col, lw=1.0, label=label)

    axes[1].axhline(45.0, color="r", ls="--", lw=1.0, label="prog tumblingu (45°/s)")
    axes[0].set_ylabel("predkosc [m/s]"); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Predkosc(t) per {sweep_name}")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("|qr| = sqrt(q²+r²) [°/s]"); axes[1].set_xlabel("czas od zaplonu [s]")
    axes[1].grid(alpha=0.3); axes[1].set_title(f"Predkosc katowa pitch/yaw(t) per {sweep_name}")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"Lot {fno}: sweep {sweep_name} -- {title_extra}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


def plot_gust_period_sweep(fno, periods_s, model_runs, out_png):
    """Diagnostyka rezonansu. Jesli 'tumbling' (qr > 45 deg/s przez
    >=3s, linia progowa) pojawia sie tylko dla niektorych okresow ->
    rezonans z arbitralnym (niezwalidowanym) okresem podmuchu, nie
    realny deficyt tlumienia aero. Jesli wystepuje dla wiekszosci/
    wszystkich okresow -> raczej realny deficyt tlumienia post-apogeum
    w modelu aero."""
    labels = [f"T={p:.1f}s" for p in periods_s]
    plot_gust_sweep(fno, labels, model_runs, out_png, "okresu podmuchu (GUST_PERIOD_S)",
                     "diagnostyka rezonansu vs realny deficyt tlumienia post-apogeum")


def plot_gust_phase_sweep(fno, phases_rad, model_runs, out_png):
    """Sweep fazy podmuchu przy STALYM okresie (domyslnie GUST_PERIOD_S
    z analyze_per_flight_6dof.py). Jesli tumbling wystepuje tylko dla
    wybranych faz -> uklad jest na granicy stabilnosci (marginalne
    tlumienie post-apogeum), sama faza/timing podmuchu decyduje czy
    qr przekroczy prog -- to wzmacnia hipoteze 'slabego tlumienia',
    nie 'wezkopasmowego rezonansu z konkretnym okresem'."""
    labels = [f"phi={math.degrees(p):.0f}°" for p in phases_rad]
    plot_gust_sweep(fno, labels, model_runs, out_png, "fazy podmuchu (GUST_PHASE_RAD)",
                     "diagnostyka marginalnego tlumienia (stale T, zmienna faza)")


def plot_roll_yaw_resonance(fno, model, out_png):
    """Diagnostyka rezonansu roll-pitch/yaw ('catastrophic yaw'): p(t)
    (predkosc toczenia, napedzana momentem od zaklinowania platów wg
    cant_angle) i |qr|(t) na wspolnym wykresie czasowym. Jesli wzrost
    |qr| pokrywa sie w czasie ze spadkiem/przejsciem p przez okreslona
    wartosc -- to podpis rezonansu roll-yaw (p przechodzi przez
    czestosc wlasna pitch/yaw, gdy ta spada wraz z cisnieniem
    dynamicznym po burnout). Jesli |qr| rosnie niezaleznie od
    zachowania p -- to raczej inny mechanizm (np. zle dobrana
    sztywnosc/tlumienie yaw)."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(model["t"], model["p_deg_s"], color="tab:red", lw=1.2)
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].set_ylabel("p (roll) [°/s]"); axes[0].grid(alpha=0.3)
    axes[0].set_title("Predkosc toczenia p(t)")

    axes[1].plot(model["t"], model["qr_deg_s"], color="tab:blue", lw=1.2)
    axes[1].axhline(45.0, color="r", ls="--", lw=1.0, label="prog tumblingu (45°/s)")
    axes[1].set_ylabel("|qr| = sqrt(q²+r²) [°/s]"); axes[1].set_xlabel("czas od zaplonu [s]")
    axes[1].grid(alpha=0.3); axes[1].set_title("Predkosc katowa pitch/yaw |qr|(t)")
    axes[1].legend(fontsize=8)

    # Adnotacja: pierwszy moment przekroczenia progu tumblingu + p w tej chwili
    above = np.where(model["qr_deg_s"] > 45.0)[0]
    if len(above) > 0:
        i0 = int(above[0])
        t0, p0 = model["t"][i0], model["p_deg_s"][i0]
        for ax in axes:
            ax.axvline(t0, color="g", ls=":", lw=1.0)
        axes[0].annotate(f"pierwsze qr>45°/s\nt={t0:.2f}s, p={p0:.0f}°/s",
                          xy=(t0, p0), xytext=(10, 10), textcoords="offset points",
                          fontsize=8, color="g")

    fig.suptitle(f"Lot {fno}: roll p(t) vs pitch/yaw |qr|(t) -- diagnostyka rezonansu "
                 f"roll-pitch/yaw (status modelu: {model['status']})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Trajektoria/predkosc(t) model 6DOF vs dane polowe")
    parser.add_argument("flights", type=int, nargs="+", help="numery lotow, np. 16 17")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa")
    parser.add_argument("--no-rerun-datcom", action="store_true")
    parser.add_argument("--gust-period-sweep", default=None,
                         help="lista okresow podmuchu [s] po przecinku, np. '1,2,3,5,8' -- "
                              "zamiast normalnego wykresu generuje sweep diagnostyczny "
                              "(patrz plot_gust_period_sweep())")
    parser.add_argument("--gust-phase-sweep", default=None,
                         help="lista faz podmuchu [deg] po przecinku, np. '0,90,180,270' -- "
                              "okres staly (GUST_PERIOD_S, nadpisywalny przez --gust-period-sweep-T) "
                              "(patrz plot_gust_phase_sweep())")
    parser.add_argument("--gust-period-sweep-T", type=float, default=GUST_PERIOD_S,
                         help="okres podmuchu [s] uzywany w --gust-phase-sweep (domyslnie GUST_PERIOD_S)")
    parser.add_argument("--roll-resonance-check", action="store_true",
                         help="zamiast normalnego wykresu generuje p(t) vs |qr|(t) -- diagnostyka "
                              "rezonansu roll-pitch/yaw ('catastrophic yaw') zamiast wplywu wiatru "
                              "(patrz plot_roll_yaw_resonance())")
    parser.add_argument("--roll-resonance-no-wind", action="store_true",
                         help="z --roll-resonance-check: pomija wiatr, zeby izolowac efekt "
                              "rezonansu roll/pitch-yaw od wymuszenia podmuchem")
    parser.add_argument("--gust-period-s", type=float, default=GUST_PERIOD_S,
                         help="okres podmuchu [s] dla normalnego wykresu trajektorii "
                              "(domyslnie GUST_PERIOD_S) -- do wyboru konkretnej "
                              "nie-tumblujacej kombinacji ze sweepu, do porownania z "
                              "danymi polowymi")
    parser.add_argument("--gust-phase-deg", type=float, default=math.degrees(GUST_PHASE_RAD),
                         help="faza podmuchu [deg] dla normalnego wykresu trajektorii "
                              "(domyslnie GUST_PHASE_RAD)")
    parser.add_argument("--gust-pulse", action="store_true",
                         help="uzyj PowerLawGustPulseWind (pojedynczy zlokalizowany w "
                              "czasie impuls gaussowski) zamiast sinusoidy trwajacej caly "
                              "lot (PowerLawGustWind) -- patrz build_wind_pulse()")
    parser.add_argument("--gust-pulse-t-center", type=float, default=5.0,
                         help="czas [s] od zaplonu szczytu impulsu (z --gust-pulse)")
    parser.add_argument("--gust-pulse-sigma", type=float, default=2.0,
                         help="szerokosc [s] impulsu gaussowskiego (z --gust-pulse)")
    parser.add_argument("--elevation-deg", type=float, default=None,
                         help="nadpisz kat elewacji [deg] z configs.txt (hipoteza: "
                              "wyrzutnia byla ustawiona pod innym katem niz zapisany -- "
                              "dla lotu 19 elew=42 odtwarza wysokosc/czas apogeum lepiej "
                              "niz zapisane 45). Nadpisuje ZAROWNO kat startowy modelu jak "
                              "i rzut grawitacji w rekonstrukcji predkosci z akcelerometru "
                              "(a_kin = a_meas - g*sin(elew)) -- oba uzywaja tej samej, "
                              "prawdziwej elewacji.")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    out_dir = Path(base) / "results"

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    cfg_base_by_nose = {
        nose: load_config(str(root / "configurations" / f"{case}.yaml"))
        for nose, case in case_by_nose.items()
    }
    t_ignition_by_nose = {nose: cfg.propulsion.t_ignition for nose, cfg in cfg_base_by_nose.items()}

    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)

    flights_by_fno = {r["fno"]: r for r in read_flights(base)}

    for fno in args.flights:
        if fno not in flights_by_fno:
            print(f"Lot {fno}: brak w configs.txt (to analyze? != yes) -- pomijam.")
            continue
        r = flights_by_fno[fno]
        case = case_by_nose[r["nose"]]
        cfg_base = cfg_base_by_nose[r["nose"]]
        t_ignition = t_ignition_by_nose[r["nose"]]
        t_burn = cfg_base.propulsion.t_burn

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
        geom.cant_angle_rad = math.radians(r["cant"])

        mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
        elev_deg = args.elevation_deg if args.elevation_deg is not None else r["elevation"]
        if args.elevation_deg is not None:
            print(f"Lot {fno}: elewacja nadpisana {r['elevation']:.1f} -> {elev_deg:.1f} deg")
        initial_state = State6DOF.initial(elevation_deg=elev_deg, azimuth_deg=r["azimuth"])

        prop_flight = build_flight_thrust(base, fno, t_ignition)
        if prop_flight is None:
            print(f"Lot {fno}: brak thrust_flight_{fno}.csv (kalibracja ciagu) -- pomijam.")
            continue

        if args.gust_period_sweep is not None:
            periods_s = [float(p) for p in args.gust_period_sweep.split(",")]
            model_runs = []
            for period in periods_s:
                wind = build_wind(base, fno, r["azimuth"], gust_period_s=period)
                if wind is None:
                    print(f"Lot {fno}: brak zmierzonego wiatru (Open-Meteo) -- pomijam.")
                    model_runs = None
                    break
                model_runs.append(model_time_series(
                    aero, geom, atm, gravity, launcher, mass, prop_flight, initial_state,
                    wind_model=wind))
            if model_runs is None:
                continue
            plot_gust_period_sweep(fno, periods_s, model_runs,
                                    out_dir / f"gust_period_sweep_flight_{fno}.png")
            continue

        if args.gust_phase_sweep is not None:
            phases_rad = [math.radians(float(p)) for p in args.gust_phase_sweep.split(",")]
            model_runs = []
            for phase in phases_rad:
                wind = build_wind(base, fno, r["azimuth"],
                                   gust_period_s=args.gust_period_sweep_T, gust_phase_rad=phase)
                if wind is None:
                    print(f"Lot {fno}: brak zmierzonego wiatru (Open-Meteo) -- pomijam.")
                    model_runs = None
                    break
                model_runs.append(model_time_series(
                    aero, geom, atm, gravity, launcher, mass, prop_flight, initial_state,
                    wind_model=wind))
            if model_runs is None:
                continue
            plot_gust_phase_sweep(fno, phases_rad, model_runs,
                                   out_dir / f"gust_phase_sweep_flight_{fno}.png")
            continue

        if args.roll_resonance_check:
            wind = None if args.roll_resonance_no_wind else build_wind(base, fno, r["azimuth"])
            if not args.roll_resonance_no_wind and wind is None:
                print(f"Lot {fno}: brak zmierzonego wiatru (Open-Meteo) -- pomijam.")
                continue
            model = model_time_series(aero, geom, atm, gravity, launcher, mass, prop_flight,
                                       initial_state, wind_model=wind)
            suffix = "_nowiatru" if args.roll_resonance_no_wind else ""
            plot_roll_yaw_resonance(fno, model,
                                     out_dir / f"roll_yaw_resonance_flight_{fno}{suffix}.png")
            continue

        if args.gust_pulse:
            wind = build_wind_pulse(base, fno, r["azimuth"],
                                     t_center_s=args.gust_pulse_t_center,
                                     sigma_s=args.gust_pulse_sigma)
        else:
            wind = build_wind(base, fno, r["azimuth"],
                               gust_period_s=args.gust_period_s,
                               gust_phase_rad=math.radians(args.gust_phase_deg))
        if wind is None:
            print(f"Lot {fno}: brak zmierzonego wiatru (Open-Meteo) -- pomijam.")
            continue

        model = model_time_series(aero, geom, atm, gravity, launcher, mass, prop_flight, initial_state,
                                   wind_model=wind)
        actual = actual_time_series(base, fno, r["azimuth"], t_burn, elev_deg)

        if args.gust_pulse:
            suffix = f"_pulse_tc{args.gust_pulse_t_center:.1f}_sig{args.gust_pulse_sigma:.1f}"
        else:
            suffix = ""
            if args.gust_period_s != GUST_PERIOD_S or args.gust_phase_deg != math.degrees(GUST_PHASE_RAD):
                suffix = f"_T{args.gust_period_s:.1f}_phi{args.gust_phase_deg:.0f}"
        if args.elevation_deg is not None:
            suffix += f"_elev{args.elevation_deg:.0f}"
        plot_flight(fno, model, actual, out_dir / f"trajectory_6dof_flight_{fno}{suffix}.png")


if __name__ == "__main__":
    main()
