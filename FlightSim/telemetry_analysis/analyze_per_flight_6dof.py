"""
analyze_per_flight_6dof.py
===========================
Per-flight deterministyczne porownanie 6DOF vs dane polowe, dla KAZDEGO
lotu z 'to analyze? == yes' w configs.txt (nie tylko 3 skorygowanych) —
zamiast usrednien per typ nosa (ktore myla loty o roznej rzeczywistej
masie w jedna grupe), dajemy jedna czysta tabele: jeden wiersz na lot.

Dla kazdego lotu izolowane sa NARAZ trzy zmienne wczesniej trzymane na
wartosciach nominalnych ze wzgledu na uproszczenie (cant_angle z YAML,
elewacja/azymut z mission_01.yaml, atmosfera ISA standard):

  1. Masa: mass_model.full/empty z YAML skalowane proporcjonalnie tak,
     by m_full_skalowane == configs.txt 'm_rocket' tego lotu (xcg/Iyy
     trzymane jako te same FRAKCJE wzgledem masy co w YAML — nie mamy
     osobnych pomiarow xcg/Iyy per lot).
  2. Elewacja/azymut: z configs.txt tego lotu (nie z mission_01.yaml).
  3. Atmosfera: ISALaunchSiteAtmosphere zakotwiczona w T/p/Rh tego lotu
     (models/atmosphere.py), zamiast ISA standard.

Cant_angle: z configs.txt tego lotu. Jesli != cant_angle juz w cache
DATCOM (z YAML), wymagany jest przebieg DATCOM dla TEGO cant_angle —
robione przez tymczasowy YAML (wzorzec z analyze_cant_sweep.py /
run_6dof_cant_montecarlo.py), usuwany po uzyciu. MAIN.py i
configurations/rocket_70mm_baseline.yaml NIE sa modyfikowane.

WAZNE: loty o cant_angle != YAML wymagaja PRAWDZIWEGO DATCOM — ten
skrypt musi byc uruchomiony LOKALNIE (Windows + MissileDATCOM.exe) dla
pelnych wynikow. W kontenerze mozna jedynie sprawdzic logike skryptu z
--no-rerun-datcom (uzywa wylacznie cache bazowego YAML — wyniki
fizycznie poprawne tylko dla lotow, ktorych cant_angle juz odpowiada
YAML).

Oprocz apogeum/V_max raportowany jest tez zasieg poziomy przy apogeum
(downrange/crossrange wzgledem azymutu strzalu, Launch Frame: X=downrange,
Y=crossrange w prawo) — PRZED rozwarciem spadochronu, wiec porownywalny
1:1 z modelem 6DOF (ktory spadochronu nie symuluje; ladowanie pod
spadochronem zalezy od wiatru podczas znoszenia i NIE jest tu uzywane
do walidacji). Actual: GPS w momencie tego samego i_apo co
trajectory_closure_summary.csv (argmax surowej wysokosci GPS), wzgledem
pierwszego probek GPS przed zaplonem (x=y=0 w modelu). Cel: sprawdzic
hipoteze trwalej niewspolosiowosci dyszy silnika — sygnatura to
SYSTEMATYCZNE odchylenie crossrange w jedna strone na wiekszosci lotow
(mean >> std), w odroznieniu od wiatru/szumu GPS (odchylenia w obie
strony, mean ~ 0). h_apo_actual_m uzyty tu pochodzi z
trajectory_closure_summary.csv i jest juz AGL (wzgledem padu) po
poprawce w validate_trajectory.py — wczesniej byl ASL (zawieral
wysokosc startowiska), co dawalo zanizony "deficyt do wyjasnienia".

Uzycie (lokalnie, z prawdziwym DATCOM):
    python analyze_per_flight_6dof.py
    python analyze_per_flight_6dof.py --case-ostra rocket_70mm_baseline --case-tepa rocket_70mm_baseline_tepa

Sanity-check w kontenerze (bez DATCOM, tylko logika kodu):
    python analyze_per_flight_6dof.py --no-rerun-datcom
"""

import sys
import csv
import argparse
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
from datcom_io.rocket_builder import build_propulsion, build_geometry, ThrustProfile, PropulsionConfig6DOFDynamic
from models.mass6 import MassModel6DOF, LinearIxx
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from core.state6 import State6DOF
from imu_reconstruction import detect_ignition
from diag_drag import detect_events
from gps_fusion import latlon_to_enu
from models.wind import create_wind
from analyze_wind_sensitivity import read_measured_wind

# Domyslny okres/faza podmuchu dla PowerLawGustWind -- NIE zwalidowane
# wzgledem rzeczywistego podmuchu (wymaga >1 strzalu, patrz models/wind.py
# i analyze_wind_sensitivity.py); tu uzywane jako JEDNA reprezentatywna
# probka (nie worst-case ze skanu), zeby pokazac rzad wielkosci wplywu
# wiatru na apogeum/zasieg, nie gorna granice.
GUST_PERIOD_S = 3.0
GUST_PHASE_RAD = 0.0


def build_flight_wind(base, fno, azimuth_deg):
    """PowerLawGustWind z faktycznie zmierzonego wiatru dla tego lotu
    (Open-Meteo, field_test_data/results/launch_weather_openmeteo.csv) --
    gust_amp = gust/mean - 1. Zwraca None jesli brak pliku/wiersza dla
    tego lotu (uruchom najpierw analyze_launch_weather.py)."""
    mw = read_measured_wind(base, fno)
    if mw is None or mw["mean_speed_mps"] <= 0:
        return None
    dir_from_deg = (azimuth_deg + mw["rel_az_deg"]) % 360.0
    gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
    return create_wind(
        "power_law_gust", speed_ref_mps=mw["mean_speed_mps"],
        dir_from_deg=dir_from_deg, azimuth_deg=azimuth_deg,
        h_ref_m=10.0, alpha_exp=0.16,
        gust_amp=gust_amp, gust_period_s=GUST_PERIOD_S, gust_phase_rad=GUST_PHASE_RAD,
    )


# --------------------------------------------------------------------------
def read_flights(base):
    """Parsuje configs.txt -> lista dict per lot (tylko 'to analyze? == yes')."""
    cfg_path = Path(base) / "configs.txt"
    lines = open(cfg_path, encoding="utf-8", errors="replace").readlines()
    header = [h.strip() for h in lines[0].strip().split("\t")]
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split("\t")]
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        try:
            fno = int(row["flight no"])
        except (ValueError, KeyError):
            continue
        if row.get("to analyze?", "no").strip().lower() != "yes":
            continue

        def f(k, default=0.0):
            try:
                return float(row.get(k, default))
            except ValueError:
                return default

        head = row.get("head configuration", "").lower().strip()
        nose = "tepa" if ("t" in head and ("pa" in head or "epa" in head)) else "ostra"

        rows.append(dict(
            fno=fno, nose=nose,
            cant=f("cant angle"), azimuth=f("azimuth"), elevation=f("elevation"),
            m_rocket=f("m_rocket"),
            T_C=f("T [C]"), p_hpa=f("p [hpa]"), RH_pct=f("Rh [%]"),
        ))
    rows.sort(key=lambda r: r["fno"])
    return rows


def build_flight_thrust(base, fno, t_ignition):
    """Buduje PropulsionConfig6DOFDynamic z thrust_flight_<fno>.csv (kolumny
    t_s, T_est_N) — rzeczywisty profil ciagu TEGO lotu z kalibracji Pc->T
    (estimate_thrust.py), zamiast usrednionego profilu z YAML. Zwraca None
    jesli plik nie istnieje (np. lot 21, wykluczony przez brama R2)."""
    csv_path = Path(base) / "results" / f"thrust_flight_{fno}.csv"
    if not csv_path.exists():
        return None
    t_list, F_list = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_list.append(float(row["t_s"]))
            F_list.append(max(float(row["T_est_N"]), 0.0))
    profile = ThrustProfile(list(zip(t_list, F_list)))
    return PropulsionConfig6DOFDynamic(thrust_profile=profile, t_ignition=t_ignition)


def actual_apogee_vmax(base, fno):
    csv_path = Path(base) / "results" / "trajectory_closure_summary.csv"
    if not csv_path.exists():
        return None, None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["flight_no"]) == fno:
                v = float(row["V_max_actual_mps"]) if row.get("V_max_actual_mps") else None
                return float(row["h_apo_actual_m"]), v
    return None, None


def downrange_crossrange(e, n, azimuth_deg):
    """Rzutuje przesuniecie ENU (e=East, n=North) na Launch Frame osi
    strzalu: downrange wzdluz azymutu, crossrange w prawo od azymutu
    (ta sama konwencja co core/state6.py: X wzdluz azymutu, Y w prawo)."""
    az = np.radians(azimuth_deg)
    downrange = e * np.sin(az) + n * np.cos(az)
    crossrange = e * np.cos(az) - n * np.sin(az)
    return downrange, crossrange


def actual_range_at_apogee(base, fno, azimuth_deg):
    """Przesuniecie poziome (downrange/crossrange wzgledem azymutu strzalu,
    Launch Frame) w momencie GPS-apogeum (ten sam indeks i_apo co
    validate_trajectory.py/trajectory_closure_summary.csv: argmax surowej
    wysokosci GPS) — PRZED rozwarciem spadochronu, wiec porownywalne z
    modelem 6DOF (ktory nie symuluje spadochronu). Referencja pozycji =
    pierwszy probek GPS przed zaplonem (tam gdzie x=y=0 w modelu)."""
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    if not fpath.exists():
        return None, None, None
    tel = parse_telemetry(fpath, verbose=False)
    t_ign_abs = detect_ignition(tel)
    i_ign, _, i_apo, _ = detect_events(tel, t_ign_abs)
    lat0, lon0 = tel.lat[i_ign], tel.lon[i_ign]
    e, n = latlon_to_enu(tel.lat[[i_apo]], tel.lon[[i_apo]], lat0, lon0)
    downrange, crossrange = downrange_crossrange(e[0], n[0], azimuth_deg)
    rng = float(np.hypot(downrange, crossrange))
    return float(downrange), float(crossrange), rng


# --------------------------------------------------------------------------
def build_scaled_mass(cfg, t_ignition, m_rocket_flight):
    """Skaluje mass_model.full/empty proporcjonalnie do m_rocket_flight,
    zachowujac frakcje xcg/Iyy/Ixx wzgledem masy nominalnej z YAML (per-lot
    pomiary xcg/Iyy nie istnieja, wiec skalujemy jednorodnie cala krzywa
    masy, nie tylko punkt full)."""
    mm = cfg.mass_model
    t_b = cfg.propulsion.t_burn
    m_full_nom = mm.full.mass
    scale = m_rocket_flight / m_full_nom if m_full_nom > 0 else 1.0

    m_full_new = mm.full.mass * scale
    m_empty_new = mm.empty.mass * scale

    return MassModel6DOF(
        m_full=m_full_new, m_empty=m_empty_new, t_burn=t_b,
        xcg_full=mm.full.xcg, xcg_empty=mm.empty.xcg,
        Iyy_full=mm.full.Iyy * scale, Iyy_empty=mm.empty.Iyy * scale,
        ixx_model=LinearIxx(Ixx_full=mm.full.Ixx * scale, Ixx_empty=mm.empty.Ixx * scale, t_burn=t_b),
        t_ignition=t_ignition,
    )


def run_one(aero, geom, atm, gravity, launcher, mass, prop, initial_state, wind_model=None):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
        wind_model=wind_model,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-6, atol=1e-8, max_step=0.05)
    h = -result.z
    i_apo = int(np.argmax(h))
    downrange, crossrange = float(result.x[i_apo]), float(result.y[i_apo])
    rng = float(np.hypot(downrange, crossrange))
    # Solver juz integruje balistycznie do ground_event (z>=0, core/solver6.py)
    # -- model NIE symuluje spadochronu, wiec result.t[-1]/x[-1]/y[-1] to
    # czysto balistyczny ("crash", bez chutu) punkt/czas upadku. Uzywane
    # tylko jako DOLNA GRANICA czasu lotu / odniesienie balistyczne w
    # analyze_impact_point.py -- prawdziwy lot ma chute, wiec nie jest to
    # predykcja realnego punktu ladowania.
    impact_downrange, impact_crossrange = float(result.x[-1]), float(result.y[-1])
    impact_rng = float(np.hypot(impact_downrange, impact_crossrange))
    impact_t = float(result.t[-1])
    impact_speed = float(result.speed[-1])
    return (float(h[i_apo]), float(np.max(result.speed)), result.status,
            downrange, crossrange, rng,
            impact_downrange, impact_crossrange, impact_rng, impact_t, impact_speed)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Per-lot 6DOF (masa+elewacja/azymut+atmosfera skorygowane) vs dane polowe")
    parser.add_argument("--case-ostra", default="rocket_70mm_baseline",
                         help="case YAML dla lotow z nosem ostrym")
    parser.add_argument("--case-tepa", default="rocket_70mm_baseline_tepa",
                         help="case YAML dla lotow z nosem tepym")
    parser.add_argument("--no-rerun-datcom", action="store_true",
                         help="uzyj wylacznie istniejacego cache bazowego YAML (bez DATCOM); "
                              "fizycznie poprawne tylko dla lotow, ktorych cant_angle == YAML")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent

    case_by_nose = {"ostra": args.case_ostra, "tepa": args.case_tepa}
    cfg_base_by_nose = {
        nose: load_config(str(root / "configurations" / f"{case}.yaml"))
        for nose, case in case_by_nose.items()
    }
    cant_yaml_by_nose = {
        nose: sorted(set(round(fin.cant_angle, 5) for fin in cfg.fins))
        for nose, cfg in cfg_base_by_nose.items()
    }
    t_ignition_by_nose = {nose: cfg.propulsion.t_ignition for nose, cfg in cfg_base_by_nose.items()}

    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)

    flights = read_flights(base)
    print(f"Loty do analizy: {[r['fno'] for r in flights]}")
    print(f"case per nos: {case_by_nose}")
    print(f"cant_angle w YAML: {cant_yaml_by_nose}  (loty o innym cant_angle wymagaja DATCOM)\n")

    print("Pelna tabela liczbowa -> CSV (per_flight_6dof_per_nose.csv); "
          "porownanie graficzne -> PNG (baseline/adjusted).")

    out_rows = []
    for r in flights:
        case = case_by_nose[r["nose"]]
        cfg_base = cfg_base_by_nose[r["nose"]]
        t_ignition = t_ignition_by_nose[r["nose"]]

        aero, _ = get_aero_for_cant(case, r["cant"], force_rerun=not args.no_rerun_datcom)
        geom = build_geometry(load_config(str(root / "configurations" / f"{case}.yaml")))
        import math
        geom.cant_angle_rad = math.radians(r["cant"])

        mass = build_scaled_mass(cfg_base, t_ignition, r["m_rocket"])
        atm = create_atmosphere("ISA_LAUNCH", T0_C=r["T_C"], p0_hpa=r["p_hpa"], RH_pct=r["RH_pct"])
        initial_state = State6DOF.initial(elevation_deg=r["elevation"], azimuth_deg=r["azimuth"])

        # Baseline: usredniony profil ciagu z YAML (ten sam dla wszystkich lotow)
        prop_base = build_propulsion(cfg_base)
        try:
            (h_base, v_base, status_base,
             dr_base, cr_base, rng_base,
             idr_base, icr_base, irng_base, it_base, isp_base) = run_one(
                aero, geom, atm, gravity, launcher, mass, prop_base, initial_state)
        except Exception as e:
            h_base, v_base, status_base = float("nan"), float("nan"), f"error:{e}"
            dr_base, cr_base, rng_base = float("nan"), float("nan"), float("nan")
            idr_base, icr_base, irng_base, it_base, isp_base = (float("nan"),) * 5

        # Adjusted: rzeczywisty profil ciagu TEGO lotu z kalibracji Pc->T
        prop_flight = build_flight_thrust(base, r["fno"], t_ignition)
        if prop_flight is not None:
            try:
                (h_adj, v_adj, status_adj,
                 dr_adj, cr_adj, rng_adj,
                 idr_adj, icr_adj, irng_adj, it_adj, isp_adj) = run_one(
                    aero, geom, atm, gravity, launcher, mass, prop_flight, initial_state)
            except Exception as e:
                h_adj, v_adj, status_adj = float("nan"), float("nan"), f"error:{e}"
                dr_adj, cr_adj, rng_adj = float("nan"), float("nan"), float("nan")
                idr_adj, icr_adj, irng_adj, it_adj, isp_adj = (float("nan"),) * 5
        else:
            h_adj, v_adj, status_adj = float("nan"), float("nan"), "no_thrust_calib"
            dr_adj, cr_adj, rng_adj = float("nan"), float("nan"), float("nan")
            idr_adj, icr_adj, irng_adj, it_adj, isp_adj = (float("nan"),) * 5

        # Adjusted + wiatr: rzeczywisty profil ciagu TEGO lotu + zmierzony
        # wiatr (PowerLawGustWind, Open-Meteo) -- izoluje wplyw wiatru na
        # gornej warstwie modelu juz skorygowanego o ciag.
        wind = build_flight_wind(base, r["fno"], r["azimuth"])
        if prop_flight is not None and wind is not None:
            try:
                (h_wind, v_wind, status_wind,
                 dr_wind, cr_wind, rng_wind,
                 idr_wind, icr_wind, irng_wind, it_wind, isp_wind) = run_one(
                    aero, geom, atm, gravity, launcher, mass, prop_flight, initial_state,
                    wind_model=wind)
            except Exception as e:
                h_wind, v_wind, status_wind = float("nan"), float("nan"), f"error:{e}"
                dr_wind, cr_wind, rng_wind = float("nan"), float("nan"), float("nan")
                idr_wind, icr_wind, irng_wind, it_wind, isp_wind = (float("nan"),) * 5
        else:
            h_wind, v_wind = float("nan"), float("nan")
            status_wind = "no_thrust_calib" if prop_flight is None else "no_measured_wind"
            dr_wind, cr_wind, rng_wind = float("nan"), float("nan"), float("nan")
            idr_wind, icr_wind, irng_wind, it_wind, isp_wind = (float("nan"),) * 5

        h_act, v_act = actual_apogee_vmax(base, r["fno"])
        dr_act, cr_act, rng_act = actual_range_at_apogee(base, r["fno"], r["azimuth"])

        dh_base = 100.0 * (h_base - h_act) / h_act if h_act else float("nan")
        dh_adj = 100.0 * (h_adj - h_act) / h_act if (h_act and not np.isnan(h_adj)) else float("nan")
        dv_base = 100.0 * (v_base - v_act) / v_act if (v_act is not None) else float("nan")
        dv_adj = 100.0 * (v_adj - v_act) / v_act if (v_act is not None and not np.isnan(v_adj)) else float("nan")
        # crossrange: bezwzgledne odchylenie [m] (a nie %) -- bliska 0 wartosc
        # rzeczywista robi % bezsensownym/niestabilnym numerycznie. Stale
        # odchylenie w jedna strone na wielu lotach = podpis niewspolosiowosci
        # dyszy (lub trwale przesuniecie xcg/asymetria plotek), w odroznieniu
        # od wiatru/turbulencji, ktore daja odchylenia w obie strony.
        dcr_base = (cr_base - cr_act) if (cr_act is not None and not np.isnan(cr_base)) else float("nan")
        dcr_adj = (cr_adj - cr_act) if (cr_act is not None and not np.isnan(cr_adj)) else float("nan")
        dcr_wind = (cr_wind - cr_act) if (cr_act is not None and not np.isnan(cr_wind)) else float("nan")

        dh_wind = 100.0 * (h_wind - h_act) / h_act if (h_act and not np.isnan(h_wind)) else float("nan")
        dv_wind = 100.0 * (v_wind - v_act) / v_act if (v_act is not None and not np.isnan(v_wind)) else float("nan")

        print(f"  lot {r['fno']:3d} ({r['nose']}, cant={r['cant']:.2f} deg): "
              f"status baseline={status_base}  adjusted={status_adj}  adjusted+wind={status_wind}")

        out_rows.append(dict(
            fno=r["fno"], nose=r["nose"], cant=r["cant"], m_rocket=r["m_rocket"],
            elevation=r["elevation"], azimuth=r["azimuth"],
            T_C=r["T_C"], p_hpa=r["p_hpa"], RH_pct=r["RH_pct"],
            h_apo_pred_baseline=h_base, v_max_pred_baseline=v_base, status_baseline=status_base,
            h_apo_pred_adjusted=h_adj, v_max_pred_adjusted=v_adj, status_adjusted=status_adj,
            h_apo_actual=h_act, v_max_actual=v_act,
            dh_pct_baseline=dh_base, dv_pct_baseline=dv_base,
            dh_pct_adjusted=dh_adj, dv_pct_adjusted=dv_adj,
            downrange_apo_pred_baseline_m=dr_base, crossrange_apo_pred_baseline_m=cr_base,
            range_apo_pred_baseline_m=rng_base,
            downrange_apo_pred_adjusted_m=dr_adj, crossrange_apo_pred_adjusted_m=cr_adj,
            range_apo_pred_adjusted_m=rng_adj,
            downrange_apo_actual_m=dr_act, crossrange_apo_actual_m=cr_act,
            range_apo_actual_m=rng_act,
            crossrange_err_baseline_m=dcr_base, crossrange_err_adjusted_m=dcr_adj,
            # Punkt/czas upadku BALISTYCZNY z modelu (bez chutu, patrz run_one) --
            # odniesienie dla analyze_impact_point.py, NIE realna predykcja
            # ladowania (prawdziwy lot ma spadochron).
            impact_downrange_baseline_m=idr_base, impact_crossrange_baseline_m=icr_base,
            impact_range_baseline_m=irng_base, impact_t_baseline_s=it_base,
            impact_speed_baseline_mps=isp_base,
            impact_downrange_adjusted_m=idr_adj, impact_crossrange_adjusted_m=icr_adj,
            impact_range_adjusted_m=irng_adj, impact_t_adjusted_s=it_adj,
            impact_speed_adjusted_mps=isp_adj,
            # Adjusted + zmierzony wiatr (PowerLawGustWind, Open-Meteo) -- patrz
            # build_flight_wind()/GUST_PERIOD_S/GUST_PHASE_RAD: faza/okres
            # podmuchu NIE zwalidowane wzgledem rzeczywistego podmuchu, to
            # JEDNA reprezentatywna probka, nie gorna granica (w odroznieniu
            # od skanu worst-case w analyze_wind_sensitivity.py).
            h_apo_pred_adjusted_wind=h_wind, v_max_pred_adjusted_wind=v_wind,
            status_adjusted_wind=status_wind,
            dh_pct_adjusted_wind=dh_wind, dv_pct_adjusted_wind=dv_wind,
            downrange_apo_pred_adjusted_wind_m=dr_wind, crossrange_apo_pred_adjusted_wind_m=cr_wind,
            range_apo_pred_adjusted_wind_m=rng_wind,
            crossrange_err_adjusted_wind_m=dcr_wind,
            impact_downrange_adjusted_wind_m=idr_wind, impact_crossrange_adjusted_wind_m=icr_wind,
            impact_range_adjusted_wind_m=irng_wind, impact_t_adjusted_wind_s=it_wind,
            impact_speed_adjusted_wind_mps=isp_wind,
        ))

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_flight_6dof_per_nose.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nZapisano: {csv_path}")

    valid_base = [r for r in out_rows if r["h_apo_actual"] and not np.isnan(r["dh_pct_baseline"])]
    if valid_base:
        dh_base_all = np.array([r["dh_pct_baseline"] for r in valid_base])
        print(f"\nBias apogeum BASELINE (usredniony silnik) na {len(valid_base)} lotach:")
        print(f"  dh_apo: mean={np.mean(dh_base_all):+.2f}%  std={np.std(dh_base_all):.2f}%  "
              f"[{np.min(dh_base_all):+.2f}%, {np.max(dh_base_all):+.2f}%]")

    valid_adj = [r for r in out_rows if r["h_apo_actual"] and not np.isnan(r["dh_pct_adjusted"])]
    if valid_adj:
        dh_adj_all = np.array([r["dh_pct_adjusted"] for r in valid_adj])
        print(f"\nBias apogeum ADJUSTED (rzeczywisty silnik tego lotu) na {len(valid_adj)} lotach:")
        print(f"  dh_apo: mean={np.mean(dh_adj_all):+.2f}%  std={np.std(dh_adj_all):.2f}%  "
              f"[{np.min(dh_adj_all):+.2f}%, {np.max(dh_adj_all):+.2f}%]")

    valid_wind = [r for r in out_rows if r["h_apo_actual"] and not np.isnan(r["dh_pct_adjusted_wind"])]
    if valid_wind:
        dh_wind_all = np.array([r["dh_pct_adjusted_wind"] for r in valid_wind])
        print(f"\nBias apogeum ADJUSTED+WIND (silnik tego lotu + zmierzony wiatr Open-Meteo) "
              f"na {len(valid_wind)} lotach:")
        print(f"  dh_apo: mean={np.mean(dh_wind_all):+.2f}%  std={np.std(dh_wind_all):.2f}%  "
              f"[{np.min(dh_wind_all):+.2f}%, {np.max(dh_wind_all):+.2f}%]")
    else:
        print("\n[brak lotow z policzonym ADJUSTED+WIND -- sprawdz czy istnieje "
              "results/launch_weather_openmeteo.csv (analyze_launch_weather.py)]")

    # Crossrange (odchylenie boczne od azymutu strzalu przy apogeum, model
    # vs GPS): test niewspolosiowosci dyszy. Wiatr/turbulencja daje odchylenia
    # w obie strony (mean ~ 0, std duze); trwala niewspolosiowosc dyszy daje
    # odchylenie SYSTEMATYCZNE (mean wyraznie != 0, w jedna strone na
    # WIEKSZOSCI lotow, niezaleznie od azymutu/dnia/warunkow).
    valid_cr = [r for r in out_rows if not np.isnan(r["crossrange_err_adjusted_m"])]
    if valid_cr:
        cr_err = np.array([r["crossrange_err_adjusted_m"] for r in valid_cr])
        n_same_sign = max(np.sum(cr_err > 0), np.sum(cr_err < 0))
        print(f"\nBlad crossrange (model_adjusted - actual) przy apogeum na {len(valid_cr)} lotach:")
        print(f"  mean={np.mean(cr_err):+.1f}m  std={np.std(cr_err):.1f}m  "
              f"[{np.min(cr_err):+.1f}m, {np.max(cr_err):+.1f}m]  "
              f"zgodny znak: {n_same_sign}/{len(valid_cr)} lotow")
        if abs(np.mean(cr_err)) > np.std(cr_err) and n_same_sign >= 0.7 * len(valid_cr):
            print("  -> ODCHYLENIE SYSTEMATYCZNE (mean >> std, zgodny znak na "
                  "wiekszosci lotow): zgodne z trwala niewspolosiowoscia dyszy "
                  "lub asymetria geometrii, NIE z wiatrem/szumem GPS.")
        else:
            print("  -> brak wyraznego systematycznego odchylenia (znak/wielkosc "
                  "niestabilne miedzy lotami) -> bardziej zgodne z wiatrem/szumem "
                  "GPS niz z trwala niewspolosiowoscia dyszy.")

    # To samo, ale PO dolozeniu zmierzonego wiatru do modelu -- jesli
    # odchylenie systematyczne PRZETRWA (mean/std/zgodny znak podobne lub
    # silniejsze niz bez wiatru), to wiatr NIE wyjasnia crossrange i
    # hipoteza trwalej niewspolosiowosci/asymetrii pozostaje w grze;
    # jesli zniknie/zmniejszy sie wyraznie, to wiatr byl glownym powodem.
    valid_cr_wind = [r for r in out_rows if not np.isnan(r["crossrange_err_adjusted_wind_m"])]
    if valid_cr_wind:
        cr_err_w = np.array([r["crossrange_err_adjusted_wind_m"] for r in valid_cr_wind])
        n_same_sign_w = max(np.sum(cr_err_w > 0), np.sum(cr_err_w < 0))
        print(f"\nBlad crossrange (model_adjusted+wiatr - actual) przy apogeum na "
              f"{len(valid_cr_wind)} lotach:")
        print(f"  mean={np.mean(cr_err_w):+.1f}m  std={np.std(cr_err_w):.1f}m  "
              f"[{np.min(cr_err_w):+.1f}m, {np.max(cr_err_w):+.1f}m]  "
              f"zgodny znak: {n_same_sign_w}/{len(valid_cr_wind)} lotow")
        if abs(np.mean(cr_err_w)) > np.std(cr_err_w) and n_same_sign_w >= 0.7 * len(valid_cr_wind):
            print("  -> ODCHYLENIE SYSTEMATYCZNE PRZETRWALO po dolozeniu zmierzonego "
                  "wiatru -> wiatr NIE wyjasnia crossrange, hipoteza trwalej "
                  "niewspolosiowosci/asymetrii pozostaje aktualna.")
        else:
            print("  -> po dolozeniu zmierzonego wiatru odchylenie systematyczne "
                  "zniknelo/zmniejszylo sie -> wiatr jest wystarczajacym wyjasnieniem "
                  "crossrange, bez potrzeby trwalej niewspolosiowosci dyszy.")

    # ------------------------------------------------------------------
    make_comparison_figure(out_rows, "baseline", out_dir / "per_flight_6dof_baseline.png")
    make_comparison_figure(out_rows, "adjusted", out_dir / "per_flight_6dof_adjusted.png")
    make_comparison_figure(out_rows, "adjusted_wind", out_dir / "per_flight_6dof_adjusted_wind.png")


def make_comparison_figure(out_rows, variant, out_png):
    """Jedna figura (4 panele) pred vs actual dla danego wariantu silnika
    ('baseline'=usredniony profil ciagu z YAML, 'adjusted'=rzeczywisty
    profil ciagu tego lotu): apogeum, V_max, downrange@apogeum,
    crossrange@apogeum -- wszystkie loty na jednym wykresie per panel."""
    fnos = [r["fno"] for r in out_rows]
    x = np.arange(len(fnos))
    width = 0.38

    h_pred = [r[f"h_apo_pred_{variant}"] for r in out_rows]
    h_act = [r["h_apo_actual"] if r["h_apo_actual"] else float("nan") for r in out_rows]
    v_pred = [r[f"v_max_pred_{variant}"] for r in out_rows]
    v_act = [r["v_max_actual"] if r["v_max_actual"] else float("nan") for r in out_rows]
    dr_pred = [r[f"downrange_apo_pred_{variant}_m"] for r in out_rows]
    dr_act = [r["downrange_apo_actual_m"] if r["downrange_apo_actual_m"] else float("nan") for r in out_rows]
    cr_pred = [r[f"crossrange_apo_pred_{variant}_m"] for r in out_rows]
    cr_act = [r["crossrange_apo_actual_m"] if r["crossrange_apo_actual_m"] else float("nan") for r in out_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], h_pred, h_act, "Apogeum [m AGL]"),
        (axes[0, 1], v_pred, v_act, "V_max [m/s]"),
        (axes[1, 0], dr_pred, dr_act, "Downrange @ apogeum [m]"),
        (axes[1, 1], cr_pred, cr_act, "Crossrange @ apogeum [m]"),
    ]
    for ax, pred, act, label in panels:
        ax.bar(x - width / 2, pred, width, color="tab:blue", alpha=0.8, label="model")
        ax.bar(x + width / 2, act, width, color="tab:orange", alpha=0.8, label="actual (GPS)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in fnos])
        ax.set_xlabel("Lot")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    nice_names = {
        "baseline": "BASELINE (usredniony profil ciagu)",
        "adjusted": "ADJUSTED (profil ciagu tego lotu)",
        "adjusted_wind": "ADJUSTED + ZMIERZONY WIATR (profil ciagu tego lotu + Open-Meteo gust)",
    }
    nice_name = nice_names.get(variant, variant)
    fig.suptitle(f"Model 6DOF vs dane polowe — {nice_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
