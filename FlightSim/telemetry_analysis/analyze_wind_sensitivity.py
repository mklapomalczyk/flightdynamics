"""
analyze_wind_sensitivity.py
============================
Sprawdza, czy realistyczny wiatr (HorizontalWind, na razie BEZ
PowerLawWind/PowerLawGustWind — patrz models/wind.py) moze wyjasnic
obserwowany deficyt apogeum lotow 18 i 20 (oba ostre, el=45 deg,
az=325 deg, ten sam dzien/miejsce startu), ktory NIE zostal wyjasniony
przez korekte ciagu per-lot (analyze_per_flight_6dof.py) ani przez
zmierzona atmosfere dnia startu (analyze_atmosphere_impact.py: tylko
ok. -3% dla obu lotow).

Metoda — ten sam pipeline 6DOF co MAIN.py, izolowana zmienna = wiatr:
  Dla kazdego lotu (18, 20): jeden przebieg referencyjny BEZ wiatru
  (atmosfera dnia startu, ciag nominalny z YAML), nastepnie siatka
  przebiegow z HorizontalWind dla roznych predkosci (2..15 m/s) i
  kierunkow (wzgledem azymutu strzalu: pod wiatr/z wiatrem/burtowy),
  aby znalezc jaka predkosc/kierunek odtwarza zmierzony deficyt
  apogeum.

  Dodatkowo: zmierzony wiatr WCZYTANY z
  field_test_data/results/launch_weather_openmeteo.csv (wygenerowany
  przez analyze_launch_weather.py — wind_speed_10m_mps,
  wind_dir_rel_azimuth_deg, wind_gust_10m_mps per lot, NIE
  hardcodowany w tym skrypcie) — jeden przebieg steady-state przy
  zmierzonej predkosci/kierunku, plus skan PowerLawGustWind po
  fazie/okresie podmuchu (amplituda = zmierzony gust/sredni - 1)
  szukajacy NAJGORSZEGO przypadku (gust w trakcie max-Q/wysokiego AoA)
  jako GORNA GRANICE wplywu wiatru zmiennego w czasie — patrz UWAGA w
  models/wind.py: faza/okres nie sa zwalidowane wzgledem
  rzeczywistego podmuchu (wymaga >1 strzalu). Jesli CSV nie istnieje,
  ta czesc jest pomijana (uruchom najpierw analyze_launch_weather.py).

  cant_angle: per-lot (lot 18 = 0 deg, lot 20 = 0.6 deg, z
  field_test_data/configs.txt) — generowany jest TYMCZASOWY YAML
  (kopia configurations/<case>.yaml z nadpisanym cant_angle plotek),
  DATCOM jest przeliczany dla niego (force_rerun=True, wymaga
  MissileDATCOM.exe lokalnie na Windows — patrz CLAUDE.md), a
  tymczasowy plik YAML jest usuwany po przebiegu. Nie modyfikuje
  MAIN.py ani configurations/*.yaml.

Uzycie:
    python analyze_wind_sensitivity.py
"""

import sys
import csv
import argparse
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir
from datcom_io.config_reader import load_config
from datcom_io.rocket_builder import (
    build_mass_model, build_propulsion, build_geometry, build_initial_state
)
from models.atmosphere import create_atmosphere
from models.gravity import create_gravity
from models.launcher import LauncherConfig
from models.wind import create_wind
from forces.force_model6 import ForceModel6DOF
from core.solver6 import run_simulation_6dof
from geo.geographic import MissionConfig
from aero import get_aero_model


# h_apo_actual: AGL (wzgledem padu), z field_test_data/results/
# trajectory_closure_summary.csv -> h_apo_actual_m (po poprawce ASL->AGL
# w validate_trajectory.py — patrz tamtejszy komentarz w simulate_ascent()).
FLIGHTS = {
    18: dict(T_C=-4.1, p_hpa=994.0, RH_pct=80.0, elevation_deg=45.0,
             azimuth_deg=325.0, h_apo_actual=1133.5, cant_angle_deg=0.0),
    20: dict(T_C=-4.1, p_hpa=994.0, RH_pct=77.0, elevation_deg=45.0,
             azimuth_deg=325.0, h_apo_actual=1228.7, cant_angle_deg=0.6),
}

WIND_SPEEDS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
# Kierunek MET (skad wieje) wzgledem azymutu strzalu: 0=z czola (naprzeciw
# strzalu, "pod wiatr" w sensie oporu), 90/270=burtowy, 180=w ogon (z wiatrem)
WIND_HEADINGS_REL = {
    "headwind": 0.0,
    "crosswind": 90.0,
    "tailwind": 180.0,
}

GUST_PERIODS_S  = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
GUST_PHASES_RAD = [i * np.pi / 4.0 for i in range(8)]   # 0..7pi/4, krok 45deg


def read_measured_wind(base, fno):
    """Wczytuje zmierzony wiatr (Open-Meteo) dla lotu fno z
    field_test_data/results/launch_weather_openmeteo.csv (wygenerowany
    przez analyze_launch_weather.py). Zwraca None jesli plik/wiersz nie
    istnieje (np. analyze_launch_weather.py nie zostal jeszcze uruchomiony
    - wymaga internetu, patrz jego docstring)."""
    csv_path = Path(base) / "results" / "launch_weather_openmeteo.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["fno"]) == fno:
                return dict(
                    mean_speed_mps=float(row["wind_speed_10m_mps"]),
                    gust_speed_mps=float(row["wind_gust_10m_mps"]),
                    rel_az_deg=float(row["wind_dir_rel_azimuth_deg"]),
                )
    return None


def build_initial_state_for(elevation_deg, azimuth_deg):
    """Stan startowy dla zadanej elewacji/azymutu (jak State6DOF.initial)."""
    from core.state6 import State6DOF
    return State6DOF.initial(elevation_deg=elevation_deg, azimuth_deg=azimuth_deg)


def make_temp_config(base_yaml_path, cant_angle_deg, tmp_yaml_path):
    """Kopia configurations/<case>.yaml z nadpisanym cant_angle wszystkich
    plotek — TYMCZASOWY plik, ma zostac usuniety po przebiegu (patrz
    modul docstring / CLAUDE.md: nigdy nie modyfikujemy configurations/*.yaml)."""
    with open(base_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for fin in data.get("fins", []):
        fin["cant_angle"] = cant_angle_deg
    with open(tmp_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return tmp_yaml_path


def run_one(atm, mass, prop, geom, gravity, launcher, aero, initial_state, wind_model):
    force_model = ForceModel6DOF(
        atmosphere=atm, mass_model=mass, aero_model=aero,
        gravity=gravity, geometry=geom, propulsion=prop, launcher=launcher,
        wind_model=wind_model,
    )
    result = run_simulation_6dof(force_model, initial_state, t_max=100, dt_output=0.02,
                                  rtol=1e-6, atol=1e-8, max_step=0.05)
    h = -result.z
    i_apo = int(np.argmax(h))
    return float(h[i_apo]), float(np.max(result.speed)), result.status


def main():
    parser = argparse.ArgumentParser(description="Czulosc apogeum na wiatr (loty 18, 20)")
    parser.add_argument("--case", default="rocket_70mm_baseline")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    base_yaml = root / "configurations" / f"{args.case}.yaml"
    gravity = create_gravity("constant")
    launcher = LauncherConfig(L_rail=3.0)

    out_rows = []
    for fno, fl in FLIGHTS.items():
        cant = fl["cant_angle_deg"]
        tmp_case = f"_tmp_{args.case}_cant{cant:g}"
        tmp_yaml = root / "configurations" / f"{tmp_case}.yaml"
        make_temp_config(base_yaml, cant, tmp_yaml)
        try:
            cfg = load_config(str(tmp_yaml))
            aero = get_aero_model(tmp_case, method="missile_datcom", force_rerun=True)
        finally:
            tmp_yaml.unlink(missing_ok=True)
        mass = build_mass_model(cfg)
        prop = build_propulsion(cfg)
        geom = build_geometry(cfg)

        atm = create_atmosphere("ISA_LAUNCH", T0_C=fl["T_C"], p0_hpa=fl["p_hpa"],
                                 RH_pct=fl["RH_pct"])
        x0 = build_initial_state_for(fl["elevation_deg"], fl["azimuth_deg"])

        h_nowind, v_nowind, status0 = run_one(atm, mass, prop, geom, gravity, launcher,
                                               aero, x0, wind_model=None)
        deficit_actual_pct = 100.0 * (h_nowind - fl["h_apo_actual"]) / h_nowind
        print(f"\n=== Lot {fno} (el={fl['elevation_deg']:.0f} deg, "
              f"az={fl['azimuth_deg']:.0f} deg, cant={cant:g} deg) ===")
        print(f"  Bez wiatru:     h_apo={h_nowind:7.1f} m  V_max={v_nowind:6.1f} m/s  "
              f"status={status0}")
        print(f"  GPS rzeczywiste: h_apo={fl['h_apo_actual']:7.1f} m  "
              f"-> deficyt do wyjasnienia = {deficit_actual_pct:+.1f}%")

        for heading_name, heading_rel in WIND_HEADINGS_REL.items():
            dir_from_deg = (fl["azimuth_deg"] + heading_rel) % 360.0
            for speed in WIND_SPEEDS:
                wind = create_wind("horizontal", speed_mps=speed,
                                    dir_from_deg=dir_from_deg,
                                    azimuth_deg=fl["azimuth_deg"])
                h_w, v_w, status_w = run_one(atm, mass, prop, geom, gravity, launcher,
                                              aero, x0, wind_model=wind)
                dh_pct = 100.0 * (h_w - h_nowind) / h_nowind
                out_rows.append(dict(
                    fno=fno, cant_angle_deg=cant, model="horizontal",
                    heading=heading_name,
                    speed_mps=speed, h_apo_m=h_w, v_max_mps=v_w, status=status_w,
                    dh_pct_vs_nowind=dh_pct,
                    h_apo_actual=fl["h_apo_actual"],
                    deficit_to_explain_pct=deficit_actual_pct,
                ))
                print(f"    horizontal  {heading_name:10s} {speed:5.1f} m/s -> "
                      f"h_apo={h_w:7.1f} m  ({dh_pct:+6.1f}% vs no-wind)  "
                      f"status={status_w}")

        # ---- Zmierzony wiatr (Open-Meteo): steady + skan podmuchu ---- #
        mw = read_measured_wind(base, fno)
        if mw is None:
            print(f"  [pominieto] brak results/launch_weather_openmeteo.csv dla lotu {fno} "
                  f"-> uruchom najpierw analyze_launch_weather.py")
            continue
        dir_from_meas = (fl["azimuth_deg"] + mw["rel_az_deg"]) % 360.0

        wind_steady = create_wind("horizontal", speed_mps=mw["mean_speed_mps"],
                                   dir_from_deg=dir_from_meas, azimuth_deg=fl["azimuth_deg"])
        h_steady, v_steady, status_steady = run_one(atm, mass, prop, geom, gravity, launcher,
                                                     aero, x0, wind_model=wind_steady)
        dh_steady_pct = 100.0 * (h_steady - h_nowind) / h_nowind
        out_rows.append(dict(
            fno=fno, cant_angle_deg=cant, model="measured_steady", heading="measured",
            speed_mps=mw["mean_speed_mps"], h_apo_m=h_steady, v_max_mps=v_steady,
            status=status_steady, dh_pct_vs_nowind=dh_steady_pct,
            h_apo_actual=fl["h_apo_actual"], deficit_to_explain_pct=deficit_actual_pct,
        ))
        print(f"  Zmierzony wiatr (steady {mw['mean_speed_mps']:.0f} m/s, "
              f"rel_az={mw['rel_az_deg']:.0f} deg) -> h_apo={h_steady:7.1f} m "
              f"({dh_steady_pct:+.1f}% vs no-wind)  status={status_steady}")

        gust_amp = mw["gust_speed_mps"] / mw["mean_speed_mps"] - 1.0
        worst_h, worst_T, worst_phase = h_steady, None, None
        for T in GUST_PERIODS_S:
            for phase in GUST_PHASES_RAD:
                wind_gust = create_wind(
                    "power_law_gust", speed_ref_mps=mw["mean_speed_mps"],
                    dir_from_deg=dir_from_meas, azimuth_deg=fl["azimuth_deg"],
                    h_ref_m=10.0, alpha_exp=0.16,
                    gust_amp=gust_amp, gust_period_s=T, gust_phase_rad=phase,
                )
                h_g, v_g, status_g = run_one(atm, mass, prop, geom, gravity, launcher,
                                              aero, x0, wind_model=wind_gust)
                dh_g_pct = 100.0 * (h_g - h_nowind) / h_nowind
                out_rows.append(dict(
                    fno=fno, cant_angle_deg=cant, model="measured_gust",
                    heading=f"T={T:g}s_phase={phase:.2f}rad",
                    speed_mps=mw["mean_speed_mps"], h_apo_m=h_g, v_max_mps=v_g,
                    status=status_g, dh_pct_vs_nowind=dh_g_pct,
                    h_apo_actual=fl["h_apo_actual"], deficit_to_explain_pct=deficit_actual_pct,
                ))
                if h_g < worst_h:
                    worst_h, worst_T, worst_phase = h_g, T, phase
        worst_dh_pct = 100.0 * (worst_h - h_nowind) / h_nowind
        print(f"  Najgorszy podmuch (gust_amp={gust_amp:.2f}, T={worst_T:g}s, "
              f"phase={worst_phase:.2f} rad) -> h_apo={worst_h:7.1f} m "
              f"({worst_dh_pct:+.1f}% vs no-wind)")

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "wind_sensitivity_18_20.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nZapisano: {csv_path}")


if __name__ == "__main__":
    main()
