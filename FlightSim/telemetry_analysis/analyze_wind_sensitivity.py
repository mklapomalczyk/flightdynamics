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


FLIGHTS = {
    18: dict(T_C=-4.1, p_hpa=994.0, RH_pct=80.0, elevation_deg=45.0,
             azimuth_deg=325.0, h_apo_actual=1291.3, cant_angle_deg=0.0),
    20: dict(T_C=-4.1, p_hpa=994.0, RH_pct=77.0, elevation_deg=45.0,
             azimuth_deg=325.0, h_apo_actual=1393.9, cant_angle_deg=0.6),
}

WIND_SPEEDS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
# Kierunek MET (skad wieje) wzgledem azymutu strzalu: 0=z czola (naprzeciw
# strzalu, "pod wiatr" w sensie oporu), 90/270=burtowy, 180=w ogon (z wiatrem)
WIND_HEADINGS_REL = {
    "headwind": 0.0,
    "crosswind": 90.0,
    "tailwind": 180.0,
}


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
