"""
analyze_launch_weather.py
==========================
Pobiera historyczna pogode (Open-Meteo Historical Weather API,
https://archive-api.open-meteo.com — bez klucza API) dla daty/godziny
kazdego lotu z field_test_data/configs.txt (kolumny "date", "time") i
lokalizacji wyrzutni (missions/mission_01.yaml: lat/lon), zeby uzyskac
PRAWDZIWA predkosc/kierunek wiatru w dniu startu — niezalezna od
zmierzonych na ziemi T/p/Rh, ktore juz mamy w configs.txt.

Wynik (predkosc_wiatru, kierunek_z [deg MET]) mozna wstawic
bezposrednio do analyze_wind_sensitivity.py (HorizontalWind,
dir_from_deg = kierunek z API) zamiast skanowac 2..15 m/s na slepo.

Wymaga dostepu do internetu (zewnetrzne API) — NIE bedzie dzialac w
zdalnym/sandboxowym kontenerze z ograniczona siecia; uruchom lokalnie.

Nie modyfikuje MAIN.py ani configurations/*.yaml.

Uzycie:
    python analyze_launch_weather.py
    python analyze_launch_weather.py --only-analyzed
"""

import sys
import csv
import argparse
import datetime as dt
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "windspeed_10m", "winddirection_10m", "windgusts_10m",
    "temperature_2m", "surface_pressure", "relativehumidity_2m",
]


def read_launch_site(root):
    with open(root / "missions" / "mission_01.yaml", encoding="utf-8") as f:
        mission = yaml.safe_load(f)
    launcher = mission["launcher"] if "launcher" in mission else mission
    return float(launcher["latitude"]), float(launcher["longitude"])


def read_flights(base):
    """Parsuje configs.txt — kolumny date (dd.mm.yyyy), time (hh.mm),
    azimuth, T/p/Rh zmierzone na ziemi. Zwraca liste dict."""
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
        if "date" not in row or "time" not in row or not row["date"] or not row["time"]:
            continue

        def f(k, default=0.0):
            try:
                return float(row.get(k, default))
            except ValueError:
                return default

        try:
            d = dt.datetime.strptime(row["date"], "%d.%m.%Y").date()
            hh, mm = row["time"].split(".")
            t = dt.time(int(hh), int(mm))
        except ValueError:
            continue

        rows.append(dict(
            fno=fno, date=d, time=t,
            azimuth_deg=f("azimuth"), elevation_deg=f("elevation"),
            T_C_ground=f("T [C]"), p_hpa_ground=f("p [hpa]"), RH_pct_ground=f("Rh [%]"),
            analyze=row.get("to analyze?", "no").strip().lower() == "yes",
        ))
    rows.sort(key=lambda r: r["fno"])
    return rows


def fetch_weather(lat, lon, date, tz="Europe/Warsaw"):
    params = dict(
        latitude=lat, longitude=lon,
        start_date=date.isoformat(), end_date=date.isoformat(),
        hourly=",".join(HOURLY_VARS), timezone=tz,
    )
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def nearest_hour_value(weather_json, var, target_time):
    times = weather_json["hourly"]["time"]
    values = weather_json["hourly"][var]
    target_minutes = target_time.hour * 60 + target_time.minute
    best_i, best_diff = 0, None
    for i, t in enumerate(times):
        hh, mm = int(t[11:13]), int(t[14:16])
        diff = abs(hh * 60 + mm - target_minutes)
        if best_diff is None or diff < best_diff:
            best_diff, best_i = diff, i
    return values[best_i]


def main():
    parser = argparse.ArgumentParser(
        description="Rzeczywista pogoda (Open-Meteo) w dniu/godzinie kazdego lotu")
    parser.add_argument("--only-analyzed", action="store_true",
                         help="tylko loty z 'to analyze? == yes' w configs.txt")
    args = parser.parse_args()

    base = get_data_dir()
    root = Path(base).parent
    lat, lon = read_launch_site(root)

    flights = read_flights(base)
    if args.only_analyzed:
        flights = [r for r in flights if r["analyze"]]

    print(f"Wyrzutnia: lat={lat:.5f}, lon={lon:.5f}")
    print(f"\n{'lot':>4} {'data':>10} {'czas':>5} {'az':>5} "
          f"{'V_wiatr':>8} {'kierunek_z':>10} {'gust':>6} {'rel_az':>7} "
          f"{'T_API':>6} {'T_grnd':>6} {'p_API':>7} {'p_grnd':>7}")

    out_rows = []
    weather_cache = {}
    for r in flights:
        key = r["date"]
        if key not in weather_cache:
            weather_cache[key] = fetch_weather(lat, lon, r["date"])
        wj = weather_cache[key]

        v_wind = nearest_hour_value(wj, "windspeed_10m", r["time"])
        dir_from = nearest_hour_value(wj, "winddirection_10m", r["time"])
        gust = nearest_hour_value(wj, "windgusts_10m", r["time"])
        t_api = nearest_hour_value(wj, "temperature_2m", r["time"])
        p_api = nearest_hour_value(wj, "surface_pressure", r["time"])

        # Kierunek wiatru wzgledem azymutu strzalu: 0=headwind, 90/270=
        # crosswind, 180=tailwind — ta sama konwencja co
        # analyze_wind_sensitivity.py (WIND_HEADINGS_REL)
        rel_az = (dir_from - r["azimuth_deg"]) % 360.0

        print(f"{r['fno']:4d} {r['date'].isoformat():>10} "
              f"{r['time'].strftime('%H:%M'):>5} {r['azimuth_deg']:5.0f} "
              f"{v_wind:8.1f} {dir_from:10.0f} {gust:6.1f} {rel_az:7.0f} "
              f"{t_api:6.1f} {r['T_C_ground']:6.1f} {p_api:7.1f} {r['p_hpa_ground']:7.1f}")

        out_rows.append(dict(
            fno=r["fno"], date=r["date"].isoformat(), time=r["time"].strftime("%H:%M"),
            azimuth_deg=r["azimuth_deg"],
            wind_speed_10m_mps=v_wind, wind_dir_from_deg=dir_from,
            wind_gust_10m_mps=gust, wind_dir_rel_azimuth_deg=rel_az,
            T_C_api=t_api, T_C_ground=r["T_C_ground"],
            p_hpa_api=p_api, p_hpa_ground=r["p_hpa_ground"],
        ))

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "launch_weather_openmeteo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nZapisano: {csv_path}")


if __name__ == "__main__":
    main()
