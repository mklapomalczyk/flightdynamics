"""
analyze_impact_point.py
=========================
Testuje hipoteze "loty byly krotsze/nizsze niz model bo czegos brakuje w
modelu" przez analize PRAWDZIWEGO punktu ladowania z GPS — do samego konca
danych telemetrii, z FLAGA SPADOCHRONU CALKOWICIE ZIGNOROWANA (nie sluzy do
obciecia/wyboru danych; tak jak diag_drag.detect_events() i tak juz
definiuje "koniec danych" = ostatnia probka, bez uzycia flagi).

Co skrypt liczy, per lot (z configs.txt, 'to analyze? == yes'):
  1. Punkt ladowania z GPS (downrange/crossrange/range wzgledem azymutu
     strzalu, ta sama konwencja Launch Frame co analyze_per_flight_6dof.py)
     — niezaleznie od flag_flight.
  2. WIARYGODNOSC tego punktu — DWA niezalezne sprawdzenia:
     a) Empiryczna predkosc opadania (dh/dt, wygladzona, diag_drag.
        smooth_gps_derivative) w ostatnich sekundach przed ladowaniem.
        Spadochron => stabilna, NISKA predkosc (rzedu kilku-kilkunastu
        m/s). Szybkie/przyspieszajace opadanie => podejrzenie awarii
        spadochronu LUB zlej probki koncowej (zamrozenie/skok GPS).
     b) Energia w apogeum jako sanity-check przemieszczenia poziomego
        miedzy apogeum i ladowaniem: idealizowany (bez oporu) czas
        spadania t_fall=sqrt(2*h_apo/g) razy resztkowa predkosc
        horyzontalna w apogeum Vh_apo daje GORNA GRANICE przemieszczenia
        BEZWIETRZNEGO/bezoporowego. Do tego dodajemy znoszenie wiatrem
        (zmierzony gust z launch_weather_openmeteo.csv jesli istnieje,
        inaczej konserwatywne 15 m/s) * czas opadania pod chute. Jesli
        zaobserwowane przemieszczenie > ta SUMA -> fizycznie niemozliwe
        bez dodatkowej energii -> najpewniej zly punkt GPS (skok/
        zamrozenie), NIE realny lot.
     UWAGA: (b) NIE jest predykcja punktu ladowania (chute+wiatr to
     zewnetrzna praca, nieograniczona energia rakiety w apogeum) — to
     tylko test "czy to jest fizycznie mozliwe", nie "czy to jest
     prawdopodobne".
  3. Jesli istnieje field_test_data/results/per_flight_6dof_per_nose.csv
     (z analyze_per_flight_6dof.py, wymaga DATCOM -> lokalnie/Windows):
     dolaczany jest BALISTYCZNY (bez chutu) punkt/czas upadku z modelu
     6DOF, wylacznie jako odniesienie czasu lotu/zasiegu DOLNEJ GRANICY
     -- model nie ma chutu, wiec nie jest to oczekiwany rownowazny wynik.

Ten skrypt NIE wymaga DATCOM — moze byc uruchomiony w pelni w kontenerze.

Uzycie:
    python analyze_impact_point.py            # wszystkie loty z configs.txt
    python analyze_impact_point.py 18 20
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_parser import get_data_dir, parse_telemetry, resolve_data_file
from imu_reconstruction import detect_ignition, G0
from diag_drag import detect_events, smooth_gps_derivative
from gps_fusion import latlon_to_enu
from analyze_per_flight_6dof import read_flights, downrange_crossrange, actual_apogee_vmax
from analyze_wind_sensitivity import read_measured_wind

DEFAULT_WIND_GUST_MPS = 15.0   # gdy brak launch_weather_openmeteo.csv dla lotu


# --------------------------------------------------------------------------
def horizontal_speed_at(tel, e, n, idx, window_s=0.6):
    """Predkosc horyzontalna w probce idx z regresji liniowej e(t)/n(t)
    w oknie +/-window_s wokol idx (stabilniejsze niz pojedyncza roznica
    schodkowego GPS)."""
    t = tel.time
    mask = np.abs(t - t[idx]) <= window_s
    if np.sum(mask) < 3:
        mask = slice(max(0, idx - 2), idx + 3)
    tt = t[mask]
    if np.ptp(tt) <= 0:
        return 0.0
    ve = np.polyfit(tt, e[mask], 1)[0]
    vn = np.polyfit(tt, n[mask], 1)[0]
    return float(np.hypot(ve, vn))


def terminal_descent_rate(tel, i_apo, i_end, frac=0.2):
    """Srednia predkosc opadania (m/s, dodatnia=opada) w ostatnim `frac`
    odcinku czasu miedzy apogeum i koncem danych -- empiryczny test
    'czy to wyglada na lot pod spadochronem'."""
    seg = slice(i_apo, i_end + 1)
    if i_end - i_apo < 5:
        return float("nan")
    dhdt = smooth_gps_derivative(tel.alt_onboard[seg], tel.time[seg], window_s=0.6)
    n = len(dhdt)
    tail = dhdt[int(n * (1 - frac)):]
    return float(-np.mean(tail))   # dh/dt < 0 podczas opadania -> dodatnia predkosc opadania


def gps_quality_near(tel, idx, n_check=5):
    """Sprawdza czy probka idx (lub jej bezposrednie sasiedztwo) lezy w
    zamrozeniu/po duzym skoku GPS (ten sam mechanizm co diag_gps_jumps.py).
    Zwraca opis tekstowy."""
    lat = tel.lat; lon = tel.lon
    e, n = latlon_to_enu(lat, lon, lat[0], lon[0])
    lo, hi = max(0, idx - n_check), min(len(e) - 1, idx + n_check)
    de = np.diff(e[lo:hi + 1]); dn = np.diff(n[lo:hi + 1])
    dist = np.hypot(de, dn)
    dt = np.diff(tel.time[lo:hi + 1])
    dt = np.where(dt > 0, dt, np.nan)
    v_impl = dist / dt
    frozen = np.sum(dist < 1e-6) >= (len(dist) - 1)
    jumpy = np.any(v_impl > 400)
    if frozen:
        return "ZAMROZONE (brak fixa GPS w tej probce)"
    if jumpy:
        return f"SKOK (implikowana V={np.nanmax(v_impl):.0f} m/s w sasiedztwie)"
    return "ok"


# --------------------------------------------------------------------------
def load_model_impact(base, fno):
    """Czyta z per_flight_6dof_per_nose.csv (analyze_per_flight_6dof.py,
    wymaga DATCOM -> lokalnie/Windows) wszystkie wielkosci modelu 6DOF
    (silnik 'adjusted' = rzeczywisty profil ciagu tego lotu) potrzebne do
    pelnego porownania z danymi polowymi: apogeum, V_max, downrange/
    crossrange przy apogeum, oraz BALISTYCZNY (bez chutu) punkt/czas/
    predkosc upadku (= koniec calkowania solvera, ground_event z>=0 --
    NIE jest to przewidywanie realnego ladowania pod spadochronem, tylko
    odniesienie/dolna granica zasiegu i czasu lotu). Zwraca None jesli
    plik/wiersz nie istnieje."""
    csv_path = Path(base) / "results" / "per_flight_6dof_per_nose.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["fno"]) == fno:
                try:
                    return dict(
                        h_apo_m=float(row["h_apo_pred_adjusted"]),
                        v_max_mps=float(row["v_max_pred_adjusted"]),
                        downrange_apo_m=float(row["downrange_apo_pred_adjusted_m"]),
                        crossrange_apo_m=float(row["crossrange_apo_pred_adjusted_m"]),
                        range_apo_m=float(row["range_apo_pred_adjusted_m"]),
                        downrange_impact_m=float(row["impact_downrange_adjusted_m"]),
                        crossrange_impact_m=float(row["impact_crossrange_adjusted_m"]),
                        rng_m=float(row["impact_range_adjusted_m"]),
                        t_s=float(row["impact_t_adjusted_s"]),
                        speed_impact_mps=float(row["impact_speed_adjusted_mps"]),
                    )
                except (ValueError, KeyError):
                    return None
    return None


# --------------------------------------------------------------------------
def process_flight(base, r):
    fno = r["fno"]
    fpath = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    if not fpath.exists():
        return None
    tel = parse_telemetry(fpath, verbose=False)
    t_ign_abs = detect_ignition(tel)
    i_ign, i_bo, i_apo, i_end = detect_events(tel, t_ign_abs)
    t = tel.time

    lat0, lon0 = tel.lat[i_ign], tel.lon[i_ign]
    e, n = latlon_to_enu(tel.lat, tel.lon, lat0, lon0)

    h_apo_agl = float(tel.alt_onboard[i_apo] - tel.alt_onboard[i_ign])
    h_land_agl = float(tel.alt_onboard[i_end] - tel.alt_onboard[i_ign])

    dr_apo, cr_apo = downrange_crossrange(e[i_apo], n[i_apo], r["azimuth"])
    dr_land, cr_land = downrange_crossrange(e[i_end], n[i_end], r["azimuth"])
    rng_apo = float(np.hypot(dr_apo, cr_apo))
    rng_land = float(np.hypot(dr_land, cr_land))
    disp = float(np.hypot(dr_land - dr_apo, cr_land - cr_apo))

    t_apo = float(t[i_apo] - t[i_ign])
    t_land = float(t[i_end] - t[i_ign])
    t_descent = t_land - t_apo

    Vh_apo = horizontal_speed_at(tel, e, n, i_apo)
    t_fall_ideal = float(np.sqrt(2.0 * h_apo_agl / G0)) if h_apo_agl > 0 else float("nan")
    bound_inertial = Vh_apo * t_fall_ideal if np.isfinite(t_fall_ideal) else float("nan")

    mw = read_measured_wind(base, fno)
    wind_gust = mw["gust_speed_mps"] if mw is not None else DEFAULT_WIND_GUST_MPS
    bound_wind = wind_gust * max(t_descent, 0.0)
    bound_total = (bound_inertial if np.isfinite(bound_inertial) else 0.0) + bound_wind

    energy_ok = disp <= bound_total if np.isfinite(bound_total) else None

    v_descent = terminal_descent_rate(tel, i_apo, i_end)
    if not np.isfinite(v_descent):
        descent_class = "n/a"
    elif v_descent < 20.0:
        descent_class = "spadochron OK (wolne opadanie)"
    elif v_descent < 40.0:
        descent_class = "umiarkowane -- sprawdzic"
    else:
        descent_class = "SZYBKIE -- mozliwa awaria chute / zly punkt"

    gps_q = gps_quality_near(tel, i_end)

    # Predkosc calkowita (horyzontalna+wertykalna) w ostatniej znanej probce
    # (i_end) -- "predkosc przy uderzeniu" w sensie uzytkownika = predkosc w
    # ostatniej znanej pozycji, NIE przy realnym ladowaniu (patrz docstring
    # modulu / gps_quality_at_landing: GPS tu zwykle juz nieaktualne).
    Vh_land = horizontal_speed_at(tel, e, n, i_end)
    speed_impact_actual = float(np.hypot(Vh_land, v_descent)) if np.isfinite(v_descent) else float(Vh_land)

    h_apo_act, v_max_act = actual_apogee_vmax(base, fno)

    model = load_model_impact(base, fno)

    return dict(
        fno=fno, azimuth=r["azimuth"], h_apo_agl_m=h_apo_agl, h_land_agl_m=h_land_agl,
        t_apo_s=t_apo, t_land_s=t_land, t_descent_s=t_descent,
        downrange_apo_m=dr_apo, crossrange_apo_m=cr_apo, range_apo_m=rng_apo,
        downrange_land_m=dr_land, crossrange_land_m=cr_land, range_land_m=rng_land,
        disp_apo_to_land_m=disp,
        Vh_apo_mps=Vh_apo, t_fall_ideal_s=t_fall_ideal,
        bound_inertial_m=bound_inertial, wind_gust_mps=wind_gust,
        bound_wind_m=bound_wind, bound_total_m=bound_total,
        energy_plausible=energy_ok,
        descent_rate_mps=v_descent, descent_class=descent_class,
        gps_quality_at_landing=gps_q,
        v_max_actual_mps=(v_max_act if v_max_act is not None else float("nan")),
        speed_at_impact_actual_mps=speed_impact_actual,
        h_apo_model_m=(model["h_apo_m"] if model else float("nan")),
        v_max_model_mps=(model["v_max_mps"] if model else float("nan")),
        downrange_apo_model_m=(model["downrange_apo_m"] if model else float("nan")),
        crossrange_apo_model_m=(model["crossrange_apo_m"] if model else float("nan")),
        range_apo_model_m=(model["range_apo_m"] if model else float("nan")),
        downrange_impact_model_m=(model["downrange_impact_m"] if model else float("nan")),
        crossrange_impact_model_m=(model["crossrange_impact_m"] if model else float("nan")),
        model_impact_range_m=(model["rng_m"] if model else float("nan")),
        model_impact_t_s=(model["t_s"] if model else float("nan")),
        speed_at_impact_model_mps=(model["speed_impact_mps"] if model else float("nan")),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Punkt ladowania z GPS (flaga spadochronu zignorowana) "
                     "+ sanity-check energii w apogeum")
    parser.add_argument("flights", nargs="*", type=int)
    args = parser.parse_args()

    base = get_data_dir()
    flights = read_flights(base)
    if args.flights:
        flights = [r for r in flights if r["fno"] in args.flights]

    print(f"{'lot':>4} {'h_apo':>7} {'t_apo':>6} {'t_land':>7} {'t_desc':>7} "
          f"{'rng_apo':>8} {'rng_land':>9} {'disp':>7} {'bound':>7} {'energia':>8} "
          f"{'v_desc':>7} {'h_land':>7}  GPS@land  klasa opadania")
    out_rows = []
    for r in flights:
        res = process_flight(base, r)
        if res is None:
            print(f"{r['fno']:4d}  brak danych telemetrii -- pomijam")
            continue
        out_rows.append(res)
        ok_str = "TAK" if res["energy_plausible"] else ("NIE" if res["energy_plausible"] is False else "n/a")
        print(f"{res['fno']:4d} {res['h_apo_agl_m']:7.1f} {res['t_apo_s']:6.2f} "
              f"{res['t_land_s']:7.2f} {res['t_descent_s']:7.2f} "
              f"{res['range_apo_m']:8.1f} {res['range_land_m']:9.1f} "
              f"{res['disp_apo_to_land_m']:7.1f} {res['bound_total_m']:7.1f} {ok_str:>8} "
              f"{res['descent_rate_mps']:7.1f} {res['h_land_agl_m']:+7.1f}  "
              f"{res['gps_quality_at_landing']:<9} {res['descent_class']}")
        if not np.isnan(res["model_impact_range_m"]):
            print(f"      [model balistyczny bez chutu: zasieg={res['model_impact_range_m']:.1f}m "
                  f"t={res['model_impact_t_s']:.2f}s -- TYLKO odniesienie, nie ma chutu]")

    has_model = any(not np.isnan(res["h_apo_model_m"]) for res in out_rows)
    if has_model:
        print("\n--- Porownanie aktualny lot (GPS) vs model 6DOF (silnik 'adjusted') ---")
        print("    'impact' = ostatnia znana pozycja/predkosc (aktualny: ostatnia probka "
              "telemetrii, model: koniec calkowania balistycznego bez chutu -- NIE jest to "
              "ten sam fizyczny moment, porownanie orientacyjne, patrz docstring modulu)")
        print(f"{'lot':>4} {'h_apo_act':>9} {'h_apo_mod':>9} {'dh%':>6}  "
              f"{'Vmax_act':>8} {'Vmax_mod':>8} {'dV%':>6}  "
              f"{'dr_apo_a':>8} {'dr_apo_m':>8} {'cr_apo_a':>8} {'cr_apo_m':>8}  "
              f"{'dr_imp_a':>8} {'dr_imp_m':>8} {'cr_imp_a':>8} {'cr_imp_m':>8}  "
              f"{'rng_imp_a':>9} {'rng_imp_m':>9}  {'Vimp_act':>8} {'Vimp_mod':>8}")
        for res in out_rows:
            if np.isnan(res["h_apo_model_m"]):
                print(f"{res['fno']:4d}  -- brak modelu (uruchom analyze_per_flight_6dof.py lokalnie z DATCOM) --")
                continue
            dh = 100.0 * (res["h_apo_model_m"] - res["h_apo_agl_m"]) / res["h_apo_agl_m"]
            dv = (100.0 * (res["v_max_model_mps"] - res["v_max_actual_mps"]) / res["v_max_actual_mps"]
                  if np.isfinite(res["v_max_actual_mps"]) and res["v_max_actual_mps"] else float("nan"))
            print(f"{res['fno']:4d} {res['h_apo_agl_m']:9.1f} {res['h_apo_model_m']:9.1f} {dh:+6.1f}  "
                  f"{res['v_max_actual_mps']:8.1f} {res['v_max_model_mps']:8.1f} {dv:+6.1f}  "
                  f"{res['downrange_apo_m']:8.1f} {res['downrange_apo_model_m']:8.1f} "
                  f"{res['crossrange_apo_m']:+8.1f} {res['crossrange_apo_model_m']:+8.1f}  "
                  f"{res['downrange_land_m']:8.1f} {res['downrange_impact_model_m']:8.1f} "
                  f"{res['crossrange_land_m']:+8.1f} {res['crossrange_impact_model_m']:+8.1f}  "
                  f"{res['range_land_m']:9.1f} {res['model_impact_range_m']:9.1f}  "
                  f"{res['speed_at_impact_actual_mps']:8.1f} {res['speed_at_impact_model_mps']:8.1f}")
    else:
        print("\n[brak per_flight_6dof_per_nose.csv -- uruchom analyze_per_flight_6dof.py "
              "lokalnie (wymaga DATCOM) zeby uzyskac porownanie z modelem]")

    if not out_rows:
        print("Brak wynikow.")
        return

    n_implausible = sum(1 for r in out_rows if r["energy_plausible"] is False)
    n_fast_descent = sum(1 for r in out_rows if r["descent_class"].startswith("SZYBKIE"))
    print(f"\nPodsumowanie ({len(out_rows)} lotow):")
    print(f"  Punkty ladowania fizycznie NIEMOZLIWE (disp > bound, prawdopodobnie "
          f"zly GPS fix): {n_implausible}/{len(out_rows)}")
    print(f"  Szybkie opadanie (mozliwa awaria chute / zly punkt koncowy): "
          f"{n_fast_descent}/{len(out_rows)}")

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "impact_point_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nZapisano: {csv_path}")


if __name__ == "__main__":
    main()
