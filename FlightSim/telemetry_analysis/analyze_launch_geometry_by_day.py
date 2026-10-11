"""
analyze_launch_geometry_by_day.py
==================================
Rozdziela obserwowane odchylenie azymutu (GPS vs configs.txt) na DWIE
przyczyny, grupujac loty PO DNIU STARTU (zalozenie: wyrzutnia ustawiona
raz na dzien, nie ruszana miedzy lotami tego samego dnia -- configs.txt
potwierdza, ze kazdy dzien ma jeden zapisany azymut/elewacje):

  1. Blad ustawienia wyrzutni (STALY danego dnia, niezalezny od wiatru)
     -- np. zla kalibracja kompasu/celownika.
  2. Znoszenie wiatrem (ZMIENNE lot od lotu, zalezne od predkosci wiatru
     tego dnia) -- weathercocking podczas wznoszenia + dryf w fazie
     balistycznej.

Metoda:
  - Dla kazdego lotu z telemetria: azymut GPS (heading zaplon->apogeum,
    patrz estimate_azimuth_from_telemetry) i elewacja IMU (patrz
    estimate_elevation_from_telemetry) -- REUZYTE z
    plot_flight_trajectory_6dof.py, zadnej nowej fizyki.
  - d_az = wrap180(GPS - configs), per lot.
  - Zmienna niezalezna to SKLADOWA BOCZNA wiatru (crosswind), nie surowa
    predkosc: crosswind = V_wiatr * sin(rel_az), gdzie rel_az = kierunek
    wiatru wzgledem zapisanego azymutu (wind_dir_rel_azimuth_deg z
    launch_weather_openmeteo.csv). Headwind/tailwind (rel_az~0/180) nie
    powinien obracac heading lotu, tylko crosswind (rel_az~90/270) —
    sprawdzone empirycznie: dla wszystkich lotow z wiatrem korelacja
    d_az~crosswind = 0.65, d_az~surowa_predkosc = 0.07 (patrz notatki
    sesji). Uzycie samej predkosci (bez kierunku) myli loty z silnym
    wiatrem od tylu/przodu (maly efekt heading) z lotami z silnym
    wiatrem bocznym (duzy efekt) -- np. lot 13 mial najsilniejszy wiatr
    (20.5 m/s) ale blisko tylnego (rel_az=159 deg), wiec crosswind byl
    umiarkowany i nie pasowalby do trendu "wiecej wiatru = wieksze
    odchylenie" gdyby uzyc surowej predkosci.
  - Per dzien z >=2 lotami I wystarczajaca wariancja crosswind:
    regresja liniowa d_az ~ a + b*crosswind. Intercept "a" = odchylenie
    przy crosswind=0 -> szacunek bledu USTAWIENIA wyrzutni tego dnia
    (osobno od wiatru). Slope "b" = wrazliwosc znoszenia [deg / (m/s)].
  - Gdy wariancja crosswind w danym dniu za mala (typowe dla danych
    Open-Meteo -- rozdzielczosc godzinowa, podobny odczyt dla lotow z
    tego samego dnia) -- regresja per-dzien NIE separuje przyczyn;
    zamiast tego raportujemy srednia/odchylenie std per dzien.

WAZNE OGRANICZENIE (przeczytaj przed interpretacja "trendu globalnego"):
kazdy dzien startu to ROWNOCZESNIE inny fizyczny setup wyrzutni (ustawiana
od nowa) I inne warunki wiatrowe -- te dwa efekty sa ZE SOBA SKOREOWANE
(confounded) w danych z 4 dni. "Trend globalny" (regresja przez wszystkie
dni razem) jest wiec TYLKO orientacyjny, NIE dowodem przyczynowosci: nie
mozna wykluczyc, ze roznica miedzy dniami wynika glownie z innego bledu
ustawienia kazdego dnia, a nie z wiatru. Najbardziej wiarygodny sygnal to
ROZRZUT W OBREBIE dnia (miedzy powtorzonymi lotami tego samego ustawienia)
-- ten jest niezalezny od bledu ustawienia z definicji.

Wymaga: field_test_data/results/launch_weather_openmeteo.csv (wygenerowany
lokalnie przez analyze_launch_weather.py -- wymaga internetu, wiec loty
bez wpisu w tym CSV (np. "to analyze? = no") sa pokazywane bez wiatru.

Wyjscie (field_test_data/results/):
  - launch_geometry_by_day.csv          -- per-lot: dzien, az/el GPS+IMU,
                                            odchylenia, wiatr (jesli jest)
  - launch_geometry_day_summary.csv     -- per-dzien: srednia/std,
                                            regresja (jesli mozliwa)
  - launch_geometry_wind_vs_azdev.png   -- d_az vs crosswind, kolor=dzien,
                                            linie regresji per-dzien (gdy
                                            mozliwe) + globalny trend
                                            (orientacyjny, patrz WAZNE
                                            OGRANICZENIE powyzej)
  - launch_geometry_day_summary.png     -- bar chart: d_az i d_el per dzien

Uzycie:
    python analyze_launch_geometry_by_day.py
"""

import sys
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry_parser import get_data_dir, resolve_data_file
from analyze_launch_weather import read_flights
from analyze_wind_sensitivity import read_measured_wind
from plot_flight_trajectory_6dof import (
    estimate_azimuth_from_telemetry,
    estimate_elevation_from_telemetry,
)
from compare_launch_geometry import wrap180, LOW_BASELINE_M, HIGH_OFFPLANE_DEG

MIN_WIND_STD_FOR_FIT = 0.5   # m/s (crosswind) -- ponizej tego regresja per-dzien niewiarygodna
MIN_FLIGHTS_FOR_FIT  = 2


def collect_flight_rows(base):
    """Zbiera per-lot: dzien, configs az/el, GPS az, IMU el, wiatr (jesli
    dostepny), flagi jakosci. Pomija loty bez pliku telemetrii."""
    rows = []
    for r in sorted(read_flights(base), key=lambda x: (x["date"], x["time"])):
        fno = r["fno"]
        if not resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt").exists():
            continue
        try:
            gps_az, baseline = estimate_azimuth_from_telemetry(base, fno)
            imu_el, off_plane = estimate_elevation_from_telemetry(base, fno)
        except Exception as e:
            print(f"Lot {fno}: blad estymacji ({e}) -- pomijam.")
            continue

        mw = read_measured_wind(base, fno)
        if mw is not None:
            wind_mps = mw["mean_speed_mps"]
            # crosswind = skladowa boczna wzgledem ZAPISANEGO azymutu
            # (rel_az_deg juz liczony wzgledem configs.txt azymutu, patrz
            # analyze_wind_sensitivity.read_measured_wind/analyze_launch_weather.py)
            crosswind_mps = wind_mps * math.sin(math.radians(mw["rel_az_deg"]))
            headwind_mps  = wind_mps * math.cos(math.radians(mw["rel_az_deg"]))
        else:
            wind_mps = crosswind_mps = headwind_mps = float("nan")

        d_az = wrap180(gps_az - r["azimuth_deg"])
        d_el = imu_el - r["elevation_deg"]

        flags = []
        if baseline < LOW_BASELINE_M:
            flags.append("krotka-baza")
        if off_plane > HIGH_OFFPLANE_DEG:
            flags.append("duzy-off-plane")
        if mw is None:
            flags.append("brak-wiatru")

        rows.append(dict(
            fno=fno, date=r["date"].isoformat(), time=r["time"].strftime("%H:%M"),
            cfg_az=r["azimuth_deg"], gps_az=gps_az, d_az=d_az, baseline_m=baseline,
            cfg_el=r["elevation_deg"], imu_el=imu_el, d_el=d_el, off_plane_deg=off_plane,
            wind_mps=wind_mps, crosswind_mps=crosswind_mps, headwind_mps=headwind_mps,
            analyze=r["analyze"], flags=",".join(flags),
        ))
    return rows


def summarize_by_day(rows):
    """Grupuje per dzien; gdzie mozliwe, dopasowuje d_az ~ a + b*crosswind
    (skladowa boczna wiatru, patrz docstring modulu -- korelacja z d_az
    duzo silniejsza niz dla surowej predkosci). Zwraca liste dictow (jeden
    na dzien), posortowana chronologicznie."""
    by_day = defaultdict(list)
    for row in rows:
        by_day[row["date"]].append(row)

    summaries = []
    for date, day_rows in sorted(by_day.items()):
        d_az_all = np.array([r["d_az"] for r in day_rows])
        d_el_all = np.array([r["d_el"] for r in day_rows])
        wind_all = np.array([r["wind_mps"] for r in day_rows])
        cross_all = np.array([r["crosswind_mps"] for r in day_rows])
        cfg_az = day_rows[0]["cfg_az"]
        cfg_el = day_rows[0]["cfg_el"]

        has_wind = ~np.isnan(cross_all)
        n_wind = int(has_wind.sum())
        cross_std = float(np.std(cross_all[has_wind])) if n_wind >= 2 else 0.0

        fit = None
        if n_wind >= MIN_FLIGHTS_FOR_FIT and cross_std >= MIN_WIND_STD_FOR_FIT:
            slope, intercept = np.polyfit(cross_all[has_wind], d_az_all[has_wind], 1)
            fit = dict(intercept=float(intercept), slope=float(slope))

        summaries.append(dict(
            date=date, n_flights=len(day_rows), cfg_az=cfg_az, cfg_el=cfg_el,
            d_az_mean=float(np.mean(d_az_all)), d_az_std=float(np.std(d_az_all)),
            d_el_mean=float(np.mean(d_el_all)), d_el_std=float(np.std(d_el_all)),
            wind_mean=float(np.nanmean(wind_all)) if n_wind else float("nan"),
            crosswind_mean=float(np.nanmean(cross_all)) if n_wind else float("nan"),
            crosswind_std=cross_std, n_wind=n_wind,
            fit_intercept=fit["intercept"] if fit else float("nan"),
            fit_slope=fit["slope"] if fit else float("nan"),
            fit_note=("regresja OK" if fit else
                      ("za malo lotow z wiatrem" if n_wind < MIN_FLIGHTS_FOR_FIT
                       else "za mala wariancja crosswind w tym dniu")),
        ))
    return summaries


def plot_wind_vs_azdev(rows, summaries, out_png):
    """d_az vs V_wiatr, kolor=dzien. Linia regresji per-dzien gdy
    wystarczajaca wariancja wiatru tego dnia (patrz summarize_by_day);
    dodatkowo globalny trend (WSZYSTKIE dni polaczone) jako orientacyjna
    linia przerywana -- ALE dni startu i warunki wiatrowe sa ze soba
    skorelowane (kazdy dzien = inny setup wyrzutni I inny wiatr), wiec
    ten trend NIE jest dowodem przyczynowosci, patrz WAZNE OGRANICZENIE
    w docstringu modulu."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = plt.cm.tab10.colors
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)

    all_cross, all_daz = [], []
    fit_by_date = {s["date"]: s for s in summaries}
    for i, (date, day_rows) in enumerate(sorted(by_day.items())):
        c = colors[i % len(colors)]
        cfg_az = day_rows[0]["cfg_az"]
        cross = np.array([r["crosswind_mps"] for r in day_rows])
        daz = np.array([r["d_az"] for r in day_rows])
        mask = ~np.isnan(cross)
        if mask.sum() == 0:
            continue
        ax.scatter(cross[mask], daz[mask], color=c, s=70, zorder=3,
                   label=f"{date} (az_cfg={cfg_az:.0f}°, n={int(mask.sum())})")
        for r in [dr for dr, m in zip(day_rows, mask) if m]:
            ax.annotate(str(r["fno"]), (r["crosswind_mps"], r["d_az"]),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
        all_cross.extend(cross[mask].tolist())
        all_daz.extend(daz[mask].tolist())

        s = fit_by_date.get(date)
        if s is not None and not math.isnan(s["fit_intercept"]):
            xs = np.linspace(min(0, min(cross[mask])), max(cross[mask]) * 1.15, 20)
            ys = s["fit_intercept"] + s["fit_slope"] * xs
            ax.plot(xs, ys, color=c, lw=1.2, ls="--", alpha=0.7)

    if len(set(all_cross)) >= 2:
        slope, intercept = np.polyfit(all_cross, all_daz, 1)
        r = np.corrcoef(all_cross, all_daz)[0, 1]
        xs = np.linspace(min(0, min(all_cross)), max(all_cross) * 1.1, 20)
        ax.plot(xs, intercept + slope * xs, color="k", lw=1.8, ls=":",
                label=f"trend globalny (orientacyjny, r={r:.2f}): "
                      f"{intercept:+.1f} + {slope:+.2f}·crosswind")

    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("skladowa boczna wiatru (crosswind) tego lotu [m/s]\n"
                 "(dodatnia = wiatr z prawej strony toru lotu)")
    ax.set_ylabel("odchylenie azymutu: GPS (zaplon->apogeum) - configs.txt [deg]")
    ax.set_title("Odchylenie azymutu vs skladowa boczna wiatru (crosswind)\n"
                "(etykiety punktow = nr lotu; linie przerywane = regresja per-dzien "
                "gdy mozliwa; trend globalny orientacyjny -- dzien i wiatr sa skorelowane)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    base = get_data_dir()
    rows = collect_flight_rows(base)
    if not rows:
        print("Brak lotow z telemetria.")
        return
    summaries = summarize_by_day(rows)

    print(f"{'flt':>3} {'data':>10} {'cfg_az':>6} {'GPS_az':>7} {'d_az':>6} "
          f"{'wiatr':>6} {'cross':>6} | {'cfg_el':>6} {'IMU_el':>7} {'d_el':>6}  flagi")
    print("-" * 96)
    for r in rows:
        w  = f"{r['wind_mps']:.1f}"      if not math.isnan(r["wind_mps"])      else "  --"
        cw = f"{r['crosswind_mps']:+.1f}" if not math.isnan(r["crosswind_mps"]) else "  --"
        print(f"{r['fno']:>3} {r['date']:>10} {r['cfg_az']:>6.0f} {r['gps_az']:>7.1f} "
              f"{r['d_az']:>+6.1f} {w:>6} {cw:>6} | {r['cfg_el']:>6.0f} {r['imu_el']:>7.1f} "
              f"{r['d_el']:>+6.1f}  {r['flags']}")

    print(f"\n{'=== Podsumowanie per dzien ===':^96}")
    print(f"{'data':>10} {'n':>2} {'cfg_az':>6} {'d_az_mean':>10} {'d_az_std':>9} "
          f"{'cross_mean':>10} | {'d_el_mean':>10} {'d_el_std':>9} | dopasowanie")
    for s in summaries:
        print(f"{s['date']:>10} {s['n_flights']:>2} {s['cfg_az']:>6.0f} "
              f"{s['d_az_mean']:>+10.1f} {s['d_az_std']:>9.1f} "
              f"{s['crosswind_mean']:>+10.1f} | {s['d_el_mean']:>+10.1f} {s['d_el_std']:>9.1f} | ",
              end="")
        if not math.isnan(s["fit_intercept"]):
            print(f"a(cross=0)={s['fit_intercept']:+.1f} deg, b={s['fit_slope']:+.2f} deg/(m/s)")
        else:
            print(s["fit_note"])

    out_dir = Path(base) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "launch_geometry_by_day.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(out_dir / "launch_geometry_day_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)

    plot_wind_vs_azdev(rows, summaries, out_dir / "launch_geometry_wind_vs_azdev.png")
    plot_day_summary(summaries, out_dir / "launch_geometry_day_summary.png")

    print(f"\nZapisano:\n  {out_dir/'launch_geometry_by_day.csv'}\n"
          f"  {out_dir/'launch_geometry_day_summary.csv'}\n"
          f"  {out_dir/'launch_geometry_wind_vs_azdev.png'}\n"
          f"  {out_dir/'launch_geometry_day_summary.png'}")


def plot_day_summary(summaries, out_png):
    dates = [s["date"] for s in summaries]
    d_az = [s["d_az_mean"] for s in summaries]
    d_az_err = [s["d_az_std"] for s in summaries]
    d_el = [s["d_el_mean"] for s in summaries]
    d_el_err = [s["d_el_std"] for s in summaries]
    wind = [s["wind_mean"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    x = np.arange(len(dates))

    bars = ax1.bar(x, d_az, yerr=d_az_err, capsize=4, color="tab:blue", alpha=0.8)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("srednie odchylenie azymutu\nGPS - configs.txt [deg]")
    ax1.set_title("Odchylenie azymutu i elewacji per dzien startu\n"
                  "(slupki bledu = odchylenie std miedzy lotami tego dnia)")
    for xi, s in zip(x, summaries):
        cw = s["crosswind_mean"]
        wl = (f"wiatr={s['wind_mean']:.1f} m/s\ncrosswind={cw:+.1f} m/s"
              if not math.isnan(cw) else "wiatr: brak")
        ax1.annotate(f"n={s['n_flights']}\n{wl}", (xi, d_az[x.tolist().index(xi)]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x, d_el, yerr=d_el_err, capsize=4, color="tab:green", alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("srednie odchylenie elewacji\nIMU - configs.txt [deg]")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d}\n(az={s['cfg_az']:.0f}°)" for d, s in zip(dates, summaries)])
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
