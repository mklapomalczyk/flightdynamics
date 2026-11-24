"""
roll_accuracy_table.py
======================
Liczbowa dokładność przewidywania prędkości obrotowej (roll) przez model, w
kilku charakterystycznych punktach każdego lotu.

Dla każdego lotu bierzemy 5 punktów:
  1. maks roll              — szczyt prędkości obrotowej (przy wypaleniu),
  2. 1/2 w narastaniu       — pierwsza chwila, gdy roll = 50% maks (faza ciągu),
  3. apogeum,
  4. 1/2 opadania           — w połowie czasu od apogeum do upadku,
  5. upadek − 2 s.

Porównujemy POMIAR vs MODEL (całka równania rolki, CLL×scale, Clp×scale,
napędzana rzeczywistą obwiednią V(t)). Żeby WYKLUCZYĆ wpływ profilu prędkości
lotu, kluczową miarą jest BEZWYMIAROWE p·d/2V — w nim równowaga roll nie zależy
od V, więc porównanie testuje samą aerodynamikę roll. Podajemy też surowe p.

UWAGA: model płetwowy obowiązuje do apogeum (napęd zaklinowania). Punkty w
OPADANIU (4,5) są zwykle poza jego ważnością (spadochron / dekoniugacja) —
pokazujemy je, żeby zobaczyć, gdzie i jak model przestaje działać.

Użycie
------
    python telemetry_analysis/roll_accuracy_table.py
    python telemetry_analysis/roll_accuracy_table.py 20 22 25
    python telemetry_analysis/roll_accuracy_table.py --cll-scale 4 --clp-scale 1
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telemetry_parser import parse_telemetry, resolve_data_file, read_config_rows
from imu_reconstruction import detect_ignition
from analyze_spin_cant import (axial_speed, _smooth, extract_spin, detect_burnout,
                               isa_sound_speed, D_REF, get_data_dir)
from compare_roll_model import (model_roll_interpolators, integrate_roll,
                                _gps_speed)

RESULTS = get_data_dir() / "results"


def full_envelope(tel, t_ign, t_apo):
    """V(t) na cały lot: max(acc_x,GPS) do apogeum, GPS w opadaniu."""
    t = tel.time
    Vgps = _smooth(_gps_speed(tel), 25)
    Vax = axial_speed(tel, t_ign)
    good = np.isfinite(Vgps) & (Vgps > 5) & (t > t_ign)
    if Vax is None:
        return np.where(good, Vgps, np.nan)
    V = np.where(good, np.maximum(Vax, Vgps), Vax)         # ciąg+coast
    V = np.where((t >= t_apo) & good, Vgps, V)             # opadanie -> GPS
    return V


def detect_impact(tel, t_ign, t_apo):
    """Czas upadku: pierwszy powrót wysokości do poziomu startu po apogeum."""
    t = tel.time
    alt = tel.alt_onboard
    if alt is None or not np.any(np.isfinite(alt)):
        return float(t[-1])
    pre = t < t_ign
    h0 = float(np.nanmedian(alt[pre])) if np.any(pre) else float(np.nanmin(alt))
    below = (t > t_apo) & np.isfinite(alt) & (alt <= h0 + 5.0)
    if np.any(below):
        return float(t[np.argmax(below)])
    return float(t[-1])


def flight_points(t, p_meas, t_ign, t_apo, t_imp):
    """Zwraca listę (etykieta, czas) dla 5 punktów."""
    w = (t > t_ign) & (t <= t_apo + 0.2)
    idx = np.where(w)[0]
    imax = idx[int(np.nanargmax(p_meas[idx]))]
    t_max, p_max = t[imax], p_meas[imax]
    rise = (t > t_ign) & (t < t_max) & (p_meas >= 0.5 * p_max)
    t_half = float(t[np.where(rise)[0][0]]) if np.any(rise) else np.nan
    return [("maks roll",   float(t_max)),
            ("1/2 narast.",  t_half),
            ("apogeum",      float(t_apo)),
            ("1/2 opadania", float(t_apo + 0.5 * (t_imp - t_apo))),
            ("upadek-2s",    float(t_imp - 2.0))]


def analyze_flight(fno, cant, cll_s, clp_s, cll_of, clp_of):
    fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
    if not fname.exists():
        return None
    tel = parse_telemetry(fname, verbose=False)
    t = tel.time
    t_ign = detect_ignition(tel)
    p_meas, ok = extract_spin(tel, t_ign)
    if not ok:
        return None
    p_meas = _smooth(p_meas, 25)
    alt = tel.alt_onboard
    t_apo = float(t[int(np.nanargmax(alt))]) if (
        alt is not None and np.any(np.isfinite(alt))) else float(t[-1])
    t_imp = detect_impact(tel, t_ign, t_apo)
    t_bo = detect_burnout(tel, t_ign) or (t_ign + 2.0)

    V = full_envelope(tel, t_ign, t_apo)
    p_model = integrate_roll(t, V, alt, cant, t_ign, t_bo, t_imp,
                             cll_of, clp_of, cll_s, clp_s) * 180.0 / np.pi

    rows = []
    for label, tp in flight_points(t, p_meas, t_ign, t_apo, t_imp):
        if not np.isfinite(tp):
            continue
        i = int(np.argmin(np.abs(t - tp)))
        Vi = V[i]
        pm, pc = p_meas[i], p_model[i]
        nd_m = (pm * np.pi / 180.0) * (D_REF / 2.0) / Vi if (Vi and Vi > 5) else np.nan
        nd_c = (pc * np.pi / 180.0) * (D_REF / 2.0) / Vi if (Vi and Vi > 5) else np.nan
        rows.append(dict(fno=fno, cant=cant, point=label, t=tp - t_ign,
                         V=Vi, p_meas=pm, p_model=pc,
                         pd2V_meas=nd_m, pd2V_model=nd_c,
                         ratio=(pc / pm if pm > 30 else np.nan),
                         phase=("opadanie" if tp > t_apo else "wznoszenie")))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", type=int)
    ap.add_argument("--cll-scale", type=float, default=4.0)
    ap.add_argument("--clp-scale", type=float, default=1.0)
    ap.add_argument("--nose", default="ostra")
    args = ap.parse_args(argv)

    cll_of, clp_of, _ = model_roll_interpolators()
    if cll_of is None:
        print("Brak tabel DATCOM roll.")
        return 1

    cfg = {}
    for row in read_config_rows("configs.txt"):
        try:
            fno = int(str(row.get("flight no", "")).strip())
        except (ValueError, TypeError):
            continue
        head = str(row.get("head configuration", "")).strip().lower()
        try:
            cant = float(str(row.get("cant angle", "")).strip())
        except ValueError:
            cant = np.nan
        cfg[fno] = dict(cant=cant, nose="tępa" if head.startswith("t") else "ostra",
                        analyze=str(row.get("to analyze?", "")).strip().lower() == "yes")

    flights = args.flights or [f for f, c in cfg.items()
                               if c["analyze"] and c["nose"] == args.nose]

    print(f"Dokładność roll modelu (CLL×{args.cll_scale:g}, Clp×{args.clp_scale:g}); "
          f"p·d/2V wyklucza prędkość lotu.\n")
    hdr = (f"{'lot':>3} {'cant':>4} {'punkt':<13}{'t[s]':>6}{'V':>5}"
           f"{'p_test':>8}{'p_mod':>8}{'pd2V_t':>9}{'pd2V_m':>9}{'mod/test':>9}")
    all_rows = []
    for fno in flights:
        c = cfg.get(fno)
        if not c or not np.isfinite(c["cant"]):
            continue
        rows = analyze_flight(fno, c["cant"], args.cll_scale, args.clp_scale,
                              cll_of, clp_of)
        if not rows:
            continue
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            mark = " *" if r["phase"] == "opadanie" else ""
            print(f"{r['fno']:>3} {r['cant']:>4} {r['point']:<13}{r['t']:>6.1f}"
                  f"{r['V']:>5.0f}{r['p_meas']:>8.0f}{r['p_model']:>8.0f}"
                  f"{r['pd2V_meas']:>9.4f}{r['pd2V_model']:>9.4f}"
                  f"{r['ratio']:>8.2f}{mark}")
        print()
        all_rows.extend(rows)

    # agregat dokładności: tylko wznoszenie (model ważny), po p·d/2V
    asc = [r for r in all_rows if r["phase"] == "wznoszenie"
           and np.isfinite(r["ratio"])]
    if asc:
        err = np.array([abs(r["ratio"] - 1.0) for r in asc])
        print("=" * 56)
        print(f"Dokładność na WZNOSZENIU (model ważny), n={len(asc)} punktów:")
        print(f"  średni |model/test − 1| = {100*np.mean(err):.0f}%   "
              f"mediana = {100*np.median(err):.0f}%")
        for lab in ("maks roll", "1/2 narast.", "apogeum"):
            e = [abs(r["ratio"]-1) for r in asc if r["point"] == lab]
            if e:
                print(f"    {lab:<13}: {100*np.mean(e):.0f}%")
        print("  (* = opadanie: poza ważnością modelu płetwowego)")

    if all_rows:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / "roll_accuracy_table.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\nZapisano: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
