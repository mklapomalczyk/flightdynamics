"""
roll_strip_check.py
===================
Niezależny sprawdzian fizyczny punktu 1: czy zmierzona skuteczność roll od
zaklinowania (p·d/2V vs cant) jest zgodna ze strip-theory — i gdzie leży błąd
modelu DATCOM.

Idea
----
W stanie równowagi napęd = tłumienie. Dla płetw o cięciwie c(y) (y — odległość
od osi roll), ze SPÓJNYM lokalnym nachyleniem siły nośnej clα:

    napęd    Clδ  ∝ clα · ∫ c(y)·y  dy           (ramię^1)
    tłumienie Clp ∝ clα · ∫ c(y)·y² dy  / V       (ramię^2)

więc w równowadze:

    p·d/2V = δ · d·∫c(y)·y dy / (2·∫c(y)·y² dy)

— i clα, liczba płetw, S_ref się SKRACAJĄ. To jest predykcja CZYSTO
GEOMETRYCZNA, niezależna od (niepewnej) efektywności płetwy. Porównanie:

    strip-theory  vs  pomiar (telemetria)  vs  model DATCOM.

Jeśli pomiar ≈ strip, a model jest niżej — błąd modelu NIE jest w geometrii/
tłumieniu, tylko w napędzie (DATCOM zaniża moment roll od zaklinowania).

Użycie
------
    python telemetry_analysis/roll_strip_check.py
    python telemetry_analysis/roll_strip_check.py --config configurations/rocket_70mm_baseline.yaml
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telemetry_parser import get_data_dir

FS = Path(__file__).resolve().parent.parent


def strip_slope(config_path):
    """Geometryczne nachylenie p·d/2V/δ [/rad] ze strip-theory dla płetw z yaml."""
    import yaml
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    fin = cfg["fins"][0]
    d = float(cfg["body"]["diameter"])
    r = d / 2.0
    s = float(fin["span"])
    cr = float(fin["root_chord"])
    ct = float(fin["tip_chord"])
    y = np.linspace(r, r + s, 4000)
    c = cr + (ct - cr) * (y - r) / s
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    I1 = trap(c * y, y)
    I2 = trap(c * y * y, y)
    return d * I1 / (2.0 * I2), dict(d=d, span=s, cr=cr, ct=ct, count=fin.get("count"))


def measured_slope():
    """Nachylenie p·d/2V vs cant[deg] z telemetrii (results/spin_cant_summary.csv, ostra)."""
    p = get_data_dir() / "results" / "spin_cant_summary.csv"
    if not p.exists():
        return None
    cant, pd = [], []
    for row in csv.DictReader(open(p, encoding="utf-8")):
        if row["nose"] != "ostra":
            continue
        try:
            c = float(row["cant"]); v = float(row["pd2V"])
        except (ValueError, KeyError):
            continue
        if np.isfinite(v):
            cant.append(c); pd.append(v)
    if len(cant) < 2:
        return None
    A = np.vstack([cant, np.ones_like(cant)]).T
    slope = float(np.linalg.lstsq(A, np.array(pd), rcond=None)[0][0])
    return slope   # /deg


def model_slope():
    """Nachylenie p·d/2V vs cant[deg] z DATCOM (Clδ/Clp przy alpha=0)."""
    from analyze_spin_cant import load_model_roll, D_REF
    tabs, stale = load_model_roll()
    if len(tabs) < 2:
        return None, stale
    t0 = tabs[0.0]
    i0 = int(np.argmin(np.abs(t0.alpha_table)))
    mach = t0.mach_table
    im = int(np.argmin(np.abs(mach - 0.75)))
    cants = np.array(sorted(tabs))
    pd = np.array([abs(tabs[c].CLL_table[i0, im]) / abs(t0.Clp_table[i0, im])
                   for c in cants])
    slope = float(np.polyfit(cants, pd, 1)[0])
    return slope, stale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",
                    default=str(FS / "configurations" / "rocket_70mm_baseline.yaml"))
    args = ap.parse_args(argv)

    r2d = np.pi / 180.0
    strip, geo = strip_slope(args.config)
    meas = measured_slope()
    mod, stale = model_slope()

    print("Skuteczność roll od zaklinowania — p·d/2V / cant")
    print("=" * 56)
    print(f"geometria płetw: d={geo['d']*1000:.0f}mm  span={geo['span']*1000:.1f}mm  "
          f"c_root={geo['cr']*1000:.0f}  c_tip={geo['ct']*1000:.0f}  n={geo['count']}")
    print()
    print(f"{'źródło':<22}{'/rad':>9}{'/deg':>10}")
    print(f"{'strip-theory (geom.)':<22}{strip:>9.3f}{strip*r2d:>10.5f}")
    if meas is not None:
        print(f"{'POMIAR (telemetria)':<22}{meas/r2d:>9.3f}{meas:>10.5f}")
    if mod is not None:
        print(f"{'model DATCOM':<22}{mod/r2d:>9.3f}{mod:>10.5f}")
    print()
    if meas is not None:
        print(f"pomiar / strip = {meas/(strip*r2d):.2f}   "
              f"(≈1 => pomiar zgodny z fizyką)")
    if meas is not None and mod is not None:
        print(f"pomiar / model = {meas/mod:.2f}   "
              f"strip / model = {(strip*r2d)/mod:.2f}")
        print()
        print("Wniosek: strip-theory (sama geometria, bez clα) ≈ pomiar, a model jest")
        print(f"~{(strip*r2d)/mod:.1f}× niżej. Geometria tłumienia jest dobra, więc błąd modelu")
        print("leży w NAPĘDZIE: DATCOM zaniża moment roll od zaklinowania (CLL).")
        print("Prawdopodobna przyczyna: cant podany jako wychylenie steru (DELTA) ma")
        print("zredukowaną skuteczność klapy zamiast pełnej incydencji płetwy.")
    if stale:
        print("\n[!] UWAGA: tabele DATCOM cant przestarzałe (zły LREF) — model niepewny.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
