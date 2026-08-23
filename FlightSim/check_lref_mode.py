"""
check_lref_mode.py
==================
Diagnostyka: jaka dlugosc odniesienia momentu jest FAKTYCZNIE w obiegu?

Powstalo, bo porownanie walidacji "przed/po" (compare_validation.py) dawalo
identyczne wyniki mimo ustawienia FLIGHTSIM_LREF_MODE=length. Skrypt sprawdza
CALY lancuch, po kolei, zeby bylo widac ktore ogniwo nie przenosi zmiany:

    zmienna srodowiskowa  ->  wygenerowany deck (for005.dat)
                          ->  wynik DATCOM (datcom.out)
                          ->  tablica aero uzywana przez model (pkl)

Najczestsza przyczyna: zmienna nie dotarla do Pythona. W PowerShell `set` to
alias Set-Variable i NIE tworzy zmiennej srodowiskowej — trzeba $env:NAZWA="...".
Dlatego punkt 1 pokazuje to, co widzi sam Python, a nie to, co wpisano w shellu.

Uzycie:
    python check_lref_mode.py [nazwa_case]        # domyslnie rocket_70mm_baseline
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CASE = sys.argv[1] if len(sys.argv) > 1 else "rocket_70mm_baseline"


def lref_in_deck(p: Path):
    for line in p.read_text(errors="replace").splitlines():
        if "LREF" in line.upper():
            try:
                return float(line.upper().split("LREF")[1].lstrip("= ").split(",")[0])
            except (ValueError, IndexError):
                return None
    return None


def main():
    print("=" * 72)
    print(f"DIAGNOSTYKA DLUGOSCI ODNIESIENIA — case '{CASE}'")
    print("=" * 72)

    from datcom_io.config_reader import load_config
    cfg = load_config(str(ROOT / "configurations" / f"{CASE}.yaml"))
    d, L = float(cfg.body.diameter), float(cfg.body.length)
    print(f"\nGeometria: srednica = {d:.4f} m,  dlugosc kadluba = {L:.4f} m"
          f"   (stosunek {L/d:.1f}x)")

    # 1. co widzi Python -------------------------------------------------- #
    mode = os.environ.get("FLIGHTSIM_LREF_MODE")
    stale = os.environ.get("FLIGHTSIM_ALLOW_STALE_LREF")
    print("\n1) Zmienne srodowiskowe WIDZIANE PRZEZ PYTHONA")
    print(f"   FLIGHTSIM_LREF_MODE       = {mode!r}")
    print(f"   FLIGHTSIM_ALLOW_STALE_LREF= {stale!r}")
    want = L if (mode or "").lower() == "length" else d
    print(f"   => oczekiwane LREF = {want:.4f} m "
          f"({'dlugosc (tryb PRZED)' if want == L else 'srednica (tryb PO)'})")
    if mode is None:
        print("   UWAGA: zmiennej NIE MA. Jesli ustawiales ja w tym samym oknie,")
        print("          to znaczy ze shell jej nie wyeksportowal:")
        print("            cmd.exe     : set FLIGHTSIM_LREF_MODE=length")
        print("            PowerShell  : $env:FLIGHTSIM_LREF_MODE=\"length\"")
        print("            git-bash    : export FLIGHTSIM_LREF_MODE=length")

    # 2. deck ------------------------------------------------------------- #
    from datcom_io.missile_datcom_generator import generate_missile_datcom_input
    tmp = ROOT / "datcom_runs" / "_lrefcheck"
    tmp.mkdir(parents=True, exist_ok=True)
    p = generate_missile_datcom_input(cfg, tmp / "probe.dat")
    deck = lref_in_deck(p)
    p.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"\n2) LREF w SWIEZO wygenerowanym decku = {deck}")
    print("   " + ("zgodne z oczekiwaniem" if deck and abs(deck - want) < 1e-4
                   else "!!! NIEZGODNE — generator nie widzi zmiennej"))

    # 3. ostatni wynik DATCOM na dysku ------------------------------------ #
    run_dir = ROOT / "datcom_runs" / CASE
    used = run_dir / "for005.dat"
    print(f"\n3) Pliki na dysku w {run_dir}")
    if used.exists():
        u = lref_in_deck(used)
        print(f"   for005.dat  LREF = {u}   (deck UZYTY do ostatniego przebiegu)")
        if u is not None and deck is not None and abs(u - deck) > 1e-4:
            print("   !!! Rozni sie od punktu 2 — walidacja NIE zostala przeliczona")
            print("       w tym trybie (albo uzyto --no-rerun-datcom).")
    else:
        print("   brak for005.dat")

    out = run_dir / "datcom.out"
    if out.exists():
        from datcom_io.missile_datcom_reader import parse_missile_datcom_output
        try:
            res = parse_missile_datcom_output(out)
            if res.cases:
                lo = float(res.cases[0].lref)
                print(f"   datcom.out  LREF = {lo:.4f}   "
                      f"({'dlugosc = PRZED' if abs(lo - L) < 1e-3 else 'srednica = PO'})")
        except Exception as exc:
            print(f"   datcom.out — blad parsowania: {type(exc).__name__}: {exc}")
    else:
        print("   brak datcom.out")

    # 3b. WSZYSTKIE katalogi przebiegow ----------------------------------- #
    # Walidacja per lot NIE liczy w katalogu bazowego case'u: get_aero_for_cant
    # zapisuje tymczasowy YAML per cant i liczy w '<case>_cantXpYYY'. Patrzenie
    # tylko na katalog bazowy pokazuje wiec stare smieci, a nie to, czego
    # walidacja naprawde uzyla. Tutaj skanujemy wszystko + czasy modyfikacji,
    # zeby bylo widac, ktory przebieg jest z ktorej proby.
    print("\n3b) Wszystkie katalogi datcom_runs (LREF + czas modyfikacji)")
    from datetime import datetime
    runs = sorted((ROOT / "datcom_runs").glob("*/"))
    if not runs:
        print("   brak")
    print(f"   {'katalog':<38} {'for005':>8} {'zapisany':>17}  {'datcom.out':>10}")
    for rd in runs:
        f5, dout = rd / "for005.dat", rd / "datcom.out"
        if not f5.exists() and not dout.exists():
            continue
        lv = lref_in_deck(f5) if f5.exists() else None
        ts = (datetime.fromtimestamp(f5.stat().st_mtime).strftime("%m-%d %H:%M:%S")
              if f5.exists() else "-")
        ov = ""
        if dout.exists():
            ov = datetime.fromtimestamp(dout.stat().st_mtime).strftime("%m-%d %H:%M:%S")
        print(f"   {rd.name:<38} {str(lv):>8} {ts:>17}  {ov:>10}")
    print("   Oczekiwanie: katalogi '<case>_cant...' to te, ktorych uzywa")
    print("   walidacja per lot. Jesli ich for005.dat maja LREF=0.07 mimo")
    print("   trybu PRZED, to zmienna nie dotarla do TEGO przebiegu.")

    # 4. co realnie trafia do modelu -------------------------------------- #
    print("\n4) Tablica aero podawana modelowi (przez cache/pkl)")
    try:
        from aero import get_aero_model
        aero = get_aero_model(CASE, method="missile_datcom", force_rerun=False)
        lr = getattr(aero, "lref_ref", None)
        print(f"   TableAero.lref_ref = {lr}")
        import numpy as np
        a = np.asarray(aero.alpha_table); cm = np.asarray(aero.Cm_table)
        i = int(np.argmin(np.abs(a - np.deg2rad(5.0))))
        if a[i] != 0:
            print(f"   Cm_alpha (przy alpha={np.degrees(a[i]):.1f} deg, srednio po Mach)"
                  f" = {float(np.mean(cm[i])) / float(a[i]):+.1f} /rad")
        print("   Rzad wielkosci: ~-3 /rad => STARA normalizacja (dlugosc),"
              " ~-54 /rad => NOWA (srednica).")
    except Exception as exc:
        print(f"   nie udalo sie zaladowac: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 72)
    print("Jesli punkt 1 pokazuje None, a chciales tryb PRZED — problem jest")
    print("w shellu, nie w modelu. Punkt 4 mowi, ktora normalizacja jest")
    print("naprawde uzywana przez ostatnio policzona walidacje.")
    print("=" * 72)


if __name__ == "__main__":
    main()
