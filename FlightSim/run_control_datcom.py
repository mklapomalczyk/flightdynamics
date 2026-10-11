"""
run_control_datcom.py
=====================
Generuje decki DATCOM potrzebne dla modulu sterowania i (jesli DATCOM jest
dostepny) uruchamia je, a nastepnie buduje tablice pochodnych sterowania.

Trzy zadania:

  1. ctrl   — sweep wychylen canardow (jeden deck skladany SAVE/NEXT CASE:
              3 wzorce x 11 wychylen -10..+10 co 2 deg)
  2. gam    — para decków bazowych: BEZ GAM (nowe, poprawne) i Z GAM (stare,
              z podwojonym zaklinowaniem) — do testu A/B, patrz
              tests/test_gam_baseline_compare.py
  3. all    — oba powyzsze

Na Linuksie (brak MissileDATCOM.exe) skrypt tylko GENERUJE decki i wypisuje,
co uruchomic na Windows. Na Windows uruchamia DATCOM i od razu buduje tablice.

Uzycie:
    python run_control_datcom.py            # = all, tylko generacja jesli brak exe
    python run_control_datcom.py ctrl
    python run_control_datcom.py gam
    python run_control_datcom.py all --run  # wymus uruchomienie DATCOM
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from control.derivatives import PANEL_PATTERNS
from datcom_io.config_reader import load_config
from datcom_io.missile_datcom_generator import (_effective_fin_sets,
                                                generate_missile_datcom_input)

CANARD_CASE  = "rocket_70mm_canards"
BASELINE_CASE = "rocket_70mm_baseline"
BASELINE_CANT = 0.6      # ten sam cant co zwalidowany przypadek lotu 19


def _runs_dir(case: str) -> Path:
    d = ROOT / "datcom_runs" / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def _datcom_available() -> bool:
    return (ROOT / "datcom" / "MissileDATCOM.exe").exists() and sys.platform.startswith("win")


def build_delta_cases(cfg, sweep_deg):
    """Lista przypadkow $DEFLCT: kazdy wzorzec x kazde wychylenie."""
    _eff, ctrl_idx = _effective_fin_sets(cfg)
    if not ctrl_idx:
        raise SystemExit(f"{cfg.name}: brak control_surfaces w konfiguracji")
    ci = ctrl_idx[0]
    cases = []
    for chan, pattern in PANEL_PATTERNS.items():
        for d in sweep_deg:
            cases.append({
                "label": f"{chan.upper()} DELTA={d:+.1f}",
                "delta": {ci: [s * d for s in pattern]},
            })
    return cases, ci


def task_ctrl(do_run: bool):
    """
    Sweep wychylen — JEDEN DECK NA KANAL, nie jeden na wszystko.

    Zmierzone empirycznie: DATCOM wywala sie (0xC00000A1) po ~20 przypadkach
    w jednym przebiegu skladanym — pierwszy pelny przebieg konczyl sie w
    polowie przypadku 21 z 34. Podzial na kanaly daje 12 przypadkow na deck
    (baza + 11 wychylen), czyli z duzym zapasem ponizej progu, a i tak jest to
    3 uruchomienia zamiast 33.
    """
    cfg = load_config(str(ROOT / "configurations" / f"{CANARD_CASE}.yaml"))
    sweep = getattr(cfg, "control_sweep_deg", None) or \
        [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    all_cases, ci = build_delta_cases(cfg, sweep)

    run_dir = _runs_dir(CANARD_CASE)
    eff, _ = _effective_fin_sets(cfg)
    print(f"\n[ctrl] zestawy pletw (od nosa): "
          f"{[(f.name, round(f.position,3)) for f, _ in eff]}")
    print(f"[ctrl] zestaw sterowy = $FINSET{ci}")
    print(f"[ctrl] sweep: {sweep}")

    decks, outs = [], []
    for chan in PANEL_PATTERNS:
        cases = [c for c in all_cases if c["label"].lower().startswith(chan)]
        if not cases:
            continue
        deck = run_dir / f"for005_ctrl_{chan}.dat"
        generate_missile_datcom_input(cfg, deck, delta_cases=cases)
        decks.append(deck)
        outs.append(run_dir / f"datcom_ctrl_{chan}.out")
        print(f"[ctrl] {chan:8s}: {len(cases)} przypadkow (+1 bazowy) -> {deck.name}")

    if do_run:
        from datcom_io.missile_datcom_runner import run_missile_datcom
        for deck, out in zip(decks, outs):
            run_missile_datcom(deck, run_dir, output_filename=out.name)
            print(f"[ctrl] wynik: {out}")
    return decks


def task_gam(do_run: bool):
    """Para decków do testu A/B usuniecia GAM."""
    cfg = load_config(str(ROOT / "configurations" / f"{BASELINE_CASE}.yaml"))
    for f in cfg.fins:
        f.cant_angle = BASELINE_CANT

    run_dir = _runs_dir(f"{BASELINE_CASE}_gamtest")
    d_no = run_dir / "for005_nogam.dat"
    d_gm = run_dir / "for005_gam.dat"
    generate_missile_datcom_input(cfg, d_no, emit_gam=False)
    generate_missile_datcom_input(cfg, d_gm, emit_gam=True)

    print(f"\n[gam] cant = {BASELINE_CANT} deg (jak zwalidowany lot 19)")
    print(f"[gam] BEZ GAM (nowe): {d_no}")
    print(f"[gam] Z GAM (stare):  {d_gm}")
    print("[gam] Roznica w decku to DOKLADNIE jedna linia (GAM=...).")

    if do_run:
        from datcom_io.missile_datcom_runner import run_missile_datcom
        run_missile_datcom(d_no, run_dir, output_filename="datcom_nogam.out")
        run_missile_datcom(d_gm, run_dir, output_filename="datcom_gam.out")
        print(f"[gam] wyniki w {run_dir}")
    return d_no, d_gm


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    task = args[0] if args else "all"
    do_run = "--run" in sys.argv or _datcom_available()

    if task not in ("ctrl", "gam", "all"):
        raise SystemExit(__doc__)

    print("=" * 70)
    print("DATCOM dla modulu sterowania")
    print("=" * 70)
    if task in ("ctrl", "all"):
        task_ctrl(do_run)
    if task in ("gam", "all"):
        task_gam(do_run)

    if not do_run:
        print("\n" + "=" * 70)
        print("MissileDATCOM.exe niedostepny (Linux) — decki wygenerowane.")
        print("Na Windows uruchom:  python run_control_datcom.py all --run")
        print("Potem zacommituj pliki datcom_*.out — reszte da sie policzyc")
        print("i sprawdzic juz bez DATCOM.")
        print("=" * 70)


if __name__ == "__main__":
    main()
