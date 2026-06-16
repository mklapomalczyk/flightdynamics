"""
datcom_io/barrowman_aero.py
===========================
Konwertuje wyniki Barrowmana do TableAero i integruje z pipeline aero.py.

Pipeline identyczny jak dla DATCOM:
  1. Sprawdź cache (aero_table.pkl)
  2. Oblicz metodą Barrowmana z pliku YAML
  3. Zapisz cache i zwróć TableAero

Użycie:
  from datcom_io.barrowman_aero import get_aero_barrowman
  aero = get_aero_barrowman("rocket_70mm_baseline")
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.aerodynamics import TableAero
from datcom_io.config_reader import load_config
from datcom_io.barrowman import BarrowmanCalculator


# Foldery projektu
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR  = PROJECT_ROOT / "configurations"
RUNS_DIR     = PROJECT_ROOT / "datcom_runs"


def get_aero_barrowman(
    case_name:   str,
    Cmq:         float = -20.0,
    force_rerun: bool  = False,
    alpha_deg:   Optional[list] = None,
) -> TableAero:
    """
    Oblicza model aerodynamiczny metodą Barrowmana.

    Pipeline:
      1. Sprawdź cache pkl
      2. Wczytaj YAML i oblicz analitycznie
      3. Zapisz cache i zwróć TableAero

    Parameters
    ----------
    case_name : str
        Nazwa konfiguracji — folder w datcom_runs/ i plik w configurations/.
    Cmq : float
        Współczynnik tłumienia kątowego [1/rad].
    force_rerun : bool
        Wymuś ponowne obliczenie (ignoruj cache).
    alpha_deg : list, optional
        Kąty natarcia do tabeli [deg]. Domyślnie ±15° co 2°.
    """
    case_dir  = RUNS_DIR / case_name
    pkl_path  = case_dir / "aero_table_barrowman.pkl"
    yaml_path = CONFIGS_DIR / f"{case_name}.yaml"

    # ---- Krok 1: cache ------------------------------------------------- #
    if pkl_path.exists() and not force_rerun:
        print(f"[Barrowman] Wczytano z cache: {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # ---- Krok 2: oblicz ------------------------------------------------ #
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono konfiguracji '{case_name}'.\n"
            f"Oczekiwano: {yaml_path}"
        )

    print(f"[Barrowman] Obliczam: {yaml_path.name}")
    config = load_config(yaml_path)
    calc   = BarrowmanCalculator(config)

    # Zakres Macha z YAML
    mach_values = config.flight_conditions.mach

    # Zakres alpha
    if alpha_deg is None:
        alpha_deg = config.flight_conditions.alpha
    alpha_rad = np.deg2rad(alpha_deg)

    n_alpha = len(alpha_rad)
    n_mach  = len(mach_values)

    CA_table  = np.zeros((n_alpha, n_mach))
    CN_table  = np.zeros((n_alpha, n_mach))
    xcp_table = np.zeros((n_alpha, n_mach))

    print(f"  Mach:  {mach_values}")
    print(f"  Alpha: {alpha_deg} deg")
    print()
    print(f"  {'Mach':>6} {'CNα':>8} {'xcp [m]':>10} {'CA':>8}")

    for j, mach in enumerate(mach_values):
        res = calc.compute(mach)
        print(f"  {mach:>6.2f} {res.CNa:>8.4f} {res.xcp:>10.4f} {res.CA:>8.4f}")

        for i, alpha in enumerate(alpha_rad):
            # CN = CNα * alpha  (liniowe, małe kąty)
            # Korekta nieliniowa dla większych alpha (Niskanen):
            # CN = CNα * sin(alpha) * cos(alpha) + ...
            # Uproszczone: CN = CNα * sin(alpha) dla większych alpha
            if abs(np.degrees(alpha)) < 10.0:
                CN = res.CNa * alpha           # liniowe
            else:
                CN = res.CNa * np.sign(alpha) * np.sin(abs(alpha))

            CA_table[i, j]  = res.CA
            CN_table[i, j]  = CN
            xcp_table[i, j] = res.xcp

    aero = TableAero(
        alpha_table = alpha_rad,
        mach_table  = np.array(mach_values),
        CA_table    = CA_table,
        CN_table    = CN_table,
        xcp_table   = xcp_table,
        Cmq         = Cmq,
    )

    # ---- Krok 3: cache ------------------------------------------------- #
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(aero, f)
    print(f"\n[Barrowman] Zapisano cache: {pkl_path}")

    return aero


def summary(case_name: str) -> str:
    """Wypisuje podsumowanie obliczeń Barrowmana."""
    yaml_path = CONFIGS_DIR / f"{case_name}.yaml"
    config    = load_config(yaml_path)
    calc      = BarrowmanCalculator(config)

    lines = [f"Barrowman — {case_name}"]
    lines.append(f"  Średnica:    {config.body.diameter*100:.1f} cm")
    lines.append(f"  Długość:     {config.body.length:.3f} m")
    lines.append(f"  xcg_ref:     {config.mass.xcg_ref:.3f} m")
    lines.append("")
    lines.append(f"  {'Mach':>6} {'CNα [1/rad]':>12} {'xcp [m]':>10} {'CA':>8} {'SM [kalibrów]':>15}")

    for mach in config.flight_conditions.mach:
        res = calc.compute(mach)
        d   = config.body.diameter
        xcg = config.mass.xcg_ref
        sm  = (res.xcp - xcg) / d   # margines statyczny w kalibrach
        lines.append(
            f"  {mach:>6.2f}"
            f"  {res.CNa:>12.4f}"
            f"  {res.xcp:>10.4f}"
            f"  {res.CA:>8.4f}"
            f"  {sm:>15.2f}"
        )

    return "\n".join(lines)
