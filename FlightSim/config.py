"""
config.py
=========
Konfiguracja projektu FlightSim.

Ustaw raz ścieżki do narzędzi zewnętrznych.
Plik nie powinien być commitowany do repozytorium (dodaj do .gitignore)
jeśli ścieżki są specyficzne dla Twojego systemu.
"""

import os
from pathlib import Path

# ============================================================================
# Ścieżki główne
# ============================================================================

# Główny folder projektu (automatycznie — katalog tego pliku)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Folder z wynikami DATCOM
# Każda konfiguracja dostaje własny podfolder
DATCOM_RUNS_DIR = PROJECT_ROOT / "datcom_runs"

# ============================================================================
# Ścieżka do DATCOM
# ============================================================================

# Przykład Windows: r"C:\Users\mklap\Desktop\datcom\datcom.exe"
# Przykład Linux/MSYS2: "/home/user/datcom/datcom"

# Missile DATCOM — czyta for005.dat z katalogu roboczego
MISSILE_DATCOM_EXE = Path(r"C:\Users\mklap\Desktop\Python_projekty\FlightSim\datcom\MissileDATCOM.exe")

# Katalog roboczy Missile DATCOM — tu musi być for005.dat i tu trafia for006.dat
MISSILE_DATCOM_WORKDIR = Path(r"C:\Users\mklap\Desktop\Python_projekty\FlightSim\datcom")

# ============================================================================
# Walidacja przy imporcie
# ============================================================================

def validate():
    """Sprawdza czy konfiguracja jest poprawna."""
    errors = []

    if not DATCOM_EXE.exists():
        errors.append(f"datcom.exe nie znaleziony: {DATCOM_EXE}")

    if errors:
        print("OSTRZEZENIE config.py:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"config.py OK: datcom.exe = {DATCOM_EXE}")

    return len(errors) == 0


if __name__ == "__main__":
    validate()
