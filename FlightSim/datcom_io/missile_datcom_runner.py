"""
datcom_io/missile_datcom_runner.py
====================================
Runner dla Missile DATCOM.

Missile DATCOM czyta plik wejściowy jako 'for005.dat'
z katalogu roboczego i zapisuje wyniki do 'for006.dat'.

Workflow:
  1. Skopiuj wygenerowany .inp do {WORKDIR}/for005.dat
  2. Uruchom MissileDATCOM.exe w WORKDIR
  3. Odczytaj wyniki z {WORKDIR}/for006.dat
  4. Skopiuj for006.dat do katalogu docelowego jako datcom.out
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path


def run_missile_datcom(
    inp_path:        str | Path,
    output_dir:      str | Path,
    exe_path:        str | Path | None = None,
    work_dir:        str | Path | None = None,
    timeout:         int = 120,
    output_filename: str = "datcom.out",
) -> Path:
    """
    Uruchamia Missile DATCOM dla podanego pliku wejściowego.

    Parameters
    ----------
    inp_path : str | Path
        Ścieżka do wygenerowanego pliku .inp (będzie skopiowany jako for005.dat).
    output_dir : str | Path
        Katalog gdzie zostanie zapisany for006.dat jako 'datcom.out'.
    exe_path : str | Path, optional
        Ścieżka do MissileDATCOM.exe. Domyślnie z config.py.
    work_dir : str | Path, optional
        Katalog roboczy (gdzie for005.dat/for006.dat). Domyślnie z config.py.
    timeout : int
        Timeout w sekundach.

    Returns
    -------
    Path  Ścieżka do pliku wynikowego (datcom.out).

    Raises
    ------
    FileNotFoundError   Jeśli exe nie istnieje.
    RuntimeError        Jeśli DATCOM zakończył się błędem.
    """
    inp_path   = Path(inp_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pobierz konfigurację
    if exe_path is None or work_dir is None:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import MISSILE_DATCOM_EXE, MISSILE_DATCOM_WORKDIR
            if exe_path is None:
                exe_path = MISSILE_DATCOM_EXE
            if work_dir is None:
                work_dir = MISSILE_DATCOM_WORKDIR
        except ImportError:
            raise ImportError(
                "Ustaw MISSILE_DATCOM_EXE i MISSILE_DATCOM_WORKDIR w config.py"
            )

    exe_path = Path(exe_path)
    work_dir = Path(work_dir)

    if not exe_path.exists():
        raise FileNotFoundError(f"MissileDATCOM.exe nie znaleziony: {exe_path}")

    # 1. Skopiuj .inp → for005.dat
    for005 = work_dir / "for005.dat"
    shutil.copy2(inp_path, for005)
    print(f"[MissileDatcom] Skopiowano: {inp_path.name} → {for005}")

    # 2. Uruchom Missile DATCOM
    print(f"[MissileDatcom] Uruchamiam: {exe_path.name} w {work_dir}")
    try:
        proc = subprocess.run(
            [str(exe_path)],
            cwd         = str(work_dir),
            capture_output = True,
            text        = True,
            timeout     = timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Missile DATCOM przekroczył timeout {timeout}s")
    except Exception as e:
        raise RuntimeError(f"Błąd uruchamiania Missile DATCOM: {e}")

    # 3. Sprawdź wynik
    for006 = work_dir / "for006.dat"
    if not for006.exists():
        stdout = proc.stdout[-500:] if proc.stdout else ""
        stderr = proc.stderr[-500:] if proc.stderr else ""
        raise RuntimeError(
            f"Missile DATCOM nie wygenerował for006.dat\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

    # 4. Skopiuj wyniki
    out_path = output_dir / output_filename
    shutil.copy2(for006, out_path)

    # Zachowaj też kopię wejścia
    inp_name = Path(inp_path).name
    shutil.copy2(for005, output_dir / inp_name)

    # Kod powrotu sprawdzamy PRZED wypisaniem "OK" — wczesniej komunikat o
    # sukcesie szedl pierwszy, a ostrzezenie o awarii ginelo pod nim, przez co
    # crash DATCOM-a wygladal jak udany przebieg (a plik .out byl urwany).
    if proc.returncode != 0:
        rc = proc.returncode
        # Windows zwraca kody NTSTATUS jako duze liczby dodatnie (0xC... = blad).
        hexrc = f" (0x{rc & 0xFFFFFFFF:08X})" if rc > 0xFFFF else ""
        print(f"[MissileDatcom] *** BLAD: DATCOM zakonczyl sie kodem {rc}{hexrc}")
        print(f"[MissileDatcom] *** Plik {out_path.name} moze byc NIEKOMPLETNY "
              f"— sprawdz koncowke pliku i liczbe przypadkow.")
    else:
        print(f"[MissileDatcom] OK — wyniki: {out_path}")

    return out_path
