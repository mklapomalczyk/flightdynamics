"""
telemetry_parser.py
===================
Parser plikow telemetrii ARTEMIDA. Obsluguje rozna kolejnosc kolumn
przez mapowanie po nazwach naglowkow (nie po pozycji).

Plik CSV rozdzielany srednikami, naglowek w pierwszej linii.
Kodowanie moze byc cp1250/utf-8 — czytamy z errors='replace'.
"""

import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass


def get_data_dir():
    """
    Zwraca katalog z danymi (field_test_data).

    Struktura:
        FlightSim/
        ├── field_test_data/      <- dane + configs.txt
        └── telemetry_analysis/   <- ten modul
    Oba foldery sa obok siebie w FlightSim, wiec dane sa w
    ../field_test_data wzgledem tego modulu.
    """
    return Path(__file__).resolve().parent.parent / "field_test_data"


def resolve_data_file(name):
    """Znajduje plik danych — najpierw w biezacym katalogu, potem w data_dir."""
    p = Path(name)
    if p.exists():
        return p
    alt = get_data_dir() / name
    if alt.exists():
        return alt
    return p   # zwroc oryginalna (zglosi blad pozniej)


# Mapowanie: znormalizowana nazwa -> mozliwe naglowki w pliku
# Klucze sa odporne na drobne roznice w nazwach kolumn
COLUMN_ALIASES = {
    "time":       ["Czas lotu [s]", "Czas lotu", "Czas_lotu_[s]", "time", "t"],
    "lon":        ["Dlugosc geograficzna [st.]", "Dlugosc geograficzna",
                   "Dlugość_geograficzna_[st.]", "Dlugosc_geograficzna_[st.]",
                   "Dlugość geograficzna [st.]"],
    "ns":         ["N/S"],
    "lat":        ["Szerokosc geograficzna [st.]", "Szerokosc geograficzna",
                   "Szerokość_geograficzna_[st.]", "Szerokosc_geograficzna_[st.]",
                   "Szerokość geograficzna [st.]"],
    "we":         ["W/E"],
    "alt":        ["Wysokosc lotu [m]", "Wysokosc lotu",
                   "Wysokość_lotu_[m]", "Wysokosc_lotu_[m]",
                   "Wysokość lotu [m]"],
    "vel":        ["Predkosc lotu [m/s]", "Predkosc lotu",
                   "Prędkość_lotu_[m/s]", "Predkosc_lotu_[m/s]",
                   "Prędkość lotu [m/s]"],
    "acc_x":      ["Przyśpieszenie X [g]", "Przyspieszenie X [g]", "Przyśpieszenie X",
                   "Przyśpieszenie_X_[g]", "Przyspieszenie_X_[g]"],
    "acc_y":      ["Przyśpieszenie Y [g]", "Przyspieszenie Y [g]", "Przyśpieszenie Y",
                   "Przyśpieszenie_Y_[g]", "Przyspieszenie_Y_[g]"],
    "acc_z":      ["Przyśpieszenie Z [g]", "Przyspieszenie Z [g]", "Przyśpieszenie Z",
                   "Przyśpieszenie_Z_[g]", "Przyspieszenie_Z_[g]"],
    "acc_centr":  ["Przyśpieszenie odśrodkowe [g]", "Przyspieszenie odśrodkowe [g]",
                   "Przyśpieszenie_odśrodkowe_[g]", "Przyspieszenie_odśrodkowe_[g]"],
    "gyro_ax":    ["Prędkość obrotowa AX [st./s]", "Predkosc obrotowa AX [st./s]",
                   "Prędkość_obrotowa_AX_[st./s]", "Predkosc_obrotowa_AX_[st./s]"],
    "gyro_x":     ["Prędkość obrotowa X [st./s]", "Predkosc obrotowa X [st./s]",
                   "Prędkość_obrotowa_X_[st./s]", "Predkosc_obrotowa_X_[st./s]"],
    "gyro_y":     ["Prędkość obrotowa Y [st./s]", "Predkosc obrotowa Y [st./s]",
                   "Prędkość_obrotowa_Y_[st./s]", "Predkosc_obrotowa_Y_[st./s]"],
    "gyro_z":     ["Prędkość obrotowa Z [st./s]", "Predkosc obrotowa Z [st./s]",
                   "Prędkość_obrotowa_Z_[st./s]", "Predkosc_obrotowa_Z_[st./s]"],
    "mag_y":      ["Indukcja magnetyczna Y [G]", "Indukcja magnetyczna Y",
                   "Magnetyzm_[G]", "MAG_[G]"],
    "press_amb":  ["Cisnienie otoczenia [hPa]", "Cisnienie otoczenia"],
    "press_cham": ["Cisnienia w komorze silnika [bar]", "Cisnienie w komorze silnika [bar]",
                   "Ciśnienie_w_komorze_spalania_[bar]", "Cisnienie_w_komorze_spalania_[bar]",
                   "Ciśnienie w komorze spalania [bar]"],
    "rudder":     ["Kąt wychylenia steru [st.]", "Kat wychylenia steru [st.]"],
    "flag_flight":["FLAGA LOT", "FLAGA_LOT"],
    "flag_apogee":["FLAGA PULAP", "FLAGA_PULAP"],
    "flag_chute": ["FLAGA SPADOCHRON", "FLAGA_SPADOCHRON"],
}


# ===========================================================================
# KONFIGURACJA OSI — telemetria ARTEMIDA ma zamienione osie zyroskopu Y<->Z
# Ustalone empirycznie przez test_axis_assignment.py (RMS 535m -> 59m)
# Aby wylaczyc zamiane: SWAP_GYRO_YZ = False
# ===========================================================================
SWAP_GYRO_YZ = True    # zamien osie pitch/yaw zyroskopu
SWAP_ACC_YZ  = False   # akcelerometr — osie wygladaja poprawnie


@dataclass
class Telemetry:
    """Sparsowane dane telemetrii jako tablice numpy.

    Pola gyro_x/y/z i acc_x/y/z sa PO ewentualnej zamianie osi (wg flag
    SWAP_GYRO_YZ / SWAP_ACC_YZ). Surowe dane bez zamiany dostepne sa
    w raw_columns (klucze 'gyro_y', 'gyro_z' itd. — oryginalne kolumny).
    """
    time:      np.ndarray   # [s]
    acc_x:     np.ndarray   # [g] — os podluzna
    acc_y:     np.ndarray   # [g]
    acc_z:     np.ndarray   # [g]
    gyro_x:    np.ndarray   # [st/s] — roll
    gyro_y:    np.ndarray   # [st/s] — pitch (po zamianie)
    gyro_z:    np.ndarray   # [st/s] — yaw  (po zamianie)
    # Telemetria pokladowa (do walidacji)
    alt_onboard: np.ndarray # [m]
    vel_onboard: np.ndarray # [m/s]
    press_cham:  np.ndarray # [bar]
    press_amb:   np.ndarray # [hPa]
    # GPS
    lat:       np.ndarray   # [st]
    lon:       np.ndarray   # [st]
    # Flagi
    flag_flight: np.ndarray
    # Metadane
    n_samples: int
    dt_mean:   float
    raw_columns: dict       # pelny slownik wszystkich kolumn (numpy)


def _build_column_map(header):
    """Buduje mapowanie znormalizowana_nazwa -> indeks kolumny."""
    col_map = {}
    for norm_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in header:
                col_map[norm_name] = header.index(alias)
                break
    return col_map


def detect_delimiter(filepath):
    """
    Wykrywa separator pliku:
      ';'   — loty 13-14
      ','   — loty 15-17
      None  — whitespace (loty 18-21)

    Decyduje na podstawie liczby kolumn jaka daje kazdy separator
    (nie liczby znakow — nazwy kolumn typu "Czas lotu [s]" zawieraja spacje).
    """
    with open(filepath, encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n\r")
    n_by_semi  = len(first.split(";"))
    n_by_comma = len(first.split(","))
    # Wybierz separator dajacy najwiecej kolumn (>1)
    if n_by_semi > 1:
        return ";"
    if n_by_comma > 1:
        return ","
    return None   # whitespace (split())


def parse_telemetry(filepath, verbose=True):
    """
    Parsuje plik telemetrii. Zwraca obiekt Telemetry.
    Obsluguje rozna kolejnosc kolumn przez mapowanie po nazwach.
    """
    filepath = Path(filepath)
    delim = detect_delimiter(filepath)
    with open(filepath, encoding="utf-8", errors="replace") as f:
        if delim is None:
            # Separator = dowolny whitespace (split bez argumentu)
            lines = f.read().splitlines()
            header = lines[0].split()
            raw_rows = [ln.split() for ln in lines[1:] if ln.strip()]
            raw_rows = [r for r in raw_rows if len(r) >= len(header) - 2]
        else:
            reader = csv.reader(f, delimiter=delim)
            header = [h.strip() for h in next(reader)]
            raw_rows = [r for r in reader if len(r) >= len(header) - 2]

    col_map = _build_column_map(header)

    # Sprawdz czy mamy wszystkie kluczowe kolumny
    required = ["time", "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {missing}\n"
                         f"Dostepne naglowki: {header}")

    def col(name, default=np.nan):
        """Wyciaga kolumne jako tablice float, NaN dla pustych."""
        if name not in col_map:
            return None
        idx = col_map[name]
        vals = []
        for r in raw_rows:
            try:
                vals.append(float(r[idx]))
            except (ValueError, IndexError):
                vals.append(default)
        return np.array(vals)

    # Pelny slownik wszystkich kolumn (dla diagnostyki)
    raw_columns = {}
    for norm_name in col_map:
        raw_columns[norm_name] = col(norm_name)

    time = col("time")
    dt   = np.diff(time)
    dt_mean = float(np.median(dt))

    # Surowe osie (przed zamiana)
    raw_gyro_y = col("gyro_y")
    raw_gyro_z = col("gyro_z")
    raw_acc_y  = col("acc_y")
    raw_acc_z  = col("acc_z")

    # Zastosuj zamiane osi wg flag konfiguracyjnych
    gyro_y_out, gyro_z_out = raw_gyro_y, raw_gyro_z
    acc_y_out,  acc_z_out  = raw_acc_y, raw_acc_z
    if SWAP_GYRO_YZ:
        gyro_y_out, gyro_z_out = raw_gyro_z, raw_gyro_y
    if SWAP_ACC_YZ:
        acc_y_out, acc_z_out = raw_acc_z, raw_acc_y

    if verbose:
        print(f"Sparsowano: {filepath.name}")
        print(f"  Kolumny rozpoznane: {len(col_map)}/{len(COLUMN_ALIASES)}")
        print(f"  Probek: {len(raw_rows)}")
        print(f"  Czas: {time[0]:.3f}s do {time[-1]:.3f}s")
        print(f"  Krok: {dt_mean:.4f}s ({1/dt_mean:.0f} Hz)")
        if SWAP_GYRO_YZ:
            print(f"  [!] Osie zyroskopu Y<->Z zamienione (SWAP_GYRO_YZ=True)")
        if SWAP_ACC_YZ:
            print(f"  [!] Osie akcelerometru Y<->Z zamienione")

    # UWAGA: w telemetrii ARTEMIDA kolumny lat/lon sa BLEDNIE OPISANE.
    # "Dlugosc geograficzna" zawiera ~50.7 z oznaczeniem N -> to szerokosc (lat)
    # "Szerokosc geograficzna" zawiera ~21.9 z oznaczeniem E -> to dlugosc (lon)
    # Korygujemy: lat <- kolumna "lon", lon <- kolumna "lat"
    _lat_col = col("lon")   # "Dlugosc geograficzna" = faktyczna szerokosc
    _lon_col = col("lat")   # "Szerokosc geograficzna" = faktyczna dlugosc

    return Telemetry(
        time=time,
        acc_x=col("acc_x"), acc_y=acc_y_out, acc_z=acc_z_out,
        gyro_x=col("gyro_x"), gyro_y=gyro_y_out, gyro_z=gyro_z_out,
        alt_onboard=col("alt"), vel_onboard=col("vel"),
        press_cham=col("press_cham"), press_amb=col("press_amb"),
        lat=_lat_col, lon=_lon_col,
        flag_flight=col("flag_flight"),
        n_samples=len(raw_rows), dt_mean=dt_mean,
        raw_columns=raw_columns,
    )


if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "ARTEMIDA_13_LOT.txt"
    tel = parse_telemetry(fp)
    print(f"\nPierwsze 3 probki acc_x: {tel.acc_x[:3]}")
    print(f"Pierwsze 3 probki gyro_x: {tel.gyro_x[:3]}")
