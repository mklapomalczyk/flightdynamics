"""
telemetry_parser.py
===================
Parser plikow telemetrii ARTEMIDA. Obsluguje rozna kolejnosc kolumn
przez mapowanie po nazwach naglowkow (nie po pozycji).

Plik CSV rozdzielany srednikami, naglowek w pierwszej linii.
Kodowanie moze byc cp1250/utf-8 — czytamy z errors='replace'.
"""

import csv
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# Znacznik w pierwszej linii pliku zbiorczego (nowy format, loty 22+).
# Pliki z tym znacznikiem sa juz w poprawnej orientacji fizycznej —
# parser NIE stosuje do nich korekt SWAP_GYRO_YZ ani zamiany lat/lon.
# Zob. aggregate_flight_data.py.
AGG_FORMAT_MARKER = "# FORMAT: ARTEMIDA_AGG v1"


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
    """Znajduje plik danych — najpierw w biezacym katalogu, potem w data_dir.

    Dla lotow w nowym formacie (22+) dane dostarczane sa jako plik zbiorczy
    ARTEMIDA_<NN>_AGG.txt (patrz aggregate_flight_data.py). Gdy skrypt prosi
    o stary schemat ARTEMIDA_<NN>_LOT.txt, a go nie ma, podstawiamy plik AGG,
    dzieki czemu wszystkie skrypty analizy dzialaja bez zmian.
    """
    p = Path(name)
    if p.exists():
        return p
    alt = get_data_dir() / name
    if alt.exists():
        return alt

    # Fallback: ARTEMIDA_<NN>_LOT.txt -> ARTEMIDA_<NN>_AGG.txt
    m = re.fullmatch(r"ARTEMIDA_(\d+)_LOT\.txt", Path(name).name)
    if m:
        agg = f"ARTEMIDA_{m.group(1)}_AGG.txt"
        for cand in (Path(agg), get_data_dir() / agg):
            if cand.exists():
                return cand

    return p   # zwroc oryginalna (zglosi blad pozniej)


# ===========================================================================
# Odporne czytanie configs.txt (wspolne dla skryptow analizy)
# ===========================================================================
def read_config_rows(config_path="configs.txt"):
    """
    Czyta configs.txt odpornie i zwraca liste wierszy jako dict (klucz=naglowek).

    Odporne na typowe problemy przy recznej edycji:
      - wiersz KROTSZY niz naglowek (brak kolumn koncowych, np. masy/pogoda)
        — brakujace kolumny dopelniane pustymi polami, zamiast pomijac wiersz;
      - wiersz rozdzielony SPACJAMI zamiast tabulatorami — gdy brak tabulatorow,
        dzielony po bialych znakach (wartosci w configs sa jednowyrazowe);
      - puste linie pomijane; biale znaki obcinane.

    Zwraca [] gdy pliku nie ma. Naglowek zawsze dzielony po tabulatorach.
    """
    p = resolve_data_file(config_path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if not lines:
        return []
    header = [h.strip() for h in lines[0].rstrip("\n\r").split("\t")]
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [x.strip() for x in line.rstrip("\n\r").split("\t")]
        if len(parts) < 2:                       # brak tabulatorow -> biale znaki
            parts = line.split()
        if len(parts) < len(header):             # dopelnij brakujace kolumny
            parts = parts + [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return rows


def _cfg_float(row, key, default):
    try:
        return float(str(row.get(key, "")).strip())
    except (ValueError, TypeError):
        return default


def load_flight_config(flight_no, config_path="configs.txt"):
    """
    Zwraca konfiguracje danego lotu z configs.txt (azymut, elewacja, cant,
    glowica, masy) lub None gdy nie znaleziono. Odporne na niepelne wiersze.
    """
    for row in read_config_rows(config_path):
        try:
            fno = int(str(row.get("flight no", "")).strip())
        except (ValueError, TypeError):
            continue
        if fno == flight_no:
            return {
                "cant":      _cfg_float(row, "cant angle", 0.0),
                "head":      str(row.get("head configuration", "?")).strip(),
                "azimuth":   _cfg_float(row, "azimuth", 0.0),
                "elevation": _cfg_float(row, "elevation", 45.0),
                "m_empty":   _cfg_float(row, "m_empty", float("nan")),
                "m_full":    _cfg_float(row, "m_full", float("nan")),
                "m_rocket":  _cfg_float(row, "m_rocket", float("nan")),
                "row":       row,
            }
    return None


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


def is_aggregated_format(filepath):
    """Czy plik jest zbiorczy (nowy format, loty 22+) — po znaczniku w 1. linii."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return f.readline().rstrip("\n\r") == AGG_FORMAT_MARKER
    except OSError:
        return False


def parse_telemetry(filepath, verbose=True):
    """
    Parsuje plik telemetrii. Zwraca obiekt Telemetry.
    Obsluguje rozna kolejnosc kolumn przez mapowanie po nazwach.

    Automatycznie rozpoznaje nowy format zbiorczy (loty 22+) i wczytuje go
    bez korekt osi/lat-lon (dane sa juz w poprawnej orientacji).
    """
    filepath = Path(filepath)
    if is_aggregated_format(filepath):
        return _parse_aggregated(filepath, verbose=verbose)

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


def _parse_aggregated(filepath, verbose=True):
    """
    Wczytuje plik zbiorczy nowego formatu (ARTEMIDA_<NN>_AGG.txt).

    Dane sa juz w poprawnej orientacji fizycznej — NIE stosujemy korekt
    SWAP_GYRO_YZ ani zamiany lat/lon. Kolumny kanoniczne mapowane sa wprost
    na pola Telemetry; pelny zestaw kanalow (w tym nadmiarowe czujniki)
    dostepny jest w raw_columns pod oryginalnymi nazwami z pliku.
    """
    filepath = Path(filepath)
    meta = []
    header = None
    rows = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith("#"):
                meta.append(line)
                continue
            if header is None:
                header = line.split("\t")
                continue
            if line.strip():
                rows.append(line.split("\t"))

    if header is None:
        raise ValueError(f"{filepath.name}: brak wiersza naglowka")

    str_cols = {"faza_lotu"}
    cols = {}
    for j, name in enumerate(header):
        if name in str_cols:
            cols[name] = np.array(
                [r[j] if j < len(r) else "" for r in rows], dtype=object)
        else:
            vals = np.empty(len(rows), dtype=float)
            for i, r in enumerate(rows):
                try:
                    vals[i] = float(r[j])
                except (ValueError, IndexError):
                    vals[i] = np.nan
            cols[name] = vals

    def g(name):
        return cols.get(name, np.full(len(rows), np.nan))

    time = g("time_s")
    dt = np.diff(time)
    dt_mean = float(np.median(dt)) if len(dt) else float("nan")

    # Flaga "w locie": z fazy lotu, awaryjnie z flagi startu / znaku czasu
    faza = cols.get("faza_lotu")
    airborne = {"WZNOSZENIE", "SPADOCHRON_ZADANY", "OPADANIE", "PO_STARCIE"}
    if faza is not None:
        flag_flight = np.array([1.0 if str(v) in airborne else 0.0 for v in faza])
    elif "flaga_startu" in cols:
        flag_flight = (np.nan_to_num(g("flaga_startu")) >= 0.5).astype(float)
    else:
        flag_flight = (time >= 0).astype(float)

    # raw_columns: wszystkie kolumny pliku + aliasy znormalizowane dla zgodnosci
    raw_columns = dict(cols)
    raw_columns.update({
        "time": time,
        "acc_x": g("acc_x_g"), "acc_y": g("acc_y_g"), "acc_z": g("acc_z_g"),
        "gyro_x": g("gyro_x_degps"), "gyro_y": g("gyro_y_degps"),
        "gyro_z": g("gyro_z_degps"),
        "alt": g("alt_msl_m"), "vel": g("vel_horiz_ms"),
        "press_cham": g("press_chamber_bar"),
        "lat": g("lat_deg"), "lon": g("lon_deg"),
        "mag_y": g("mag_x_G"),
        "flag_flight": flag_flight,
    })

    if verbose:
        nmod = sum(1 for ln in meta if ln.startswith("#   "))
        print(f"Sparsowano (format zbiorczy): {filepath.name}")
        print(f"  Kolumny: {len(header)}  Probek: {len(rows)}")
        if len(time):
            print(f"  Czas: {time[0]:.3f}s do {time[-1]:.3f}s  "
                  f"Krok: {dt_mean:.4f}s ({1/dt_mean:.0f} Hz)")
        print(f"  Moduly zrodlowe: {nmod}  "
              f"[bez korekt osi — dane w poprawnej orientacji]")

    return Telemetry(
        time=time,
        acc_x=g("acc_x_g"), acc_y=g("acc_y_g"), acc_z=g("acc_z_g"),
        gyro_x=g("gyro_x_degps"), gyro_y=g("gyro_y_degps"),
        gyro_z=g("gyro_z_degps"),
        alt_onboard=g("alt_msl_m"), vel_onboard=g("vel_horiz_ms"),
        press_cham=g("press_chamber_bar"), press_amb=np.full(len(rows), np.nan),
        lat=g("lat_deg"), lon=g("lon_deg"),
        flag_flight=flag_flight,
        n_samples=len(rows), dt_mean=dt_mean,
        raw_columns=raw_columns,
    )


if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "ARTEMIDA_13_LOT.txt"
    tel = parse_telemetry(fp)
    print(f"\nPierwsze 3 probki acc_x: {tel.acc_x[:3]}")
    print(f"Pierwsze 3 probki gyro_x: {tel.gyro_x[:3]}")
