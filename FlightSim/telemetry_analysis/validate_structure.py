"""
validate_structure.py
=====================
Sprawdza spojnosc struktury plikow telemetrycznych ARTEMIDA.

Porownuje naglowki kolumn miedzy plikami, wykrywa:
- rozna liczbe kolumn
- rozna kolejnosc kolumn
- brakujace kluczowe kolumny
- rozne czestotliwosci probkowania

Uruchomienie:
    python validate_structure.py            # wszystkie pliki ARTEMIDA_*.txt
    python validate_structure.py 13 14 15   # konkretne loty
"""

import sys
import csv
from pathlib import Path

# Import z tego samego pakietu
sys.path.insert(0, str(Path(__file__).parent))
from telemetry_parser import (COLUMN_ALIASES, _build_column_map,
                              resolve_data_file, get_data_dir,
                              detect_delimiter, read_config_rows,
                              is_aggregated_format, parse_telemetry)


# Kluczowe kolumny wymagane do rekonstrukcji IMU
REQUIRED = ["time", "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z"]
# Opcjonalne ale przydatne
OPTIONAL = ["lat", "lon", "alt", "vel", "press_cham", "flag_flight"]


def read_header(filepath):
    """Czyta tylko naglowek pliku (auto-detekcja separatora)."""
    delim = detect_delimiter(filepath)
    with open(filepath, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if delim is None:
        return first.split()
    return [h.strip() for h in first.rstrip("\n\r").split(delim)]


def count_rows_and_rate(filepath):
    """Liczy wiersze i szacuje czestotliwosc probkowania."""
    times = []
    delim = detect_delimiter(filepath)
    with open(filepath, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    header = lines[0].split() if delim is None else lines[0].split(delim)
    t_idx = None
    for alias in COLUMN_ALIASES["time"]:
        if alias in header:
            t_idx = header.index(alias)
            break
    n_rows = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        row = line.split() if delim is None else line.split(delim)
        n_rows += 1
        if t_idx is not None and len(row) > t_idx:
            try:
                times.append(float(row[t_idx]))
            except ValueError:
                pass
    rate = None
    t_span = None
    if len(times) > 2:
        import statistics
        dts = [times[i+1]-times[i] for i in range(len(times)-1)]
        dt_med = statistics.median(dts)
        rate = 1.0 / dt_med if dt_med > 0 else None
        t_span = (times[0], times[-1])
    return n_rows, rate, t_span


def _validate_aggregated(filepath, result, verbose=True):
    """Walidacja pliku zbiorczego (nowy format) przez parser."""
    import numpy as np
    tel = parse_telemetry(filepath, verbose=False)
    present = {
        "time": tel.time, "acc_x": tel.acc_x, "acc_y": tel.acc_y,
        "acc_z": tel.acc_z, "gyro_x": tel.gyro_x, "gyro_y": tel.gyro_y,
        "gyro_z": tel.gyro_z,
    }
    opt = {"lat": tel.lat, "lon": tel.lon, "alt": tel.alt_onboard,
           "vel": tel.vel_onboard, "press_cham": tel.press_cham,
           "flag_flight": tel.flag_flight}

    def _absent(a):
        return a is None or (hasattr(a, "size") and
                             (a.size == 0 or np.all(np.isnan(a.astype(float)))
                              if a.dtype.kind == "f" else False))

    result["missing_required"] = [k for k, v in present.items() if _absent(v)]
    result["missing_optional"] = [k for k, v in opt.items() if _absent(v)]
    # liczba kolumn = pelen zestaw kanalow w pliku zbiorczym
    result["n_columns"] = len(tel.raw_columns)
    result["n_rows"] = tel.n_samples
    result["rate_hz"] = (1.0 / tel.dt_mean) if tel.dt_mean else None
    result["t_span"] = (float(tel.time[0]), float(tel.time[-1])) if tel.n_samples else None
    result["column_order_match"] = None
    result["ok"] = (len(result["missing_required"]) == 0)

    if verbose:
        status = "OK" if result["ok"] else "PROBLEM"
        print(f"  [{status}] {filepath.name}  [format zbiorczy]")
        rate = result["rate_hz"]
        print(f"      kanaly={result['n_columns']}  wiersze={result['n_rows']}  "
              f"czestotliwosc={rate:.0f}Hz" if rate else "      (brak czasu)")
        if result["t_span"]:
            print(f"      czas: {result['t_span'][0]:.2f}s do {result['t_span'][1]:.2f}s")
        if result["missing_required"]:
            print(f"      ! BRAK kluczowych: {result['missing_required']}")
        if result["missing_optional"]:
            print(f"      ~ brak opcjonalnych: {result['missing_optional']} "
                  f"(np. lot bez GPS/MGLO)")
    return result


def validate_file(filepath, reference_header=None, verbose=True):
    """
    Waliduje pojedynczy plik. Zwraca slownik z wynikami.
    """
    filepath = Path(filepath)
    result = {
        "file": filepath.name,
        "exists": filepath.exists(),
        "n_columns": None,
        "n_rows": None,
        "rate_hz": None,
        "t_span": None,
        "missing_required": [],
        "missing_optional": [],
        "column_order_match": None,
        "ok": False,
    }

    if not filepath.exists():
        if verbose:
            print(f"  [BRAK PLIKU] {filepath.name}")
        return result

    # Nowy format zbiorczy (loty 22+): inna struktura — walidujemy przez parser,
    # ktory mapuje kanoniczne kolumny na pola Telemetry. Nie porownujemy kolejnosci
    # kolumn z referencja starego formatu.
    if is_aggregated_format(filepath):
        return _validate_aggregated(filepath, result, verbose)

    header = read_header(filepath)
    col_map = _build_column_map(header)
    result["n_columns"] = len(header)

    # Kluczowe kolumny
    result["missing_required"] = [r for r in REQUIRED if r not in col_map]
    result["missing_optional"] = [o for o in OPTIONAL if o not in col_map]

    # Liczba wierszy i czestotliwosc
    n_rows, rate, t_span = count_rows_and_rate(filepath)
    result["n_rows"] = n_rows
    result["rate_hz"] = rate
    result["t_span"] = t_span

    # Porownanie kolejnosci kolumn z referencja
    if reference_header is not None:
        result["column_order_match"] = (header == reference_header)

    result["ok"] = (len(result["missing_required"]) == 0)

    if verbose:
        status = "OK" if result["ok"] else "PROBLEM"
        print(f"  [{status}] {filepath.name}")
        print(f"      kolumny={result['n_columns']}  wiersze={n_rows}  "
              f"czestotliwosc={rate:.0f}Hz" if rate else "      (brak czasu)")
        if t_span:
            print(f"      czas: {t_span[0]:.2f}s do {t_span[1]:.2f}s")
        if result["missing_required"]:
            print(f"      ! BRAK kluczowych kolumn: {result['missing_required']}")
        if result["missing_optional"]:
            print(f"      ~ brak opcjonalnych: {result['missing_optional']}")
        if result["column_order_match"] is False:
            print(f"      ~ inna kolejnosc kolumn niz referencja (parser to obsluzy)")

    return result


def load_flights_to_analyze(config_path="configs.txt"):
    """Wczytuje numery lotow oznaczone 'yes' w configs.txt (odporny czytnik)."""
    flights = []
    rows = read_config_rows(config_path)
    if not rows:
        print(f"[WARN] Brak lub pusty {config_path}")
        return flights
    # Znajdz nazwe kolumny "to analyze?"
    analyze_key = next((k for k in rows[0] if "analyze" in k.lower()), None)
    for row in rows:
        try:
            fno = int(str(row.get("flight no", "")).strip())
        except (ValueError, TypeError):
            continue
        if analyze_key and str(row.get(analyze_key, "")).strip().lower() == "yes":
            flights.append(fno)
    return flights


def main():
    data_dir = get_data_dir()

    if len(sys.argv) > 1:
        flight_nos = [int(a) for a in sys.argv[1:]]
    else:
        flight_nos = load_flights_to_analyze(str(resolve_data_file("configs.txt")))
        print(f"Loty do analizy z configs.txt (yes): {flight_nos}\n")

    print("="*60)
    print("WALIDACJA STRUKTURY PLIKOW TELEMETRYCZNYCH")
    print("="*60)

    results = []
    reference_header = None
    for fno in flight_nos:
        fname = resolve_data_file(f"ARTEMIDA_{fno}_LOT.txt")
        # Referencja kolejnosci kolumn tylko ze starego formatu (zbiorczy ma inna strukture)
        if (reference_header is None and fname.exists()
                and not is_aggregated_format(fname)):
            reference_header = read_header(fname)
            print(f"\nReferencja kolumn: {fname.name} ({len(reference_header)} kolumn)\n")
        res = validate_file(fname, reference_header)
        res["flight_no"] = fno
        results.append(res)

    # Podsumowanie
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    ok_count = sum(1 for r in results if r["ok"])
    print(f"Plikow OK: {ok_count}/{len(results)}")

    # Spojnosc kolumn
    n_cols = set(r["n_columns"] for r in results if r["n_columns"])
    if len(n_cols) == 1:
        print(f"Wszystkie pliki maja {n_cols.pop()} kolumn — spojne")
    else:
        print(f"! Rozna liczba kolumn miedzy plikami: {sorted(n_cols)}")

    rates = set(round(r["rate_hz"]) for r in results if r["rate_hz"])
    if len(rates) == 1:
        print(f"Wszystkie pliki: {rates.pop()} Hz — spojne")
    else:
        print(f"! Rozne czestotliwosci: {sorted(rates)} Hz")

    problems = [r for r in results if not r["ok"]]
    if problems:
        print(f"\nPliki z problemami:")
        for r in problems:
            print(f"  Lot {r['flight_no']}: {r['missing_required'] or 'brak pliku'}")

    return results


if __name__ == "__main__":
    main()
