"""
aggregate_flight_data.py
========================
Scala rozproszone pliki modulow telemetrii (nowy format ARTEMIDA, loty 22+)
w jeden plik zbiorczy na lot, ktory wczytuje telemetry_parser.

Nowy format rejestratora dzieli dane jednego lotu na osobne pliki wg modulu
nadawczego (kazdy ma wlasna czestotliwosc ramek i wlasny licznik ramek, ale
WSPOLNA os czasu `czas_od_startu_s [s]` liczona od pierwszej ramki z flaga
startu):

    lot_<NN>_MGLO_PRZELICZONE.txt   ID 1, 200 Hz  — modul glowny:
                                     GNSS, scalone IMU (acc_R*, giro_R*),
                                     mag_RX, cisnienie komory, faza lotu
    lot_<NN>_MPOM_PRZELICZONE.txt   ID 2, ~340 Hz — 4 akcelerometry
                                     (ZP/ZM/YP/YM) x3 osie + zyroskop X/Y/Z
    lot_<NN>_MMAG_PRZELICZONE.txt   ID 3, 500 Hz  — 4 magnetometry x3 osie
    lot_<NN>_MPCS_PRZELICZONE.txt   ID 4, 500 Hz  — cisnienie komory (LSB+bar)

Nie kazdy lot ma komplet modulow (np. lot 23 nie ma MGLO).

Scalanie
--------
1. Os odniesienia (siatka czasu): MGLO gdy jest, inaczej MPOM.
2. Kanaly z pozostalych modulow interpolowane liniowo na siatke odniesienia
   (np.interp po `czas_od_startu_s`). Wszystkie surowe kanaly zachowane pod
   nazwami z prefiksem modulu (mpom_*, mmag_*, mpcs_*).
3. Kanoniczna triada IMU wymagana przez skrypty analizy:
     - loty z MGLO (22/24/25): acc_x/y/z = acc_RX/RY/RZ, gyro = giro_RX/RY/RZ
     - lot bez MGLO (23):      acc_x/y/z = srednia 4 akcelerometrow MPOM,
                               gyro = giro_X/Y/Z z MPOM
   Srednia 4 symetrycznie rozmieszczonych akcelerometrow (ZP/ZM, YP/YM) znosi
   w pierwszym rzedzie skladowe odsrodkowe/styczne, zostawiajac sile wlasciwa
   korpusu. Zgodnosc tej sredniej z MGLO acc_R* jest weryfikowana na lotach,
   gdzie dostepne sa oba (patrz --check).

UWAGA: nowy format ma POPRAWNIE opisane kolumny (szerokosc=lat, dlugosc=lon)
i — zakladamy — poprawne osie zyroskopu. Dlatego plik zbiorczy jest zapisany
w orientacji fizycznej gotowej do uzycia; telemetry_parser rozpoznaje znacznik
formatu i NIE stosuje korekt osi/lat-lon przewidzianych dla starego sprzetu.

Uzycie
------
    python telemetry_analysis/aggregate_flight_data.py            # wszystkie wykryte loty
    python telemetry_analysis/aggregate_flight_data.py 22 24      # wybrane loty
    python telemetry_analysis/aggregate_flight_data.py --check    # + walidacja sredniej MPOM vs MGLO
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telemetry_parser import get_data_dir, AGG_FORMAT_MARKER  # noqa: E402


# Kolejnosc preferencji modulu odniesienia (siatka czasu)
REFERENCE_PREFERENCE = ["MGLO", "MPOM", "MMAG", "MPCS"]

# Kolumny wspolne na poczatku kazdego pliku modulu (pomijane przy prefiksowaniu)
LEAD_COLS = {"rakieta", "numer_lotu", "nadawca", "flaga_startu",
             "nr_ramki", "czas_od_startu_s"}

TIME_COL = "czas_od_startu_s"


# --------------------------------------------------------------------------
def _strip_unit(name):
    """"acc_RX [g]" -> "acc_RX"; "czas_od_startu_s [s]" -> "czas_od_startu_s"."""
    return name.split("[")[0].strip()


def read_module(path):
    """
    Wczytuje jeden plik modulu.

    Zwraca dict:
        meta        — linie komentarza (#) jako lista
        rate        — ramek/s (float) jesli podane w naglowku
        module      — kod modulu (MGLO/MPOM/...) z kolumny 'nadawca'
        cols        — lista nazw kolumn (bez jednostek)
        units       — dict nazwa->jednostka (np. 'acc_RX'->'g')
        data        — dict nazwa->np.ndarray (float; faza_lotu zostaje str)
        time        — np.ndarray czasu [s]
        n           — liczba wierszy
    """
    path = Path(path)
    meta, rate = [], None
    header = None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith("#"):
                meta.append(line)
                if "Ramek na sekunde" in line:
                    try:
                        rate = float(line.split(":")[1])
                    except (ValueError, IndexError):
                        pass
                continue
            if header is None:
                header = line.split("\t")
                continue
            if line.strip():
                rows.append(line.split("\t"))

    if header is None:
        raise ValueError(f"{path.name}: brak wiersza naglowka")

    cols = [_strip_unit(h) for h in header]
    units = {}
    for h in header:
        c = _strip_unit(h)
        if "[" in h:
            units[c] = h[h.index("[") + 1: h.rindex("]")] if "]" in h else ""
    ncol = len(cols)

    # faza_lotu jest tekstowa — reszta liczbowa
    str_cols = {"faza_lotu", "rakieta", "nadawca"}
    data = {}
    for j, c in enumerate(cols):
        if c in str_cols:
            data[c] = np.array(
                [r[j] if j < len(r) else "" for r in rows], dtype=object)
        else:
            vals = np.empty(len(rows), dtype=float)
            for i, r in enumerate(rows):
                try:
                    vals[i] = float(r[j])
                except (ValueError, IndexError):
                    vals[i] = np.nan
            data[c] = vals

    module = None
    if "nadawca" in data and len(rows):
        module = str(data["nadawca"][0])
    time = data[TIME_COL]

    # Upewnij sie, ze czas jest rosnacy (interpolacja tego wymaga)
    order = np.argsort(time, kind="stable")
    if not np.all(order == np.arange(len(time))):
        for c in data:
            data[c] = data[c][order]
        time = data[TIME_COL]

    return dict(meta=meta, rate=rate, module=module, cols=cols,
                units=units, data=data, time=time, n=len(rows), path=path)


def discover_flights(data_dir):
    """Wykrywa numery lotow z plikow lot_<NN>_<MOD>_PRZELICZONE.txt."""
    flights = {}
    for p in sorted(Path(data_dir).glob("lot_*_PRZELICZONE.txt")):
        parts = p.name.split("_")
        try:
            fno = int(parts[1])
            mod = parts[2]
        except (IndexError, ValueError):
            continue
        flights.setdefault(fno, {})[mod] = p
    return dict(sorted(flights.items()))


def _interp(ref_time, mod_time, values):
    """Interpolacja liniowa na siatke ref_time; poza zakresem -> NaN."""
    good = np.isfinite(mod_time) & np.isfinite(values)
    if good.sum() < 2:
        return np.full(ref_time.shape, np.nan)
    out = np.interp(ref_time, mod_time[good], values[good],
                    left=np.nan, right=np.nan)
    return out


# --------------------------------------------------------------------------
def aggregate_flight(fno, module_paths, verbose=True):
    """
    Scala moduly jednego lotu. Zwraca (colnames, columns, meta_lines).
    columns: dict nazwa_kanoniczna -> np.ndarray (lub object dla faza_lotu).
    """
    modules = {mod: read_module(p) for mod, p in module_paths.items()}

    # Modul odniesienia
    ref_mod = next((m for m in REFERENCE_PREFERENCE if m in modules), None)
    if ref_mod is None:
        raise ValueError(f"Lot {fno}: brak znanego modulu odniesienia")
    ref = modules[ref_mod]
    ref_time = ref["time"]
    n = len(ref_time)

    out = {}           # nazwa -> tablica
    order = []         # kolejnosc kolumn w pliku

    def add(name, arr):
        if name not in out:
            order.append(name)
        out[name] = arr

    # --- kolumny bazowe ---
    add("time_s", ref_time)
    add("flight", np.full(n, fno, dtype=int))
    add("nr_ramki_ref", ref["data"].get("nr_ramki", np.full(n, np.nan)))
    add("flaga_startu", ref["data"].get("flaga_startu", np.full(n, np.nan)))

    # faza lotu (tylko MGLO); gdy brak -> z flaga_startu
    if "faza_lotu" in ref["data"]:
        add("faza_lotu", ref["data"]["faza_lotu"])
    else:
        fs = ref["data"].get("flaga_startu", np.zeros(n))
        phase = np.where(np.nan_to_num(fs) >= 0.5, "PO_STARCIE", "PRZED_STARTEM")
        add("faza_lotu", phase.astype(object))

    # --- kanoniczna triada IMU ---
    if ref_mod == "MGLO":
        for ax, src in (("x", "acc_RX"), ("y", "acc_RY"), ("z", "acc_RZ")):
            add(f"acc_{ax}_g", ref["data"][src])
        for ax, src in (("x", "giro_RX"), ("y", "giro_RY"), ("z", "giro_RZ")):
            add(f"gyro_{ax}_degps", ref["data"][src])
        add("acc_RXO_degps", ref["data"].get("acc_RXO", np.full(n, np.nan)))
        add("mag_x_G", ref["data"].get("mag_RX", np.full(n, np.nan)))
        imu_source = "MGLO (scalone pokladowo acc_R*/giro_R*)"
    else:
        # Lot bez MGLO (np. 23): triada z MPOM. UWAGA — 4 akcelerometry MPOM sa
        # zamontowane w ROZNYCH orientacjach (sciany ZP/ZM/YP/YM), wiec zwykla
        # srednia po osiach czujnika NIE daje poprawnej triady korpusu. Ustalono
        # empirycznie (loty 22/24/25, gdzie jest tez MGLO):
        #   - os PODLUZNA korpusu (thrust, = MGLO acc_RX) odpowiada SREDNIEJ osi
        #     Z czujnikow MPOM; korelacja ~0.98, nachylenie ~1.0 — WIARYGODNA.
        #   - osie poprzeczne sa wymieszane przez montaz i bez geometrii montazu
        #     nie da sie ich jednoznacznie odtworzyc — traktowane orientacyjnie.
        #   - zyroskop MPOM: brak modulu odniesienia do walidacji; dodatkowo
        #     znaczniki czasu MPOM sa OSZACOWANE (nie mierzone), wiec szybkie
        #     sygnaly obrotowe maja ograniczona wiarygodnosc.
        mp = modules.get("MPOM")
        if mp is None:
            raise ValueError(f"Lot {fno}: brak MGLO i MPOM — nie ma zrodla IMU")

        def smean(sax):   # srednia danej osi czujnika po 4 modulach
            return np.nanmean(
                np.vstack([mp["data"][f"acc_{m}_{sax}"]
                           for m in ("ZP", "ZM", "YP", "YM")]), axis=0)

        add("acc_x_g", smean("Z"))        # PODLUZNA (thrust) — wiarygodna
        add("acc_y_g", smean("X"))        # poprzeczna — orientacyjna
        add("acc_z_g", smean("Y"))        # poprzeczna — orientacyjna
        for ax in ("X", "Y", "Z"):
            add(f"gyro_{ax.lower()}_degps", mp["data"][f"giro_{ax}"])
        add("mag_x_G", np.full(n, np.nan))
        imu_source = ("MPOM bez MGLO: acc_x (podluzna)=srednia osi Z czujnikow "
                      "[wiarygodna, ~0.98 korelacji z MGLO]; acc_y/acc_z "
                      "(poprzeczne) i giro — ORIENTACYJNE (nieznana geometria "
                      "montazu, czas MPOM oszacowany)")

    # --- GNSS / nawigacja (tylko MGLO) ---
    gnss_map = {
        "lat_deg": "szerokosc", "lon_deg": "dlugosc",
        "alt_msl_m": "wysokosc_msl", "alt_ellip_m": "wysokosc_elipsoida",
        "vel_horiz_ms": "predkosc_pozioma", "vel_north_ms": "predkosc_polnoc",
        "vel_east_ms": "predkosc_wschod", "vel_down_ms": "predkosc_dol",
        "hacc_m": "dokladnosc_pozioma", "vacc_m": "dokladnosc_pionowa",
        "fix_type": "typ_fixa", "n_sat": "liczba_satelitow", "itow_ms": "itow",
    }
    if ref_mod == "MGLO":
        for out_name, src in gnss_map.items():
            add(out_name, ref["data"].get(src, np.full(n, np.nan)))
    else:
        for out_name in gnss_map:
            add(out_name, np.full(n, np.nan))

    # --- cisnienie komory (kanoniczne): MGLO gdy jest, inaczej z MPCS ---
    if ref_mod == "MGLO" and "cisnienie_komory" in ref["data"]:
        add("press_chamber_bar", ref["data"]["cisnienie_komory"])
    elif "MPCS" in modules:
        mp = modules["MPCS"]
        add("press_chamber_bar",
            _interp(ref_time, mp["time"], mp["data"]["cisnienie_komory"]))
    else:
        add("press_chamber_bar", np.full(n, np.nan))

    # --- surowe kanaly z pozostalych modulow (prefiks modulu) ---
    for mod, m in modules.items():
        if mod == ref_mod:
            # Z modulu odniesienia dokladamy tylko kanaly jeszcze nie-kanoniczne
            # (np. pelen blok GNSS juz dodany); surowe IMU zostawiamy jako ref_*
            for c in m["cols"]:
                if c in LEAD_COLS or c == "faza_lotu":
                    continue
                if c.startswith("zew1_b"):   # nieodkodowane bajty payloadu — pomijamy
                    continue
                pname = f"{mod.lower()}_{c}"
                if pname not in out and f"ref_{c}" not in out:
                    add(f"ref_{c}", m["data"][c])
            continue
        for c in m["cols"]:
            if c in LEAD_COLS or c == "faza_lotu":
                continue
            if c.startswith("zew1_b"):
                continue
            add(f"{mod.lower()}_{c}", _interp(ref_time, m["time"], m["data"][c]))

    # --- blok komentarza / proweniencji ---
    meta = [
        AGG_FORMAT_MARKER,
        f"# Lot: {fno}   Rakieta: Artemida",
        f"# Modul odniesienia (siatka czasu): {ref_mod} "
        f"({ref['rate']:.2f} Hz, {n} probek)" if ref["rate"] else
        f"# Modul odniesienia (siatka czasu): {ref_mod} ({n} probek)",
        f"# Zrodlo kanonicznej triady IMU: {imu_source}",
        "# Moduly zrodlowe:",
    ]
    for mod in REFERENCE_PREFERENCE:
        if mod in modules:
            m = modules[mod]
            rate = f"{m['rate']:.2f} Hz" if m["rate"] else "? Hz"
            meta.append(f"#   {mod}: {m['path'].name}  ({rate}, {m['n']} probek)")
    meta.append("# Kanaly z modulow innych niz odniesienia interpolowane liniowo "
                "na os czasu odniesienia.")
    meta.append("# Os czasu: sekundy od pierwszej ramki z flaga startu "
                "(wspolna dla wszystkich modulow).")

    if verbose:
        miss = [m for m in REFERENCE_PREFERENCE if m not in modules]
        print(f"Lot {fno}: ref={ref_mod}, {n} probek, "
              f"{len(order)} kolumn, moduly={list(modules)}"
              + (f", BRAK={miss}" if miss else ""))

    return order, out, meta


def write_aggregated(fno, order, columns, meta, data_dir):
    """Zapisuje plik zbiorczy ARTEMIDA_<NN>_AGG.txt (TSV)."""
    out_path = Path(data_dir) / f"ARTEMIDA_{fno}_AGG.txt"
    n = len(columns["time_s"])

    def fmt(name, i):
        v = columns[name][i]
        if isinstance(v, str):
            return v
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:.6g}"
        return str(v)

    with open(out_path, "w", encoding="utf-8") as fh:
        for line in meta:
            fh.write(line + "\n")
        fh.write("\t".join(order) + "\n")
        for i in range(n):
            fh.write("\t".join(fmt(name, i) for name in order) + "\n")
    return out_path


def check_mpom_vs_mglo(fno, module_paths):
    """Walidacja: srednia 4 akcelerometrow MPOM vs scalone MGLO acc_R*.

    Potwierdza, ze fallback dla lotu 23 (srednia MPOM) jest zgodny ze
    scaleniem pokladowym tam, gdzie dostepne sa oba moduly.
    """
    if "MGLO" not in module_paths or "MPOM" not in module_paths:
        print(f"Lot {fno}: pomijam check (brak MGLO lub MPOM)")
        return
    mg = read_module(module_paths["MGLO"])
    mp = read_module(module_paths["MPOM"])
    print(f"\nLot {fno}: zgodnosc sredniej MPOM vs MGLO acc_R* "
          f"(interp MGLO->MPOM):")
    for ax, src in (("X", "acc_RX"), ("Y", "acc_RY"), ("Z", "acc_RZ")):
        mean_mpom = np.nanmean(
            np.vstack([mp["data"][f"acc_{m}_{ax}"]
                       for m in ("ZP", "ZM", "YP", "YM")]), axis=0)
        mglo_on_mp = _interp(mp["time"], mg["time"], mg["data"][src])
        d = mean_mpom - mglo_on_mp
        good = np.isfinite(d)
        rms = float(np.sqrt(np.nanmean(d[good] ** 2))) if good.any() else np.nan
        bias = float(np.nanmean(d[good])) if good.any() else np.nan
        print(f"   {ax}: RMS={rms:.4f} g  bias={bias:+.4f} g")


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flights", nargs="*", type=int,
                    help="numery lotow (domyslnie: wszystkie wykryte)")
    ap.add_argument("--check", action="store_true",
                    help="walidacja sredniej MPOM vs MGLO")
    ap.add_argument("--data-dir", default=None, help="katalog danych")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else get_data_dir()
    found = discover_flights(data_dir)
    if not found:
        print(f"Brak plikow lot_*_PRZELICZONE.txt w {data_dir}")
        return 1

    targets = args.flights or list(found)
    print(f"Katalog danych: {data_dir}")
    print(f"Wykryte loty (nowy format): {list(found)}")
    print(f"Do scalenia: {targets}\n")

    for fno in targets:
        if fno not in found:
            print(f"Lot {fno}: brak plikow — pomijam")
            continue
        order, cols, meta = aggregate_flight(fno, found[fno])
        out = write_aggregated(fno, order, cols, meta, data_dir)
        print(f"  -> {out.name}")
        if args.check:
            check_mpom_vs_mglo(fno, found[fno])

    print("\nGotowe. Pliki zbiorcze rozpoznawane automatycznie przez "
          "telemetry_parser (resolve_data_file / parse_telemetry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
