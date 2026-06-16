# Modul analizy danych telemetrycznych ARTEMIDA

Rekonstrukcja trajektorii lotu rakiety z danych IMU (akcelerometry +
zyroskopy) metoda strapdown inertial navigation, z opcjonalna fuzja GPS.

## Gdzie umiescic pliki

Skopiuj folder `telemetry_analysis/` do `FlightSim/field_test_data/`:

```
FlightSim/
└── field_test_data/
    ├── configs.txt                     <- konfiguracje lotow (juz masz)
    ├── ARTEMIDA_13_LOT.txt             <- dane telemetryczne (juz masz)
    ├── ARTEMIDA_14_LOT.txt
    ├── ...
    ├── telemetry_analysis/             <- TEN folder skopiuj tutaj
    │   ├── __init__.py
    │   ├── telemetry_parser.py
    │   ├── imu_reconstruction.py
    │   ├── gps_fusion.py
    │   ├── reconstruct_flight.py
    │   ├── validate_structure.py
    │   ├── batch_process.py
    │   └── test_axis_assignment.py
    └── results/                        <- tworzony automatycznie
        ├── reconstruction_flight_*.png
        └── validation_summary.csv
```

## Jak uruchomic

Wszystkie komendy uruchamiaj **z katalogu `field_test_data`**:

### Walidacja struktury wszystkich plikow
```
python telemetry_analysis/validate_structure.py
```
Sprawdza czy wszystkie pliki maja te same kolumny, czestotliwosc, itd.

### Analiza pojedynczego lotu
```
python telemetry_analysis/reconstruct_flight.py 13
```
Generuje `reconstruction_flight_13.png` z pelna rekonstrukcja.

### Przetworzenie wszystkich lotow (yes w configs.txt)
```
python telemetry_analysis/batch_process.py
```
Waliduje, rekonstruuje wszystkie loty, zapisuje wykresy + zbiorczy CSV.

## Konfiguracja (telemetry_parser.py)

Na gorze pliku sa flagi do latwej zmiany:

```python
SWAP_GYRO_YZ = True    # zamien osie pitch/yaw zyroskopu (ARTEMIDA)
SWAP_ACC_YZ  = False   # akcelerometr — osie poprawne
```

UWAGA: w telemetrii ARTEMIDA wykryto dwie anomalie (obsluzone automatycznie):
1. Osie zyroskopu Y i Z sa zamienione (ustalone empirycznie, RMS 535m -> 59m)
2. Kolumny lat/lon sa blednie opisane (parser koryguje przy wczytaniu)

## Metodyka

1. **Kalibracja** — bias zyroskopu i skala akcelerometru z danych spoczynkowych
   (rakieta nieruchoma na wyrzutni, t < -1s).
2. **Orientacja poczatkowa** — z azymutu i elewacji wyrzutni (configs.txt).
3. **Strapdown INS** — calkowanie zyroskopow (kwaternion) -> obrot przyspieszen
   do ukladu ENU -> odjecie grawitacji -> calkowanie do predkosci i pozycji.
4. **Fuzja GPS** — filtr komplementarny korygujacy dryf IMU danymi GPS
   (bramkowanie w fazie napedowej gdzie GPS niewiarygodny).

## Walidacja dla lotu 13

| Metryka  | IMU    | IMU+GPS | GPS (odniesienie) |
|----------|--------|---------|-------------------|
| Apogeum  | 1018m  | 1108m   | 1113m             |
| Vmax     | 295m/s | 288m/s  | 212m/s (ograniczony) |

GPS cywilny ogranicza Vmax w fazie napedowej — rzeczywista predkosc
maksymalna jest blizsza rekonstrukcji IMU (~290 m/s).
