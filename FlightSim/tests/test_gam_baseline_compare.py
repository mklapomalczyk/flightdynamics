"""
tests/test_gam_baseline_compare.py
===================================
Test A/B usuniecia GAM z decka DATCOM.

Kat zaklinowania pletw byl zapisywany DWUKROTNIE: jako GAM= (diedra/incydencja
pletwy) ORAZ jako DELTA= w $DEFLCT, wiec DATCOM widzial ~2x zamierzone
zaklinowanie. Podrecznik definiuje $DEFLCT jako "the incidence angle for each
panel in each fin set", wiec zaklinowanie nalezy TAM — GAM zostalo usuniete.

Test ma dwie czesci:

  A) STRUKTURALNA — dziala zawsze, takze bez DATCOM:
     rozniца decka BEZ GAM i Z GAM to dokladnie jedna linia (GAM=...),
     a wariant Z GAM odtwarza zacommitowany deck bajt w bajt.

  B) NUMERYCZNA — uruchamia sie dopiero gdy istnieja pliki wyjsciowe
     datcom_nogam.out / datcom_gam.out (wygeneruj je na Windows przez
     `python run_control_datcom.py gam --run`). Porownuje wspolczynniki,
     zwlaszcza CLL, ktory jest napedzany zaklinowaniem.

Uruchomienie:  python3 tests/test_gam_baseline_compare.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datcom_io.config_reader import load_config
from datcom_io.missile_datcom_generator import generate_missile_datcom_input

PASS = 0
FAIL = 0
SKIP = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name} — {why}")


print("=" * 66)
print("TEST A/B: usuniecie GAM z decka DATCOM")
print("=" * 66)

# --------------------------------------------------------------------------
print("\nA. Roznica strukturalna decka (bez DATCOM)")
cfg = load_config(str(ROOT / "configurations" / "rocket_70mm_baseline.yaml"))
for f in cfg.fins:
    f.cant_angle = 0.6      # jak zwalidowany przypadek lotu 19

tmp = ROOT / "datcom_runs" / "_gamtest_tmp"
tmp.mkdir(parents=True, exist_ok=True)
p_no = generate_missile_datcom_input(cfg, tmp / "nogam.dat", emit_gam=False)
p_gm = generate_missile_datcom_input(cfg, tmp / "gam.dat",   emit_gam=True)

l_no = p_no.read_text().splitlines()
l_gm = p_gm.read_text().splitlines()
only_gm = [l for l in l_gm if l not in l_no]
only_no = [l for l in l_no if l not in l_gm]

check("deck Z GAM ma dokladnie 1 linie wiecej", len(l_gm) - len(l_no) == 1,
      f"({len(l_gm)} vs {len(l_no)})")
check("ta linia to GAM=",
      len(only_gm) == 1 and only_gm[0].strip().startswith("GAM="),
      f"({only_gm})")
check("deck BEZ GAM nie wnosi zadnej nowej linii", len(only_no) == 0,
      f"({only_no})")
check("DELTA obecne w obu (zaklinowanie nie zginelo)",
      any("DELTA1=" in l for l in l_no) and any("DELTA1=" in l for l in l_gm))

ref = ROOT / "datcom_runs" / "rocket_70mm_baseline_cant0p600" / "for005.dat"
if ref.exists():
    check("emit_gam=True odtwarza zacommitowany deck bajt w bajt",
          p_gm.read_text().strip() == ref.read_text().strip())
else:
    skip("porownanie z zacommitowanym deckiem", f"brak {ref}")

for p in (p_no, p_gm):
    p.unlink(missing_ok=True)
tmp.rmdir()

# --------------------------------------------------------------------------
print("\nB. Roznica numeryczna wynikow DATCOM")
run_dir = ROOT / "datcom_runs" / "rocket_70mm_baseline_gamtest"
o_no = run_dir / "datcom_nogam.out"
o_gm = run_dir / "datcom_gam.out"

if not (o_no.exists() and o_gm.exists()):
    skip("porownanie wspolczynnikow",
         "brak datcom_nogam.out / datcom_gam.out "
         "(uruchom na Windows: python run_control_datcom.py gam --run)")
else:
    from datcom_io.missile_datcom_reader import (missile_datcom_to_table_aero,
                                                 parse_missile_datcom_output)
    t_no = missile_datcom_to_table_aero(parse_missile_datcom_output(o_no))
    t_gm = missile_datcom_to_table_aero(parse_missile_datcom_output(o_gm))

    check("siatki alpha/Mach identyczne",
          np.allclose(t_no["alpha_deg"], t_gm["alpha_deg"]) and
          np.allclose(t_no["mach"], t_gm["mach"]))

    print("\n  Wzgledna zmiana wspolczynnikow (bez GAM vs z GAM):")
    for key in ("CN", "CM", "CA", "CLL"):
        a, b = np.asarray(t_no[key]), np.asarray(t_gm[key])
        scale = max(float(np.max(np.abs(b))), 1e-12)
        rel = float(np.max(np.abs(a - b))) / scale
        print(f"    {key:4s} max |zmiana| / max|z GAM| = {rel*100:7.2f} %")

    a, b = np.asarray(t_no["CLL"]), np.asarray(t_gm["CLL"])
    moved = not np.allclose(a, b, rtol=1e-6, atol=1e-12)
    check("CLL SIE ZMIENIL (usuniecie podwojonego zaklinowania)", moved,
          "— jesli nie, GAM nie wplywal na CLL i hipoteza podwojenia jest bledna")

    if moved:
        ratio = float(np.max(np.abs(b))) / max(float(np.max(np.abs(a))), 1e-12)
        print(f"    stosunek max|CLL| (z GAM / bez GAM) = {ratio:.3f}")
        print("    (blisko 2.0 => zaklinowanie faktycznie liczylo sie podwojnie)")

    check("CA praktycznie bez zmian (GAM nie zmienia oporu osiowego)",
          np.allclose(t_no["CA"], t_gm["CA"], rtol=0.02),
          "— duza zmiana CA bylaby zaskoczeniem, sprawdz deck")

print("\n" + "=" * 66)
total = PASS + FAIL
print(f"  Wynik: {PASS}/{total} testow zaliczonych" + (f", {SKIP} pominietych" if SKIP else ""))
print("  STATUS: OK" if FAIL == 0 else f"  STATUS: {FAIL} FAIL")
print("=" * 66)
sys.exit(1 if FAIL else 0)
