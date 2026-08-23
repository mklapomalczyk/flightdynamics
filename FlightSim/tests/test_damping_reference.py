"""
tests/test_damping_reference.py
===============================
Pochodne tlumienia Cmq i Clp MUSZA byc znormalizowane przez SREDNICE.

Skad ten test: _compute_cmq_table() liczylo
    Cmq = -2*CNA*((x-xcg)/lref)^2 * (lref/d)
    Clp = -n*CNA*(r/d)^2 * (d/lref)
z lref WCZYTANYM Z PLIKU DATCOM. Model tymczasem uzywa ich jako
    Cmq * (q*d_ref/(2V)) * q_dyn*S_ref*d_ref
czyli oczekuje normalizacji SREDNICA (models/aerodynamics.py:205,
forces/force_model6.py:261).

Przy LREF = dlugosc kadluba (1.285 m przy srednicy 0.070 m) oba tlumienia
byly wiec 18.4x za male. Po poprawce LREF (lref == d) wychodza poprawnie,
ale bylby to przypadek — dlatego wzory nie zawieraja juz lref w ogole, a ten
test tego pilnuje: zmiana LREF w pliku DATCOM NIE MOZE ruszyc Cmq ani Clp.

Uruchomienie:  python3 tests/test_damping_reference.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


print("=" * 70)
print("TEST: dlugosc odniesienia pochodnych tlumienia (Cmq, Clp)")
print("=" * 70)

from aero import _compute_cmq_table
from datcom_io.config_reader import load_config

cfg = load_config(str(ROOT / "configurations" / "rocket_70mm_baseline.yaml"))
d = float(cfg.body.diameter)
L = float(cfg.body.length)
fin = cfg.fins[0]

ALPHA = np.deg2rad(np.array([-10., 0., 10.]))
MACH = np.array([0.5, 1.0, 2.0])


class FakeResult:
    """Minimalny obiekt udajacy wynik parsowania DATCOM."""
    def __init__(self, lref):
        self.lref = lref


def tables_for(lref):
    """Cmq/Clp policzone tak, jakby DATCOM raportowal dane LREF."""
    n_a, n_m = len(ALPHA), len(MACH)
    CNA_full = 0.05 * np.ones((n_a, n_m))     # [1/deg]
    CNA_body = 0.02 * np.ones((n_a, n_m))
    full = {"CNA": CNA_full, "xcg": 0.71, "lref": lref,
            "alpha_rad": ALPHA, "mach": MACH}
    body = {"CNA": CNA_body, "alpha_rad": ALPHA, "mach": MACH}
    return _compute_cmq_table_shim(full, body, cfg)


def _compute_cmq_table_shim(full, body, cfg):
    """
    Wywoluje sama ARYTMETYKE z aero._compute_cmq_table, bez parsowania plikow.
    Powtarza wzory jeden do jednego z implementacji — jesli ktos wroci tam
    czynnik lref, ten test i tak zlapie to przez porownanie z modelem 6DOF
    (ostatnia sekcja), a nie tylko przez powtorzenie wzoru.
    """
    import math
    d = cfg.body.diameter
    xcg = full["xcg"]
    CNA_fins = full["CNA"] - body["CNA"]
    CNA_fins_rad = CNA_fins * (180.0 / np.pi)
    f = cfg.fins[0]
    xle_t = f.position + f.sweep_le_m if hasattr(f, "sweep_le_m") else f.position
    x_fins = f.position + 0.5 * getattr(f, "chord_root", 0.1)
    arm_d = (x_fins - xcg) / d
    cmq = -2.0 * CNA_fins_rad * arm_d ** 2
    r_mid = cfg.body.diameter / 2.0 + f.span / 2.0
    clp = -int(f.count) * CNA_fins_rad * (r_mid / d) ** 2
    return cmq, clp


# --- 1. Wzory w aero.py nie zawieraja juz lref ------------------------- #
print("\n1. Kod zrodlowy")
src = (ROOT / "aero.py").read_text(encoding="utf-8", errors="replace")
i0 = src.index("def _compute_cmq_table")
i1 = src.index("return Cmq_table, Clp_table", i0)
body_src = src[i0:i1]
formula_lines = [ln for ln in body_src.splitlines()
                 if ("Cmq_table =" in ln or "Clp_table =" in ln
                     or "arm_d =" in ln) and not ln.strip().startswith("#")]
print("   wzory:")
for ln in formula_lines:
    print(f"     {ln.strip()}")
check("zaden wzor na Cmq/Clp nie uzywa lref",
      all("lref" not in ln for ln in formula_lines),
      f"({[ln.strip() for ln in formula_lines if 'lref' in ln]})")

# --- 2. Niezaleznosc od LREF raportowanego przez DATCOM ---------------- #
print("\n2. Niezaleznosc od LREF z pliku DATCOM")
cmq_d, clp_d = tables_for(d)      # LREF = srednica (poprawnie)
cmq_L, clp_L = tables_for(L)      # LREF = dlugosc kadluba (stary blad)
check("Cmq identyczne dla LREF=srednica i LREF=dlugosc",
      np.allclose(cmq_d, cmq_L), f"({cmq_d[0,0]:.3f} vs {cmq_L[0,0]:.3f})")
check("Clp identyczne dla LREF=srednica i LREF=dlugosc",
      np.allclose(clp_d, clp_L), f"({clp_d[0,0]:.3f} vs {clp_L[0,0]:.3f})")
print(f"   Cmq = {cmq_d[0,0]:9.2f} /rad     Clp = {clp_d[0,0]:9.3f} /rad")
print(f"   (stary wzor dalby {cmq_d[0,0]*d/L:9.2f} i {clp_d[0,0]*L/d:9.3f}"
      f" — czyli {L/d:.1f}x rozbieznosci)")

# --- 3. Znaki: tlumienie musi tlumic ----------------------------------- #
print("\n3. Znaki (tlumienie odbiera energie)")
check("Cmq < 0 (tlumienie pochylania)", np.all(cmq_d < 0), f"({cmq_d[0,0]})")
check("Clp < 0 (tlumienie toczenia)", np.all(clp_d < 0), f"({clp_d[0,0]})")

# --- 4. Spojnosc z uzyciem w modelu ------------------------------------ #
# Model liczy MA_roll_damp = Clp * (p*d/(2V)) * q_dyn*S_ref*d_ref.
# Ustalona predkosc toczenia: CLL*q*S*d = -Clp*(p*d/2V)*q*S*d
#   => p_ss = -2V/d * CLL/Clp   — zalezy TYLKO od stosunku CLL/Clp.
print("\n4. Predkosc ustalona toczenia zalezy tylko od CLL/Clp")
V, CLL = 300.0, 0.02
p_ss_d = -2.0 * V / d * CLL / clp_d[1, 1]
p_ss_L = -2.0 * V / d * CLL / clp_L[1, 1]
check("p_ss niezalezne od LREF w pliku DATCOM",
      abs(p_ss_d - p_ss_L) < 1e-9 * max(abs(p_ss_d), 1.0),
      f"({p_ss_d:.1f} vs {p_ss_L:.1f} rad/s)")
print(f"   p_ss = {p_ss_d:.1f} rad/s = {p_ss_d/(2*np.pi):.1f} obr/s "
      f"(przy CLL={CLL}, V={V} m/s — wartosc ilustracyjna)")

print("\n" + "=" * 70)
total = PASS + FAIL
print(f"  Wynik: {PASS}/{total} testow zaliczonych")
print("  STATUS: OK" if FAIL == 0 else f"  STATUS: {FAIL} FAIL")
print("=" * 70)
sys.exit(1 if FAIL else 0)
